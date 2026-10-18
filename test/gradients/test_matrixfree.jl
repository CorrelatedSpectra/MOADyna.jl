@testset "Gradients with matrix-free operators" begin
    using LinearAlgebra, SparseArrays
    G = MOADyna.Gradients
    s = SpinSite{1}(:s)
    b = basis(Hilbert(:s => s))
    sx, sy, sz = S(s)
    ops = [sz * sz + 0.6sz, sx + 0.2sy]
    terms = [matrixfree(op, b; threaded=false) for op in ops]
    cmap = G.AffineMap([1.0, 0.0], reshape([0.0, 1.0], 2, 1))
    model = G.AffineModel(terms, cmap, [:mix])
    reference = G.AffineModel(sparse.(terms), cmap, [:mix])
    @test_throws ArgumentError G.AffineModel(Hermitian.(terms), cmap, [:mix])
    θ = [0.3]
    H = G.hamiltonian(model, θ)
    @test H isa MatrixFreeOperator
    @test G.dhamiltonian(model, θ, [1.0]) isa MatrixFreeOperator
    @test Matrix(H) ≈ Matrix(G.hamiltonian(reference, θ))
    E, ψ, gap = G.groundstate(model, θ; dense_max=0)
    Er, ψr, gapr = G.groundstate(reference, θ)
    @test E ≈ Er atol=1e-11
    @test gap ≈ gapr atol=1e-11
    @test G.grad_E0(model, θ, ψ) ≈ G.grad_E0(reference, θ, ψr) atol=1e-10
    dψ = G.sternheimer_dψ0(model, θ, E, ψ, [1.0])
    dH = G.dhamiltonian(model, θ, [1.0])
    rhs = dH * ψ - ψ * dot(ψ, dH * ψ)
    @test norm(H * dψ - E * dψ + rhs) < 1e-10
    frequencies = collect(range(0.1, 1.5; length=3))
    @test G.response_jvp(model, θ, [1.0], ψ, frequencies; E0=E, Γ=0.3) ≈
          G.response_jvp(reference, θ, [1.0], ψ, frequencies; E0=E, Γ=0.3) atol=1e-10
    @test_throws ArgumentError G.groundstate_manifold(model, θ)
    @test_throws ArgumentError terms[1] + matrixfree(ops[2], basis(Hilbert(:s => s)))
    wrong = matrixfree(ops[2], basis(Hilbert(:s => s)))
    @test_throws ArgumentError G.XASGradientModel(model, model, (wrong,), (terms[2],),
        Matrix{Float64}(I, 3, 3), frequencies; Γ=0.3)

    # Combine entries before multiplying by the input, preserving cancellation.
    f = FermionSite{1}(:f)
    bf = basis(Hilbert(:f => f), n_fermion([f]) == 1)
    A = matrixfree(1e308n(f, 1), bf)
    @test (A - A) * [2.0] == [0.0]
end
