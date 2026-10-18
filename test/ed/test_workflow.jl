@testset "Eigenstates with automatic bases" begin
    using LinearAlgebra, Random

    sites = [FermionSite{2}(Symbol("s$i")) for i in 1:4]
    h = Hilbert(s.name => s for s in sites)
    hopping = sum(cdag(sites[i], σ) * c(sites[i+1], σ) for i in 1:3, σ in 1:2)
    H = -(hopping + hopping') + 4sum(n(s, 1) * n(s, 2) for s in sites)
    allowed = [n_fermion(h) == 4, WeightedParticleCount(sites, repeat([1, -1], 4)) == 0]
    b = basis(h, allowed...)
    exact = eigen(H, b; n=2)
    fixed = eigen(H, h; restrictions=allowed, n=2)
    @test fixed isa Eigenstates
    @test fixed.basis.states == b.states
    @test fixed.values ≈ exact.values
    @test fixed.selection === nothing && fixed.backend === :sparse
    selected = eigen(H, h; restrictions=allowed, adaptive=true,
                     backend=:matrixfree, tol=1e-8, seed_size=3, max_states=36)
    @test selected.values[1] ≈ exact.values[1] atol=1e-8
    @test selected.selection.residual_norm <= 1e-8
    @test size(selected.vectors, 1) == length(selected.basis)
    v = embed(selected.vectors[:, 1], selected.basis => b)
    @test norm(matrixfree(H, b) * v - selected.values[1] * v) <= 1e-8
    explicit = eigen(H, h; restrictions=allowed, adaptive=true, seed=selected.basis,
                     seed_max_work=1, max_states=36, tol=1e-8)
    @test explicit.values[1] ≈ exact.values[1] atol=1e-8
    @test_throws ArgumentError eigen(H, h; restrictions=allowed, seed=selected.basis)
    @test_throws ArgumentError eigen(H, h; restrictions=allowed, adaptive=true, which=:LR)
    @test_throws ArgumentError eigen(H, h; restrictions=n_fermion(h) == 9, adaptive=true)

    model = ShellModel(:Ni_3d)
    D = sum(i * n(site_of(model, :Ni_3d), i) for i in 1:10)
    result = eigen(D, model; restrictions=n_fermion(model.hilbert) == 1,
                   adaptive=true, backend=:matrixfree)
    @test result.values[1] ≈ 1.0
end
