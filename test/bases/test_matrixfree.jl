using Test, MOADyna, LinearAlgebra, SparseArrays, Random

@testset "Matrix-free operator actions" begin
    rng = MersenneTwister(14)
    f = FermionSite{3}(:f)
    p = BosonSite{2}(:p)
    s = SpinSite{1}(:s)
    padding = FermionSite{64}(:padding)
    h = Hilbert(:f => f, :p => p, :padding => padding, :s => s)
    bb = basis(h, n_fermion([padding]) == 0)
    # The final spin chain can fail after changing the temporary state.
    H = (0.3 + 0.4im) * cdag(f, 3) * c(f, 1) * bdag(p) +
        (0.7 - 0.2im) * c(f, 2) * Splus(s, 0) + 0.9 * Sz(s, -1) + 0.2 +
        Splus(s, -1) + Sminus(s, 0) * Splus(s, 0) * Splus(s, -1)
    M = assemble(compile(H, bb), bb)
    X = randn(rng, ComplexF64, length(bb), 3)
    original_states = copy(bb.states)
    for threaded in (false, true)
        A = matrixfree(H, bb; threaded)
        B = matrixfree(compile(H, bb), bb; threaded)
        @test size(A) == size(M)
        @test eltype(A) === eltype(M)
        @test A * X ≈ M * X
        @test A * X[:, 1] ≈ M * X[:, 1]
        @test B * X ≈ M * X
        @test A' * X ≈ M' * X
        @test A'' * X ≈ M * X
        combined = 0.4im * A + (1.2 - 0.3im) * A'
        @test combined * X ≈ (0.4im * M + (1.2 - 0.3im) * M') * X
        @test combined' * X ≈ (0.4im * M + (1.2 - 0.3im) * M')' * X
        @test sparse(A) ≈ M
        @test A[3, 2] ≈ M[3, 2]
        @test !ishermitian(A)
        Y = ones(ComplexF64, size(X))
        @test mul!(Y, A, X, 0.4im, -0.7) ≈ 0.4im * M * X .- 0.7
        fill!(Y, complex(NaN, NaN))
        @test mul!(Y, A, X) ≈ M * X
        @test all(iszero, mul!(Y, A, fill(complex(NaN, NaN), size(X)), 0, 0))
        task = Threads.@spawn A * X
        Y_concurrent = A * X
        @test fetch(task) ≈ Y_concurrent
        storage = zeros(ComplexF64, 2length(bb), 3)
        output = view(storage, 2:2:2length(bb), :)
        @test mul!(output, A, X) ≈ M * X
        @test all(iszero, storage[1:2:end, :])
    end
    @test bb.states == original_states

    # An independent two-mode matrix fixes signs, ordering and small coefficients.
    g = FermionSite{2}(:g)
    hg = Hilbert(:g => g)
    bg = basis(hg, n_fermion(hg) == 1)
    H2 = (0.2 + 0.3im) * cdag(g, 1) * c(g, 2) +
         (0.2 - 0.3im) * cdag(g, 2) * c(g, 1) + 1e-15 * n(g, 2)
    A2 = matrixfree(H2, bg)
    @test Matrix(A2) ≈ [0 0.2+0.3im; 0.2-0.3im 1e-15]
    @test ishermitian(A2)
    # Projection can make a globally non-Hermitian operator identically zero.
    @test ishermitian(matrixfree(c(g, 1), bg))
    occupied = basis(hg, n_fermion(hg) == 2)
    cancelled = matrixfree(1e308 * n(g, 1) - 1e308, occupied)
    @test cancelled * [2.0] == [0.0]
    @test nnz(sparse(cancelled)) == 0
    x = ones(ComplexF64, 2)
    @test (0.4im * A2 + (1.2 - 0.3im) * A2') * x ≈
        (0.4im * Matrix(A2) + (1.2 - 0.3im) * Matrix(A2)') * x
    @test_throws ArgumentError mul!(x, A2, x)
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, 3), A2, x)
    @test_throws ArgumentError matrixfree(H2, bg; threaded=:invalid)
    @test_throws ArgumentError matrixfree(compile(H2, bg), basis(hg))

    # Creation maps the vacuum sector to the one-particle sector.
    vacuum = basis(hg, n_fermion(hg) == 0)
    # Projected-out coefficients need not have a representable sum.
    large = (typemax(Int) ÷ 2 + 1) // 1
    rational_terms = OperatorSum{Rational{Int}}()
    for mode in 1:2
        MOADyna.Algebra._push!(rational_terms, large, first(n(g, mode)).chain)
    end
    @test matrixfree(rational_terms, occupied, vacuum) * [2.0] == [0.0]
    @test nnz(sparse(matrixfree(rational_terms, occupied, vacuum))) == 0
    T = matrixfree(cdag(g, 1) + 2im * cdag(g, 2), bg, vacuum)
    @test size(T) == (2, 1)
    @test T * [3.0] ≈ [3.0, 6im]
    @test T' * [1.0, 2im] ≈ [5.0]
    @test Matrix(T') ≈ Matrix(T)'
    @test !ishermitian(T)
    other = FermionSite{2}(:other)
    @test_throws ArgumentError matrixfree(cdag(g, 1), bg, basis(Hilbert(:other => other)))
    empty_basis = basis(hg, n_fermion(hg) == 0, n_fermion(hg) == 1)
    @test size(matrixfree(H2, empty_basis) * zeros(0, 2)) == (0, 2)
    empty_actions = (matrixfree(H2, empty_basis), matrixfree(H2, empty_basis, bg),
                     matrixfree(H2, bg, empty_basis))
    @test size.(sparse.(empty_actions)) == ((0, 0), (0, 2), (2, 0))

    # Real operators keep their storage type and promote complex right-hand sides.
    for R in (Float32, Float64)
        op = one(OperatorSum{R})
        A = matrixfree(op, bg)
        @test eltype(A) === R
        @test eltype(A * Complex{R}[1, 2im]) === Complex{R}
        @test A * Complex{R}[1, 2im] == Complex{R}[1, 2im]
    end
end
