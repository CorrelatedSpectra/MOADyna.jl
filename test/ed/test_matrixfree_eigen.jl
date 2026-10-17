@testset "Sparse and matrix-free eigenproblems" begin
    using LinearAlgebra

    site = FermionSite{8}(:f)
    h = Hilbert(:f => site)
    b = basis(h, n_fermion(h) == 1)
    H = sum((k / 7) * n(site, k) for k in 1:8) +
        sum((0.3 + 0.2im) * cdag(site, k) * c(site, k + 1) +
            (0.3 - 0.2im) * cdag(site, k + 1) * c(site, k) for k in 1:7)
    A = matrixfree(H, b; threaded=false)
    M = assemble(compile(H, b), b)
    reference = eigen(Hermitian(Matrix(M)))
    for backend in (:sparse, :matrixfree), count in (1, 3)
        E = eigen(H, b; backend, n=count, dense_below=0,
                  krylovdim=6, tol=1e-11, threaded=false)
        @test E.values ≈ reference.values[1:count] atol=1e-10
        @test norm(M * E.vectors - E.vectors * Diagonal(E.values)) < 1e-9
        @test E.vectors' * E.vectors ≈ I atol=1e-10
    end
    E = eigen(A, b; n=2, dense_below=1024, x0=ones(8), threaded=true)
    @test E.values ≈ reference.values[1:2] atol=1e-9
    @test eltype(E.vectors) == eltype(A)
    initial = copy(E.vectors)
    warm = eigen(A, b; n=3, x0=initial, threaded=false)
    @test warm.values ≈ reference.values[1:3] atol=1e-9
    @test initial == E.vectors
    @test_throws DimensionMismatch eigen(A, b; n=2, x0=ones(7, 2))
    @test_throws ArgumentError eigen(A, b; n=2, x0=ones(8, 3))
    @test_throws ArgumentError eigen(H, b; backend=:unknown)
    @test_throws ArgumentError eigen(A, basis(h, n_fermion(h) == 1))
    @test_throws ArgumentError eigen(matrixfree(cdag(site, 1) * c(site, 2), b), b)

    s1, s2 = SpinSite{1//2}(:s1), SpinSite{1//2}(:s2)
    spin_basis = basis(Hilbert(:s1 => s1, :s2 => s2))
    J = sum(S(s1)[k] * S(s2)[k] for k in 1:3)
    Jmatrix = assemble(compile(J, spin_basis), spin_basis)
    for backend in (:sparse, :matrixfree)
        E = eigen(J, spin_basis; backend, n=4, dense_below=0, tol=1e-12)
        @test E.values ≈ [-0.75, 0.25, 0.25, 0.25] atol=1e-11
        @test E.vectors' * E.vectors ≈ I atol=1e-11
        @test norm(Jmatrix * E.vectors - E.vectors * Diagonal(E.values)) < 1e-10
    end

    @testset "Scaled Hamiltonian" begin
        E = eigen(1e-13 * H, b; backend=:matrixfree, n=2,
                  tol=1e-15, krylovdim=6, threaded=false)
        @test E.values ≈ 1e-13 * reference.values[1:2] atol=1e-15 rtol=0
        @test norm(1e-13 * M * E.vectors - E.vectors * Diagonal(E.values)) ≤ sqrt(2) * 1e-15
    end
    @testset "Seed normalization and block rank" begin
        identity_on_sector = sum(n(site, k) for k in 1:8)
        seed = fill(1e-20, 8)
        E = eigen(identity_on_sector, b; backend=:matrixfree, n=2,
                  x0=seed, threaded=false)
        @test E.values ≈ ones(2) atol=1e-10
        @test E.vectors' * E.vectors ≈ I atol=1e-10
        @test seed == fill(1e-20, 8)
        @test_throws ArgumentError eigen(A, b; x0=zeros(8))
    end
end
