@testset "ED.eigen — basics" begin
    using LinearAlgebra
    using SparseArrays

    # Heisenberg dimer (G5): clean degenerate-cluster test case.
    s1 = SpinSite{1//2}(:s1)
    s2 = SpinSite{1//2}(:s2)
    h_sp = Hilbert(:s1 => s1, :s2 => s2)
    local_dot(a::NTuple{3, OperatorSum}, b::NTuple{3, OperatorSum}) =
        a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    H_heis = local_dot(S(s1), S(s2))
    basis_sp = EagerBasis(h_sp)

    # E0a — eigvals match dense reference; orthonormality
    @testset "E0a: G1-G6 eigvals match dense reference; orthonormality" begin
        # G1: one-mode number op
        s_g1 = FermionSite{1}(:s)
        h_g1 = Hilbert(:s => s_g1)
        b_g1 = EagerBasis(h_g1)
        H_g1 = n(s_g1, 1)
        E = eigen(H_g1, b_g1; n = 2)
        @test E.values ≈ [0.0, 1.0]
        @test E.vectors' * E.vectors ≈ I(2) atol = 1e-12

        # G3: U n_↑ n_↓
        s_g3 = FermionSite{2}(:s)
        h_g3 = Hilbert(:s => s_g3)
        b_g3 = EagerBasis(h_g3)
        U = 4.0
        H_g3 = U * n(s_g3, 1) * n(s_g3, 2)
        E = eigen(H_g3, b_g3; n = 4)
        @test sort(E.values) ≈ [0.0, 0.0, 0.0, U]
        @test E.vectors' * E.vectors ≈ I(4) atol = 1e-12

        # G5: Heisenberg dimer (degenerate triplet)
        E = eigen(H_heis, basis_sp; n = 4)
        @test E.values[1] ≈ -0.75 atol = 1e-10
        @test all(E.values[2:4] .≈ 0.25)
        @test E.vectors' * E.vectors ≈ I(4) atol = 1e-12
        # Each column is still an eigenvector after cluster-QR.
        H_sparse = assemble(compile(H_heis, basis_sp), basis_sp)
        @test maximum(norm(H_sparse * E.vectors[:, k] - E.values[k] * E.vectors[:, k])
                      for k in 1:4) < 1e-10
    end

    # E0b — dense and Krylov agree on the GROUND STATE
    # On a degenerate spectrum, single-vector Lanczos returns at most one
    # vector per degenerate eigenspace (this is a fundamental property,
    # not a v0.1 limitation we can fix without block Lanczos). We test
    # agreement on the non-degenerate ground state of a 4-site Heisenberg
    # chain, where both paths must produce identical numbers.
    @testset "E0b: dense and Krylov agree on non-degenerate GS" begin
        sites = [SpinSite{1//2}(Symbol("s$i")) for i in 1:4]
        h = Hilbert(s.name => s for s in sites)
        H = sum(local_dot(S(sites[i]), S(sites[i + 1])) for i in 1:3)
        bb = EagerBasis(h)

        E_dense = eigen(H, bb; n = 1, dense_below = typemax(Int))
        E_kryl  = eigen(H, bb; n = 1, dense_below = 0,
                        krylovdim = 32, tol = 1e-12)
        @test E_dense.values[1] ≈ E_kryl.values[1] atol = 1e-10
        # Eigenvector projectors match (vectors agree up to phase).
        P_d = E_dense.vectors[:, 1] * E_dense.vectors[:, 1]'
        P_k = E_kryl.vectors[:, 1]  * E_kryl.vectors[:, 1]'
        @test norm(P_d - P_k) < 1e-9
    end

    # E0c — `which` selection rules
    @testset "E0c: which = :SR / :LR / :LM" begin
        # G3 spectrum is {0, 0, 0, U}. SR → 0, LR → U, LM → U (largest |λ|).
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)
        U = 4.0
        H = U * n(s, 1) * n(s, 2)

        @test eigen(H, b; n = 1, which = :SR).values[1] ≈ 0.0 atol = 1e-12
        @test eigen(H, b; n = 1, which = :LR).values[1] ≈ U
        @test eigen(H, b; n = 1, which = :LM).values[1] ≈ U

        # Rejected which values throw with helpful messages.
        @test_throws ArgumentError eigen(H, b; n = 1, which = :SI)
        @test_throws ArgumentError eigen(H, b; n = 1, which = :LI)
        @test_throws ArgumentError eigen(H, b; n = 1, which = :SM)
    end

    # E0d — ConvergenceError fires when forced
    @testset "E0d: ConvergenceError on too-tight maxiter" begin
        # Force Krylov path with a Hamiltonian that needs more than 1 iter.
        # 6-site Heisenberg chain — just big enough that maxiter=1 won't
        # converge.
        sites = [SpinSite{1//2}(Symbol("s$i")) for i in 1:6]
        h = Hilbert(s.name => s for s in sites)
        H = sum(local_dot(S(sites[i]), S(sites[i + 1])) for i in 1:5)
        bb = EagerBasis(h)
        try
            eigen(H, bb; n = 3, dense_below = 0,
                  maxiter = 1, krylovdim = 6, tol = 1e-14)
            @test false  # should have thrown
        catch e
            @test e isa ConvergenceError
            @test e.normres isa Vector{Float64}
            @test length(e.normres) ≥ 1
            @test max_normres(e) > 0
            @test e.converged < 3
        end
    end

    # E0e — reproducibility of the default Krylov initial vector
    @testset "E0e: reproducibility of default x0" begin
        # 6-site Heisenberg chain — Krylov path
        sites = [SpinSite{1//2}(Symbol("s$i")) for i in 1:6]
        h = Hilbert(s.name => s for s in sites)
        H = sum(local_dot(S(sites[i]), S(sites[i + 1])) for i in 1:5)
        bb = EagerBasis(h)
        E1 = eigen(H, bb; n = 3, dense_below = 0)
        E2 = eigen(H, bb; n = 3, dense_below = 0)
        @test E1.values == E2.values   # bit-for-bit
        # Eigenvectors equal up to per-column sign flip.
        for k in 1:3
            ratio = E1.vectors[:, k] ./ E2.vectors[:, k]
            ratio = filter(isfinite, ratio)
            # All ratios should be the same scalar (1 or -1 modulo phase).
            @test all(abs.(abs.(ratio) .- 1) .< 1e-10)
        end
    end

    # E0f — input validation
    @testset "E0f: input validation" begin
        # Set up a small valid problem to break in different ways.
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)
        H = n(s, 1)

        # Mismatched sizes (assemble against a different basis)
        b_other = EagerBasis(h, n_fermion(h) == 1)   # smaller basis
        H_sparse_other = assemble(compile(H, b_other), b_other)
        @test_throws DimensionMismatch eigen(H_sparse_other, b; n = 1)

        # n out of range
        @test_throws ArgumentError eigen(H, b; n = 0)
        @test_throws ArgumentError eigen(H, b; n = length(b) + 1)

        # Bad which
        @test_throws ArgumentError eigen(H, b; n = 1, which = :BOGUS)

        # Undersized krylovdim
        @test_throws ArgumentError eigen(H, b; n = 3, krylovdim = 2)

        # Wrong-length x0
        @test_throws DimensionMismatch eigen(H, b; n = 1,
                                             x0 = randn(length(b) + 1))

        # Complex x0 on real H — rejected
        @test_throws ArgumentError eigen(H, b; n = 1,
                                         x0 = ComplexF64.(randn(length(b))))
    end

    # E0g — cluster-aware orthogonalization helper, tested directly.
    # KrylovKit's single-vector Lanczos doesn't expose multiplet
    # degeneracies (one vector per eigenspace from one initial vector),
    # so we can't trigger _orthonormalize_clusters! through a physical
    # eigsolve in v0.1. Test the helper directly on a synthesized
    # degenerate problem: H = diag(...) with explicit clusters; messy
    # eigvec input within each cluster; verify the helper produces
    # orthonormal columns that remain eigenvectors of H.
    @testset "E0g: _orthonormalize_clusters! preserves eigenvector property" begin
        using MOAD.ED: _orthonormalize_clusters!
        using LinearAlgebra: Diagonal, qr

        # Spectrum: -1 (×3), 0 (×1), 0.5 (×2), 1 (×2).
        H = Diagonal([-1.0, -1.0, -1.0, 0.0, 0.5, 0.5, 1.0, 1.0])
        λ = collect(H.diag)

        # Build "messy" eigenvectors: standard basis mixed within clusters
        # by random orthogonal then random non-orthogonal upper-triangular.
        V = Matrix{Float64}(I, 8, 8)
        # Cluster 1:3 — Q rotation within the eigenspace.
        Q1, _ = qr(randn(3, 3));  V[:, 1:3] = V[:, 1:3] * Matrix(Q1)
        # Then a non-orthogonal mix (upper triangular, column-normalized).
        R = Float64[1 0.5 0.0; 0 1 0.5; 0 0 1]
        V[:, 1:3] = V[:, 1:3] * R
        # Renormalize (otherwise columns aren't unit vectors going in).
        for k in 1:size(V, 2)
            V[:, k] ./= norm(V[:, k])
        end
        # At this point V[:, 1:3] still spans the eigenspace of -1 (only
        # rows 1:3 have weight) but isn't orthonormal.

        _orthonormalize_clusters!(V, λ, 1e-8)

        # Orthonormal columns
        @test V' * V ≈ I(8) atol = 1e-10
        # Each column still an eigenvector of H
        for k in 1:8
            res = H * V[:, k] - λ[k] * V[:, k]
            @test norm(res) < 1e-10
        end
    end

    # E0h — x0 eltype rules
    @testset "E0h: x0 eltype handling" begin
        # Real H: real x0 OK, complex x0 throws (covered in E0f for the throw).
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)
        H_real = n(s, 1)                # real Hamiltonian
        E_real = eigen(H_real, b; n = 1, x0 = randn(length(b)))
        @test eltype(E_real.vectors) == Float64

        # Complex H (Sy in the operator): complex eigvecs; real x0 promotes.
        sp = SpinSite{1//2}(:sp)
        h_sp = Hilbert(:sp => sp)
        b_sp = EagerBasis(h_sp)
        H_complex = Sy(sp, -1//2) + Sy(sp, 1//2)        # full Sy on the site
        x0_real = randn(length(b_sp))                    # real x0 with complex H
        E_complex = eigen(H_complex, b_sp; n = 1, x0 = x0_real)
        @test eltype(E_complex.vectors) <: Complex
    end

    # E0i — Higher-precision x0 promotes (BigFloat → Float64 etc.)
    # 6-site Heisenberg chain to land on the Krylov path.
    @testset "E0i: x0 BigFloat / Float32 → Float64" begin
        sites = [SpinSite{1//2}(Symbol("s$i")) for i in 1:6]
        h = Hilbert(s.name => s for s in sites)
        H = sum(local_dot(S(sites[i]), S(sites[i + 1])) for i in 1:5)
        bb = EagerBasis(h)

        # BigFloat x0 used to fail with a backend MethodError; now converts.
        x0_bf = BigFloat.(randn(length(bb)))
        E1 = eigen(H, bb; n = 1, dense_below = 0, x0 = x0_bf)
        @test E1.values[1] isa Float64

        # Float32 x0 also converts cleanly.
        x0_f32 = Float32.(randn(length(bb)))
        E2 = eigen(H, bb; n = 1, dense_below = 0, x0 = x0_f32)
        @test E2.values[1] isa Float64

        # Reject explicitly non-Number element types.
        x0_str = String["x" for _ in 1:length(bb)]
        @test_throws ArgumentError eigen(H, bb; n = 1, x0 = x0_str)
    end

    # E0j — Public eigen() rejects non-Hermitian input loudly.
    # Per the chapter contract H is Hermitian by assumption. The public
    # API accepts a SparseMatrixCSC; Quietly mirroring one triangle (as
    # `Hermitian(M)` would do) produces wrong eigenvalues for a different
    # operator than what the user passed. We check `ishermitian` upfront
    # and throw.
    @testset "E0j: public eigen() throws on non-Hermitian input" begin
        # Need a basis to call the public API; size has to match H.
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)                              # 4 states
        # Non-Hermitian 4×4 sparse matrix.
        H_bad = sparse([0.0 10.0 0.0 0.0;
                        0.0  0.0 0.0 0.0;
                        0.0  0.0 1.0 0.0;
                        0.0  0.0 0.0 2.0])
        @test_throws ArgumentError eigen(H_bad, b; n = 1)
    end
end
