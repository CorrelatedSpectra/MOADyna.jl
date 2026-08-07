@testset "Responses block Lanczos + cf primitive" begin
    using LinearAlgebra
    using SparseArrays
    using Random: MersenneTwister
    using MOADyna.Responses: block_lanczos, cf_block
    using MOADyna.Spectroscopy: evaluate_on_grid, LanczosChunk

    # ---------------------------------------------------------------
    # S4 — block Lanczos cf matches the exact dense resolvent
    # The original spec compared to "3 independent scalar
    # Lanczos cf evaluations"; we use the dense exact resolvent
    # because it is unambiguously correct to LAPACK precision and
    # avoids confounding the test with scalar-Lanczos numerical
    # breakdown at large K (no reorthogonalisation = ghost
    # eigenvalues). The two tests are equivalent in intent: verify
    # that block-Lanczos + cf reproduces ⟨x|(zI-H)⁻¹|x⟩ when the
    # Krylov subspace is exhausted.
    # ---------------------------------------------------------------
    @testset "S4: block-3 diag matches dense exact resolvent" begin
        rng = MersenneTwister(0)
        N   = 200
        A   = randn(rng, ComplexF64, N, N)
        Hm  = (A + A') / 2
        H   = sparse(Hm)

        Braw = 3
        X    = randn(rng, ComplexF64, N, Braw)

        # Block Lanczos with FRO — full Krylov coverage for B=3 at K = ⌈N/B⌉.
        K_block = N + 5             # comfortably exhausts the subspace
        result  = block_lanczos(H, X;
                                krylovdim = K_block,
                                reorth   = :full,
                                tol      = 1e-14, min_iter = 5,
                                deflate_tol = 1e-14)

        chunk = LanczosChunk{ComplexF64}(
            result.α, result.β, result.R,
            Braw, 1, 0, result.n_iter, result.converged)

        Γ   = 0.5
        ωs  = -5.0:2.5:5.0
        χ   = evaluate_on_grid(chunk, ωs; Γ = Γ, Eg = 0.0)
        @test size(χ) == (Braw, Braw, length(ωs))

        # Exact reference: dense (zI − H)⁻¹, full block χ_ref[a,b,i].
        χ_ref = [adjoint(X[:, a]) * inv(complex(ω, Γ/2) * I - Hm) * X[:, b]
                 for a in 1:Braw, b in 1:Braw, ω in ωs]
        @test maximum(abs, χ .- χ_ref) < 1e-9
    end

    # ---------------------------------------------------------------
    # Block-1 Lanczos = scalar Lanczos (algorithmic equivalence)
    # ---------------------------------------------------------------
    @testset "block-1 with B=1 starting vector matches scalar form" begin
        rng = MersenneTwister(2)
        N   = 80
        A   = randn(rng, ComplexF64, N, N)
        Hm  = (A + A') / 2
        H   = sparse(Hm)
        x   = randn(rng, ComplexF64, N)

        # Block-1 Lanczos.
        result = block_lanczos(H, reshape(x, N, 1);
                               krylovdim = N + 5,
                               reorth   = :full,
                               tol      = 1e-14,
                               deflate_tol = 1e-14)
        chunk = LanczosChunk{ComplexF64}(
            result.α, result.β, result.R, 1, 1, 0,
            result.n_iter, result.converged)

        Γ  = 0.3
        ωs = -3.0:1.5:3.0
        χ  = evaluate_on_grid(chunk, ωs; Γ = Γ, Eg = 0.0)

        ref = [adjoint(x) * inv(complex(ω, Γ/2) * I - Hm) * x for ω in ωs]
        @test maximum(abs, [χ[1, 1, i] - ref[i] for i in eachindex(ωs)]) < 1e-10
    end

    # ---------------------------------------------------------------
    # Sanity: cf_block reduces to scalar cf for B = 1
    # ---------------------------------------------------------------
    @testset "cf_block: B=1 reduces to scalar cf" begin
        rng = MersenneTwister(42)
        αs = [randn(rng, ComplexF64, 1, 1) for _ in 1:5]
        βs = [randn(rng, ComplexF64, 1, 1) for _ in 1:4]
        function _scalar_cf(αs, βs, z)
            G = 1 / (z - αs[end][1, 1])
            for k in (length(αs)-1):-1:1
                G = 1 / (z - αs[k][1, 1] - conj(βs[k][1, 1]) * G * βs[k][1, 1])
            end
            return G
        end
        ωs = (-2.0, 0.0, 1.0, 3.0)
        @test maximum(abs(cf_block(αs, βs, complex(ω, 0.05))[1, 1] -
                          _scalar_cf(αs, βs, complex(ω, 0.05))) for ω in ωs) < 1e-12
    end

    # ---------------------------------------------------------------
    # cf at z = ω + iΓ/2 produces a Hermitian block Re part
    # ---------------------------------------------------------------
    @testset "cf_block: Hermitian αs → Hermitian Re(G), anti-Herm Im(G)" begin
        rng = MersenneTwister(7)
        Bact = 3
        K = 6
        αs = Matrix{ComplexF64}[]
        βs = Matrix{ComplexF64}[]
        for _ in 1:K
            A = randn(rng, ComplexF64, Bact, Bact)
            push!(αs, (A + A') / 2)        # Hermitian
        end
        for _ in 1:(K-1)
            push!(βs, randn(rng, ComplexF64, Bact, Bact))
        end
        z = complex(0.5, 0.05)
        G = cf_block(αs, βs, z)
        # For Hermitian αs but complex z, G itself is not Hermitian, but
        # G(z̄) = G(z)†.
        G_conj = cf_block(αs, βs, conj(z))
        @test G_conj ≈ adjoint(G)   atol = 1e-12
    end

    # ---------------------------------------------------------------
    # Block Lanczos with rank-deflation: pass linearly-dependent block
    # ---------------------------------------------------------------
    @testset "Initial QR rank deflation (B_active < B_raw)" begin
        rng = MersenneTwister(101)
        N = 200
        A = randn(rng, ComplexF64, N, N)
        H = sparse((A + A') / 2)
        # 3-column starting block where col 3 = col 1 + col 2 (rank 2).
        v1 = randn(rng, ComplexF64, N)
        v2 = randn(rng, ComplexF64, N)
        X  = hcat(v1, v2, v1 + v2)
        result = block_lanczos(H, X; krylovdim = 30, reorth = :none,
                               deflate_tol = 1e-10)
        @test size(result.R, 1) == 2     # B_active = 2 (rank-deflated)
        @test size(result.R, 2) == 3     # B_raw    = 3
        @test result.n_iter ≥ 25         # should still progress
        # Each α is 2 × 2; sanity.
        @test all(size(α) == (2, 2) for α in result.α)
    end

    # ---------------------------------------------------------------
    # Block Lanczos with retain_basis returns full V stack
    # ---------------------------------------------------------------
    @testset "retain_basis stores V_blocks; orthonormality holds" begin
        rng = MersenneTwister(99)
        N = 100
        A = randn(rng, ComplexF64, N, N)
        H = sparse((A + A') / 2)
        X = randn(rng, ComplexF64, N, 2)

        result = block_lanczos(H, X; krylovdim = 10, retain_basis = true,
                               reorth = :none)
        @test result.V_basis !== nothing
        Vs = result.V_basis
        # Each V block must be orthonormal within itself.
        @test maximum(opnorm(adjoint(V) * V - I(size(V, 2))) for V in Vs) < 1e-10
        # Adjacent V blocks should be orthogonal: V_k† · V_{k+1} ≈ 0.
        # (For non-FRO this isn't tested across non-adjacent k.)
        @test maximum(norm(adjoint(Vs[k]) * Vs[k+1]) for k in 1:length(Vs)-1) < 1e-9
    end

    # ---------------------------------------------------------------
    # FRO: full reorthogonalisation makes ALL V blocks pairwise orthogonal
    # ---------------------------------------------------------------
    @testset "reorth = :full: all V blocks pairwise orthogonal" begin
        rng = MersenneTwister(11)
        N = 80
        A = randn(rng, ComplexF64, N, N)
        H = sparse((A + A') / 2)
        X = randn(rng, ComplexF64, N, 2)

        result = block_lanczos(H, X; krylovdim = 12, retain_basis = true,
                               reorth = :full)
        @test result.V_basis !== nothing
        Vs = result.V_basis
        @test maximum(norm(adjoint(Vs[i]) * Vs[j])
                      for i in 1:length(Vs), j in 1:length(Vs) if i < j) < 1e-9
    end

    # ---------------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # retain_basis structural invariant: V_basis stores K+1 blocks but
    # only the first K participate in the cf representation. This
    # caught a real RIXS/FY bug where `V_int = hcat(V_basis...)` shipped
    # an extra block, causing a DimensionMismatch downstream when the
    # Krylov subspace was NOT exhausted (krylovdim < N).
    # ---------------------------------------------------------------
    @testset "retain_basis: V_basis has K+1 blocks; V_int sandwich uses K" begin
        using MOADyna.Spectroscopy: _build_T_K
        rng = MersenneTwister(123)
        N = 200                           # large enough that K = 50 < N
        A = randn(rng, ComplexF64, N, N)
        H = sparse((A + A') / 2)
        Braw = 2
        X = randn(rng, ComplexF64, N, Braw)

        K_request = 50
        result = block_lanczos(H, X;
                               krylovdim    = K_request,
                               reorth       = :none,
                               retain_basis = true,
                               tol          = 1e-14,
                               deflate_tol  = 1e-14)

        # Subspace not exhausted: K iterations completed, V_basis has K+1
        # blocks (initial V_1 + one V_{k+1} pushed at each iteration).
        @test result.n_iter == K_request
        @test length(result.V_basis) == K_request + 1
        @test length(result.α)       == K_request

        # Building T_K from α/β gives a (K · Bact) × (K · Bact) matrix.
        # V_int for the cf representation must take exactly K blocks
        # (the first K of V_basis); using all K+1 would dimension-mismatch.
        T_K = _build_T_K(result.α, result.β)
        T_K_dim = size(T_K, 1)
        V_int_correct = hcat(result.V_basis[1:length(result.α)]...)
        @test size(V_int_correct, 2) == T_K_dim
        @test size(V_int_correct, 1) == N

        # The naive (K+1)-block hcat would give T_K_dim + B_active extra
        # columns and break a downstream `V_int * w` for any K-vector w.
        V_int_wrong = hcat(result.V_basis...)
        @test size(V_int_wrong, 2) > T_K_dim
    end

    # ---------------------------------------------------------------
    # End-to-end RIXS / FY no-dimension-mismatch when krylovdim < N
    # ---------------------------------------------------------------
    # The 3-level smoke tests in test_rixs.jl / test_fluorescence_yield.jl
    # use a Hilbert space small enough that the inner Lanczos exhausts
    # the Krylov subspace (V_basis ends up with exactly K entries via
    # the rank-deflation soft-stop), masking the K+1 issue. To bite the
    # bug path, we need:
    #   - krylovdim_request < dim(Krylov subspace), AND
    #   - the inner Lanczos to actually reach krylovdim_request without
    #     soft-stopping early (so V_basis grows to K + 1).
    #
    # A diagonal H with an eigenvector ψ falls into a 1-D Krylov subspace
    # (n_iter = 1, V_basis size 1) — that misses the bug entirely. Use a
    # non-diagonal H so T·ψ_g spreads over a multi-dimensional Krylov
    # subspace.
    @testset "rixs/FY survive krylovdim < dim(Krylov subspace) (regression)" begin
        using MOADyna: rixs, fluorescence_yield
        using MOADyna.Spectroscopy: SpectraTensor
        using Random: MersenneTwister

        # 6 fermion modes with 1 particle → 6 states. Random Hermitian H
        # with off-diagonal hopping; T_in = c†_2 c_1; T_out = c†_2 c_3.
        s = FermionSite{6}(:s)
        hil = Hilbert(:s => s)
        b = EagerBasis(hil, n_fermion(hil) == 1)
        @test length(b) == 6

        # H_diag (on-site) + H_hop (random hopping in upper triangle,
        # symmetrised). Builds a 6×6 dense Hermitian.
        rng = MersenneTwister(0xCAFE)
        H_op = sum(i * n(s, i) for i in 1:6)
        for i in 1:6, j in (i+1):6
            t_ij = randn(rng)
            H_op += t_ij * (cdag(s, i) * c(s, j) + cdag(s, j) * c(s, i))
        end
        H_sp  = assemble(compile(H_op, b), b)
        T_in  = cdag(s, 2) * c(s, 1)
        T_out = cdag(s, 2) * c(s, 3)

        # ψ = mode-1-occupied (already in basis); spreads over the full
        # 6-D Krylov subspace under H. T_in · ψ → mode-2-occupied,
        # which under H mixes with all other 5 states.
        ψ = zeros(ComplexF64, length(b))
        for i in 1:length(b)
            w = get_state(b, i)[1]
            (w & UInt64(0x01)) != 0 && (w & UInt64(0x3E)) == 0 && (ψ[i] = 1.0; break)
        end

        # First confirm the bug path is actually live: a fresh inner
        # Lanczos with krylovdim = 3 should reach n_iter = 3 and produce
        # length(V_basis) = K + 1 = 4. A diagonal-H test would give 1.
        X_int = reshape(assemble(compile(T_in, b), b) * ψ, length(b), 1)
        inner_test = block_lanczos(H_sp, X_int;
                                    krylovdim   = 3,
                                    retain_basis = true,
                                    reorth      = :none,
                                    tol         = 1e-14,
                                    deflate_tol = 1e-14)
        @test inner_test.n_iter == 3                     # ran to krylovdim
        @test length(inner_test.V_basis) == 4            # K + 1 path is live

        # rixs with explicit krylovdim < dim(Krylov subspace): must run
        # without DimensionMismatch in `Y = V_int * W`.
        r = rixs(H_sp, H_sp, b, T_in, T_out, ψ;
                 ω_in_grid = 0.0:1.0:5.0, ω_out_grid = 0.0:0.1:5.0,
                 Γ_intermediate = 0.5, Γ_final = 0.05,
                 Eg = 1.0, krylovdim = 3)
        @test r isa SpectraTensor
        @test ndims(r.tensor) == 2

        # fluorescence_yield with krylovdim < dim(Krylov subspace).
        f = fluorescence_yield(H_sp, b, T_in, T_out, ψ;
                                ω_in_grid = 0.0:1.0:5.0,
                                Γ_intermediate = 0.5,
                                Eg = 1.0, krylovdim = 3)
        @test f isa SpectraTensor
        @test ndims(f.tensor) == 1
    end

    @testset "input validation" begin
        H = sparse(ComplexF64[1.0 0; 0 -1.0])
        X = ComplexF64[1.0 0; 0 1.0]
        @test_throws ArgumentError block_lanczos(H, X; reorth = :bogus)
        @test_throws ArgumentError block_lanczos(H, X; krylovdim = 0)
        @test_throws ArgumentError block_lanczos(H, X; max_iter = 0)
        # Hamiltonian / starting block dimension mismatch.
        H2 = sparse(ComplexF64[1.0 0 0; 0 -1.0 0; 0 0 0.0])
        @test_throws DimensionMismatch block_lanczos(H2, X)
    end

    # ---------------------------------------------------------------
    # Rank-zero starting block: pivots[1] == 0 guard at the initial QR
    # (src/responses/block_lanczos.jl:132). Covered end-to-end by the
    # correlator rank-zero test; this is the focused unit-level check.
    # ---------------------------------------------------------------
    @testset "rank-zero starting block raises" begin
        H = sparse(ComplexF64[1.0 0; 0 -1.0])
        # All-zero starting block (degenerate).
        X_zero = zeros(ComplexF64, 2, 2)
        @test_throws ArgumentError block_lanczos(H, X_zero)
        # Single zero column also raises.
        @test_throws ArgumentError block_lanczos(H, zeros(ComplexF64, 2, 1))
    end

    # ---------------------------------------------------------------
    # Krylov subspace exhaustion: when the starting block plus its
    # H-iterates span the full N-dimensional Hilbert space, the
    # recurrence must soft-stop with converged = true at n_iter ≤ N
    # (either via residual_norm < threshold or via B_next == 0; both
    # paths set converged = true and break — line 195 or 207).
    # ---------------------------------------------------------------
    @testset "Krylov subspace exhaustion → converged=true" begin
        # Tiny 4×4 diagonal H; uniform starting vector mixes all 4
        # eigenstates, so the Krylov dimension is exactly 4.
        H = sparse(ComplexF64.(Diagonal([1.0, 2.0, 3.0, 4.0])))
        X = reshape(ComplexF64[1, 1, 1, 1] ./ 2, 4, 1)
        result = block_lanczos(H, X; krylovdim = 50, deflate_tol = 1e-12)
        @test result.converged
        @test result.n_iter ≤ 4
    end
end
