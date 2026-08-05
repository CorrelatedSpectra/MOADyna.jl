@testset "Spectroscopy XAS smoke tests" begin
    using LinearAlgebra
    using SparseArrays
    using MOAD: xas
    using MOAD.Spectroscopy: SpectraTensor, LanczosChunk, DEFAULTS

    # Closed-form XAS reference for an isolated two-level transition:
    # H = ε_c n_1 + ε_v n_2, T = c†_2 c_1, ψ_g = |10⟩ (mode 1 occupied).
    # Peak at ω = ε_v − ε_c with χ(ω) = 1/(ω − (ε_v − ε_c) + iΓ/2).
    #
    # The 2-mode fermion site has 4 states; H + T live on this 4-state
    # basis and the analytical answer ignores all the empty / fully-
    # occupied configurations because they aren't reached by T·ψ_g.

    function _build_2level_system(; ε_c = -1.0, ε_v = 4.0)
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)                        # 4 states
        H = ε_c * n(s, 1) + ε_v * n(s, 2)
        H_sp = assemble(compile(H, b), b)
        T = cdag(s, 2) * c(s, 1)                 # core → valence promotion
        # Find the |10⟩ state index (mode 1 occupied, mode 2 empty: encoded 0b01).
        ψ = zeros(ComplexF64, length(b))
        idx = 0
        for i in 1:length(b)
            w = get_state(b, i)[1]
            (w & 0x01) != 0 && (w & 0x02) == 0 && (idx = i; break)
        end
        idx == 0 && error("2-level basis: ground state not found")
        ψ[idx] = 1.0
        return (s = s, basis = b, H = H_sp, T = T, ψ = ψ, Eg = ε_c, peak = ε_v - ε_c)
    end

    @testset "scalar T, scalar ψ → 1-D tensor; peak at ω = ε_v - ε_c" begin
        sys = _build_2level_system()
        Γ  = 0.2
        ωs = -1.0:0.05:8.0
        result = xas(sys.H, sys.basis, sys.T, sys.ψ;
                     ω_grid = ωs, Γ = Γ, Eg = sys.Eg)

        @test result isa SpectraTensor
        @test ndims(result.tensor) == 1
        @test size(result.tensor) == (length(ωs),)
        @test result.Eg == sys.Eg
        @test result.metadata[:Γ] == Γ
        @test length(result.chunks) == 1

        # Compare to analytic χ(ω) = 1/(ω - peak + iΓ/2)
        ref = [1 / (ω - sys.peak + im * Γ / 2) for ω in ωs]
        @test maximum(abs, result.tensor .- ref) < 1e-10
    end

    @testset "Eigen overload extracts Eg + GS, gives same answer" begin
        sys = _build_2level_system()
        Γ  = 0.2
        ωs = -1.0:0.5:8.0

        H_dense = Matrix(sys.H)
        E = eigen(Hermitian(H_dense))

        # Both paths should give the same spectrum.
        r_eigen = xas(sys.H, sys.basis, sys.T, E;     ω_grid = ωs, Γ = Γ)
        r_psi   = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = ωs, Γ = Γ, Eg = sys.Eg)
        @test r_eigen.tensor ≈ r_psi.tensor   atol = 1e-12
        @test r_eigen.Eg == r_psi.Eg
    end

    @testset "Eigen + Eg both raises ArgumentError" begin
        sys = _build_2level_system()
        H_dense = Matrix(sys.H)
        E = eigen(Hermitian(H_dense))
        @test_throws ArgumentError xas(sys.H, sys.basis, sys.T, E;
                                        ω_grid = -1:0.5:1, Γ = 0.2, Eg = 0.0)
    end

    @testset "Bare ψ without Eg raises ArgumentError" begin
        sys = _build_2level_system()
        @test_throws ArgumentError xas(sys.H, sys.basis, sys.T, sys.ψ;
                                        ω_grid = -1:0.5:1, Γ = 0.2)
    end

    @testset "vector T, scalar ψ → (N_T, N_T, n_ω) tensor" begin
        sys = _build_2level_system()
        Γ  = 0.2
        ωs = -1.0:0.5:8.0
        T_vec = [sys.T, 0.5 * sys.T]            # second op = scaled first
        result = xas(sys.H, sys.basis, T_vec, sys.ψ;
                     ω_grid = ωs, Γ = Γ, Eg = sys.Eg)
        @test ndims(result.tensor) == 3
        @test size(result.tensor) == (2, 2, length(ωs))
        @test length(result.chunks) == 1

        # Cross check: χ_aa = ⟨ψ|T†_a G T_a|ψ⟩ = (with second op = 0.5·first) χ_11 vs 0.25·χ_22.
        @test result.tensor[2, 2, :] ≈ 0.25 .* result.tensor[1, 1, :]   atol = 1e-10
        @test result.tensor[1, 2, :] ≈ 0.5  .* result.tensor[1, 1, :]   atol = 1e-10
        @test result.tensor[2, 1, :] ≈ 0.5  .* result.tensor[1, 1, :]   atol = 1e-10
    end

    @testset "scalar T, list ψ → (N_ψ, n_ω) tensor" begin
        sys = _build_2level_system()
        Γ  = 0.2
        ωs = -1.0:0.5:8.0
        # Second ψ = sqrt(2) · first → spectrum is 2× first's.
        ψ_list = [sys.ψ, sqrt(2) * sys.ψ]
        result = xas(sys.H, sys.basis, sys.T, ψ_list;
                     ω_grid = ωs, Γ = Γ, Eg = sys.Eg)
        @test ndims(result.tensor) == 2
        @test size(result.tensor) == (2, length(ωs))
        @test length(result.chunks) == 2
        @test result.tensor[2, :] ≈ 2 .* result.tensor[1, :]   atol = 1e-10
        @test result.metadata[:n_ψ] == 2
    end

    @testset "vector T, list ψ → (N_T, N_T, N_ψ, n_ω) tensor" begin
        sys = _build_2level_system()
        Γ  = 0.2
        ωs = -1.0:0.5:8.0
        T_vec = [sys.T, 0.5 * sys.T]
        ψ_list = [sys.ψ, sqrt(2) * sys.ψ]
        result = xas(sys.H, sys.basis, T_vec, ψ_list;
                     ω_grid = ωs, Γ = Γ, Eg = sys.Eg)
        @test ndims(result.tensor) == 4
        @test size(result.tensor) == (2, 2, 2, length(ωs))
        @test length(result.chunks) == 2

        # Cross check tensor[a, b, ψ_idx, ω] for two ψ's.
        @test result.tensor[1, 1, 2, :] ≈ 2 .* result.tensor[1, 1, 1, :]   atol = 1e-10
        @test result.tensor[2, 2, 1, :] ≈ 0.25 .* result.tensor[1, 1, 1, :] atol = 1e-10
    end

    @testset "ω_grid = :auto produces a reasonable range" begin
        sys = _build_2level_system()
        Γ  = 0.2
        result = xas(sys.H, sys.basis, sys.T, sys.ψ;
                     Γ = Γ, Eg = sys.Eg)        # ω_grid defaults to :auto
        @test result.ω_grid isa AbstractRange
        @test result.metadata[:auto_range]
        # Auto grid should bracket the peak (ε_v - ε_c = 5).
        @test first(result.ω_grid) < sys.peak
        @test last(result.ω_grid)  > sys.peak
        # Peak should be sampled by enough points for a Lorentzian visible.
        @test step(result.ω_grid) ≤ Γ / DEFAULTS.auto_range_density + eps()
    end

    @testset "ASCII alias resolves to the same answer" begin
        sys = _build_2level_system()
        ωs = -1.0:0.5:8.0
        r_greek = xas(sys.H, sys.basis, sys.T, sys.ψ;
                      ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        r_ascii = xas(sys.H, sys.basis, sys.T, sys.ψ;
                      omega_grid = ωs, Gamma = 0.2, Eg = sys.Eg)
        @test r_greek.tensor ≈ r_ascii.tensor    atol = 1e-12

        # Mixing both raises.
        @test_throws ArgumentError xas(sys.H, sys.basis, sys.T, sys.ψ;
                                        ω_grid = ωs, omega_grid = ωs,
                                        Γ = 0.2, Eg = sys.Eg)
        @test_throws ArgumentError xas(sys.H, sys.basis, sys.T, sys.ψ;
                                        ω_grid = ωs, Γ = 0.2, Gamma = 0.2,
                                        Eg = sys.Eg)
    end

    @testset "xas — pre-applied source matches T·ψ form" begin
        sys = _build_2level_system()
        Γ  = 0.2
        ωs = -1.0:0.05:8.0

        # Reference: standard 4-arg form.
        spec_a = xas(sys.H, sys.basis, sys.T, sys.ψ;
                     ω_grid = ωs, Γ = Γ, Eg = sys.Eg)

        # Pre-apply T manually, then call the 3-arg form.
        T_sp   = assemble(compile(sys.T, sys.basis), sys.basis)
        source = T_sp * sys.ψ
        spec_b = xas(sys.H, sys.basis, source;
                     ω_grid = ωs, Γ = Γ, Eg = sys.Eg)

        @test spec_b isa SpectraTensor
        @test ndims(spec_b.tensor) == 1
        @test size(spec_b.tensor) == (length(ωs),)
        @test maximum(abs, spec_a.tensor .- spec_b.tensor) < 1e-13
    end

    @testset "xas — pre-applied source: Eg / size validation" begin
        sys = _build_2level_system()
        T_sp   = assemble(compile(sys.T, sys.basis), sys.basis)
        source = T_sp * sys.ψ
        ωs = -1.0:0.5:8.0

        # Eg required.
        @test_throws ArgumentError xas(sys.H, sys.basis, source;
                                       ω_grid = ωs, Γ = 0.2)

        # Length mismatch.
        bad_source = vcat(source, zero(eltype(source)))
        @test_throws DimensionMismatch xas(sys.H, sys.basis, bad_source;
                                           ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)

        # ASCII alias works on the new form too.
        r_greek = xas(sys.H, sys.basis, source;
                      ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        r_ascii = xas(sys.H, sys.basis, source;
                      omega_grid = ωs, Gamma = 0.2, Eg = sys.Eg)
        @test r_greek.tensor ≈ r_ascii.tensor    atol = 1e-12
    end

    @testset "metadata fields populated" begin
        sys = _build_2level_system()
        ωs = -1.0:0.5:8.0
        result = xas(sys.H, sys.basis, sys.T, sys.ψ;
                     ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        m = result.metadata
        @test m[:function] === :xas
        @test m[:Γ] == 0.2
        @test m[:Eg] == sys.Eg
        @test m[:n_T] == 1
        @test m[:n_ψ] == 1
        @test m[:converged] isa Bool
        @test m[:basis_id] isa UInt64
        @test haskey(m, :n_iter_per_ψ)
    end
end
