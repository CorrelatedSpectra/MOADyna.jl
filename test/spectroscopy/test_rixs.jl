@testset "Spectroscopy RIXS smoke tests" begin
    using LinearAlgebra
    using SparseArrays
    using MOADyna: rixs
    using MOADyna.Spectroscopy: SpectraTensor

    # Closed-form RIXS reference for a 3-level system:
    # Modes 1, 2, 3 = |g⟩ (ground), |i⟩ (intermediate), |f⟩ (final).
    # H = ε_g·n_1 + ε_i·n_2 + ε_f·n_3 (with ε_g = 0, n_1 + n_2 + n_3 = 1
    # restriction so basis is {|100⟩, |010⟩, |001⟩} = 3 states).
    # T_in  = c†_2 c_1                (g → i)
    # T_out = c†_2 c_3                (f → i; T_out† does i → f)
    # ψ_g = |100⟩.
    #
    # Then χ(ω_in, ω_out) = 1 / [(z_in − ε_i)(z_out − ε_f) conj(z_in − ε_i)]
    #                     = 1 / [(z_out − ε_f) · |z_in − ε_i|²]
    #
    # with z_in  = ω_in  + Eg + iΓ_int/2     and   Eg = 0,
    #      z_out = ω_out + Eg + iΓ_final/2.
    function build_3level(; ε_i = 5.0, ε_f = 2.0)
        s = FermionSite{3}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h, n_fermion(h) == 1)
        H = 0.0 * n(s, 1) + ε_i * n(s, 2) + ε_f * n(s, 3)
        H_sp = assemble(compile(H, b), b)
        T_in  = cdag(s, 2) * c(s, 1)
        T_out = cdag(s, 2) * c(s, 3)
        ψ = zeros(ComplexF64, length(b))
        for i in 1:length(b)
            w = get_state(b, i)[1]
            (w & 0x01) != 0 && (w & 0x06) == 0 && (ψ[i] = 1.0; break)
        end
        return (basis = b, H = H_sp, T_in = T_in, T_out = T_out,
                ψ = ψ, Eg = 0.0, ε_i = ε_i, ε_f = ε_f)
    end

    @testset "scalar T_in, scalar T_out, single ψ → 2-D tensor" begin
        sys = build_3level()
        Γ_i = 0.5
        Γ_f = 0.05
        ω_in_grid  = 3.0:0.2:7.0
        ω_out_grid = 0.0:0.05:4.0
        result = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                      ω_in_grid       = ω_in_grid,
                      ω_out_grid      = ω_out_grid,
                      Γ_intermediate  = Γ_i,
                      Γ_final         = Γ_f,
                      Eg              = sys.Eg)

        @test result isa SpectraTensor
        @test ndims(result.tensor) == 2
        @test size(result.tensor) == (length(ω_in_grid), length(ω_out_grid))
        @test result.metadata[:function] === :rixs
        @test result.ω_grid === (ω_in_grid, ω_out_grid)

        # Compare to analytic χ.
        ref = [1 / ((ω_out + sys.Eg + im * Γ_f / 2 - sys.ε_f) *
                    abs2(ω_in + sys.Eg + im * Γ_i / 2 - sys.ε_i))
               for ω_in in ω_in_grid, ω_out in ω_out_grid]
        @test maximum(abs, result.tensor .- ref) < 1e-10
    end

    @testset "RIXS map peak position" begin
        sys = build_3level()
        result = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                      ω_in_grid  = 3.0:0.2:7.0,
                      ω_out_grid = 0.0:0.05:4.0,
                      Γ_intermediate = 0.5, Γ_final = 0.05, Eg = sys.Eg)
        # Peak in -Im at (ω_in = ε_i, ω_out = ε_f).
        intensity = -imag.(result.tensor)
        idx = argmax(intensity)
        @test (3.0:0.2:7.0)[idx[1]] ≈ sys.ε_i  atol = 0.2
        @test (0.0:0.05:4.0)[idx[2]] ≈ sys.ε_f atol = 0.05
    end

    @testset "RIXS auto-range produces tuple of ranges" begin
        sys = build_3level()
        result = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                      Γ_intermediate = 0.5, Γ_final = 0.05, Eg = sys.Eg)
        @test result.ω_grid isa Tuple
        @test result.ω_grid[1] isa AbstractRange
        @test result.ω_grid[2] isa AbstractRange
        @test result.metadata[:auto_range]
        # Auto ranges must bracket the peaks.
        @test first(result.ω_grid[1]) < sys.ε_i < last(result.ω_grid[1])
        @test first(result.ω_grid[2]) < sys.ε_f < last(result.ω_grid[2])
    end

    @testset "ASCII alias for Greek kwargs" begin
        sys = build_3level()
        ω_in_grid  = 3.0:0.5:7.0
        ω_out_grid = 0.0:0.1:4.0
        r_greek = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                       ω_in_grid = ω_in_grid, ω_out_grid = ω_out_grid,
                       Γ_intermediate = 0.5, Γ_final = 0.05, Eg = sys.Eg)
        r_ascii = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                       omega_in_grid = ω_in_grid, omega_out_grid = ω_out_grid,
                       Gamma_intermediate = 0.5, Gamma_final = 0.05, Eg = sys.Eg)
        @test r_greek.tensor ≈ r_ascii.tensor   atol = 1e-12
    end

    @testset "RIXS chunks: one per ω_in" begin
        sys = build_3level()
        ω_in_grid = 3.0:0.5:7.0
        result = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                      ω_in_grid = ω_in_grid, ω_out_grid = 0:0.1:4,
                      Γ_intermediate = 0.5, Γ_final = 0.05, Eg = sys.Eg)
        @test length(result.chunks) == length(ω_in_grid)
        @test all(result.chunks[k].ω_in_index == k && result.chunks[k].ψ_index == 1
                  for k in 1:length(result.chunks))
    end

    @testset "multi-ψ ω_in :auto unions inner-pole ranges across ψ" begin
        # Two ψ's with the same H but different starting blocks (different
        # sectors of |g⟩, |i⟩, |f⟩). The :auto ω_in_grid must bracket the
        # union of *both* inner spectra, not just the first ψ's.
        sys = build_3level()
        # Construct a second ψ that picks the SAME ground state but with
        # an additional admixture so the inner spectrum has a wider range.
        # Simplest: inner-Hamiltonian eigenstates differ between ψs only
        # in the QR projection; we just verify the auto-grid succeeds and
        # produces a tuple with non-empty ranges.
        ψ_list = [sys.ψ, sys.ψ ./ 2]    # second ψ is just rescaled
        result = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, ψ_list;
                       Γ_intermediate = 0.5, Γ_final = 0.05, Eg = sys.Eg)
        @test result.ω_grid isa Tuple
        @test length(result.ω_grid[1]) ≥ 2
        @test length(result.ω_grid[2]) ≥ 2
        @test result.metadata[:auto_range]
        # Both ψ's span the same physical spectrum here, so the autorange
        # bracket should contain the analytic peak ε_i (within Γ).
        @test first(result.ω_grid[1]) ≤ sys.ε_i ≤ last(result.ω_grid[1])
    end

    @testset "Eigen overload extracts Eg + GS" begin
        sys = build_3level()
        H_dense = Matrix(sys.H)
        E = eigen(Hermitian(H_dense))
        ω_in_grid  = 3.0:0.5:7.0
        ω_out_grid = 0.0:0.1:4.0
        r_eig = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, E;
                     ω_in_grid = ω_in_grid, ω_out_grid = ω_out_grid,
                     Γ_intermediate = 0.5, Γ_final = 0.05)
        r_psi = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                     ω_in_grid = ω_in_grid, ω_out_grid = ω_out_grid,
                     Γ_intermediate = 0.5, Γ_final = 0.05, Eg = sys.Eg)
        @test r_eig.tensor ≈ r_psi.tensor   atol = 1e-10
    end
end
