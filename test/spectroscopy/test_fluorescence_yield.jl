@testset "Spectroscopy fluorescence_yield smoke tests" begin
    using LinearAlgebra
    using SparseArrays
    using MOAD: fluorescence_yield, rixs
    using MOAD.Spectroscopy: SpectraTensor

    # Same 3-level system as test_rixs.jl. With T_excite, T_decay
    # producing |i⟩ and |f⟩ respectively, the analytic FY is a single
    # Lorentzian centred at ω_in = ε_i with FWHM = Γ_intermediate:
    #
    #   FY(ω_in) = π / [(ω_in − ε_i)² + (Γ_int/2)²]
    function build_3level(; ε_i = 5.0, ε_f = 2.0)
        s = FermionSite{3}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h, n_fermion(h) == 1)
        H = 0.0 * n(s, 1) + ε_i * n(s, 2) + ε_f * n(s, 3)
        H_sp = assemble(compile(H, b), b)
        T_excite = cdag(s, 2) * c(s, 1)
        T_decay  = cdag(s, 2) * c(s, 3)
        ψ = zeros(ComplexF64, length(b))
        for i in 1:length(b)
            w = get_state(b, i)[1]
            (w & 0x01) != 0 && (w & 0x06) == 0 && (ψ[i] = 1.0; break)
        end
        return (basis = b, H = H_sp, T_excite = T_excite, T_decay = T_decay,
                ψ = ψ, Eg = 0.0, ε_i = ε_i, ε_f = ε_f)
    end

    @testset "scalar T_excite, scalar T_decay, single ψ → 1-D real tensor" begin
        sys = build_3level()
        Γ_i = 0.5
        ω_in_grid = 0.0:0.05:10.0
        result = fluorescence_yield(sys.H, sys.basis,
                                     sys.T_excite, sys.T_decay, sys.ψ;
                                     ω_in_grid      = ω_in_grid,
                                     Γ_intermediate = Γ_i,
                                     Eg             = sys.Eg)

        @test result isa SpectraTensor
        @test eltype(result.tensor) === Float64
        @test ndims(result.tensor) == 1
        @test size(result.tensor) == (length(ω_in_grid),)
        @test result.metadata[:function] === :fluorescence_yield

        # Compare to analytic FY(ω_in) = π / [(ω_in − ε_i)² + (Γ_i/2)²].
        ref = [π / ((ω_in - sys.ε_i)^2 + (Γ_i / 2)^2) for ω_in in ω_in_grid]
        @test maximum(abs, result.tensor .- ref) < 1e-10
    end

    @testset "FY peak at ω_in = ε_i" begin
        sys = build_3level()
        result = fluorescence_yield(sys.H, sys.basis,
                                     sys.T_excite, sys.T_decay, sys.ψ;
                                     Γ_intermediate = 0.5, Eg = sys.Eg)
        @test result.ω_grid isa AbstractRange
        idx = argmax(result.tensor)
        @test result.ω_grid[idx] ≈ sys.ε_i   atol = step(result.ω_grid)
    end

    @testset "ASCII alias for Greek kwargs" begin
        sys = build_3level()
        ω_in_grid = 0.0:0.5:10.0
        r_a = fluorescence_yield(sys.H, sys.basis, sys.T_excite, sys.T_decay, sys.ψ;
                                  ω_in_grid = ω_in_grid,
                                  Γ_intermediate = 0.5, Eg = sys.Eg)
        r_b = fluorescence_yield(sys.H, sys.basis, sys.T_excite, sys.T_decay, sys.ψ;
                                  omega_in_grid = ω_in_grid,
                                  Gamma_intermediate = 0.5, Eg = sys.Eg)
        @test r_a.tensor ≈ r_b.tensor   atol = 1e-12
    end

    @testset "S3 sanity: FY ≈ ∫ Σ_kl(-Im RIXS) dω_out" begin
        # Chapter §11.S3: FY equals the analytic integral of the
        # corresponding RIXS map (Σ over decay channels k of −Im of
        # the diagonal in the kkll formula). With single T_excite +
        # single T_decay this reduces to π · ⟨ψ|...|ψ⟩, which is the
        # FY direct algorithm. Empirically: FY(ω_in) should match
        # ∫ -Im χ_RIXS(ω_in, ω_out) dω_out within 1% over a wide ω_out window.
        sys = build_3level()
        Γ_i = 0.8
        Γ_f = 0.05
        ω_in_grid  = 3.5:0.25:6.5
        ω_out_grid = -10.0:0.02:14.0     # very wide so integral is essentially exact
        r_rixs = rixs(sys.H, sys.H, sys.basis, sys.T_excite, sys.T_decay, sys.ψ;
                      ω_in_grid       = ω_in_grid,
                      ω_out_grid      = ω_out_grid,
                      Γ_intermediate  = Γ_i,
                      Γ_final         = Γ_f,
                      Eg              = sys.Eg)
        r_fy = fluorescence_yield(sys.H, sys.basis,
                                   sys.T_excite, sys.T_decay, sys.ψ;
                                   ω_in_grid       = ω_in_grid,
                                   Γ_intermediate  = Γ_i,
                                   Eg              = sys.Eg)

        # Trapezoid integral of -Im RIXS over ω_out at each ω_in.
        Δ = step(ω_out_grid)
        integrals = [sum(-imag.(r_rixs.tensor[i, :])) * Δ for i in eachindex(ω_in_grid)]
        @test maximum(abs(integrals[i] - r_fy.tensor[i]) /
                      max(abs(r_fy.tensor[i]), eps()) for i in eachindex(ω_in_grid)) < 1e-2
    end

    @testset "list ψ → (N_ψ, n_ω_in)" begin
        sys = build_3level()
        ω_in_grid = 0.0:0.5:10.0
        ψ_list = [sys.ψ, sqrt(2) * sys.ψ]
        r = fluorescence_yield(sys.H, sys.basis, sys.T_excite, sys.T_decay, ψ_list;
                                ω_in_grid = ω_in_grid,
                                Γ_intermediate = 0.5, Eg = sys.Eg)
        @test ndims(r.tensor) == 2
        @test size(r.tensor) == (2, length(ω_in_grid))
        # FY scales linearly with ‖ψ‖² (Y = G_int · T·ψ scales linearly,
        # so ‖Z‖² scales as ‖ψ‖²). ‖√2·ψ‖² / ‖ψ‖² = 2 → FY ratio = 2.
        @test r.tensor[2, :] ≈ 2 .* r.tensor[1, :]   atol = 1e-10
    end

    @testset "Eigen overload extracts Eg" begin
        sys = build_3level()
        H_dense = Matrix(sys.H)
        E = eigen(Hermitian(H_dense))
        r_e = fluorescence_yield(sys.H, sys.basis, sys.T_excite, sys.T_decay, E;
                                  ω_in_grid = 3.0:0.5:7.0,
                                  Γ_intermediate = 0.5)
        r_p = fluorescence_yield(sys.H, sys.basis, sys.T_excite, sys.T_decay, sys.ψ;
                                  ω_in_grid = 3.0:0.5:7.0,
                                  Γ_intermediate = 0.5, Eg = sys.Eg)
        @test r_e.tensor ≈ r_p.tensor   atol = 1e-10
    end
end
