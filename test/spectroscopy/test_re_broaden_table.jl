@testset "re_broaden_table — energy-dependent Lorentzian" begin
    using LinearAlgebra
    using MOAD: xas
    using MOAD.Spectroscopy: SpectraTensor, re_broaden, re_broaden_table

    # Reuse the 2-level XAS fixture from test_helpers.jl.
    function build_2level()
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)
        H = -1.0 * n(s, 1) + 4.0 * n(s, 2)
        H_sp = assemble(compile(H, b), b)
        T = cdag(s, 2) * c(s, 1)
        ψ = zeros(ComplexF64, length(b))
        for i in 1:length(b)
            w = get_state(b, i)[1]
            (w & 0x01) != 0 && (w & 0x02) == 0 && (ψ[i] = 1.0; break)
        end
        return (basis = b, H = H_sp, T = T, ψ = ψ, Eg = -1.0, peak = 5.0)
    end

    @testset "constant table reproduces re_broaden(spec; Γ)" begin
        sys = build_2level()
        ωs  = -1.0:0.2:8.0
        Γ_const = 0.5
        spec = xas(sys.H, sys.basis, sys.T, sys.ψ;
                   ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)

        spec_ref   = re_broaden(spec; Γ = Γ_const)
        spec_table = re_broaden_table(spec; lorentz_table = [
            first(spec.ω_grid) => Γ_const,
            last(spec.ω_grid)  => Γ_const,
        ])

        # Lehmann-sum identity vs cf evaluation: agreement to FP tolerance.
        @test maximum(abs, spec_ref.tensor .- spec_table.tensor) < 1e-12

        # Single-point table is also "constant"; same agreement.
        spec_table_one = re_broaden_table(spec;
            lorentz_table = [first(spec.ω_grid) => Γ_const])
        @test maximum(abs, spec_ref.tensor .- spec_table_one.tensor) < 1e-12
    end

    @testset "linear interpolation between two anchors" begin
        # MOAD.Spectroscopy._interp_clamped is the internal helper; reach
        # in via the submodule. With (ω₁, Γ₁) and (ω₂, Γ₂), Γ at the
        # midpoint must be (Γ₁ + Γ₂)/2; at the endpoints exactly Γ_i;
        # outside the table the values clamp.
        interp = MOAD.Spectroscopy._interp_clamped
        ωs = [0.0, 4.0]
        Γs = [0.1, 0.5]
        @test interp(0.0, ωs, Γs) ≈ 0.1
        @test interp(4.0, ωs, Γs) ≈ 0.5
        @test interp(2.0, ωs, Γs) ≈ 0.3            # midpoint
        @test interp(1.0, ωs, Γs) ≈ 0.2            # quarter
        @test interp(-5.0, ωs, Γs) ≈ 0.1           # clamp left
        @test interp(99.0, ωs, Γs) ≈ 0.5           # clamp right
    end

    @testset "energy-dependent Γ shifts pole height" begin
        # Single isolated pole at ω = 5.0. Use a table that puts the
        # pole at Γ = Γ_pole and compare to a constant scalar re_broaden
        # at the same Γ_pole — they must agree pointwise.
        sys = build_2level()
        ωs  = -1.0:0.2:8.0
        spec = xas(sys.H, sys.basis, sys.T, sys.ψ;
                   ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        Γ_pole = 0.3
        # Table with non-trivial slope, but evaluating to Γ_pole at the
        # pole's ω = peak = 5.0 (sys.peak − Eg − Eg = 5 − (−1)? No — the
        # spectroscopic ω of the pole is E − Eg. The peak in the spec
        # sits at ω = peak = sys.peak (test_helpers convention).
        ω_pole = sys.peak                     # 5.0
        # Linear table: Γ(0) = 0.05, Γ(10) = 0.05 + (10/ω_pole)*Γ_slope ...
        # Easier: anchor exactly at ω_pole so Γ(ω_pole) = Γ_pole, with
        # different values elsewhere just to prove "varying" works.
        spec_table = re_broaden_table(spec; lorentz_table = [
            -1.0    => 0.7,
            ω_pole  => Γ_pole,
            8.0     => 1.5,
        ])
        spec_const = re_broaden(spec; Γ = Γ_pole)
        # Only the pole at ω_pole contributes to the spectrum (other poles
        # have zero weight in this 2-level fixture). So both spectra must
        # agree pointwise to FP tolerance.
        @test maximum(abs, spec_table.tensor .- spec_const.tensor) < 1e-12
    end

    @testset "input validation" begin
        sys = build_2level()
        ωs  = -1.0:0.5:8.0
        spec = xas(sys.H, sys.basis, sys.T, sys.ψ;
                   ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)

        @test_throws ArgumentError re_broaden_table(spec)                       # missing table
        @test_throws ArgumentError re_broaden_table(spec; lorentz_table = Pair{Float64,Float64}[])
        @test_throws ArgumentError re_broaden_table(spec;
            lorentz_table = [0.0 => -0.1, 5.0 => 0.3])                          # negative Γ
        @test_throws ArgumentError re_broaden_table(spec;
            lorentz_table = [0.0 => 0.3, 5.0 => 0.5], gauss = -0.1)             # negative gauss
    end

    @testset "ASCII alias Lorentz_table works" begin
        sys = build_2level()
        ωs  = -1.0:0.2:8.0
        spec = xas(sys.H, sys.basis, sys.T, sys.ψ;
                   ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        a = re_broaden_table(spec; lorentz_table = [-1.0 => 0.3, 8.0 => 0.3])
        b = re_broaden_table(spec; Lorentz_table = [-1.0 => 0.3, 8.0 => 0.3])
        @test a.tensor ≈ b.tensor   atol = 1e-12
        # Specifying both raises.
        @test_throws ArgumentError re_broaden_table(spec;
            lorentz_table = [0.0 => 0.3], Lorentz_table = [0.0 => 0.3])
    end

    @testset "Gaussian convolution applied when gauss > 0" begin
        sys = build_2level()
        ωs  = -1.0:0.1:8.0
        spec = xas(sys.H, sys.basis, sys.T, sys.ψ;
                   ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        no_g  = re_broaden_table(spec;
            lorentz_table = [first(ωs) => 0.3, last(ωs) => 0.3])
        with_g = re_broaden_table(spec;
            lorentz_table = [first(ωs) => 0.3, last(ωs) => 0.3], gauss = 0.4)
        # Convolved spectrum must differ from the unconvolved one but
        # preserve the integrated weight (Gaussian is normalized, real-
        # space convolution along the ω axis).
        @test maximum(abs, with_g.tensor .- no_g.tensor) > 1e-3
        @test isapprox(sum(with_g.tensor), sum(no_g.tensor); rtol = 1e-2)
        @test with_g.metadata[:σ_gauss] == 0.4
    end
end
