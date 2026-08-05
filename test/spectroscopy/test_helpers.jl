@testset "Spectroscopy helpers" begin
    using LinearAlgebra
    using MOAD: xas
    using MOAD.Spectroscopy: SpectraTensor, find_chunk, tridiagonal,
                              BlockTriDiagonal, poles, re_broaden, polarise,
                              average, weighted_sum, restrict_to_window,
                              plot_range

    # Reuse the 2-level system from test_xas.jl with a unique GS.
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

    @testset "find_chunk + tridiagonal" begin
        sys = build_2level()
        ωs  = -1.0:0.5:8.0
        result = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        c = find_chunk(result; ψ_index = 1, ω_in_index = 0)
        @test c.ψ_index == 1
        @test c.raw_block_size == 1
        @test_throws ArgumentError find_chunk(result; ψ_index = 2)

        td = tridiagonal(c)
        @test td isa BlockTriDiagonal
        @test td.α === c.α
        @test td.β === c.β
        @test td.R === c.R
    end

    @testset "poles: Lehmann form reconstructs the cf" begin
        sys = build_2level()
        ωs  = -1.0:0.5:8.0
        Γ   = 0.2
        result = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = ωs, Γ = Γ, Eg = sys.Eg)
        ps = poles(result; ψ_index = 1)
        # 2-level system → after T·ψ_g, only one pole survives in the chunk.
        # (Block Lanczos may produce a few iterations even for trivial systems
        # because of basis padding; nontrivial poles must include peak = 5.0
        # at energy E = 4.0 above Eg.)
        peak_E = sys.Eg + sys.peak           # absolute energy of the peak pole
        # At least one pole must match `peak_E` with non-trivial weight.
        idx = findall(p -> abs(p.E - peak_E) < 1e-9, ps)
        @test !isempty(idx)
        # Reconstruct χ from poles and compare to the tensor.
        recon = [sum(p.weight[1, 1] / (ω + sys.Eg + im * Γ / 2 - p.E) for p in ps)
                 for ω in ωs]
        @test maximum(abs, recon .- result.tensor) < 1e-10
    end

    @testset "re_broaden: Lorentzian re-evaluation matches direct call" begin
        sys = build_2level()
        ωs  = -1.0:0.2:8.0
        Γ1  = 0.2
        Γ2  = 0.5
        r1  = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = ωs, Γ = Γ1, Eg = sys.Eg)
        r2_direct  = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = ωs, Γ = Γ2, Eg = sys.Eg)
        r2_rebroad = re_broaden(r1; Γ = Γ2)
        @test r2_rebroad.tensor ≈ r2_direct.tensor   atol = 1e-12
        @test r2_rebroad.metadata[:Γ] == Γ2
        @test r2_rebroad.metadata[:rebroadened_from] == (Γ = Γ1,)
    end

    @testset "re_broaden: ASCII alias works; Γ_intermediate rejected" begin
        sys = build_2level()
        ωs  = -1.0:0.5:8.0
        result = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        r_a = re_broaden(result; Γ = 0.4)
        r_b = re_broaden(result; Gamma = 0.4)
        @test r_a.tensor ≈ r_b.tensor   atol = 1e-12
        @test_throws ArgumentError re_broaden(result; Γ_intermediate = 0.5)
        @test_throws ArgumentError re_broaden(result; Γ_final = 0.5)        # XAS rejects Γ_final
    end

    @testset "polarise on scalar XAS: -Im of tensor" begin
        sys = build_2level()
        ωs  = -1.0:0.2:8.0
        result = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        s = polarise(result)
        @test s isa Vector{Float64}
        @test s ≈ -imag.(result.tensor)
        # Lorentzian peak is positive at ω = peak.
        peak_idx = argmax(s)
        @test ωs[peak_idx] ≈ sys.peak    atol = step(ωs)
    end

    @testset "polarise on tensor-form XAS contracts to scalar" begin
        sys = build_2level()
        ωs  = -1.0:0.2:8.0
        T_vec = [sys.T, 0.5 * sys.T]
        r_tensor = xas(sys.H, sys.basis, T_vec, sys.ψ; ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        # Contract with ε = (1, 0) — should equal the scalar XAS with operator T₁.
        ε = ComplexF64[1, 0]
        s_contracted = polarise(r_tensor, ε)
        r_scalar = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        @test s_contracted ≈ -imag.(r_scalar.tensor)   atol = 1e-10

        # Contract with ε = (0, 1) — operator is 0.5·T, so spectrum is 0.25·scalar.
        ε2 = ComplexF64[0, 1]
        s2 = polarise(r_tensor, ε2)
        @test s2 ≈ 0.25 .* (-imag.(r_scalar.tensor))   atol = 1e-10
    end

    @testset "polarise input validation" begin
        sys = build_2level()
        ωs  = -1.0:0.5:8.0
        r_scalar = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        # Calling polarise(scalar, ε) raises.
        @test_throws ArgumentError polarise(r_scalar, ComplexF64[1, 0])

        T_vec = [sys.T, 0.5 * sys.T]
        r_tensor = xas(sys.H, sys.basis, T_vec, sys.ψ; ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        # Wrong-length ε raises.
        @test_throws DimensionMismatch polarise(r_tensor, ComplexF64[1, 0, 0])
    end

    @testset "Spectrum algebra: +, -, *scalar, average, weighted_sum" begin
        sys = build_2level()
        ωs  = -1.0:0.5:8.0
        a   = xas(sys.H, sys.basis, sys.T, sys.ψ;       ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        b   = xas(sys.H, sys.basis, 0.5 * sys.T, sys.ψ; ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)

        s = a + b
        @test s.tensor ≈ a.tensor .+ b.tensor
        @test s.chunks === nothing
        @test s.metadata[:function] === :algebraic_combination

        d = a - b
        @test d.tensor ≈ a.tensor .- b.tensor

        m = 2.0 * a
        @test m.tensor ≈ 2.0 .* a.tensor

        m2 = a * 2.0
        @test m2.tensor ≈ m.tensor

        avg = average([a, b])
        @test avg.tensor ≈ 0.5 .* (a.tensor .+ b.tensor)

        w = weighted_sum([a, b], [0.3, 0.7])
        @test w.tensor ≈ 0.3 .* a.tensor .+ 0.7 .* b.tensor
    end

    @testset "Spectrum algebra: grid mismatch raises" begin
        sys = build_2level()
        a = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = -1.0:0.5:8.0, Γ = 0.2, Eg = sys.Eg)
        c = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = -1.0:0.25:8.0, Γ = 0.2, Eg = sys.Eg)
        @test_throws ArgumentError a + c
    end

    @testset "restrict_to_window crops ω axis" begin
        sys = build_2level()
        ωs  = -1.0:0.2:8.0
        result = xas(sys.H, sys.basis, sys.T, sys.ψ; ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        cropped = restrict_to_window(result, (4.0, 6.0))
        @test all(4.0 .≤ cropped.ω_grid .≤ 6.0)
        @test cropped.metadata[:restrict_window] == (4.0, 6.0)
        # ω_grid endpoints lie inside the window.
        @test first(cropped.ω_grid) ≥ 4.0
        @test last(cropped.ω_grid)  ≤ 6.0
        @test_throws ArgumentError restrict_to_window(result, (100.0, 200.0))
    end

    @testset "plot_range :peak rule" begin
        sys = build_2level()
        result = xas(sys.H, sys.basis, sys.T, sys.ψ; Γ = 0.2, Eg = sys.Eg)
        lo, hi = plot_range(result; threshold = 0.01)
        @test lo ≤ sys.peak ≤ hi
        @test lo ≥ first(result.ω_grid)
        @test hi ≤ last(result.ω_grid)
    end

    @testset "plot_range :integrated rule" begin
        sys = build_2level()
        result = xas(sys.H, sys.basis, sys.T, sys.ψ; Γ = 0.2, Eg = sys.Eg)
        lo, hi = plot_range(result; rule = :integrated, q = 0.05)
        @test lo ≤ sys.peak ≤ hi
    end

    # ---------------------------------------------------------------
    # Round-2 review fixes — RIXS polarise + RIXS re_broaden +
    # restrict_to_window for RIXS
    # ---------------------------------------------------------------

    function build_3level_rixs(; ε_i = 5.0, ε_f = 2.0)
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
        return (basis = b, H = H_sp, T_in = T_in, T_out = T_out, ψ = ψ,
                Eg = 0.0, ε_i = ε_i, ε_f = ε_f)
    end

    @testset "polarise(rixs, ε_in, ε_out): tensor-form contraction" begin
        using MOAD: rixs
        sys = build_3level_rixs()
        ω_in_grid  = 3.0:0.5:7.0
        ω_out_grid = 0.0:0.1:4.0
        Γ_i = 0.5; Γ_f = 0.05

        # Tensor RIXS with [T_in, 0.5·T_in] and [T_out, 0.5·T_out].
        T_in_v  = [sys.T_in,  0.5 * sys.T_in]
        T_out_v = [sys.T_out, 0.5 * sys.T_out]
        r_tensor = rixs(sys.H, sys.H, sys.basis, T_in_v, T_out_v, sys.ψ;
                        ω_in_grid = ω_in_grid, ω_out_grid = ω_out_grid,
                        Γ_intermediate = Γ_i, Γ_final = Γ_f, Eg = sys.Eg)
        @test ndims(r_tensor.tensor) == 6

        # Scalar reference: same H, scalar T_in, scalar T_out.
        r_scalar = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                        ω_in_grid = ω_in_grid, ω_out_grid = ω_out_grid,
                        Γ_intermediate = Γ_i, Γ_final = Γ_f, Eg = sys.Eg)

        # Contract tensor with ε_in = (1, 0), ε_out = (1, 0) → first
        # operator of each → matches scalar RIXS.
        ε_first = ComplexF64[1, 0]
        s_contracted = polarise(r_tensor, ε_first, ε_first)
        @test s_contracted ≈ -imag.(r_scalar.tensor)   atol = 1e-9

        # Contract with ε = (0, 1) → second operator (0.5·T) on both sides.
        # Scaling: each ε contributes its operator scale through χ.
        # The 4-index χ_{ijkl} with operators [T, 0.5T] has T_a = T*scale[a].
        # ε*_in[i] T_a has scale[i] (for ε=(0,1) → scale[2] = 0.5).
        # ε_out[j] T_b has scale[j] = 0.5. ε*_out[k] T_c has scale[k] = 0.5.
        # ε_in[l] T_d has scale[l] = 0.5. Total: 0.5⁴ = 1/16.
        ε_second = ComplexF64[0, 1]
        s_second = polarise(r_tensor, ε_second, ε_second)
        @test s_second ≈ (1/16) .* (-imag.(r_scalar.tensor))   atol = 1e-9
    end

    @testset "polarise(rixs, ε_in, ε_out): input validation" begin
        using MOAD: rixs
        sys = build_3level_rixs()
        # Scalar T_in + scalar T_out RIXS: polarise(_, ε, ε) should reject.
        r = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                 ω_in_grid = 3.0:0.5:7.0, ω_out_grid = 0.0:0.1:4.0,
                 Γ_intermediate = 0.5, Γ_final = 0.05, Eg = sys.Eg)
        @test_throws ArgumentError polarise(r, ComplexF64[1], ComplexF64[1])

        # Tensor case: wrong-length ε raises.
        r_tensor = rixs(sys.H, sys.H, sys.basis,
                         [sys.T_in], [sys.T_out, 0.5 * sys.T_out], sys.ψ;
                         ω_in_grid = 3.0:0.5:7.0, ω_out_grid = 0.0:0.1:4.0,
                         Γ_intermediate = 0.5, Γ_final = 0.05, Eg = sys.Eg)
        @test_throws DimensionMismatch polarise(r_tensor,
                                                ComplexF64[1, 0],   # wrong: N_in=1
                                                ComplexF64[1, 0])
    end

    @testset "re_broaden(rixs; Γ_final): replays cf at new Γ" begin
        using MOAD: rixs
        sys = build_3level_rixs()
        ω_in_grid  = 3.5:0.5:6.5
        ω_out_grid = 1.0:0.1:3.0

        r_05 = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                    ω_in_grid = ω_in_grid, ω_out_grid = ω_out_grid,
                    Γ_intermediate = 0.5, Γ_final = 0.05, Eg = sys.Eg)
        # Re-broaden at Γ_final = 0.2 should match a fresh rixs() call.
        r_20_direct = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                           ω_in_grid = ω_in_grid, ω_out_grid = ω_out_grid,
                           Γ_intermediate = 0.5, Γ_final = 0.2, Eg = sys.Eg)
        r_20_rb = re_broaden(r_05; Γ_final = 0.2)
        @test r_20_rb.tensor ≈ r_20_direct.tensor   atol = 1e-10
        @test r_20_rb.metadata[:Γ_final] == 0.2
        @test r_20_rb.metadata[:rebroadened_from] == (Γ_final = 0.05,)

        # `Γ` alias is accepted on RIXS too (canonical kwarg there is `Γ_final`).
        r_20_alias = re_broaden(r_05; Γ = 0.2)
        @test r_20_alias.tensor ≈ r_20_rb.tensor   atol = 1e-12
    end

    @testset "restrict_to_window: RIXS keeps ω_in axis" begin
        using MOAD: rixs
        sys = build_3level_rixs()
        ω_in_grid  = 3.0:0.5:7.0
        ω_out_grid = 0.0:0.1:4.0
        result = rixs(sys.H, sys.H, sys.basis, sys.T_in, sys.T_out, sys.ψ;
                       ω_in_grid = ω_in_grid, ω_out_grid = ω_out_grid,
                       Γ_intermediate = 0.5, Γ_final = 0.05, Eg = sys.Eg)
        cropped = restrict_to_window(result, (1.0, 3.0))
        # ω_grid is now still a tuple.
        @test cropped.ω_grid isa Tuple
        # ω_in unchanged.
        @test cropped.ω_grid[1] === ω_in_grid
        # ω_out cropped.
        @test all(1.0 .≤ cropped.ω_grid[2] .≤ 3.0)
        # Tensor shape: ω_in dim preserved, ω_out dim shrunk.
        @test size(cropped.tensor, 1) == length(ω_in_grid)
        @test size(cropped.tensor, 2) == length(cropped.ω_grid[2])
    end
end
