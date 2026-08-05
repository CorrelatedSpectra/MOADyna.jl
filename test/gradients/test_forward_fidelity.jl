# test/gradients/test_forward_fidelity.jl
#
# Build-step 7 of the v0.3 differentiable forward model (MOAD.Gradients): the
# forward-fidelity / "useful, not vacuum" gate. The differentiable XAS forward
# (XASGradientModel.spectrum, built from preassembled operator matrices) must
# reproduce MOAD's own validated physical xas()/Lanczos path on the SAME ingredients.
#
# B1 — 2-level system, embedding E = I (exact, single basis): the complex correlator
#      and the intensity match xas() and the analytic single-pole result.
# B2 — a genuine core→valence multiplet (Ni d⁸ → 2p⁵d⁹ L-edge) WITH a real ground→
#      core-hole embedding: spectrum matches xas() (pinned to exact Lanczos) per
#      polarization, in both the complex correlator and the intensity.
#
# (Minimal-fixture note: d⁰→d¹ would be smaller, but a d⁰ ground sector is a SINGLE
# state — no gap for `groundstate`. d⁸ is the canonical NiO anchor and its ³A₂g ground
# term is orbital-singlet, so a tiny Sz field gives a clean non-degenerate ground.)
# Reproducible (no RNG).

using MOAD
using LinearAlgebra
using SparseArrays
using Test

const Gr = MOAD.Gradients

@testset "Gradients build-step 7: forward-fidelity vs xas()" begin

    # ------------------------------------------------------------------
    # B1 — 2-level system, E = I (mirrors test/spectroscopy/test_xas.jl)
    # ------------------------------------------------------------------
    @testset "B1: 2-level, E=I, exact" begin
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)                          # 4 states
        ε_c, ε_v = -1.0, 4.0
        H   = ε_c * n(s, 1) + ε_v * n(s, 2)
        Hsp = assemble(compile(H, b), b)
        T   = cdag(s, 2) * c(s, 1)                 # core → valence
        Tsp = assemble(compile(T, b), b)
        # ground |10⟩ (mode 1 occupied, mode 2 empty)
        ψ = zeros(ComplexF64, length(b))
        idx = findfirst(i -> (get_state(b, i)[1] & 0x01) != 0 &&
                             (get_state(b, i)[1] & 0x02) == 0, 1:length(b))
        ψ[idx] = 1.0
        Eg = ε_c

        Γ = 0.2;  ωs = collect(-1.0:0.25:8.0)

        # differentiable path: constant affine wrappers, E = I, A = B = (T,)
        gm = Gr.AffineModel([Matrix{ComplexF64}(Hsp)], Gr.AffineMap([1.0], zeros(1, 1)), [:x])
        model = Gr.XASGradientModel(gm, gm, (Matrix{ComplexF64}(Tsp),),
                                    (Matrix{ComplexF64}(Tsp),),
                                    Matrix{ComplexF64}(I, 4, 4), ωs; Γ = Γ)
        C = Gr.xas_response_C(model.g_model, model.f_model, model.A, model.B,
                              model.embedding, [0.0], ωs; Γ = Γ)

        # physical path
        res = xas(Hsp, b, T, ψ; ω_grid = ωs, Γ = Γ, Eg = Eg)

        # analytic single pole χ(ω) = 1/(ω − (ε_v−ε_c) + iΓ/2)
        peak = ε_v - ε_c
        ref = [1 / (ω - peak + im * Γ / 2) for ω in ωs]

        @test maximum(abs, C[1, 1, :] .- res.tensor) < 1e-8        # vs xas() (complex C)
        @test maximum(abs, C[1, 1, :] .- ref) < 1e-8               # vs analytic
        S = Gr.spectrum(model, [0.0])
        @test maximum(abs, S[1, 1, :] .- (-imag(ref) ./ π)) < 1e-8 # intensity
    end

    # ------------------------------------------------------------------
    # B2 — Ni d⁸ → 2p⁵d⁹ L-edge WITH a real embedding
    # ------------------------------------------------------------------
    @testset "B2: d⁸→d⁹ multiplet, embedding, vs exact Lanczos" begin
        m = ShellModel([:Ni_2p, :Ni_3d])
        # ground (2p⁶ d⁸) and core-hole (2p∈{5,6}, total = 14 ⇒ 2p⁶d⁸ ⊕ 2p⁵d⁹) bases
        basis_gs  = basis(m, nshells(m, :Ni_2p) == 6, nshells(m, :Ni_3d) == 8)
        basis_xas = basis(m, nshells(m, :Ni_2p) in 5:6, total(m) == 14)
        @test length(basis_gs) == 45
        @test length(basis_xas) == 45 + 60

        # one operator expression, assembled on BOTH bases (so the n_2p=6 block of
        # H_XAS equals H_GS ⇒ the embedded ground state is an H_XAS eigenstate with the
        # same E0). A tiny Sz field lifts the ³A₂g spin triplet → non-degenerate ground.
        H = coulomb(m, :Ni_3d; U = 0.0, F = (5.0, 3.0)) +
            0.8 * Akm(m, :Ni_3d, :Oh, [0.6, -0.4]) +
            1.0 * LS(m, :Ni_2p) +
            5.0 * n(m, :Ni_2p) +
            0.05 * Sz(m, :Ni_3d)
        H_gs  = assemble(compile(H, basis_gs), basis_gs)
        H_xas = assemble(compile(H, basis_xas), basis_xas)
        T = dipole(m, :Ni_2p => :Ni_3d)            # [T_x, T_y, T_z] OperatorSums

        # explicit embedding isometry E (N_f × N_g)
        Ng, Nf = length(basis_gs), length(basis_xas)
        E = zeros(ComplexF64, Nf, Ng)
        for i in 1:Ng
            j = get_index(basis_xas, get_state(basis_gs, i))
            j > 0 && (E[j, i] = 1.0)
        end
        @test norm(E' * E - I) < 1e-12

        # ground state (non-degenerate thanks to the Sz field)
        gm = Gr.AffineModel([Matrix{ComplexF64}(H_gs)], Gr.AffineMap([1.0], zeros(1, 1)), [:x])
        E0, ψ0, gap = Gr.groundstate(gm, [0.0])
        @test gap > 1e-6
        @test E * ψ0 ≈ embed(ψ0, basis_gs => basis_xas)            # hand-built E == embed

        fm = Gr.AffineModel([Matrix{ComplexF64}(H_xas)], Gr.AffineMap([1.0], zeros(1, 1)), [:x])
        Γ = 0.6;  ωs = collect(range(-8.0, 14.0; length = 45))
        kd = length(basis_xas)                     # pin Lanczos to exact

        ψ0_f = E * ψ0
        for (α, Tα) in enumerate(T)
            Tα_sp = assemble(compile(Tα, basis_xas), basis_xas)
            model = Gr.XASGradientModel(gm, fm, (Matrix{ComplexF64}(Tα_sp),),
                                        (Matrix{ComplexF64}(Tα_sp),), E, ωs; Γ = Γ)
            C = Gr.xas_response_C(model.g_model, model.f_model, model.A, model.B,
                                  model.embedding, [0.0], ωs; Γ = Γ)
            res = xas(H_xas, basis_xas, Tα, ψ0_f; ω_grid = ωs, Γ = Γ, Eg = E0,
                      krylovdim = kd, reorth = :full, tol = 0.0)
            @test maximum(abs, C[1, 1, :] .- res.tensor) < 1e-6    # complex correlator
            S = Gr.spectrum(model, [0.0])
            @test maximum(abs, S[1, 1, :] .- (-imag(res.tensor) ./ π)) < 1e-6  # intensity
        end
    end
end
