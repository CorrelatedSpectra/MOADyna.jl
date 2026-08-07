# test/gradients/test_affine_model.jl
#
# Build-step 1 of the v0.3 differentiable forward model (MOADyna.Gradients): the
# affine Hamiltonian model H(θ) = Σ_i c_i(θ) M_i and its directional derivative,
# finite-difference-validated on a compact NiO ligand-field ground-state
# Hamiltonian. No eigensolver here. Reproducible: every parameter is fixed below
# (no RNG).

using MOADyna
using LinearAlgebra
using SparseArrays
using Test

const G = MOADyna.Gradients

@testset "Gradients build-step 1: affine model" begin
    # --- Compact NiO ligand-field ground-state model (two shells) -----------
    m = ShellModel([:Ni_3d, :L_3d])
    b = basis(m, total(m) == 18)            # d⁸ + L¹⁰ manifold (190 states)

    # Reference physical parameters (eV). Representative values; the test only
    # needs them nonzero and an affine map around them.
    U_dd = 7.3
    F2_0 = 10.0
    F4_0 = 6.2
    ζ_0  = 0.083
    Δ0   = 4.7
    r0   = 0.8                              # Slater reduction (scaled_80)
    TenDqNi0, TenDqL0 = 0.56, 1.44
    Veg0, Vt2g0       = 2.06, 1.21

    asm(op) = assemble(compile(op, b), b)

    # --- Preassembled unit-coefficient term matrices (built ONCE) -----------
    # Directional Coulomb operators: one per independent Slater coefficient.
    # coulomb(U,(f2,f4)) = U·M_U + f2·M_F2 + f4·M_F4 (induced F⁰ captured in each).
    M_U   = asm(coulomb(m, :Ni_3d; U = 1.0, F = (0.0, 0.0)))
    M_F2  = asm(coulomb(m, :Ni_3d; U = 0.0, F = (1.0, 0.0)))
    M_F4  = asm(coulomb(m, :Ni_3d; U = 0.0, F = (0.0, 1.0)))
    M_AkN = asm(Akm(m, :Ni_3d, :Oh, [0.6, -0.4]))
    M_AkL = asm(Akm(m, :L_3d,  :Oh, [0.6, -0.4]))
    M_hEg = asm(hop(m, :Ni_3d, :L_3d, :Oh; irrep = :Eg))
    M_hT2 = asm(hop(m, :Ni_3d, :L_3d, :Oh; irrep = :T2g))
    M_LS  = asm(LS(m, :Ni_3d))
    N_Ni  = asm(n(m, :Ni_3d))
    N_L   = asm(n(m, :L_3d))

    terms = [M_U, M_F2, M_F4, M_AkN, M_AkL, M_hEg, M_hT2, M_LS, N_Ni, N_L]
    @test all(t -> size(t) == size(M_U), terms)     # conformable (same basis)

    # --- ZSA onsite energies and their constant Δ-Jacobian ------------------
    anchors = [(Ni_3d = 8, L_3d = 10) => 0.0,
               (Ni_3d = 9, L_3d = 9)  => Δ0]
    ε, dε_dE, A = G.zsa_onsite([:Ni_3d, :L_3d], anchors; U = (Ni_3d = U_dd,))
    # Cross-check the reconstruction against the shipped solver.
    es_ref = onsite_energies(m; U = (Ni_3d = U_dd,), anchors = anchors)
    @test ε[1] ≈ es_ref.Ni_3d
    @test ε[2] ≈ es_ref.L_3d

    dεNi_dΔ = dε_dE[1, 2]       # Ni_3d onsite vs anchor-2 energy (= Δ)
    dεL_dΔ  = dε_dE[2, 2]

    # --- Affine coefficient map  c(θ) = c0 + J·θ ----------------------------
    # θ = [r, Δ, TenDqNi, TenDqL, Veg, Vt2g, ζ]   (U_dd fixed, not a knob)
    names = [:r, :Δ, :TenDqNi, :TenDqL, :Veg, :Vt2g, :ζ]
    nθ, nt = length(names), length(terms)
    c0 = zeros(nt)
    J  = zeros(nt, nθ)
    c0[1] = U_dd                            # M_U: fixed U
    J[2, 1] = F2_0;  J[3, 1] = F4_0         # M_F2, M_F4: r·F_k0
    J[4, 3] = 1.0;   J[5, 4] = 1.0          # Akm: 10Dq (Ni, L)
    J[6, 5] = 1.0;   J[7, 6] = 1.0          # hop: V (eg, t2g)
    J[8, 7] = 1.0                           # LS: ζ
    # onsite: ε_s(Δ) = ε0_s + (∂ε_s/∂Δ)·(Δ − Δ0)
    c0[9]  = ε[1] - dεNi_dΔ * Δ0;  J[9, 2]  = dεNi_dΔ
    c0[10] = ε[2] - dεL_dΔ  * Δ0;  J[10, 2] = dεL_dΔ

    cmap  = G.AffineMap(c0, J)
    model = G.AffineModel(terms, cmap, names)
    @test G.n_params(model) == nθ
    @test G.n_terms(model)  == nt

    θ_ref = [r0, Δ0, TenDqNi0, TenDqL0, Veg0, Vt2g0, ζ_0]
    δ = 1e-4

    # === (a) scalar coefficient-Jacobian precheck (no matrices) =============
    for k in 1:nθ
        ek = zeros(nθ); ek[k] = 1.0
        fd = (G.coeffs(cmap, θ_ref .+ δ .* ek) .-
              G.coeffs(cmap, θ_ref .- δ .* ek)) ./ (2δ)
        @test fd ≈ G.coeff_jacobian(cmap, θ_ref)[:, k] atol = 1e-9
    end

    # === (b) affinity / dH: central FD == dH to roundoff; 2nd diff ≈ 0 ======
    H0 = G.hamiltonian(model, θ_ref)
    scaleH = max(1.0, maximum(abs, H0))
    for k in 1:nθ
        ek = zeros(nθ); ek[k] = 1.0
        Hp = G.hamiltonian(model, θ_ref .+ δ .* ek)
        Hm = G.hamiltonian(model, θ_ref .- δ .* ek)
        fd  = (Hp - Hm) / (2δ)
        dHk = G.dhamiltonian(model, θ_ref, ek)
        @test maximum(abs, fd - dHk) < 1e-7 * scaleH       # affine ⇒ near-exact
        @test maximum(abs, Hp - 2 * H0 + Hm) < 1e-7 * scaleH  # 2nd difference ≈ 0
    end

    # === (c) faithfulness: decomposed H == monolithic H, ref + off-ref θ ====
    function monolithic(θ)
        r, Δ, tNi, tL, veg, vt2, ζ = θ
        es = onsite_energies(m; U = (Ni_3d = U_dd,),
                             anchors = [(Ni_3d = 8, L_3d = 10) => 0.0,
                                        (Ni_3d = 9, L_3d = 9)  => Δ])
        Hop = coulomb(m, :Ni_3d; U = U_dd, F = (r * F2_0, r * F4_0)) +
              ζ   * LS(m, :Ni_3d) +
              tNi * Akm(m, :Ni_3d, :Oh, [0.6, -0.4]) +
              tL  * Akm(m, :L_3d,  :Oh, [0.6, -0.4]) +
              veg * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :Eg) +
              vt2 * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :T2g) +
              es.Ni_3d * n(m, :Ni_3d) + es.L_3d * n(m, :L_3d)
        return asm(Hop)
    end
    @test maximum(abs, G.hamiltonian(model, θ_ref) - monolithic(θ_ref)) < 1e-8
    # Off-reference θ (fixed perturbation, no RNG) — guards against a missing
    # term that happens to cancel at θ_ref.
    θ_off = θ_ref .+ [0.03, -0.5, 0.02, -0.10, 0.40, -0.20, 0.005]
    @test maximum(abs, G.hamiltonian(model, θ_off) - monolithic(θ_off)) < 1e-8

    # hamiltonian/dhamiltonian reject wrong-length θ / θ̇ (dhamiltonian guards BOTH,
    # so a non-affine coefficient map that actually uses θ can't be mis-evaluated).
    @test_throws DimensionMismatch G.hamiltonian(model, θ_ref[1:end-1])
    @test_throws DimensionMismatch G.dhamiltonian(model, θ_ref[1:end-1], zeros(nθ))
    @test_throws DimensionMismatch G.dhamiltonian(model, θ_ref, zeros(nθ - 1))

    # === (d) rank/conditioning guard: degenerate anchors raise ==============
    bad = [(Ni_3d = 8, L_3d = 10) => 0.0,
           (Ni_3d = 8, L_3d = 10) => 1.0]   # identical configs ⇒ rank-deficient A
    @test_throws ArgumentError G.zsa_onsite([:Ni_3d, :L_3d], bad; U = (Ni_3d = U_dd,))

    # typo'd anchor key (not in target_shells) must raise, not silently mis-solve
    typo = [(Ni_3d = 8, L_3d = 10) => 0.0,
            (Ni_d3 = 9, L_3d = 9)  => Δ0]   # `Ni_d3` is a typo for `Ni_3d`
    @test_throws ArgumentError G.zsa_onsite([:Ni_3d, :L_3d], typo; U = (Ni_3d = U_dd,))
end
