# =====================================================================
# NiO L_{2,3}-edge XAS — native end-to-end acceptance test (Plan 2 / T24)
# =====================================================================
#
# Builds the canonical NiO L_{2,3} XAS Hamiltonian end-to-end using
# native MOADyna primitives (`onsite_energies`, `coulomb`, `LS`, `Akm`,
# `hop`, `dipole`) — NO `MOADyna.QuantyIO.read_quanty_operator` anywhere.
#
# Compares the resulting MOADyna spectrum tensor to the PyQuanty reference
# (`XAS_lanczos_cont_frac.txt`) across all three Cartesian polarisations.
# Plan 2 target: residual ≤ 1e-4 of peak.
#
# Reference data lives at
#     docs/dev/validation/spectroscopy/nio_xas/reference/
# and is tracked in the repo. A missing reference is reported as a
# test failure (not a skip) — the validation tier should fail loud
# if a tracked fixture has been removed.
#
# This is the LOAD-BEARING acceptance test for Plan 2: it exercises
# every Plan 2 surface (ShellModel, basis DSL, onsite_energies, Akm,
# hop, LS, coulomb, dipole, eigen, embed, xas) in a single end-to-end
# path with no QuantyIO crutch.

@testset "NiO L_{2,3} XAS — native (no read_quanty_operator)" begin
    using DelimitedFiles
    using LinearAlgebra: norm
    using MOADyna: xas

    # ---------------------------------------------------------------
    # Reference-data presence check
    # ---------------------------------------------------------------
    val_dir  = joinpath(@__DIR__, "..", "..", "..",
                        "docs", "dev", "validation", "spectroscopy", "nio_xas")
    ref_file = joinpath(val_dir, "reference", "XAS_lanczos_cont_frac.txt")
    # The reference is tracked in the repo; missing means the working tree
    # was tampered with. Reported as a test FAILURE so the validation
    # tier shows red, then short-circuit so the rest of the testset
    # doesn't avalanche into noisy follow-on errors.
    @test isfile(ref_file)
    isfile(ref_file) || return

    # ---------------------------------------------------------------
    # Shell model and parameters (verbatim from Quanty NiO.lua)
    # ---------------------------------------------------------------
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])

    F2dd, F4dd          = 11.14, 6.87
    F2pd, G1pd, G3pd    = 6.67, 4.92, 2.80
    zeta_3d, zeta_2p    = 0.081, 11.51
    tenDq, tenDqL       = 0.56, 1.44
    Veg, Vt2g           = 2.06, 1.21
    Udd, Upd, Delta     = 7.3, 8.5, 4.7
    Bz, Hz              = 0.000001, 0.120

    # ---------------------------------------------------------------
    # Onsite energies — anchor solver
    # ---------------------------------------------------------------
    # GS sub-system: Ni_3d ⊕ L_3d only (Ni_2p stays full, no anchor needed).
    es_gs = onsite_energies(m;
        shells  = (:Ni_3d, :L_3d),
        U       = (Ni_3d = Udd,),
        anchors = [(Ni_3d = 8, L_3d = 10) => 0.0,
                   (Ni_3d = 9, L_3d = 9)  => Delta])
    ed = es_gs.Ni_3d
    eL = es_gs.L_3d

    # XAS sub-system: all three shells, with Upd cross-shell coupling.
    es_xas = onsite_energies(m;
        U       = (Ni_3d = Udd,),
        pairs   = ((:Ni_2p, :Ni_3d) => Upd,),
        anchors = [(Ni_2p = 6, Ni_3d = 8, L_3d = 10) => 0.0,
                   (Ni_2p = 6, Ni_3d = 9, L_3d = 9)  => Delta,
                   (Ni_2p = 5, Ni_3d = 9, L_3d = 10) => 0.0])
    ep_x = es_xas.Ni_2p
    ed_x = es_xas.Ni_3d
    eL_x = es_xas.L_3d

    # ---------------------------------------------------------------
    # Hamiltonian — common piece (3d block; identical in GS and XAS)
    # ---------------------------------------------------------------
    H_common = coulomb(m, :Ni_3d; U = Udd, F = (F2dd, F4dd)) +
               zeta_3d * LS(m, :Ni_3d) +
               tenDq  * Akm(m, :Ni_3d, :Oh, [0.6, -0.4]) +
               tenDqL * Akm(m, :L_3d,  :Oh, [0.6, -0.4]) +
               Veg  * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :Eg)  +
               Vt2g * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :T2g) +
               Bz * (2 * Sz(m, :Ni_3d) + Lz(m, :Ni_3d)) +
               Hz * Sz(m, :Ni_3d)

    H_GS = H_common + ed * n(m, :Ni_3d) + eL * n(m, :L_3d)

    H_XAS = H_common +
            coulomb(m, :Ni_2p, :Ni_3d; U = Upd, F = (F2pd,), G = (G1pd, G3pd)) +
            zeta_2p * LS(m, :Ni_2p) +
            ep_x * n(m, :Ni_2p) + ed_x * n(m, :Ni_3d) + eL_x * n(m, :L_3d)

    # Dipole (length-3: T_x, T_y, T_z).
    T = dipole(m, :Ni_2p => :Ni_3d)

    # ---------------------------------------------------------------
    # Bases
    # ---------------------------------------------------------------
    basis_gs  = basis(m,
        nshells(m, :Ni_2p) == 6,
        nshells(m, :Ni_3d) + nshells(m, :L_3d) == 18)
    basis_xas = basis(m,
        nshells(m, :Ni_2p) in 5:6,
        total(m) == 24)

    @info "MOADyna native NiO basis sizes: GS = $(length(basis_gs)), XAS = $(length(basis_xas))"
    @test length(basis_gs)  == 190
    @test length(basis_xas) == 310

    # ---------------------------------------------------------------
    # Ground state
    # ---------------------------------------------------------------
    gs = eigen(H_GS, basis_gs; n = 3)
    Eg = gs.values[1]
    @info "MOADyna native NiO Eg = $(round(Eg; digits = 6)) eV (Quanty: ≈ -3.503)"
    @test Eg ≈ -3.503 atol = 0.01

    psi0 = embed(gs.vectors[:, 1], basis_gs => basis_xas)
    @test sum(abs2, psi0) ≈ 1.0 atol = 1e-10

    # ---------------------------------------------------------------
    # XAS spectra — 3 Cartesian polarisations vs PyQuanty reference
    # ---------------------------------------------------------------
    # `xas` expects H as an assembled matrix (block_lanczos takes a
    # matvec-able operator with `size(::, ::Int)`); compile+assemble
    # H_XAS once on the XAS basis.
    H_XAS_sp = assemble(compile(H_XAS, basis_xas), basis_xas)

    omega_grid = range(-15.0, 25.0; length = 801)
    Gamma      = 0.6

    ref      = readdlm(ref_file)
    @test size(ref, 1) == 801

    # Each polarisation is a SEPARATE LOGICAL PROPERTY: one independent
    # spectrum-vs-reference comparison per Cartesian axis. Three @test
    # calls inside the loop are deliberate, not loop-inflation.
    for (ax_idx, ax_name) in enumerate(("x", "y", "z"))
        spec = xas(H_XAS_sp, basis_xas, T[ax_idx], psi0;
                   omega_grid = omega_grid, Gamma = Gamma,
                   Eg = Eg, krylovdim = 200)

        # Reference cols: 1=ω; (2,3)=Re/Im x; (4,5)=Re/Im y; (6,7)=Re/Im z.
        ref_col  = complex.(ref[:, 2 * ax_idx], ref[:, 2 * ax_idx + 1])
        peak     = maximum(abs, ref_col)
        residual = maximum(abs, spec.tensor .- ref_col) / peak
        @info "MOADyna native NiO XAS $(ax_name): max |Δ| / peak = $(residual)"
        @test residual < 1e-4
    end
end
