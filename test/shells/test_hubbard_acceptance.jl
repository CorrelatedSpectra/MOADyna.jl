using Test
using MOAD
using LinearAlgebra: norm

@testset "two-orbital Hubbard half-filled — Kanamori J vs density-density" begin
    # 5-orbital d-shell (synthetic tag :test_3d, parsed as ell=2), restricted
    # to half-filling = 5 electrons.  Kanamori GS energy must be <= density-
    # density GS energy at the same (U, J) because the spin-flip and pair-hop
    # terms broaden the multiplet and lower the singlet GS through coupling to
    # the symmetric intra-orbital block.
    m = ShellModel([:test_3d])

    U, J = 5.0, 0.5

    H_dd = density_density(m, :test_3d; U = U, J = J)
    H_K  = kanamori(m, :test_3d; U = U, J = J)

    @test H_dd' == H_dd
    @test H_K'  == H_K

    # Half-filled (5 electrons) sector: C(10,5) = 252 states
    basis = EagerBasis(m, total(m) == 5)

    gs_dd = eigen(H_dd, basis; n = 1)
    gs_K  = eigen(H_K,  basis; n = 1)

    @test isreal(gs_dd.values[1])
    @test isreal(gs_K.values[1])
    # Kanamori GS energy is <= density-density GS energy
    @test gs_K.values[1] <= gs_dd.values[1] + 1e-10
end

@testset "single-band Hubbard dimer — known closed-form spectrum" begin
    # Two single-orbital sites (ell=0 => 2 modes each = up/dn).
    # No hopping; just on-site U on each site.  Half-filled (2 electrons).
    # Spectrum:
    #   - 4 single-occupancy states at E=0: |up,up>, |up,dn>, |dn,up>, |dn,dn>
    #   - 2 doubly-occupied states at E=U: |updn, 0>, |0, updn>
    m = ShellModel([:A_1s, :B_1s])
    U = 4.0

    H = density_density(m, :A_1s; U = U) +
        density_density(m, :B_1s; U = U)

    # Half-filled sector: C(4,2) = 6 states total
    basis = EagerBasis(m, total(m) == 2)
    sys = eigen(H, basis; n = length(basis))
    sorted_evals = sort(real.(sys.values))

    @test count(e -> abs(e) < 1e-10, sorted_evals) == 4
    @test count(e -> abs(e - U) < 1e-10, sorted_evals) == 2
    @test length(sorted_evals) == 6
end

@testset "shell-keyed onsite + Hubbard composition end-to-end" begin
    # NiO-style two-shell setup: Ni_3d hybridised with L_3d, with a
    # density-density Coulomb on the 3d shell and onsite shifts on each.
    # Onsite energies pinned to the closed-form Sawatzky-Zaanen-Allen
    # anchor solution (the full anchor solver lives in Plan 2 / Multiplets):
    #     ed = (10 Δ - n(19+n) U/2) / (10+n)   at n=8, U=7.3, Δ=4.7
    #     eL = n((1+n) U/2 - Δ) / (10+n)
    # By construction, the d⁸L¹⁰ baseline configuration has energy 0.
    m = ShellModel([:Ni_3d, :L_3d])

    Udd, Delta = 7.3, 4.7
    ed = (10 * Delta - 8 * 27 * Udd / 2) / 18
    eL = 8 * (9 * Udd / 2 - Delta) / 18

    H = density_density(m, :Ni_3d; U = Udd) +
        ed * n(m, :Ni_3d) + eL * n(m, :L_3d)

    @test H' == H

    # d⁸L¹⁰ sector: C(10,8)·C(10,10) = 45 states. By the ZSA construction
    # this baseline has GS energy ≈ 0.
    basis_d8 = EagerBasis(m, nshells(m, :Ni_3d) == 8, nshells(m, :L_3d) == 10)
    gs_d8 = eigen(H, basis_d8; n = 1)
    @test gs_d8.values[1] ≈ 0.0 atol=1e-8

    # d⁹L⁹ sector: GS energy ≈ Δ (charge-transfer anchor).
    basis_d9 = EagerBasis(m, nshells(m, :Ni_3d) == 9, nshells(m, :L_3d) == 9)
    gs_d9 = eigen(H, basis_d9; n = 1)
    @test gs_d9.values[1] ≈ Delta atol=1e-8
end

@testset "shell-keyed Sz expectation on a fully-polarised state" begin
    # Sanity: build a d-shell, take the unique state with all spins up
    # (5 up, 0 dn), verify <Sz> = 5/2.
    #
    # Mode order for ell=2 (d-shell) in m-major dn-then-up convention:
    #   mode 1 = dn(m=-2), mode 2 = up(m=-2),
    #   mode 3 = dn(m=-1), mode 4 = up(m=-1),
    #   mode 5 = dn(m=0),  mode 6 = up(m=0),
    #   mode 7 = dn(m=1),  mode 8 = up(m=1),
    #   mode 9 = dn(m=2),  mode 10 = up(m=2)
    # Weight vector for spin-up count (1 on each up mode, 0 on dn modes):
    #   w_up = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    m = ShellModel([:Ni_3d])

    site = MOAD.Shells.site_of(m, :Ni_3d)
    w_up = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]   # +1 on each up mode

    # Restrict to total 5 electrons AND all 5 are up-spin:
    #   nshells == 5 ensures total electron count = 5
    #   WeightedParticleCount(...) == 5 ensures all 5 are up-spin
    basis_5up = EagerBasis(m,
        nshells(m, :Ni_3d) == 5,
        WeightedParticleCount([site], w_up) == 5)

    @test length(basis_5up) == 1   # the unique fully-up d5 state

    Sz_op = Sz(m, :Ni_3d)
    Sz_mat = assemble(compile(Sz_op, basis_5up), basis_5up)
    psi = ones(ComplexF64, length(basis_5up))   # the single basis state
    @test (psi' * Sz_mat * psi)[1] ≈ 5/2
end
