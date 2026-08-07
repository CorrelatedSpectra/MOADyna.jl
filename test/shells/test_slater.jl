using Test
using MOADyna
using MOADyna.Algebra: OperatorSum, n_fermion
using MOADyna.Shells: ShellModel, coulomb, ell_of, site_of
using MOADyna.Bases: EagerBasis, compile, assemble
using LinearAlgebra: eigvals, tr

@testset "slater" begin

    @testset "F^0 intra correction (3j-derived a^k_intra)" begin
        # p shell: a² = 2/25
        @test MOADyna.Shells._compute_F0_intra(1, 0.0, (1.0,)) ≈ 2 / 25 atol = 1e-12
        # d shell: a² = a⁴ = 2/63
        @test MOADyna.Shells._compute_F0_intra(2, 0.0, (1.0, 0.0)) ≈ 2 / 63 atol = 1e-12
        @test MOADyna.Shells._compute_F0_intra(2, 0.0, (0.0, 1.0)) ≈ 2 / 63 atol = 1e-12
        # f shell: a² = 4/195, a⁴ = 2/143, a⁶ = 100/5577
        @test MOADyna.Shells._compute_F0_intra(3, 0.0, (1.0, 0.0, 0.0)) ≈ 4 / 195 atol = 1e-12
        @test MOADyna.Shells._compute_F0_intra(3, 0.0, (0.0, 1.0, 0.0)) ≈ 2 / 143 atol = 1e-12
        @test MOADyna.Shells._compute_F0_intra(3, 0.0, (0.0, 0.0, 1.0)) ≈ 100 / 5577 atol = 1e-12
        # Linearity: a generic combination matches the per-k sum
        F = (11.14, 6.87)
        U = 7.3
        expected = U + (2 / 63) * F[1] + (2 / 63) * F[2]
        @test MOADyna.Shells._compute_F0_intra(2, U, F) ≈ expected atol = 1e-12
    end

    @testset "Hermiticity — d shell" begin
        m = ShellModel([:Ni_3d])
        H = coulomb(m, :Ni_3d; U = 7.3, F = (11.14, 6.87))
        @test H == H'
    end

    @testset "Hermiticity — p shell" begin
        m = ShellModel([:Ni_2p])
        H = coulomb(m, :Ni_2p; U = 5.0, F = (8.5,))
        @test H == H'
    end

    @testset "Hermiticity — f shell" begin
        m = ShellModel([:Ce_4f])
        H = coulomb(m, :Ce_4f; U = 6.0, F = (10.0, 7.0, 5.0))
        @test H == H'
    end

    @testset "Pure F^0 reduces to U·N(N-1)/2 (centroid sanity)" begin
        # With F = (0.0, 0.0), F^0 = U exactly. The Slater operator
        # then collapses to the spherical centroid, whose action on a
        # number-eigenstate gives U·N(N-1)/2. We verify by assembling
        # in a fixed-N sector and checking that all eigenvalues equal
        # U·N(N-1)/2.
        m = ShellModel([:Ni_3d])
        U = 1.7
        H = coulomb(m, :Ni_3d; U = U, F = (0.0, 0.0))
        function _max_dev_at_N(N::Int)
            bas = EagerBasis(m.hilbert, n_fermion(m.hilbert) == N)
            Hmat = Matrix(assemble(compile(H, bas), bas))
            target = U * N * (N - 1) / 2
            evs = real.(eigvals(Hmat))
            return maximum(abs.(evs .- target))
        end
        @test maximum(_max_dev_at_N(N) for N in 0:3) < 1e-10
    end

    @testset "Number-conserving" begin
        # Each term in H_C has two cdag and two c on the same shell, so
        # [H_C, N_shell] = 0. Verify via an assembled commutator at N=2.
        m = ShellModel([:Ni_3d])
        H = coulomb(m, :Ni_3d; U = 7.3, F = (11.14, 6.87))
        Nshell = MOADyna.Shells.n(m, :Ni_3d)
        bas = EagerBasis(m.hilbert, n_fermion(m.hilbert) == 2)
        Hmat = Matrix(assemble(compile(H, bas), bas))
        Nmat = Matrix(assemble(compile(Nshell, bas), bas))
        @test maximum(abs, Hmat * Nmat - Nmat * Hmat) < 1e-10
    end

    # ----------------------------------------------------------------
    # Two-shell Slater Coulomb (Plan 2 Task 13)
    # ----------------------------------------------------------------
    @testset "F^0 inter correction (3j-derived, exchange-only)" begin
        # p–d (ℓ=1, ℓ'=2): k=1 → 1/15; k=3 → 3/70.
        @test MOADyna.Shells._compute_F0_inter(1, 2, 0.0, (), (1.0, 0.0)) ≈ 1 / 15 atol = 1e-12
        @test MOADyna.Shells._compute_F0_inter(1, 2, 0.0, (), (0.0, 1.0)) ≈ 3 / 70 atol = 1e-12
        # d–d' (ℓ=2, ℓ'=2): k=0 → 1/10; k=2 → 1/35; k=4 → 1/35.
        @test MOADyna.Shells._compute_F0_inter(2, 2, 0.0, (), (1.0, 0.0, 0.0)) ≈ 1 / 10 atol = 1e-12
        @test MOADyna.Shells._compute_F0_inter(2, 2, 0.0, (), (0.0, 1.0, 0.0)) ≈ 1 / 35 atol = 1e-12
        @test MOADyna.Shells._compute_F0_inter(2, 2, 0.0, (), (0.0, 0.0, 1.0)) ≈ 1 / 35 atol = 1e-12
        # p–f (ℓ=1, ℓ'=3): k=2 → 3/70; k=4 → 2/63.
        @test MOADyna.Shells._compute_F0_inter(1, 3, 0.0, (), (1.0, 0.0)) ≈ 3 / 70 atol = 1e-12
        @test MOADyna.Shells._compute_F0_inter(1, 3, 0.0, (), (0.0, 1.0)) ≈ 2 / 63 atol = 1e-12
        # Linearity: U baseline + exchange contributions add.
        U = 8.5
        Gs = (4.92, 2.80)
        expected = U + (1 / 15) * Gs[1] + (3 / 70) * Gs[2]
        @test MOADyna.Shells._compute_F0_inter(1, 2, U, (), Gs) ≈ expected atol = 1e-12
    end

    @testset "coulomb two-shell — Hermiticity (NiO 2p–3d Upd)" begin
        m = ShellModel([:Ni_2p, :Ni_3d])
        H = coulomb(m, :Ni_2p, :Ni_3d; U = 8.5, F = (6.67,), G = (4.92, 2.80))
        @test H == H'
    end

    @testset "coulomb two-shell — number conservation per shell" begin
        m = ShellModel([:Ni_2p, :Ni_3d])
        H = coulomb(m, :Ni_2p, :Ni_3d; U = 8.5, F = (6.67,), G = (4.92, 2.80))
        n_p = MOADyna.Shells.n(m, :Ni_2p)
        n_d = MOADyna.Shells.n(m, :Ni_3d)
        # Restrict to a fixed total-N sector; cross-shell term must
        # commute with each shell's number operator separately.
        bas = EagerBasis(m.hilbert, n_fermion(m.hilbert) == 3)
        Hmat = Matrix(assemble(compile(H, bas), bas))
        Npm  = Matrix(assemble(compile(n_p, bas), bas))
        Ndm  = Matrix(assemble(compile(n_d, bas), bas))
        @test maximum(abs, Hmat * Npm - Npm * Hmat) < 1e-10
        @test maximum(abs, Hmat * Ndm - Ndm * Hmat) < 1e-10
    end

    @testset "coulomb two-shell — distinct-shell guard" begin
        m = ShellModel([:Ni_3d])
        @test_throws ArgumentError coulomb(m, :Ni_3d, :Ni_3d; U = 1.0)
    end

    @testset "coulomb two-shell — pure F^0 reduces to U·N_A·N_B" begin
        # With F = G = (), only the auto-derived F^0 = U survives.
        # The cross-shell direct centroid acts on a number-eigenstate
        # as U·N_A·N_B (no self-energy correction since A ≠ B).
        m = ShellModel([:Ni_2p, :Ni_3d])
        U = 1.7
        H = coulomb(m, :Ni_2p, :Ni_3d; U = U)
        n_p = MOADyna.Shells.n(m, :Ni_2p)
        n_d = MOADyna.Shells.n(m, :Ni_3d)
        function _max_dev_at_N(N::Int)
            bas = EagerBasis(m.hilbert, n_fermion(m.hilbert) == N)
            Hmat = Matrix(assemble(compile(H, bas), bas))
            Npm  = Matrix(assemble(compile(n_p, bas), bas))
            Ndm  = Matrix(assemble(compile(n_d, bas), bas))
            target = U * Npm * Ndm
            return maximum(abs, Hmat - target)
        end
        @test maximum(_max_dev_at_N(N) for N in 0:4) < 1e-10
    end

    @testset "coulomb two-shell — centroid matches U·n_A·n_B (no double-count)" begin
        m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
        # NiO Upd parameters
        Upd  = 8.5
        G1pd = 4.92
        G3pd = 2.80
        F2pd = 6.67
        H = coulomb(m, :Ni_2p, :Ni_3d; U=Upd, F=(F2pd,), G=(G1pd, G3pd))
        # Centroid: trace of H over the (n_p=6, n_d=8, n_L=10) sector,
        # divided by sector size, should equal Upd · n_p · n_d = 8.5 · 6 · 8 = 408.0.
        bas = basis(m, nshells(m, :Ni_2p) == 6,
                       nshells(m, :Ni_3d) == 8,
                       nshells(m, :L_3d) == 10)
        H_M = Matrix(assemble(compile(H, bas), bas))
        centroid = real(tr(H_M)) / size(H_M, 1)
        @test centroid ≈ Upd * 6 * 8 atol=1e-10
    end

    @testset "coulomb two-shell — exchange-only centroid matches Quanty convention" begin
        m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
        # Just G^1 turned on, U=0, F=().
        G1 = 4.92
        H = coulomb(m, :Ni_2p, :Ni_3d; U=0.0, F=(), G=(G1, 0.0))
        bas = basis(m, nshells(m, :Ni_2p) == 6,
                       nshells(m, :Ni_3d) == 8,
                       nshells(m, :L_3d) == 10)
        H_M = Matrix(assemble(compile(H, bas), bas))
        centroid = real(tr(H_M)) / size(H_M, 1)
        # F^0_inter for G=(G1,0) is U + (1/15)·G1 = 0 + 0.328 → direct contributes +0.328·48 = +15.744
        # Exchange contributes -(1/15)·G1·n_p·n_d = -0.328·48 = -15.744
        # Total centroid: 0
        @test centroid ≈ 0.0 atol=1e-10
    end

end
