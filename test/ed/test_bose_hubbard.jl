# =====================================================================
# Bose-Hubbard model — sanity checks (boson layer end-to-end)
# =====================================================================
#
# Closed-form / algebraically tractable cases that exercise bosonic
# operators (b, bdag, n_b), boson-only Hilbert spaces, and `eigen` on
# bosonic bases. Quantitative cross-comparison vs QuSpin lives in
# `validation/test_bh_quspin.jl`.

@testset "Bose-Hubbard — sanity" begin
    using LinearAlgebra: norm

    # BH-A: single boson on L = 4 PBC reduces to single-particle tight
    # binding. Exact spectrum E_k = -2t cos(2πk/L), k = 0..L−1
    # → {-2t, 0, 0, +2t}. Hardcore Nmax = 1 is enough for one boson.
    @testset "BH-A: single boson on L=4 PBC = tight-binding cosine" begin
        L = 4
        t_hop = 1.0
        sites = [BosonSite{1}(Symbol("p$i")) for i in 1:L]
        h = Hilbert(s.name => s for s in sites)
        H_hop = sum(bdag(sites[i]) * b(sites[mod1(i + 1, L)]) for i in 1:L)
        H = -t_hop * (H_hop + H_hop')
        bb = EagerBasis(h, n_boson(h) == 1)
        E = eigen(H, bb; n = 4)
        @test E.values ≈ [-2.0, 0.0, 0.0, 2.0] atol = 1e-12
    end

    # BH-B: t = 0 atomic limit on L = 3, N = 3, Nmax = 2, U = 4, μ = 2.
    # GS = |1,1,1⟩ at energy −μ·N = −6 (each site n_i(n_i−1) = 0).
    # Lowest excited state moves one boson onto a neighbour: any
    # |2,0,1⟩-type configuration costs (U/2)·2·1 − μ·3 = U − 3μ = −2;
    # gap = U.
    @testset "BH-B: atomic limit (t = 0)" begin
        L, Nmax = 3, 2
        U, μ = 4.0, 2.0
        sites = [BosonSite{Nmax}(Symbol("p$i")) for i in 1:L]
        h = Hilbert(s.name => s for s in sites)
        H = (U / 2) * sum(n_b(s) * n_b(s) - n_b(s) for s in sites) -
            μ * sum(n_b(s) for s in sites)
        bb = EagerBasis(h, n_boson(h) == L)
        E = eigen(H, bb; n = 2)
        @test E.values[1] ≈ -μ * L atol = 1e-12
        @test E.values[2] - E.values[1] ≈ U atol = 1e-12
    end

    # BH-C: 2-site dimer at Nmax = 1 (hardcore). The on-site U term
    # vanishes identically (n(n−1) ≡ 0 for Nmax = 1). 1-boson sector:
    # bonding/antibonding at ∓t. 2-boson sector: only |1,1⟩, single
    # state at E = 0.
    @testset "BH-C: hardcore 2-site dimer (Nmax = 1)" begin
        t_hop = 1.0
        sites = [BosonSite{1}(Symbol("p$i")) for i in 1:2]
        h = Hilbert(s.name => s for s in sites)
        H = -t_hop * (bdag(sites[1]) * b(sites[2]) +
                      bdag(sites[2]) * b(sites[1]))

        b1 = EagerBasis(h, n_boson(h) == 1)
        E1 = eigen(H, b1; n = 2)
        @test E1.values ≈ [-t_hop, t_hop] atol = 1e-12

        b2 = EagerBasis(h, n_boson(h) == 2)
        @test length(b2) == 1
        @test eigen(H, b2; n = 1).values[1] ≈ 0.0 atol = 1e-12
    end
end
