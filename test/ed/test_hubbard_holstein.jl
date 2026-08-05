# =====================================================================
# Hubbard-Holstein model — sanity checks (mixed Fock layer end-to-end)
# =====================================================================
#
# Decoupled-limit and analytical cases that exercise mixed
# fermion+boson Hilbert spaces. Quantitative cross-comparison against
# QuSpin (spinless Holstein) lives in
# `validation/test_holstein_quspin.jl`.

@testset "Hubbard-Holstein — sanity" begin

    # HH-A: g = 0 factorisation. With the e-ph coupling switched off,
    # the phonon and electron sectors decouple; GS is the Hubbard-dimer
    # GS tensored with the phonon vacuum, at energy
    #   E_GS = U/2 − √((U/2)² + 4 t²) + 0.
    @testset "HH-A: g = 0 factorisation → Hubbard dimer + phonon vacuum" begin
        L = 2
        Nmax_ph = 3
        t_hop, U, ω = 1.0, 4.0, 1.0

        electrons = [FermionSite{2}(Symbol("e$i")) for i in 1:L]
        phonons   = [BosonSite{Nmax_ph}(Symbol("p$i")) for i in 1:L]
        h = Hilbert(s.name => s for s in vcat(electrons, phonons))

        H_hop = -t_hop * sum(cdag(electrons[1], σ) * c(electrons[2], σ) +
                             cdag(electrons[2], σ) * c(electrons[1], σ)
                             for σ in 1:2)
        H_U   = U * sum(n(electrons[i], 1) * n(electrons[i], 2) for i in 1:L)
        H_ph  = ω * sum(n_b(p) for p in phonons)
        H     = H_hop + H_U + H_ph

        bb = EagerBasis(h, n_fermion(h) == 2)
        E = eigen(H, bb; n = 1)
        E_exact = U / 2 - sqrt((U / 2)^2 + 4 * t_hop^2)
        @test E.values[1] ≈ E_exact atol = 1e-10
    end

    # HH-B: single site, single electron, linear coupling
    # +g(b†+b)·n_e. With one electron the operator (n_↑+n_↓) ≡ 1, so
    # H_eff = ω b†b + g(b†+b) is a displaced harmonic oscillator with
    # exact GS energy E = −g²/ω (Lang-Firsov).
    # Phonon cutoff Nmax = 20 gives convergence ≪ 1e−10 at g/ω = 0.5.
    @testset "HH-B: single-site polaron (Lang-Firsov)" begin
        ω, g_eph = 1.0, 0.5
        electron = FermionSite{2}(:e1)
        phonon   = BosonSite{20}(:p1)
        h = Hilbert(:e1 => electron, :p1 => phonon)
        H = ω * n_b(phonon) +
            g_eph * (bdag(phonon) + b(phonon)) *
            (n(electron, 1) + n(electron, 2))
        bb = EagerBasis(h, n_fermion(h) == 1)
        E = eigen(H, bb; n = 1, dense_below = typemax(Int))
        @test E.values[1] ≈ -g_eph^2 / ω atol = 1e-10
    end

    # HH-C: empty electron sector. With n_e = 0 the coupling term acts
    # as the zero operator and H = ω · n_b. GS is the phonon vacuum at
    # E = 0.
    @testset "HH-C: empty electron sector → phonon vacuum" begin
        ω, g_eph = 1.0, 0.5
        electron = FermionSite{2}(:e1)
        phonon   = BosonSite{4}(:p1)
        h = Hilbert(:e1 => electron, :p1 => phonon)
        H = ω * n_b(phonon) +
            g_eph * (bdag(phonon) + b(phonon)) *
            (n(electron, 1) + n(electron, 2))
        bb = EagerBasis(h, n_fermion(h) == 0)
        @test eigen(H, bb; n = 1).values[1] ≈ 0.0 atol = 1e-12
    end
end
