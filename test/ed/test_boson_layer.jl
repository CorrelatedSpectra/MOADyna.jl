# =====================================================================
# Boson layer — runtime primitive checks
# =====================================================================
#
# Direct end-to-end checks that the bosonic ladder b, b†, n_b
# assemble to the correct matrix elements on a finite-cutoff
# BosonSite{Nmax}, and that a single-mode displaced harmonic
# oscillator returns its analytical ground-state energy.
#
# Complements the symbolic algebra tests in
# `test/algebra/test_canonicalize.jl` (b·b† = 1 + n_b at the
# OperatorSum level) and the multi-site BH / HH coverage in
# `test_bose_hubbard.jl` and `test_hubbard_holstein.jl`.

@testset "Boson layer — primitives" begin
    using LinearAlgebra: Diagonal, diag

    # PR-A: ladder amplitudes b†|n⟩ = √(n+1)|n+1⟩, b|n⟩ = √n|n−1⟩,
    # with hard cutoff b†|Nmax⟩ = 0. We verify operator identities
    # on a single-site basis, sidestepping any dependence on the
    # internal occupation-state ordering.
    @testset "PR-A: ladder amplitudes + cutoff" begin
        Nmax = 5
        site = BosonSite{Nmax}(:p)
        h = Hilbert(:p => site)
        bb = EagerBasis(h)
        @test length(bb) == Nmax + 1

        B  = Matrix(assemble(compile(b(site),    bb), bb))
        Bd = Matrix(assemble(compile(bdag(site), bb), bb))
        Nb = Matrix(assemble(compile(n_b(site),  bb), bb))

        # n_b spectrum is exactly {0, 1, …, Nmax}.
        @test sort(round.(Int, real.(diag(Nb)))) == collect(0:Nmax)
        # b and b† are constructed independently in the algebra layer;
        # pin that they assemble as honest adjoints of one another.
        @test Bd ≈ B' atol = 1e-12
        # b†·b ≡ n_b ⇒ each column of b† has amplitude √n_target.
        @test Bd * B ≈ Nb atol = 1e-12
        # b·b†|n⟩ = (n+1)|n⟩ for n < Nmax; 0 at the cutoff.
        n_diag = round.(Int, real.(diag(Nb)))
        expected = Float64[k < Nmax ? k + 1 : 0 for k in n_diag]
        @test diag(B * Bd) ≈ expected atol = 1e-12
    end

    # PR-B: truncated commutator. For the infinite oscillator
    # [b, b†] = 1, but with hard cutoff Nmax the top state picks up
    # a defect: [b, b†]|Nmax⟩ = −Nmax|Nmax⟩. Diagonal in the
    # n-ordered basis: (1, 1, …, 1, −Nmax).
    @testset "PR-B: truncated commutator [b, b†]" begin
        Nmax = 4
        site = BosonSite{Nmax}(:p)
        h = Hilbert(:p => site)
        bb = EagerBasis(h)

        B  = Matrix(assemble(compile(b(site),    bb), bb))
        Bd = Matrix(assemble(compile(bdag(site), bb), bb))
        Nb = Matrix(assemble(compile(n_b(site),  bb), bb))
        comm = B * Bd - Bd * B

        n_diag = round.(Int, real.(diag(Nb)))
        perm   = sortperm(n_diag)
        expected = Float64[1, 1, 1, 1, -Nmax]   # n = 0..Nmax
        @test diag(comm)[perm] ≈ expected atol = 1e-12
        @test comm ≈ Diagonal(diag(comm)) atol = 1e-12
    end

    # PR-C: pure single-mode displaced harmonic oscillator
    # H = ω·n_b + λ·(b† + b). Exact GS energy is −λ²/ω; cutoff
    # truncation error decays exponentially in Nmax once Nmax ≫
    # ⟨n⟩ ≈ (λ/ω)². Nmax = 30 at λ/ω = 0.5 is well over-converged.
    @testset "PR-C: pure displaced oscillator" begin
        Nmax = 30
        ω, λ = 1.0, 0.5
        site = BosonSite{Nmax}(:p)
        h = Hilbert(:p => site)
        H = ω * n_b(site) + λ * (bdag(site) + b(site))
        bb = EagerBasis(h)
        E = eigen(H, bb; n = 1, dense_below = typemax(Int))
        @test E.values[1] ≈ -λ^2 / ω atol = 1e-10
    end
end
