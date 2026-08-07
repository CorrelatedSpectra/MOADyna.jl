# test/gradients/test_degeneracy_sector.jl
#
# Build-step 5 of the v0.3 differentiable forward model (MOADyna.Gradients): the
# degeneracy gate + sector-aware ground state. Two layers:
#   (1) matrix-only — groundstate_manifold (cluster + external-gap gate),
#       manifold_gate (U†dH U ≈ αI splitting check), lowlying_spectrum;
#   (2) opt-in physics bridge — classify_groundstate (spatial irrep via
#       classify_state + ⟨S²⟩ via a caller-supplied compiled S² matrix).
#
# Fixtures: A — a symmetry-protected 2-fold manifold (manifold-preserving vs
# splitting directions); B — a two-sector level crossing (the gate fires through
# the non-degenerate wrapper / manifold_gate, NOT through the external-gap gate,
# except in the unresolved-cluster band); C — a real cubic d⁸ model whose ground
# term is ³A₂g, labeled against the known irrep/multiplicity. Reproducible (no RNG).

using MOADyna
using LinearAlgebra
using SparseArrays
using Test

const G = MOADyna.Gradients

@testset "Gradients build-step 5: degeneracy gate + sector-aware GS" begin

    # ------------------------------------------------------------------
    # Fixture A — symmetry-protected 2-fold ground manifold (matrix-only)
    # ------------------------------------------------------------------
    # H(θ) = M0 + a·M1 + b·M2 on a 5-level diagonal model. The two lowest levels
    # are exactly equal at b = 0 (a persistent 2-fold manifold). M1 shifts the
    # manifold rigidly (preserving); M2 shifts the two partners oppositely
    # (splitting). Test at θ = [a, 0].
    @testset "A: persistent 2-fold manifold + gate" begin
        M0 = Matrix{ComplexF64}(Diagonal([-2.0, -2.0, 1.0, 2.0, 3.0]))
        M1 = Matrix{ComplexF64}(Diagonal([0.1, 0.1, 0.0, 0.0, 0.0]))   # rigid shift
        M2 = Matrix{ComplexF64}(Diagonal([0.05, -0.05, 0.0, 0.0, 0.0])) # splitting
        names = [:a, :b]
        model = G.AffineModel([M0, M1, M2],
                              G.AffineMap([1.0, 0.0, 0.0], [0.0 0.0; 1.0 0.0; 0.0 1.0]),
                              names)
        θ = [0.3, 0.0]

        gs = G.groundstate_manifold(model, θ)
        @test gs.g == 2
        @test size(gs.U) == (5, 2)
        @test norm(gs.U' * gs.U - I) < 1e-10
        @test isapprox(gs.E0, -2.0 + 0.1 * 0.3; atol = 1e-12)
        @test isapprox(gs.gap_external, (1.0) - (-2.0 + 0.1 * 0.3); atol = 1e-12)

        # the non-degenerate wrapper must reject a degenerate GS
        @test_throws ArgumentError G.groundstate(model, θ)

        # manifold-preserving direction θ̇ = [1, 0] (the M1 rigid shift)
        r_keep = G.manifold_gate(model, θ, gs.U, [1.0, 0.0])
        @test r_keep.ok
        @test isapprox(r_keep.α, 0.1; atol = 1e-10)
        @test r_keep.split < 1e-9

        # splitting direction θ̇ = [0, 1] (the M2 opposite shift)
        r_split = G.manifold_gate(model, θ, gs.U, [0.0, 1.0])
        @test !r_split.ok
        @test isapprox(r_split.α, 0.0; atol = 1e-10)
        @test r_split.split > 1e-3
        # cross-check: B = U†·M2·U has eigenvalues ±0.05 (the first-order splitting)
        B = gs.U' * (M2 * gs.U)
        @test isapprox(sort(real(eigvals(Hermitian(Matrix(B))))), [-0.05, 0.05]; atol = 1e-10)

        # U guards
        @test_throws DimensionMismatch G.manifold_gate(model, θ, ones(ComplexF64, 4, 2), [1.0, 0.0])
        bad = ComplexF64[1 0; 1 0; 0 0; 0 0; 0 0]      # non-orthonormal columns
        @test_throws ArgumentError G.manifold_gate(model, θ, bad, [1.0, 0.0])
        # orth_tol must itself be validated (a permissive tol must not slip a
        # non-orthonormal U through)
        @test_throws ArgumentError G.manifold_gate(model, θ, gs.U, [1.0, 0.0]; orth_tol = Inf)
        @test_throws ArgumentError G.manifold_gate(model, θ, gs.U, [1.0, 0.0]; orth_tol = 0.0)
    end

    # ------------------------------------------------------------------
    # Fixture B — two-sector level crossing (matrix-only)
    # ------------------------------------------------------------------
    # H(t) = diag(t, 1−t, 5): two non-mixing sectors a=t, b=1−t cross at t*=0.5;
    # the third level (5) is far above. At the crossing the ground cluster is
    # 2-fold AND externally isolated ⇒ groundstate_manifold returns g==2 (no
    # throw). Rejection is via the non-degenerate wrapper and manifold_gate.
    @testset "B: two-sector crossing" begin
        M0 = Matrix{ComplexF64}(Diagonal([0.0, 1.0, 5.0]))
        M1 = Matrix{ComplexF64}(Diagonal([1.0, -1.0, 0.0]))
        model = G.AffineModel([M0, M1], G.AffineMap([1.0, 0.0], reshape([0.0, 1.0], 2, 1)),
                              [:t])

        # away from the crossing: non-degenerate, gate trivially passes
        gsa = G.groundstate_manifold(model, [0.2])
        @test gsa.g == 1
        @test isapprox(gsa.E0, 0.2; atol = 1e-12)
        E0a, _, gapa = G.groundstate(model, [0.2])      # wrapper returns
        @test isapprox(E0a, 0.2; atol = 1e-12)
        @test isapprox(gapa, 0.6; atol = 1e-12)         # (0.8 − 0.2)

        # ground sector flips across t* = 0.5 (argmin of the diagonal)
        @test argmin(real(diag(G.hamiltonian(model, [0.2])))) == 1   # block a lowest
        @test argmin(real(diag(G.hamiltonian(model, [0.8])))) == 2   # block b lowest

        # exact crossing t = 0.5: g == 2, externally isolated (gap to the third = 4.5)
        gsx = G.groundstate_manifold(model, [0.5])
        @test gsx.g == 2
        @test isapprox(gsx.E0, 0.5; atol = 1e-12)
        @test isapprox(gsx.gap_external, 4.5; atol = 1e-12)
        # the non-degenerate wrapper rejects the crossing (E1 − E0 = 0 ≤ gap_tol)
        @test_throws ArgumentError G.groundstate(model, [0.5])
        # manifold_gate along the crossing direction θ̇ = [1] splits the manifold
        rx = G.manifold_gate(model, [0.5], gsx.U, [1.0])
        @test !rx.ok
        @test isapprox(rx.α, 0.0; atol = 1e-10)         # tr(diag(1,−1))/2 = 0
        @test rx.split > 1e-3
        Bx = gsx.U' * (M1 * gsx.U)                       # eigenvalues = HF slopes ±1
        @test isapprox(sort(real(eigvals(Hermitian(Matrix(Bx))))), [-1.0, 1.0]; atol = 1e-9)

        # unresolved-cluster band: degen_tol < E1−E0 ≤ gap_tol ⇒ the external-gap
        # gate itself throws (the two lowest are split by 5e-7, between the tols).
        @test_throws ArgumentError G.groundstate_manifold(model, [0.5 + 2.5e-7];
                                                          gap_tol = 1e-6, degen_tol = 1e-8)
    end

    # ------------------------------------------------------------------
    # Guard tests (matrix-only)
    # ------------------------------------------------------------------
    @testset "guards" begin
        M0 = Matrix{ComplexF64}(Diagonal([0.0, 1.0, 5.0]))
        M1 = Matrix{ComplexF64}(Diagonal([1.0, -1.0, 0.0]))
        model = G.AffineModel([M0, M1], G.AffineMap([1.0, 0.0], reshape([0.0, 1.0], 2, 1)),
                              [:t])
        # degen_tol ≥ gap_tol is rejected
        @test_throws ArgumentError G.groundstate_manifold(model, [0.2];
                                                          gap_tol = 1e-6, degen_tol = 1e-6)
        # whole spectrum degenerate (g == d) ⇒ no external state to gate against
        deg = G.AffineModel([Matrix{ComplexF64}(Diagonal([2.0, 2.0]))],
                            G.AffineMap([1.0], zeros(1, 1)), [:x])
        @test_throws ArgumentError G.groundstate_manifold(deg, [0.0])
    end

    # ------------------------------------------------------------------
    # Fixture C — real cubic d⁸ model: ground term ³A₂g (opt-in bridge)
    # ------------------------------------------------------------------
    # Spin-independent cubic d-shell, Ni²⁺ d⁸: the Oh ground term is ³A₂g
    # (spatial A2g, spin triplet). Build the compiled Hamiltonian + S² matrices
    # in the same basis, wrap H as a constant AffineModel, and label the ground
    # state. (Mirrors test/diagnostics/test_classify_state_fock.jl.)
    @testset "C: cubic d⁸ → ³A₂g labeling" begin
        m   = ShellModel([:Ni_3d])
        H   = coulomb(m, :Ni_3d; U = 0.0, F = (11.14, 6.87)) +
              1.0 * Akm(m, :Ni_3d, :Oh, [0.6, -0.4])
        b   = basis(m, nshells(m, :Ni_3d) == 8)
        Hsp = assemble(compile(H, b), b)
        S2  = assemble(compile(Ssqr(m, :Ni_3d), b), b)
        grp = pointgroup(:Oh)
        N   = length(b)

        model = G.AffineModel([Matrix{ComplexF64}(Hsp)],
                              G.AffineMap([1.0], zeros(1, 1)), [:x])
        θ = [0.0]

        rep = G.classify_groundstate(model, θ; basis = b, m = m, G = grp, S2 = S2,
                                     window = 5.0)
        # ground term: ³A₂g — spatial A2g, triplet (S=1 ⇒ ⟨S²⟩=2, multiplicity 3)
        @test rep.ground.degeneracy == 3
        @test rep.ground.irrep == :A2g
        @test isapprox(rep.ground.irrep_weight, 1.0; atol = 1e-6)
        @test isapprox(rep.ground.S2, 2.0; atol = 1e-6)
        @test rep.ground.multiplicity == 3
        @test rep.ground.spin == 1.0
        @test rep.ground.gap_above_E0 == 0.0
        @test rep.gs.g == 3

        # competitors are ascending in gap_above_E0 and within the window
        gaps = [c.gap_above_E0 for c in rep.competitors]
        @test issorted(gaps)
        @test all(0.0 .< gaps .≤ 5.0)

        # independent enumeration: cluster the dense spectrum with the same
        # degen_tol and check classify_groundstate reproduces the cluster energies.
        ev = sort(real(eigvals(Hermitian(Matrix(Hsp)))))
        E0 = ev[1]
        cluster_E = Float64[]
        i = 1
        while i ≤ length(ev)
            j = i
            while j < length(ev) && ev[j + 1] - ev[i] ≤ 1e-8
                j += 1
            end
            ev[i] - E0 ≤ 5.0 && push!(cluster_E, ev[i])
            i = j + 1
        end
        reported_E = [rep.ground.energy; [c.energy for c in rep.competitors]]
        @test length(reported_E) == length(cluster_E)
        @test all(isapprox.(reported_E, cluster_E; atol = 1e-8))

        # contract guard: a wrong-dimension S² matrix
        @test_throws DimensionMismatch G.classify_groundstate(model, θ; basis = b, m = m,
            G = grp, S2 = Matrix{ComplexF64}(I, N - 1, N - 1), window = 5.0)
    end
end
