# =====================================================================
# 2-site Hubbard dimer — finite-T same-sector neutral thermal average
# =====================================================================
#
# Headline validation fixture for Plan 2d (responses, finite-temperature
# `:neutral` one-sided thermal average). Compares the Boltzmann-weighted
# per-state pole-fold exposed by `MOAD.correlator(...; T, channel=:neutral,
# form=:pole)` against an INDEPENDENT dense single-sum reference built by
# direct diagonalisation (`eigen` on the dense N=2-sector Hamiltonian) and
# an explicit double Lehmann sum — i.e. without ever touching the
# block-Lanczos / continued-fraction reframe+fold path.
#
# Physics (finite-temperature thermal average), with k_B = 1, β = 1/T:
#
#   I_{ab}(ω,T) = (1/Z) Σ_m e^{−βE_m} ⟨m| A_a† (ω + E_m − H + iΓ/2)⁻¹ B_b |m⟩,
#   Z = Σ_m e^{−βE_m},
#
# where the sum runs over eigenstates |m⟩ of H on `basis` (a SINGLE
# conserved sector → same-sector, trap-free). Expanding the resolvent in
# the eigenbasis {E_n, |n⟩} of the same sector,
#
#   I_ref[a,b](ω) = (1/Z) Σ_m e^{−βE_m}
#                   Σ_n ⟨m|A_a†|n⟩⟨n|A_b|m⟩ / (ω − (E_n − E_m) + iΓ/2).
#
# System:
#
#   H = -t (c†_{1↑} c_{2↑} + c†_{1↓} c_{2↓} + h.c.)
#       + U (n_{1↑} n_{1↓} + n_{2↑} n_{2↓}),    t = 1, U = 4
#
# restricted to the N=2 particle-number sector (a single conserved sector
# → same-sector). Single `FermionSite{4}(:s)`, mode layout
#
#   mode 1 → site 1, spin ↑     mode 2 → site 1, spin ↓
#   mode 3 → site 2, spin ↑     mode 4 → site 2, spin ↓
#
# (matches the intra-spin hopping `cdag(s,1)*c(s,3)`, `cdag(s,2)*c(s,4)`.)
#
# Neutral operators (conserve N and total Sz → stay in N=2):
#
#   A_1 = B_1 = Sᶻ(site 1) = (n_{1↑} − n_{1↓})/2 = (n(s,1) − n(s,2))/2
#   A_2 = B_2 = Sᶻ(site 2) = (n_{2↑} − n_{2↓})/2 = (n(s,3) − n(s,4))/2
#
# giving a 2×2 response matrix (catches row/col/residue folding bugs).
# The N=2 sector (6 states) has a NONDEGENERATE singlet ground state, a
# 3-fold-degenerate excited triplet (E = 0 with the gauge here), and
# higher singlets from double occupancy.
#
# ---------------------------------------------------------------------
# Achieved tolerances (recorded after running, see @info lines):
#
#   Check 1 (element-wise I vs dense single-sum reference, T∈{0.5,2.0}):
#       max |Δ| = 4.1e-15   (asserted < 1e-10)
#   Check 2 (T→0 limit T=1e-6 vs T===nothing neutral):
#       max |Δ| = 0.0       (asserted < 1e-8)
#   Check 3 (Z normalisation, kernel vs hand Z-weighted fold):
#       max |Δ| = 2.2e-16   (asserted < 1e-10)
#   Check 4 (P8: triplet basis-rotation invariance, max |Δ| = 4.4e-16,
#            asserted < 1e-10; N_states truncation inside triplet RAISES).
#   Check 5 (P9: degenerate-GS 1/g averaging on a separate toy):
#       1/g manifold average match max |Δ| = 1.8e-15 (asserted < 1e-10);
#       T=0 vs single-argmin differ by 7.5 (asserted > 1e-3).
#
# This validates the 2d same-sector neutral one-sided thermal average.
# =====================================================================

@testset "Hubbard dimer finite-T neutral thermal average" begin
    using LinearAlgebra
    using Random

    # ---------------------------------------------------------------
    # Fixture: 4-mode FermionSite, dimer Hamiltonian, N=2 sector only.
    # ---------------------------------------------------------------
    s = FermionSite{4}(:s)
    h = Hilbert(:s => s)
    t, U = 1.0, 4.0

    H_hop = -t * (cdag(s, 1) * c(s, 3) + cdag(s, 2) * c(s, 4))
    H_op  = (H_hop + H_hop') + U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))

    # Neutral on-site Sᶻ operators: (n_↑ − n_↓)/2 per site.
    Sz1 = (n(s, 1) - n(s, 2)) * 0.5
    Sz2 = (n(s, 3) - n(s, 4)) * 0.5
    ops = [Sz1, Sz2]   # As === Bs (autocorrelator contract)

    Γ = 0.1

    # N=2 sector — a single conserved particle-number sector. No Sz
    # restriction: the full N=2 sector (6 states) carries the triplet
    # used in check 4. Sᶻ operators conserve N and total Sz, so they
    # stay inside this sector ⇒ same-sector / trap-free.
    basis = EagerBasis(h, n_fermion(h) == 2)
    @test length(basis) == 6   # C(4,2) = 6
    H_sp = assemble(compile(H_op, basis), basis)

    # ---------------------------------------------------------------
    # Independent dense single-sum reference.
    # ---------------------------------------------------------------
    # Diagonalise the N=2 dense H directly (NOT via block-Lanczos), and
    # assemble the Sᶻ operators on the same basis. Matrix elements are
    # evaluated in the dense eigenbasis.
    H_dense = Matrix(H_sp)
    F = eigen(Hermitian(H_dense))
    E = F.values                  # ascending eigenvalues E_n (length 6)
    V = F.vectors                 # eigenvectors (columns), basis-index space

    A1 = Matrix(assemble(compile(Sz1, basis), basis))
    A2 = Matrix(assemble(compile(Sz2, basis), basis))
    A_ops = (A1, A2)              # A_a in basis-index space

    # Eigenbasis matrix elements of A_a:  Atil[a][n,m] = ⟨n|A_a|m⟩.
    Atil = map(A_ops) do A
        V' * A * V               # Hermitian A ⇒ this is the eigenbasis op
    end

    # Dense double-sum reference at one ω and temperature T.
    #   A_a† in the eigenbasis is Atil[a]' (conjugate-transpose).
    #   I_ref[a,b](ω) = (1/Z) Σ_m e^{−βE_m}
    #                   Σ_n ⟨m|A_a†|n⟩⟨n|A_b|m⟩ / (ω − (E_n−E_m) + iΓ/2)
    function I_ref(ω::Real, Tval::Real)
        β = 1 / Tval
        E0 = E[1]
        w  = exp.(-β .* (E .- E0))    # unnormalised Boltzmann weights
        Z  = sum(w)
        Iab = zeros(ComplexF64, 2, 2)
        for m in eachindex(E)
            ρm = w[m] / Z
            for a in 1:2, b in 1:2
                acc = 0.0 + 0.0im
                for n in eachindex(E)
                    # ⟨m|A_a†|n⟩ = conj(⟨n|A_a|m⟩) = conj(Atil[a][n,m])
                    mAa_n = conj(Atil[a][n, m])
                    nAb_m = Atil[b][n, m]
                    denom = ω - (E[n] - E[m]) + im * Γ / 2
                    acc += mAa_n * nAb_m / denom
                end
                Iab[a, b] += ρm * acc
            end
        end
        return Iab
    end

    # ω points spanning the relevant range (intra-sector excitations are
    # within ±U ≈ ±4; cover 0 and both signs).
    ωs_check = [-3.0, -1.0, -0.2, 0.0, 0.5, 2.5]

    # ---------------------------------------------------------------
    # Check 1 — element-wise match against dense single-sum reference.
    # ---------------------------------------------------------------
    @testset "Check 1 — element-wise I vs dense single-sum reference" begin
        max_err = 0.0
        for Tval in (0.5, 2.0)
            P = correlator(H_sp, basis, ops, ops;
                           T = Tval, channel = :neutral, form = :pole, Γ = Γ)
            for ω in ωs_check
                I_k = P(ω)
                I_r = I_ref(ω, Tval)
                max_err = max(max_err, maximum(abs, I_k .- I_r))
            end
        end
        @info "Check 1 — finite-T neutral element-wise max |Δ| = $(max_err)"
        @test max_err < 1e-10
    end

    # ---------------------------------------------------------------
    # Check 2 — T→0 limit reproduces the T===nothing neutral result.
    # ---------------------------------------------------------------
    # The N=2 GS is a nondegenerate singlet, so as T→0 the thermal sum
    # collapses to the single ground state — identical to the T=0 path.
    @testset "Check 2 — T→0 limit vs T===nothing" begin
        P_T0 = correlator(H_sp, basis, ops, ops;
                          T = 1e-6, channel = :neutral, form = :pole, Γ = Γ)
        P_none = correlator(H_sp, basis, ops, ops;
                            channel = :neutral, form = :pole, Γ = Γ)
        max_err = 0.0
        for ω in ωs_check
            max_err = max(max_err, maximum(abs, P_T0(ω) .- P_none(ω)))
        end
        @info "Check 2 — T→0 vs T===nothing max |Δ| = $(max_err)"
        @test max_err < 1e-8
    end

    # ---------------------------------------------------------------
    # Check 3 — Z normalisation consistency.
    # ---------------------------------------------------------------
    # Recompute Z independently from the N=2 spectrum and confirm the
    # kernel's 1/Z normalisation is embedded correctly: the thermal I is
    # the Z-weighted average of the per-state (T===nothing) responses.
    # Build Σ_m ρ_m I_m(ω) by hand from single-state neutral calls and
    # compare to the kernel's thermal result. This isolates the 1/Z
    # weighting from the resolvent (independent of I_ref's double sum).
    @testset "Check 3 — Z normalisation" begin
        Tval = 1.0
        β = 1 / Tval
        E0 = E[1]
        wraw = exp.(-β .* (E .- E0))
        Z = sum(wraw)
        ρ = wraw ./ Z
        @test sum(ρ) ≈ 1.0 atol = 1e-12   # normalisation sanity

        # Manual Z-weighted fold of per-state neutral responses.
        P_kernel = correlator(H_sp, basis, ops, ops;
                              T = Tval, channel = :neutral, form = :pole, Γ = Γ)
        max_err = 0.0
        for ω in ωs_check
            I_manual = zeros(ComplexF64, 2, 2)
            for m in eachindex(E)
                ψm = V[:, m]
                Pm = correlator(H_sp, basis, ops, ops;
                                channel = :neutral, form = :pole, Γ = Γ,
                                state = ψm, Eg = E[m])
                I_manual .+= ρ[m] .* Pm(ω)
            end
            max_err = max(max_err, maximum(abs, P_kernel(ω) .- I_manual))
        end
        @info "Check 3 — kernel vs hand Z-weighted fold max |Δ| = $(max_err)"
        @test max_err < 1e-10
    end

    # ---------------------------------------------------------------
    # Check 4 — P8: triplet manifold-completeness + truncation guard.
    # ---------------------------------------------------------------
    # The N=2 spectrum (gauge here): singlet GS at E0, a 3-fold
    # degenerate triplet, then higher singlets. At a T that appreciably
    # populates the triplet, the thermal average must be invariant under
    # an arbitrary unitary rotation WITHIN the degenerate triplet
    # eigenspace (the kernel sums the full manifold ⇒ basis-independent),
    # and truncating N_states INSIDE the triplet must RAISE (P8 gate).
    @testset "Check 4 — P8 triplet split + rotation invariance" begin
        # Identify the degenerate triplet: the manifold of 3 equal-energy
        # states sitting above the nondegenerate singlet GS.
        degen_tol = 1e-8
        # Group eigenvalues into degenerate manifolds.
        manifolds = Vector{Vector{Int}}()
        for n in eachindex(E)
            placed = false
            for grp in manifolds
                if abs(E[n] - E[grp[1]]) < degen_tol
                    push!(grp, n); placed = true; break
                end
            end
            placed || push!(manifolds, [n])
        end
        # GS is nondegenerate (singlet); the triplet is the first 3-fold
        # manifold.
        @test length(manifolds[1]) == 1                 # singlet GS
        triplet = manifolds[findfirst(g -> length(g) == 3, manifolds)]
        @test length(triplet) == 3

        # (a) Rotation invariance: rotate the triplet eigenvectors by an
        # arbitrary 3×3 unitary, rebuild a dense reference with the
        # rotated eigenbasis, and confirm I is unchanged. Since the
        # rotated set spans the same eigenspace, I_ref must be invariant.
        function I_ref_rotated(ω::Real, Tval::Real, Vrot)
            β = 1 / Tval
            E0 = E[1]
            w  = exp.(-β .* (E .- E0)); Z = sum(w)
            A1r = Vrot' * A1 * Vrot
            A2r = Vrot' * A2 * Vrot
            Ar  = (A1r, A2r)
            Iab = zeros(ComplexF64, 2, 2)
            for m in eachindex(E)
                ρm = w[m] / Z
                for a in 1:2, b in 1:2
                    acc = 0.0 + 0.0im
                    for n in eachindex(E)
                        denom = ω - (E[n] - E[m]) + im * Γ / 2
                        acc += conj(Ar[a][n, m]) * Ar[b][n, m] / denom
                    end
                    Iab[a, b] += ρm * acc
                end
            end
            return Iab
        end

        # Random-but-seeded unitary within the triplet block (seed = 42,
        # documented for reproducibility).
        rng = MersenneTwister(42)
        Q, _ = qr(randn(rng, ComplexF64, 3, 3))
        # V is real (eigvecs of a real-symmetric H); promote to complex so
        # the within-manifold complex rotation can be stored.
        Vrot = Matrix{ComplexF64}(V)
        Vrot[:, triplet] = ComplexF64.(V[:, triplet]) * Matrix(Q)   # rotate within manifold

        Tval = 5.0   # high enough to populate the triplet appreciably
        max_rot_err = 0.0
        for ω in ωs_check
            max_rot_err = max(max_rot_err,
                              maximum(abs, I_ref(ω, Tval) .- I_ref_rotated(ω, Tval, Vrot)))
        end
        @info "Check 4 — triplet rotation invariance of I_ref max |Δ| = $(max_rot_err)"
        @test max_rot_err < 1e-10

        # Kernel sums the full manifold, so it equals I_ref (and hence is
        # rotation-invariant) at this T as well.
        P = correlator(H_sp, basis, ops, ops;
                       T = Tval, channel = :neutral, form = :pole, Γ = Γ)
        max_kern_err = 0.0
        for ω in ωs_check
            max_kern_err = max(max_kern_err, maximum(abs, P(ω) .- I_ref(ω, Tval)))
        end
        @test max_kern_err < 1e-10

        # (b) Truncation INSIDE the triplet must raise. Singlet GS is
        # index 1; the triplet is indices 2,3,4. N_states = 3 truncates
        # between triplet members 2 and 3 ⇒ splits the manifold ⇒ P8 raise.
        N_split = triplet[1]   # = 2: keeps singlet + 1 triplet member, cuts 2
        @test N_split == 2
        err = try
            correlator(H_sp, basis, ops, ops;
                       T = Tval, channel = :neutral, form = :pole, Γ = Γ,
                       N_states = N_split)
            nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("degenerate", err.msg)
    end

    # ---------------------------------------------------------------
    # Check 5 — P9: degenerate ground-manifold 1/g averaging (toy).
    # ---------------------------------------------------------------
    # The dimer N=2 GS is nondegenerate (singlet) ⇒ cannot test P9. We
    # build a separate toy with a DEGENERATE ground manifold: the N=1
    # sector of a NON-hopping (t=0) dimer. There all four single-particle
    # states {|1↑⟩,|2↑⟩,|1↓⟩,|2↓⟩} are degenerate at E = 0 (a 4-fold
    # ground manifold).
    #
    # Neutral probe: O = Sᶻ(1) + 2·Sᶻ(2). In the occupation basis this is
    # DIAGONAL with values +1/2 (|1↑⟩), −1/2 (|1↓⟩), +1 (|2↑⟩), −1 (|2↓⟩)
    # — all NONZERO and DISTINCT. Full-rank on the manifold ⇒ O annihilates
    # NO manifold vector (so the per-state kernel never hits a rank-zero
    # block under the kernel's arbitrary orthonormalisation), AND its
    # per-state response varies, so the 1/g manifold average DIFFERS from
    # any single argmin state. At T=0 the kernel must give the basis-
    # independent 1/g average; T===nothing picks one arbitrary argmin state.
    @testset "Check 5 — P9 degenerate-GS 1/g averaging (toy)" begin
        # t = 0 ⇒ no hopping; U term is inactive in N=1 (no double occ).
        H_toy_op = U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))   # all-zero in N=1
        basis1 = EagerBasis(h, n_fermion(h) == 1)
        @test length(basis1) == 4
        H1 = assemble(compile(H_toy_op, basis1), basis1)

        # Confirm 4-fold degenerate ground manifold at E = 0.
        F1 = eigen(Hermitian(Matrix(H1)))
        @test all(abs.(F1.values) .< 1e-12)   # all four degenerate at 0
        g = 4

        Otoy = Sz1 + 2.0 * Sz2   # full-rank neutral probe on the N=1 manifold
        ops1 = [Otoy]            # single neutral operator → 1×1 response

        # T = 0 kernel: 1/g manifold average.
        P_T0 = correlator(H1, basis1, ops1, ops1;
                          T = 0.0, channel = :neutral, form = :pole, Γ = Γ)

        # Independent 1/g reference: average the single-state neutral
        # response over the FULL degenerate eigenbasis. Basis-independent
        # because the manifold is closed.
        V1 = F1.vectors
        function I_avg(ω::Real)
            acc = zeros(ComplexF64, 1, 1)
            for m in 1:g
                Pm = correlator(H1, basis1, ops1, ops1;
                                channel = :neutral, form = :pole, Γ = Γ,
                                state = V1[:, m], Eg = F1.values[m])
                acc .+= Pm(ω) ./ g
            end
            return acc
        end

        max_err_avg = 0.0
        diff_from_single = 0.0
        # T===nothing single-argmin response (one arbitrary GS state).
        P_none = correlator(H1, basis1, ops1, ops1;
                            channel = :neutral, form = :pole, Γ = Γ)
        for ω in ωs_check
            max_err_avg = max(max_err_avg, maximum(abs, P_T0(ω) .- I_avg(ω)))
            diff_from_single = max(diff_from_single, maximum(abs, P_T0(ω) .- P_none(ω)))
        end
        @info "Check 5 — P9 1/g average match max |Δ| = $(max_err_avg); " *
              "T=0 vs T===nothing single-state max |Δ| = $(diff_from_single)"
        # T=0 reproduces the 1/g manifold average to machine precision.
        @test max_err_avg < 1e-10
        # ... and that average genuinely differs from the single-argmin
        # state (so the test exercises the averaging, not a degenerate
        # coincidence).
        @test diff_from_single > 1e-3
    end
end
