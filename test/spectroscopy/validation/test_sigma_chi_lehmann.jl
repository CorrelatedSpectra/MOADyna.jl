# =====================================================================
# Spectroscopy validation — σ(ω) and S(q,ω) wrappers vs dense Lehmann
# =====================================================================
#
# Validates the two Layer-5 spectroscopy wrappers shipped in Plan 2e,
#   MOADyna.Spectroscopy.optical_conductivity(...)   → regular Re σ_αβ(ω>0)
#   MOADyna.Spectroscopy.dynamical_structure_factor(...) → S(q,ω) (full ω)
# against INDEPENDENT dense Lehmann-sum references built here by direct
# diagonalisation of the conserved sector (`eigen`), without touching the
# block-Lanczos / continued-fraction path inside `correlator`.
#
# Physics contract (e = ℏ = k_B = 1, β = 1/T):
#
#   Re σ_αβ(ω,T) = factor(ω,T) · (−Im Λ_αβ(ω,T)) / (ω·V),     ω > 0,
#   factor(ω,T)  = (T===nothing || iszero(T)) ? 1 : −expm1(−ω/T),
#   S(q,ω,T)     = −Im C(q,ω,T) / π,                          (full ω grid)
#
# where Λ_αβ = C_αβ = ⟨A_a†(z−H)⁻¹A_b⟩ is the one-sided neutral correlator,
# z = ω + E_m − H + iΓ/2, A = j (σ) or A = O_q (S). At finite T the dense
# reference is the Boltzmann-weighted same-sector thermal average
#   C(ω,T) = (1/Z) Σ_m e^{−βE_m} ⟨m|A†(ω+E_m−H+iΓ/2)⁻¹A|m⟩,  Z = Σ_m e^{−βE_m},
# summed over the eigenstates |m⟩ of the single conserved sector H.
#
# ---------------------------------------------------------------------
# σ system — half-filled 2-site Hubbard dimer, N=2 / Sz=0 sector (U=4, t=1),
#   paramagnetic ↑-bond current  j = i·t·(c†_{1↑}c_{2↑} − c†_{2↑}c_{1↑}),
#   Hermitian, e=ℏ=1 (mirrors test/spectroscopy/test_conductivity.jl).
#   Volume V = 2.0.
#
# χ system — 4-site spin-1/2 Heisenberg chain, periodic BC,
#   H = Σ_⟨ij⟩ S_i·S_j (J=1), longitudinal probe at momentum q = π:
#   O_q = Sz_q = Σ_{r=0}^{3} e^{i q r} Sz_r  (complex ⇒ ComplexF64 matrix).
#   Allowed momentum on L=4 PBC chain: q ∈ {0, π/2, π, 3π/2}; q=π chosen.
#
# NOTE: NO f-sum-rule check for σ. The regular part alone does NOT satisfy
# the f-sum rule (the Drude/diamagnetic weight is excluded by the locked
# one-sided contract), so an f-sum-rule test would be physically wrong here.
#
# ---------------------------------------------------------------------
# Achieved tolerances (recorded after running; see @info lines):
#
#   σ Check A (T=0 element-wise, Re σ vs dense Lehmann over ω>0 grid):
#       max |Δ| = 6.9e-15     (asserted < 1e-10)
#   σ Check B (finite-T element-wise, T=1.0, Re σ vs dense thermal ref
#       incl. detailed-balance factor — the key 2d+2e end-to-end check):
#       max |Δ| = 1.2e-14     (asserted < 1e-10)
#   χ Check C (S(q=π,ω) element-wise vs dense Lehmann over full ω grid):
#       max |Δ| = 7.1e-15     (asserted < 1e-10)
#   χ Check D (static structure-factor sum rule, T=0:
#       ∫ S(q,ω) dω ≈ ⟨ψ₀|O_q†O_q|ψ₀⟩, trapezoid on ω ∈ [−12, 12],
#       Γ = 0.02, 24001 points):
#       rel. error = 5.3e-4   (asserted < 5e-2)
# =====================================================================

using Test
using LinearAlgebra
using MOADyna
using MOADyna.Spectroscopy: optical_conductivity, dynamical_structure_factor

@testset "σ / S(q,ω) wrappers vs dense Lehmann (2e)" begin

    # =================================================================
    # σ system — Hubbard dimer bond current
    # =================================================================
    s = FermionSite{4}(:s)                     # modes: 1↑, 1↓, 2↑, 2↓
    h = Hilbert(:s => s)
    t, U = 1.0, 4.0
    V = 2.0                                     # volume (e = ℏ = 1)

    H_hop = -t * (cdag(s, 1) * c(s, 3) + cdag(s, 2) * c(s, 4))
    H_op  = (H_hop + H_hop') + U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))
    basis = EagerBasis(h,
                       n_fermion(h) == 2,
                       WeightedParticleCount([s], [1, -1, 1, -1]) == 0)
    H_sp = assemble(compile(H_op, basis), basis)

    # Paramagnetic ↑-bond current j = i·t·(c†₁↑ c₂↑ − c†₂↑ c₁↑), Hermitian.
    j_op  = im * t * (cdag(s, 1) * c(s, 3) - cdag(s, 3) * c(s, 1))
    j_mat = Matrix{ComplexF64}(assemble(compile(j_op, basis), basis))

    # Dense diagonalisation of the conserved sector.
    H_dense = Matrix{Float64}(H_sp)
    Fσ = eigen(Hermitian(H_dense))
    Eσ = Fσ.values
    Vσ = Fσ.vectors                             # columns = eigenvectors

    Γ  = 0.1
    ωs = range(-1.0, 8.0, length = 120)         # straddles 0 to test the drop

    # --- dense T=0 one-sided correlator Λ(ω) = ⟨ψ₀|j†(ω+Eg−H+iΓ/2)⁻¹j|ψ₀⟩ ---
    function Λ_dense_T0(ωgrid)
        k0 = argmin(Eσ)
        ψ₀ = ComplexF64.(Vσ[:, k0])
        Eg = Eσ[k0]
        x  = j_mat * ψ₀                         # j† = j (Hermitian)
        out = Vector{ComplexF64}(undef, length(ωgrid))
        for (k, ω) in enumerate(ωgrid)
            z = ω + Eg + im * Γ / 2
            out[k] = dot(x, (z * I - H_dense) \ x)
        end
        return out
    end

    # --- dense finite-T thermal one-sided correlator C_jj(ω,T) -------------
    #   C(ω,T) = (1/Z) Σ_m e^{−βE_m} ⟨m|j(ω+E_m−H+iΓ/2)⁻¹j|m⟩, Z = Σ_m e^{−βE_m}.
    function Cjj_dense_T(ωgrid, Tval)
        β  = 1 / Tval
        E0 = minimum(Eσ)
        w  = exp.(-β .* (Eσ .- E0))             # unnormalised Boltzmann weights
        Z  = sum(w)
        out = zeros(ComplexF64, length(ωgrid))
        for m in eachindex(Eσ)
            ψm = ComplexF64.(Vσ[:, m])
            Em = Eσ[m]
            x  = j_mat * ψm
            ρm = w[m] / Z
            for (k, ω) in enumerate(ωgrid)
                z = ω + Em + im * Γ / 2
                out[k] += ρm * dot(x, (z * I - H_dense) \ x)
            end
        end
        return out
    end

    # --- σ Check A: T=0 element-wise -------------------------------------
    σ0 = optical_conductivity(H_sp, basis, j_op; ω_grid = ωs, Γ = Γ, volume = V)
    @test ndims(σ0.tensor) == 1
    ωpos = collect(σ0.ω_grid)
    @test all(>(0), ωpos)
    Λ0 = Λ_dense_T0(ωpos)
    σ_refA = (-imag.(Λ0)) ./ (ωpos .* V)        # factor = 1 at T=0
    errA = maximum(abs, σ0.tensor .- σ_refA)
    @info "σ Check A — T=0 Re σ element-wise max |Δ| = $(errA)"
    @test errA < 1e-10

    # --- σ Check B: finite-T element-wise (key 2d+2e end-to-end check) ----
    Tσ = 1.0                                    # populates the excited states
    σT = optical_conductivity(H_sp, basis, j_op; ω_grid = ωs, Γ = Γ,
                              volume = V, T = Tσ)
    ωposT = collect(σT.ω_grid)
    @test ωposT ≈ ωpos
    CjjT = Cjj_dense_T(ωposT, Tσ)
    factor = -expm1.(-ωposT ./ Tσ)              # = 1 − e^{−ω/T}
    σ_refB = factor .* (-imag.(CjjT)) ./ (ωposT .* V)
    errB = maximum(abs, σT.tensor .- σ_refB)
    @info "σ Check B — finite-T (T=$Tσ) Re σ element-wise max |Δ| = $(errB)"
    @test errB < 1e-10

    # NOTE: deliberately NO f-sum-rule check on σ — the regular part excludes
    # the Drude/diamagnetic weight, so it does not satisfy the f-sum rule.

    # =================================================================
    # χ system — 4-site Heisenberg chain S(q,ω), q = π
    # =================================================================
    L = 4
    spins = [SpinSite{1//2}(Symbol("s$i")) for i in 1:L]
    hχ = Hilbert((Symbol("s$i") => spins[i] for i in 1:L)...)

    # H = Σ_⟨ij⟩ S_i·S_j with periodic boundary conditions (J = 1).
    bonds = [(i, mod1(i + 1, L)) for i in 1:L]
    Hχ_op = sum(S(spins[i])[1] * S(spins[j])[1] +
                S(spins[i])[2] * S(spins[j])[2] +
                S(spins[i])[3] * S(spins[j])[3] for (i, j) in bonds)
    basisχ = EagerBasis(hχ)
    Hχ_sp  = assemble(compile(Hχ_op, basisχ), basisχ)

    # O_q = Sz_q = Σ_{r=0}^{L-1} e^{i q r} Sz_r at q = π (allowed on L=4 PBC).
    q   = π
    Szr = [S(spins[r + 1])[3] for r in 0:(L - 1)]      # per-site total Sz
    Oq_op = sum(exp(im * q * r) * Szr[r + 1] for r in 0:(L - 1))
    Oq_mat = Matrix{ComplexF64}(assemble(compile(Oq_op, basisχ), basisχ))

    Hχ_dense = Matrix{Float64}(Hχ_sp)
    Fχ = eigen(Hermitian(Hχ_dense))
    k0χ = argmin(Fχ.values)
    ψ0χ = ComplexF64.(Fχ.vectors[:, k0χ])
    Egχ = Fχ.values[k0χ]

    # --- dense T=0 one-sided correlator C(q,ω) = ⟨ψ₀|O_q†(z−H)⁻¹O_q|ψ₀⟩ ---
    function Cq_dense(ωgrid, Γval)
        x = Oq_mat * ψ0χ                         # O_q applied to GS
        out = Vector{ComplexF64}(undef, length(ωgrid))
        for (k, ω) in enumerate(ωgrid)
            z = ω + Egχ + im * Γval / 2
            out[k] = dot(x, (z * I - Hχ_dense) \ x)   # dot conjugates x ⇒ O_q†
        end
        return out
    end

    # --- χ Check C: S(q,ω) element-wise over full ω grid ------------------
    Γχ  = 0.1
    ωsχ = range(-4.0, 4.0, length = 200)         # full grid (no ω≤0 drop)
    Sqω = dynamical_structure_factor(Hχ_sp, basisχ, Oq_op; ω_grid = ωsχ, Γ = Γχ)
    @test ndims(Sqω.tensor) == 1
    @test length(Sqω.ω_grid) == length(ωsχ)
    Cref = Cq_dense(collect(ωsχ), Γχ)
    S_refC = (-imag.(Cref)) ./ π
    errC = maximum(abs, Sqω.tensor .- S_refC)
    @info "χ Check C — S(q=π,ω) element-wise max |Δ| = $(errC)"
    @test errC < 1e-10

    # --- χ Check D: static structure-factor sum rule (T=0) ---------------
    #   ∫ S(q,ω) dω ≈ ⟨ψ₀|O_q† O_q|ψ₀⟩  (zeroth spectral moment).
    # Use a small Γ and a wide/fine grid so the Lorentzian-broadened integral
    # recovers the static moment; trapezoid on ω ∈ [−12, 12], 24001 points.
    Γsum  = 0.02
    ωsum  = range(-12.0, 12.0, length = 24001)
    Ssum  = dynamical_structure_factor(Hχ_sp, basisχ, Oq_op; ω_grid = ωsum, Γ = Γsum)
    ωg    = collect(Ssum.ω_grid)
    integral = sum((ωg[2:end] .- ωg[1:end-1]) .*
                   (Ssum.tensor[2:end] .+ Ssum.tensor[1:end-1]) ./ 2)   # trapezoid
    static_moment = real(dot(Oq_mat * ψ0χ, Oq_mat * ψ0χ))   # ⟨ψ₀|O_q† O_q|ψ₀⟩
    rel_err = abs(integral - static_moment) / abs(static_moment)
    @info "χ Check D — sum rule ∫S dω = $(integral), ⟨O_q†O_q⟩ = $(static_moment), " *
          "rel. error = $(rel_err)"
    @test rel_err < 5e-2
end
