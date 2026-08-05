# =====================================================================
# MOAD.Spectroscopy — kubo_response: full Kubo retarded susceptibility χ_AB(ω,T)
# =====================================================================
#
# The complete two-sided Kubo retarded response function
#
#   χ_AB(ω, T) = Σ_{m,n} (w_m − w_n) ⟨m|A|n⟩⟨n|B|m⟩ / (ω + iΓ/2 − (E_n − E_m)),
#   w_k = e^{−βE_k} / Z,
#
# from a DIRECT eigenstate-pair (dense Lehmann) sum — distinct from the
# one-sided correlator wrappers `optical_conductivity` / `dynamical_structure_factor`
# (which give only Re σ(ω>0) / S(q,ω) from the one-sided current/density
# correlator). Unlike those, this is a genuine CROSS-correlator: `A` and `B` may
# differ, and the commutator weight `(w_m − w_n)` requires eigenstate-pair matrix
# elements, so the implementation diagonalises the sector rather than running a
# one-sided Lanczos resolvent.
#
# Design (v0.2 closeout Item C):
#  - `temperature` is required (`k_B = 1`); `temperature = 0` uses the same
#    degenerate-ground-manifold averaging as the rest of the finite-T layer.
#  - The truncation knob is `N_eigs` (NOT `N_states`): it is a PROJECTED Lehmann
#    response — both the initial AND final states are restricted to the lowest
#    `N_eigs` eigenstates. Default is the full spectrum (small sectors only); a
#    truncation warns.
#  - The regular broadened response drops the `n = m` and (near-)degenerate
#    `|E_m − E_n| < degen_tol` terms (their commutator numerator vanishes in
#    equilibrium). The Drude / static-δ intraband weight is therefore EXCLUDED
#    (recorded as `:drude_excluded => true` in metadata), exactly as
#    `optical_conductivity` excludes the Drude weight.
#  - Broadening convention matches the rest of the layer:
#    `… / (ω + iΓ/2 − (E_n − E_m))`, FWHM kwarg `Γ`.

"""
    kubo_response(H, basis, A, B; temperature, ω_grid, Γ, N_eigs = nothing,
                  degen_tol = 1e-8) -> SpectraTensor

Full Kubo retarded susceptibility

```math
χ_{AB}(ω, T) = \\sum_{m,n} (w_m - w_n)
               \\frac{⟨m|A|n⟩⟨n|B|m⟩}{ω + iΓ/2 - (E_n - E_m)},
\\qquad w_k = e^{-βE_k}/Z,
```

assembled directly from the eigenstate-pair matrix elements of `H` on `basis`.
This is the **full two-sided** retarded response (both reactive and dissipative
parts, all `ω`) — distinct from the one-sided [`optical_conductivity`](@ref) /
[`dynamical_structure_factor`](@ref) wrappers. A genuine cross-correlator: `A`
and `B` may differ.

# Positional arguments
- `H::AbstractMatrix`     — Hamiltonian on `basis`.
- `basis::AbstractBasis`  — the basis `H`, `A`, `B` live on.
- `A`, `B`                — `OperatorSum` (→ scalar `χ_{AB}(ω)`) or
                            `Vector{<:OperatorSum}` (→ tensor `χ_{ij}(ω)` with
                            `i` indexing `A`, `j` indexing `B`). Bring-your-own
                            operators (same convention as σ/S).

# Keyword arguments
ASCII aliases: `ω_grid ↔ omega_grid`, `Γ ↔ Gamma`.

| kwarg | default | role |
|---|---|---|
| `temperature` | required | temperature (`k_B = 1`). `0` ⇒ degenerate-GS average. |
| `ω_grid` | required | explicit ω grid (full grid retained; reactive part at all ω). |
| `Γ` | `DEFAULTS.Γ_response` | FWHM Lorentzian broadening. |
| `N_eigs` | `nothing` | project onto the lowest `N_eigs` eigenstates (initial AND final) via a **partial** eigensolve. `nothing` ⇒ full dense spectrum (small sectors only). A truncation `@warn`s; a cutoff that splits a degenerate manifold raises. |
| `degen_tol` | `1e-8` | `|E_m − E_n| < degen_tol` pairs (incl. `n = m`) are dropped from the regular response. |

# Return
`SpectraTensor{ComplexF64}` carrying the **full complex** `χ` — a 1-D tensor
`(n_ω,)` for scalar `A`,`B`, or `(N_A, N_B, n_ω)` for operator vectors. `Eg = E₀`
(ground energy). Metadata records `:observable => :kubo_chi`, `:temperature`,
`:Γ`, `:N_eigs`, and `:drude_excluded => true` (the Drude/static-δ intraband
term is not included — obtain it separately, as for `optical_conductivity`).

The dense pair sum is `O(\\dim^2)`, tractable only on small sectors.
"""
function kubo_response(H, basis::AbstractBasis,
                       A::Union{OperatorSum, Vector{<:OperatorSum}},
                       B::Union{OperatorSum, Vector{<:OperatorSum}};
                       temperature::Real,
                       ω_grid = nothing, omega_grid = nothing,
                       Γ = nothing, Gamma = nothing,
                       N_eigs = nothing, degen_tol = 1e-8)
    (temperature isa Real && temperature ≥ 0) || throw(ArgumentError(
        "kubo_response: temperature must be a real number ≥ 0; got $(repr(temperature))"))
    (degen_tol isa Real && isfinite(degen_tol) && degen_tol > 0) || throw(ArgumentError(
        "kubo_response: degen_tol must be a positive finite real; got $(repr(degen_tol))"))

    A_is_vec = !(A isa OperatorSum)
    B_is_vec = !(B isa OperatorSum)
    A_list = A isa OperatorSum ? OperatorSum[A] : collect(A)
    B_list = B isa OperatorSum ? OperatorSum[B] : collect(B)
    (isempty(A_list) || isempty(B_list)) && throw(ArgumentError(
        "kubo_response: A and B must each be a non-empty OperatorSum or Vector{<:OperatorSum}"))

    ω_use = _resolve_pair(ω_grid, omega_grid; default = nothing, name = "ω_grid")
    ω_use === nothing && throw(ArgumentError(
        "kubo_response: ω_grid is required (pass ω_grid or omega_grid)"))
    Γ_use = _resolve_pair(Γ, Gamma; default = DEFAULTS.Γ_response, name = "Γ")

    evals, V = _kubo_spectrum(H, basis, N_eigs, degen_tol)
    nstate = length(evals)
    E0 = minimum(evals)
    w  = _kubo_weights(evals, Float64(temperature), E0, degen_tol)

    # Matrix elements in the eigenbasis: Amats[i][m,n] = ⟨m|A_i|n⟩.
    Amats = [adjoint(V) * (Matrix{ComplexF64}(assemble(compile(a, basis), basis)) * V)
             for a in A_list]
    Bmats = [adjoint(V) * (Matrix{ComplexF64}(assemble(compile(b, basis), basis)) * V)
             for b in B_list]

    ω_vec = collect(Float64, ω_use)
    nω = length(ω_vec)
    nA = length(A_list); nB = length(B_list)
    out  = zeros(ComplexF64, nA, nB, nω)
    half = im * Float64(Γ_use) / 2

    # Pair-Lehmann sum. Skip n=m and |ΔE|<degen_tol (commutator numerator → 0
    # in equilibrium; the Drude/static term lives there and is excluded).
    for n in 1:nstate, m in 1:nstate
        dw = w[m] - w[n]
        dw == 0 && continue
        ΔE = evals[n] - evals[m]
        abs(ΔE) < degen_tol && continue
        for j in 1:nB, i in 1:nA
            coeff = dw * Amats[i][m, n] * Bmats[j][n, m]
            coeff == 0 && continue
            @inbounds for k in 1:nω
                out[i, j, k] += coeff / (ω_vec[k] + half - ΔE)
            end
        end
    end

    tensor = (!A_is_vec && !B_is_vec) ? out[1, 1, :] : out
    metadata = Dict{Symbol, Any}(
        :function       => :kubo_response,
        :observable     => :kubo_chi,
        :temperature    => temperature,
        :Γ              => Float64(Γ_use),
        :N_eigs         => N_eigs === nothing ? nstate : N_eigs,
        :n_states_used  => nstate,
        :drude_excluded => true,
        :degen_tol      => degen_tol,
        :n_A            => nA,
        :n_B            => nB,
    )
    F = typeof(ω_vec)
    N = ndims(tensor)
    return SpectraTensor{ComplexF64, Float64, N, F}(
        tensor, nothing, ω_vec, Float64(E0), max(nA, nB), metadata)
end

# Eigensystem for the pair-Lehmann sum.
#  - N_eigs === nothing: full dense spectrum (small sectors only).
#  - N_eigs given: a genuine PROJECTED Lehmann — a partial eigensolve for the
#    lowest N_eigs states (NOT a full solve + truncate), so large sectors stay
#    usable. A degeneracy gate forbids splitting a manifold at the cutoff (the
#    within-manifold eigenbasis is arbitrary, so a partial manifold gives a
#    basis-dependent χ).
function _kubo_spectrum(H, basis::AbstractBasis, N_eigs, degen_tol)
    if N_eigs === nothing
        length(basis) > 4096 && throw(ArgumentError(
            "kubo_response: full-spectrum (N_eigs = nothing) dense pair-Lehmann is only " *
            "tractable for length(basis) ≤ 4096 (got $(length(basis))); pass N_eigs"))
        F = eigen(Hermitian(Matrix{ComplexF64}(H)))
        evals = Float64.(real.(F.values))
        perm = sortperm(evals)
        return evals[perm], Matrix{ComplexF64}(F.vectors)[:, perm]
    end

    (N_eigs isa Integer && !(N_eigs isa Bool) && N_eigs ≥ 1) || throw(ArgumentError(
        "kubo_response: N_eigs must be an Integer ≥ 1; got $(repr(N_eigs))"))
    # Partial eigensolve: lowest N_eigs (+1 buffer state for the manifold gate).
    n_req = min(N_eigs + 1, length(basis))
    E = eigen(H, basis; n = n_req)
    evals = Float64.(real.(E.values))
    perm = sortperm(evals)
    evals = evals[perm]
    V = Matrix{ComplexF64}(E.vectors)[:, perm]

    if length(evals) ≥ N_eigs + 1 && abs(evals[N_eigs + 1] - evals[N_eigs]) < degen_tol
        throw(ArgumentError(
            "kubo_response: N_eigs=$N_eigs splits a degenerate manifold " *
            "(E[$N_eigs] and E[$(N_eigs + 1)] differ by < degen_tol=$degen_tol); " *
            "increase N_eigs to include the full manifold"))
    end

    n_keep = min(N_eigs, length(evals))
    if n_keep < length(basis)
        @warn "kubo_response: projected Lehmann onto the lowest $n_keep of " *
              "$(length(basis)) states — BOTH initial and final states are truncated " *
              "(an approximation to the full retarded χ)."
    end
    return evals[1:n_keep], V[:, 1:n_keep]
end

# Boltzmann weights; T=0 → equal weight over the degenerate ground manifold.
function _kubo_weights(evals::AbstractVector{<:Real}, temperature::Real,
                       E0::Real, degen_tol::Real)
    if temperature == 0
        g = count(e -> abs(e - E0) < degen_tol, evals)
        return [abs(e - E0) < degen_tol ? 1.0 / g : 0.0 for e in evals]
    else
        β = 1 / temperature
        raw = exp.(-β .* (evals .- E0))
        return raw ./ sum(raw)
    end
end
