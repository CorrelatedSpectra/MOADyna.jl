# =====================================================================
# MOAD.Spectroscopy — optical_conductivity (2e)
# =====================================================================
#
# Thin wrapper over MOAD.Responses.correlator(...; channel = :neutral):
# the regular optical conductivity Re σ_αβ(ω > 0) from the one-sided
# current–current correlator, with the finite-T detailed-balance factor.
# One-sided / regular
# part only — not the full Kubo retarded σ (deferred). Bring-your-own
# current operator (no current-operator / position machinery in 2e).
#
# The method is added in 2e Task 1; this file holds the generic-function
# declaration so `using MOAD` loads with the symbol exported (matching the
# `function xas end` idiom).

# --- shared 2e helpers (used by optical_conductivity + dynamical_structure_factor) ---

# Normalise a single OperatorSum or a Vector of them to a Vector.
_to_op_list(x::OperatorSum) = OperatorSum[x]
_to_op_list(xs::Vector{<:OperatorSum}) = collect(xs)

# Resolve the neutral one-sided correlator GridResponse C_{ab}(ω) on `ω_use`.
# Ground-state/ensemble path (P4): optionally pass a precomputed ψ₀+Eg for the
# T=0 path; finite-T must use `state = :ground_state`.
function _response_grid(H, basis::AbstractBasis, ops::Vector{<:OperatorSum},
                        ω_use, Γ_use, T, N_states, degen_tol, ψ₀, Eg; kwargs...)
    common = (channel = :neutral, form = :grid, ω = ω_use, Γ = Γ_use,
              T = T, N_states = N_states, degen_tol = degen_tol)
    if ψ₀ === nothing
        # ψ₀ and Eg are supplied as a pair; Eg alone would be silently dropped
        # (the kernel computes the ground state when ψ₀ is omitted), so reject it.
        Eg === nothing ||
            throw(ArgumentError("Eg was supplied without ψ₀ — they are a pair; " *
                                "omit Eg to use the computed ground state, or pass both"))
        return Responses.correlator(H, basis, ops, ops; common..., kwargs...)
    else
        Eg === nothing &&
            throw(ArgumentError("Eg is required when ψ₀ is supplied"))
        return Responses.correlator(H, basis, ops, ops;
                                    state = ψ₀, Eg = Eg, common..., kwargs...)
    end
end

"""
    optical_conductivity(H, basis, j; kwargs...) -> SpectraTensor

Regular (paramagnetic) optical conductivity `Re σ_αβ(ω, T)` for `ω > 0`,
built from the one-sided current–current correlator
`Λ_αβ(ω) = ⟨ψ₀| j_α† (ω + E₀ − H + iΓ/2)⁻¹ j_β |ψ₀⟩` (finite-T: the 2d
thermal average), via

    Re σ_αβ(ω, T) = factor(ω, T) · ( −Im Λ_αβ(ω) ) / (ω · V),
    factor(ω, T)  = (T === nothing || iszero(T)) ? 1 : -expm1(-ω/T)

with units `e = ℏ = 1` (charge / lattice-constant factors belong in the
caller's current operator `j`; `volume` is a per-cluster normalisation).

This is the **regular** part only — the zero-frequency Drude weight (and
the diamagnetic contribution) is *not* included and is *not* assumed
zero; it is model/boundary-dependent and must be obtained separately.
This is **not** the full Kubo retarded conductivity (deferred).

# Positional arguments
- `H::AbstractMatrix`     — Hamiltonian on `basis`.
- `basis::AbstractBasis`  — the basis `H` and `j` live on.
- `j`                     — `OperatorSum` (single direction → scalar `σ`)
                            or `Vector{<:OperatorSum}` (e.g. `[j_x, j_y, j_z]`
                            → ordered tensor `σ_αβ`). The same list is used
                            as both `As` and `Bs` (P3); off-diagonal `α≠β`
                            is the ordered element `C_αβ`, **not** a
                            cross-correlator. The caller symmetrises into the
                            dissipative tensor `½(σ_αβ + σ_βα*)` if wanted.

# Keyword arguments
ASCII aliases: `ω_grid ↔ omega_grid`, `Γ ↔ Gamma` (`_resolve_pair`).

| kwarg | default | role |
|---|---|---|
| `ω_grid` | required | explicit ω grid (should contain `ω > 0`; `ω ≤ 0` are dropped). |
| `Γ` | `DEFAULTS.Γ_response` | FWHM Lorentzian broadening. |
| `volume` | `1.0` | per-cluster volume normalisation; must be `> 0` and finite. |
| `temperature` | `nothing` | finite-T temperature (`k_B = 1`); `nothing`/`0.0` ⇒ factor 1. Canonical thermal kwarg (consistent with `xas`/`rixs`/`fluorescence_yield`). |
| `T` | `nothing` | accepted alias for `temperature`; specifying both raises. |
| `N_states`, `degen_tol` | thermal-only knobs forwarded to `correlator`. |
| `ψ₀`, `Eg` | `nothing` | optional precomputed ground state (T=0 path only; forbidden with `T`). |

# Return
`SpectraTensor{Float64}`: a 1-D tensor `(n_ω⁺,)` for a single operator,
or `(N, N, n_ω⁺)` for a vector of `N` operators, sampled on the filtered
`ω > 0` grid. Metadata records `:observable => :sigma_regular`, `:volume`,
`:T`, `:Γ`, and `:dropped_nonpositive_ω` (count of dropped `ω ≤ 0` points).
The complex correlator is **not** stored in metadata (call `correlator`
directly if needed).
"""
function optical_conductivity(H, basis::AbstractBasis,
                              j::Union{OperatorSum, Vector{<:OperatorSum}};
                              ω_grid = nothing, omega_grid = nothing,
                              Γ = nothing, Gamma = nothing,
                              volume::Real = 1.0,
                              T = nothing, temperature = nothing,
                              N_states = nothing, degen_tol = 1e-8,
                              ψ₀ = nothing, Eg = nothing, kwargs...)
    # `temperature` is the canonical thermal kwarg (consistent with xas/rixs/fy);
    # `T` is accepted as an alias. Specifying both raises.
    T = _resolve_pair(temperature, T; default = nothing, name = "temperature")
    ops = _to_op_list(j)
    isempty(ops) &&
        throw(ArgumentError("optical_conductivity: j must be a non-empty " *
                            "OperatorSum or Vector{<:OperatorSum}"))
    (volume > 0 && isfinite(volume)) ||
        throw(ArgumentError("optical_conductivity: volume must be positive and " *
                            "finite (got $volume)"))
    (ψ₀ === nothing || T === nothing) ||
        throw(ArgumentError("optical_conductivity: ψ₀ cannot be combined with T " *
                            "(finite-T requires the ground-state/ensemble path)"))

    ω_use = _resolve_pair(ω_grid, omega_grid; default = nothing, name = "ω_grid")
    ω_use === nothing &&
        throw(ArgumentError("optical_conductivity: ω_grid is required " *
                            "(pass ω_grid or omega_grid)"))
    Γ_use = _resolve_pair(Γ, Gamma; default = DEFAULTS.Γ_response, name = "Γ")

    G = _response_grid(H, basis, ops, ω_use, Γ_use, T, N_states, degen_tol, ψ₀, Eg;
                       kwargs...)

    # Keep only ω > 0 (Re σ is even; the one-sided spectrum carries emission
    # weight at ω ≤ 0). Drop ALL ω ≤ 0, recording the count.
    ω_full = G.ω
    keep   = findall(>(0), ω_full)
    isempty(keep) &&
        throw(ArgumentError("optical_conductivity: ω_grid has no positive " *
                            "frequencies; Re σ is defined for ω > 0 only"))
    n_drop = length(ω_full) - length(keep)
    ω_pos  = ω_full[keep]
    n_ops  = length(ops)

    Tnil = (T === nothing || iszero(T))
    out  = Array{Float64, 3}(undef, n_ops, n_ops, length(keep))
    @inbounds for (kk, kidx) in enumerate(keep)
        ω = ω_pos[kk]
        factor = Tnil ? 1.0 : -expm1(-ω / T)
        scale  = factor / (ω * volume)
        for b in 1:n_ops, a in 1:n_ops
            out[a, b, kk] = scale * (-imag(G.data[a, b, kidx]))
        end
    end

    tensor = n_ops == 1 ? out[1, 1, :] : out
    ω_grid_out = collect(Float64, ω_pos)
    metadata = Dict{Symbol, Any}(
        :function              => :optical_conductivity,
        :observable            => :sigma_regular,
        :volume                => Float64(volume),
        :T                     => T,
        :Γ                     => Float64(Γ_use),
        :dropped_nonpositive_ω => n_drop,
        :n_ops                 => n_ops,
    )
    F = typeof(ω_grid_out)
    N = ndims(tensor)
    return SpectraTensor{Float64, Float64, N, F}(
        tensor, nothing, ω_grid_out, Float64(G.Eg), n_ops, metadata)
end
