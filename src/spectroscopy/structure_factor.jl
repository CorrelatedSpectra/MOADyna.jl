# =====================================================================
# MOADyna.Spectroscopy — dynamical_structure_factor (2e)
# =====================================================================
#
# Thin wrapper over MOADyna.Responses.correlator(...; channel = :neutral):
# the dynamical structure factor S(q,ω) = −Im C(q,ω)/π from the one-sided
# correlator of a caller-supplied q-phased operator O_q.
# One-sided structure
# factor — NOT the Kubo retarded χ(q,ω) (deferred). Bring-your-own
# q-operator (no position / Fourier machinery in 2e).
#
# Implemented in 2e Task 2. Reuses the shared `_to_op_list` / `_response_grid`
# helpers defined in conductivity.jl (same module; included earlier).

"""
    dynamical_structure_factor(H, basis, O_q; kwargs...) -> SpectraTensor

Dynamical structure factor `S(q, ω, T) = −Im C(q,ω,T) / π` from the
one-sided correlator
`C(q,ω) = ⟨ψ₀| O_q† (ω + E₀ − H + iΓ/2)⁻¹ O_q |ψ₀⟩` (finite-T: the 2d
thermal average), evaluated over the **full** ω grid (anti-Stokes weight
at `ω < 0` is physical; no ω-division, no detailed-balance factor — the
thermal weights are intrinsic at finite T).

This is the one-sided structure factor, **not** the Kubo retarded
χ(q,ω) (deferred). For a generic non-Hermitian `O_q`, detailed balance
relates the `−q` / conjugate channel, so no same-`q` detailed-balance
claim is made — the wrapper simply reports `−Im C/π`.

# Positional arguments
- `H::AbstractMatrix`     — Hamiltonian on `basis`.
- `basis::AbstractBasis`  — the basis `H` and `O_q` live on.
- `O_q`                   — `OperatorSum` (single → scalar `S(q,ω)`) or
                            `Vector{<:OperatorSum}` (→ ordered matrix
                            `C_{ij}`, diagonal `i=i` are per-operator `S`,
                            off-diagonal `i≠j` are cross-terms). The same
                            list is used as both `As` and `Bs` (not a
                            cross-correlator). The caller owns the `O_q`
                            convention (e.g. `O_q = Σ_r e^{iq·r_r} O_r`).

# Keyword arguments
ASCII aliases: `ω_grid ↔ omega_grid`, `Γ ↔ Gamma` (`_resolve_pair`).

| kwarg | default | role |
|---|---|---|
| `ω_grid` | required | explicit ω grid (full grid retained — no drop). |
| `Γ` | `DEFAULTS.Γ_response` | FWHM Lorentzian broadening. |
| `temperature` | `nothing` | finite-T temperature (`k_B = 1`). Canonical thermal kwarg (consistent with `xas`/`rixs`/`fluorescence_yield`). |
| `T` | `nothing` | accepted alias for `temperature`; specifying both raises. |
| `N_states`, `degen_tol` | thermal-only knobs forwarded to `correlator`. |
| `ψ₀`, `Eg` | `nothing` | optional precomputed ground state (T=0 path only; forbidden with `T`). |

# Return
`SpectraTensor{Float64}`: a 1-D tensor `(n_ω,)` for a single operator, or
`(N, N, n_ω)` for a vector of `N` operators, on the full `ω_grid`.
Metadata records `:observable => :S_q_omega`, `:T`, `:Γ`. The complex
correlator is **not** stored in metadata (call `correlator` directly).
"""
function dynamical_structure_factor(H, basis::AbstractBasis,
                                    O_q::Union{OperatorSum, Vector{<:OperatorSum}};
                                    ω_grid = nothing, omega_grid = nothing,
                                    Γ = nothing, Gamma = nothing,
                                    T = nothing, temperature = nothing,
                                    N_states = nothing, degen_tol = 1e-8,
                                    ψ₀ = nothing, Eg = nothing, kwargs...)
    # `temperature` is the canonical thermal kwarg (consistent with xas/rixs/fy);
    # `T` is accepted as an alias. Specifying both raises.
    T = _resolve_pair(temperature, T; default = nothing, name = "temperature")
    ops = _to_op_list(O_q)
    isempty(ops) &&
        throw(ArgumentError("dynamical_structure_factor: O_q must be a non-empty " *
                            "OperatorSum or Vector{<:OperatorSum}"))
    (ψ₀ === nothing || T === nothing) ||
        throw(ArgumentError("dynamical_structure_factor: ψ₀ cannot be combined with T " *
                            "(finite-T requires the ground-state/ensemble path)"))

    ω_use = _resolve_pair(ω_grid, omega_grid; default = nothing, name = "ω_grid")
    ω_use === nothing &&
        throw(ArgumentError("dynamical_structure_factor: ω_grid is required " *
                            "(pass ω_grid or omega_grid)"))
    Γ_use = _resolve_pair(Γ, Gamma; default = DEFAULTS.Γ_response, name = "Γ")

    G = _response_grid(H, basis, ops, ω_use, Γ_use, T, N_states, degen_tol, ψ₀, Eg;
                       kwargs...)

    # S(q,ω) = −Im C(q,ω)/π over the FULL grid (no ω≤0 drop, no factor).
    n_ops = length(ops)
    nω    = length(G.ω)
    out   = Array{Float64, 3}(undef, n_ops, n_ops, nω)
    @inbounds for k in 1:nω, b in 1:n_ops, a in 1:n_ops
        out[a, b, k] = (-imag(G.data[a, b, k])) / π
    end

    tensor = n_ops == 1 ? out[1, 1, :] : out
    ω_grid_out = collect(Float64, G.ω)
    metadata = Dict{Symbol, Any}(
        :function   => :dynamical_structure_factor,
        :observable => :S_q_omega,
        :T          => T,
        :Γ          => Float64(Γ_use),
        :n_ops      => n_ops,
    )
    F = typeof(ω_grid_out)
    N = ndims(tensor)
    return SpectraTensor{Float64, Float64, N, F}(
        tensor, nothing, ω_grid_out, Float64(G.Eg), n_ops, metadata)
end
