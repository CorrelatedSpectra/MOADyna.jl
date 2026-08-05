# =====================================================================
# re_broaden_table — energy-dependent Lorentzian re-broadening
# =====================================================================
#
# Per-pole Lorentzian re-broadening with an energy-dependent width
# Γ(ω) defined by linear interpolation over a user-supplied
# (ω, Γ) anchor table. Equivalent to scalar `re_broaden` when the
# table is constant; in general handles e.g. core-hole lifetime
# broadening that varies across the absorption edge.
#
# Implementation: eigendecompose each chunk's block-tridiagonal T_K
# once (Lehmann form), then for every grid point sum
#
#     χ(ω) = R† · Σ_n |n⟩_top · (R† |n⟩_top)† / (z_n − E_n)
#     z_n  = ω + Eg + i · Γ(E_n − Eg) / 2
#
# Γ is evaluated at each pole's *spectroscopic* energy (E_n − Eg),
# matching the user's ω axis. The Lehmann path ensures bit-exact
# agreement with `re_broaden` when the table is constant.
#
# An optional Gaussian convolution (matching the existing
# `re_broaden`'s σ_gauss semantics — `gauss` is the standard
# deviation, not FWHM) is applied along the ω axis.

"""
    re_broaden_table(spec::SpectraTensor;
                     gauss::Real = 0.0,
                     lorentz_table::AbstractVector{<:Pair{<:Real,<:Real}})
        -> SpectraTensor

Re-evaluate a `SpectraTensor` with an energy-dependent Lorentzian
broadening `Γ(ω)` defined by a `(ω, Γ)` anchor table. `Γ(ω)` is
piecewise-linear between table points and clamped to the endpoint
values outside the table range.

Each Lehmann pole `E_n` is broadened with `Γ_n = Γ(E_n − Eg)`, where
`E_n − Eg` is the pole's position on the user's ω axis. After the
Lorentzian sum, an optional Gaussian convolution of standard
deviation `gauss` is applied along the ω axis (Voigt profile).

Arguments
---------
- `spec::SpectraTensor` — the result of `xas(...)`, `rixs(...)`, or
  another `re_broaden_table` call. Must carry chunks (i.e. not an
  algebra-derived spectrum). FY results are rejected (no output Γ
  to re-broaden — the final-state ω_out has been integrated out).
  RIXS results are re-broadened along the output (ω_out / Γ_final)
  axis; `Γ_intermediate` is not handled here (rerun `rixs(...)`).
- `gauss::Real = 0.0` — Gaussian standard deviation (σ_gauss
  semantics). Aliases: `σ_gauss`, `sigma_gauss`.
- `lorentz_table::AbstractVector{<:Pair{<:Real,<:Real}}` — anchor
  points `[ω₁ => Γ₁, ω₂ => Γ₂, …]`. Need not be sorted; will be
  sorted by `ω` internally. Must have ≥ 1 entry; all `Γ_i ≥ 0`.
  ASCII alias: `Lorentz_table`.

Returns
-------
A fresh `SpectraTensor` with the same chunks (the recipe is
preserved) and a new tensor evaluated at the energy-dependent
broadening.

Notes
-----
With a constant table (single point, or two endpoints with the same
Γ), this function reproduces `re_broaden(spec; Γ = Γ_const)` to
floating-point tolerance.
"""
function re_broaden_table(spec::SpectraTensor;
                          gauss::Real = 0.0,
                          σ_gauss     = nothing,
                          sigma_gauss = nothing,
                          lorentz_table = nothing,
                          Lorentz_table = nothing)
    # Resolve table kwarg pair.
    table_user = _resolve_pair(lorentz_table, Lorentz_table;
                               default = nothing, name = "lorentz_table")
    table_user === nothing &&
        throw(ArgumentError("re_broaden_table: must specify lorentz_table"))

    # Resolve Gaussian σ aliases. `gauss` (canonical, kw-only positional in
    # the docstring) plus σ_gauss / sigma_gauss aliases for parity with
    # `re_broaden`. Specifying more than one is an error.
    σ_alias = _resolve_pair(σ_gauss, sigma_gauss;
                            default = nothing, name = "σ_gauss")
    if σ_alias !== nothing && gauss != 0
        throw(ArgumentError(
            "re_broaden_table: specify Gaussian width once (gauss / σ_gauss / sigma_gauss)"))
    end
    σ_use = σ_alias === nothing ? Float64(gauss) : Float64(σ_alias)
    σ_use ≥ 0 || throw(ArgumentError("re_broaden_table: gauss must be ≥ 0"))

    # Validate table.
    isempty(table_user) &&
        throw(ArgumentError("re_broaden_table: lorentz_table must be non-empty"))
    ωs_tab = Float64[Float64(p.first)  for p in table_user]
    Γs_tab = Float64[Float64(p.second) for p in table_user]
    all(Γ -> Γ ≥ 0, Γs_tab) ||
        throw(ArgumentError("re_broaden_table: all Γ values in lorentz_table must be ≥ 0"))
    # Sort by ω.
    perm = sortperm(ωs_tab)
    ωs_tab = ωs_tab[perm]
    Γs_tab = Γs_tab[perm]

    # FY rejects re-broadening (matches `re_broaden`).
    fn_sym = Symbol(spec.metadata[:function])
    fn_sym === :fluorescence_yield &&
        throw(ArgumentError(
            "re_broaden_table: FY result has no output Γ to re-broaden " *
            "(rerun fluorescence_yield(...; Γ_intermediate) instead)"))

    Γ_fn = ω -> _interp_clamped(ω, ωs_tab, Γs_tab)

    if fn_sym === :rixs
        return _rixs_rebroaden_table(spec, Γ_fn, σ_use, ωs_tab, Γs_tab)
    else
        return _xas_rebroaden_table(spec, Γ_fn, σ_use, ωs_tab, Γs_tab)
    end
end

# Linearly interpolate Γ at ω over a sorted (ωs, Γs) anchor table.
# Endpoints clamp outside the range.
function _interp_clamped(ω::Real, ωs::Vector{Float64}, Γs::Vector{Float64})
    n = length(ωs)
    n == 1 && return Γs[1]
    ω ≤ ωs[1]   && return Γs[1]
    ω ≥ ωs[end] && return Γs[end]
    # Binary search for the bracketing interval.
    i = searchsortedlast(ωs, ω)
    # i now satisfies ωs[i] ≤ ω < ωs[i+1] (since ω < ωs[end]).
    ω1, ω2 = ωs[i], ωs[i+1]
    Γ1, Γ2 = Γs[i], Γs[i+1]
    t = (ω - ω1) / (ω2 - ω1)
    return Γ1 + t * (Γ2 - Γ1)
end

# Per-chunk evaluator: Lehmann sum with per-pole Γ_n = Γ_fn(E_n - Eg).
# Returns the (B_raw, B_raw, n_ω) tensor — same shape as
# `evaluate_on_grid` from cf.jl.
function _evaluate_on_grid_table(chunk::LanczosChunk, ω_grid;
                                 Γ_fn, Eg::Real)
    R       = chunk.R
    B_active = size(R, 1)
    B_raw    = chunk.raw_block_size
    Tc      = eltype(R)
    Tc <: Complex ||
        throw(ArgumentError("re_broaden_table: chunk eltype must be Complex; got $Tc"))

    T_K = _build_T_K(chunk.α, chunk.β)
    F   = eigen(Hermitian(T_K))
    Es  = real.(F.values)
    Vs  = F.vectors
    Rdag = adjoint(R)

    # Precompute residue matrices  W_n = (R† n_top) (R† n_top)†   (B_raw × B_raw)
    # and per-pole Γ. Float64 since Γ is real.
    n_poles = length(Es)
    W = Vector{Matrix{Tc}}(undef, n_poles)
    Γ_pole = Vector{Float64}(undef, n_poles)
    @inbounds for n in 1:n_poles
        n_top = Vs[1:B_active, n]
        v     = Rdag * n_top                     # B_raw vector
        W[n]  = v * adjoint(v)
        Γ_pole[n] = Float64(Γ_fn(Float64(Es[n]) - Float64(Eg)))
        Γ_pole[n] ≥ 0 || throw(ArgumentError(
            "re_broaden_table: interpolated Γ at ω = $(Es[n] - Eg) is negative"))
    end

    n_ω = length(ω_grid)
    out = zeros(Tc, B_raw, B_raw, n_ω)
    @inbounds for (i, ω) in enumerate(ω_grid)
        for n in 1:n_poles
            denom = Tc(Float64(ω) + Float64(Eg) - Es[n] + im * (Γ_pole[n] / 2))
            inv_denom = inv(denom)
            for b in 1:B_raw, a in 1:B_raw
                out[a, b, i] += W[n][a, b] * inv_denom
            end
        end
    end
    return out
end

# XAS / FY-shaped re-broadening with a Γ-function.
function _xas_rebroaden_table(spec::SpectraTensor, Γ_fn, σ_gauss::Real,
                              ωs_tab::Vector{Float64}, Γs_tab::Vector{Float64})
    chunks = spec.chunks
    chunks === nothing &&
        throw(ArgumentError("re_broaden_table: spec has no chunks (algebra-derived)"))
    Eg     = spec.Eg
    ω_grid = spec.ω_grid
    n_ω    = length(ω_grid)

    χs = [_evaluate_on_grid_table(c, ω_grid; Γ_fn = Γ_fn, Eg = Eg) for c in chunks]

    md = spec.metadata
    T_is_vector = md[:T_is_vector_input]::Bool
    ψ_is_list   = md[:ψ_is_list_input]::Bool
    N_T = md[:n_T]::Int
    N_ψ = md[:n_ψ]::Int

    new_tensor = _xas_pack(χs, T_is_vector, ψ_is_list, N_T, N_ψ, n_ω)

    if σ_gauss > 0
        new_tensor = _convolve_gaussian(new_tensor, ω_grid, σ_gauss)
    end

    new_meta = copy(md)
    new_meta[:Γ_table] = collect(zip(ωs_tab, Γs_tab))
    delete!(new_meta, :Γ)              # scalar Γ no longer applies
    σ_gauss > 0 && (new_meta[:σ_gauss] = σ_gauss)
    new_meta[:rebroadened_from] = (Γ = get(md, :Γ, nothing),)

    F = typeof(ω_grid)
    N = ndims(new_tensor)
    return SpectraTensor{eltype(new_tensor), Float64, N, F}(
        new_tensor, chunks, ω_grid, Eg, spec.block_size, new_meta)
end

# RIXS re-broadening along the output axis (ω_out / Γ_final).
function _rixs_rebroaden_table(spec::SpectraTensor, Γ_fn, σ_gauss::Real,
                               ωs_tab::Vector{Float64}, Γs_tab::Vector{Float64})
    chunks = spec.chunks
    chunks === nothing &&
        throw(ArgumentError("re_broaden_table: spec has no chunks (algebra-derived)"))
    Eg = spec.Eg
    ω_in_grid, ω_out_grid = spec.ω_grid::Tuple
    n_in  = length(ω_in_grid)
    n_out = length(ω_out_grid)

    md = spec.metadata
    N_in   = md[:n_T_in]::Int
    N_out  = md[:n_T_out]::Int
    N_ψ    = md[:n_ψ]::Int
    T_in_vec  = md[:T_in_is_vector_input]::Bool
    T_out_vec = md[:T_out_is_vector_input]::Bool
    ψ_is_list = md[:ψ_is_list_input]::Bool
    full6d = T_in_vec || T_out_vec

    new_tensor = if full6d && ψ_is_list
        zeros(ComplexF64, N_in, N_out, N_out, N_in, N_ψ, n_in, n_out)
    elseif full6d
        zeros(ComplexF64, N_in, N_out, N_out, N_in, n_in, n_out)
    elseif ψ_is_list
        zeros(ComplexF64, N_ψ, n_in, n_out)
    else
        zeros(ComplexF64, n_in, n_out)
    end

    for chunk in chunks
        χ̄ = _evaluate_on_grid_table(chunk, ω_out_grid; Γ_fn = Γ_fn, Eg = Eg)
        ψ_idx    = chunk.ψ_index
        ω_in_idx = chunk.ω_in_index
        for ω_idx in 1:n_out
            slab = χ̄[:, :, ω_idx]
            if full6d
                χ_jikl = reshape(slab, N_out, N_in, N_out, N_in)
                χ_user = permutedims(χ_jikl, (2, 1, 3, 4))
                if ψ_is_list
                    new_tensor[:, :, :, :, ψ_idx, ω_in_idx, ω_idx] .= χ_user
                else
                    new_tensor[:, :, :, :, ω_in_idx, ω_idx] .= χ_user
                end
            else
                if ψ_is_list
                    new_tensor[ψ_idx, ω_in_idx, ω_idx] = slab[1, 1]
                else
                    new_tensor[ω_in_idx, ω_idx] = slab[1, 1]
                end
            end
        end
    end

    if σ_gauss > 0
        new_tensor = _convolve_gaussian(new_tensor, ω_out_grid, σ_gauss)
    end

    new_meta = copy(md)
    new_meta[:Γ_final_table] = collect(zip(ωs_tab, Γs_tab))
    delete!(new_meta, :Γ_final)
    σ_gauss > 0 && (new_meta[:σ_gauss] = σ_gauss)
    new_meta[:rebroadened_from] = (Γ_final = get(md, :Γ_final, nothing),)

    F = typeof(spec.ω_grid)
    N = ndims(new_tensor)
    return SpectraTensor{eltype(new_tensor), Float64, N, F}(
        new_tensor, chunks, spec.ω_grid, Eg, spec.block_size, new_meta)
end
