# =====================================================================
# MOAD.Responses — arithmetic: evaluation, scalar multiplication,
#                  adjoint, and addition
# =====================================================================
#
# Convention pins:
#   - Frequency:    z = ω + Eg + iΓ/2
#   - Evaluation:   prefactor * R' * cf_block(α, β, z) * R
#   - adjoint(LanczosResponse) → PoleResponse via promote-then-adjoint
#   - + on PoleResponse: requires Eg/Γ matching; lazy-promote Lanczos via to_pole

# =====================================================================
# Evaluation R(ω) per concrete type
# =====================================================================

"""
    (R::LanczosResponse{T})(ω::Number) -> Matrix

Evaluate the response at frequency `ω` using the continued-fraction
recurrence. The unified sign=±1 frequency convention is:

    z = sign * (ω + im * Γ/2) + Eg

For `sign = +1` (addition / neutral channel):

    z = ω + Eg + iΓ/2

For `sign = -1` (removal channel):

    z = Eg - ω - iΓ/2

In both cases the result is:

    sign * prefactor * R' * cf_block(α, β, z) * R

See the Conventions appendix for the sign-flip algebra.

Returns a `B_raw × B_raw` matrix (shape of `R.R`'s column dimension).
"""
function (R::LanczosResponse{T})(ω::Number) where {T}
    z = R.sign * (ω + im * R.Γ / 2) + R.Eg
    G_top = cf_block(R.α, R.β, z)                      # B_active × B_active
    return R.sign * R.prefactor * R.R' * G_top * R.R   # B_raw × B_raw
end

"""
    (P::PoleResponse{T})(ω::Number) -> Matrix

Evaluate the pole-residue representation at frequency `ω`:

    P(ω) = P.a0 + Σ_n P.residues[n] / (ω − P.poles[n] + im * P.Γ / 2)

Returns a matrix of the same shape as `P.a0` and each `P.residues[n]`.
"""
function (P::PoleResponse{T})(ω::Number) where {T}
    Tc = T <: Complex ? T : Complex{T}
    result = Matrix{Tc}(P.a0)   # copy a0 into accumulator
    for n in eachindex(P.poles)
        coef = inv(Tc(ω) - Tc(P.poles[n]) + im * Tc(P.Γ) / 2)
        @. result += coef * P.residues[n]
    end
    return result
end

"""
    (G::GridResponse{T, N})(ω::Number) -> Array

Evaluate the gridded response at an exact grid point. If `ω` is not in
`G.ω`, raises `ArgumentError` (interpolation is deferred to v0.5+).

Returns a slice of `G.data` at the matching ω index along the last axis:
a `Matrix` when `N = 3` (the standard `(rows, cols, nω)` layout), or an
`(N-1)`-dimensional `Array` for higher-order grids (e.g., `N = 4` for
RIXS with layout `(rows, cols, ω_in, ω_out)`).
"""
function (G::GridResponse{T, N})(ω::Number) where {T, N}
    idx = findfirst(==(ω), G.ω)
    if idx === nothing
        throw(ArgumentError(
            "GridResponse evaluation: ω = $(ω) not in stored grid; " *
            "interpolation deferred to v0.5+"
        ))
    end
    # last axis is ω; slice it
    colons = ntuple(_ -> Colon(), N - 1)
    return G.data[colons..., idx]
end

"""
    (GF::GreensFunction)(ω::Number) -> Matrix

Evaluate the Green's function at `ω` by summing populated channels.
If both `addition` and `removal` are populated, returns their sum;
if only one is populated, returns its value.
"""
function (GF::GreensFunction)(ω::Number)
    if !isnothing(GF.addition) && !isnothing(GF.removal)
        return GF.addition(ω) + GF.removal(ω)
    elseif !isnothing(GF.addition)
        return GF.addition(ω)
    else
        return GF.removal(ω)
    end
end

# =====================================================================
# Scalar multiplication and adjoint
# =====================================================================

# --- Scalar * LanczosResponse --------------------------------------------

"""
    α * R::LanczosResponse -> LanczosResponse

Scale a `LanczosResponse` by a scalar `α`. Only the `prefactor` field
is updated; `α/β/R` are not touched. If `α` has a different eltype from
`T`, the result is promoted to `LanczosResponse{promote_type(T, typeof(α))}`.
"""
function Base.:*(α::Number, L::LanczosResponse{T}) where {T}
    S = promote_type(T, typeof(α))
    new_prefactor = S(α) * S(L.prefactor)
    if S === T
        return LanczosResponse{T}(L.α, L.β, L.R, L.Eg, L.Γ, L.sign,
                                   new_prefactor, L.converged)
    else
        # promote all blocks
        new_α = Vector{Matrix{S}}([Matrix{S}(a) for a in L.α])
        new_β = Vector{Matrix{S}}([Matrix{S}(b) for b in L.β])
        new_R = Matrix{S}(L.R)
        return LanczosResponse{S}(new_α, new_β, new_R, L.Eg, L.Γ, L.sign,
                                   new_prefactor, L.converged)
    end
end

"""
    R::LanczosResponse * α -> LanczosResponse

Commute to `α * R`.
"""
Base.:*(L::LanczosResponse, α::Number) = α * L

# --- Scalar * PoleResponse -----------------------------------------------

"""
    α * P::PoleResponse -> PoleResponse

Scale `a0` and all `residues` by `α`. Type-promotes if necessary.
"""
function Base.:*(α::Number, P::PoleResponse{T}) where {T}
    S = promote_type(T, typeof(α))
    new_a0 = S(α) .* Matrix{S}(P.a0)
    new_residues = Vector{Matrix{S}}([S(α) .* Matrix{S}(r) for r in P.residues])
    return PoleResponse{S}(new_a0, P.poles, new_residues, P.Eg, P.Γ)
end

"""
    P::PoleResponse * α -> PoleResponse
"""
Base.:*(P::PoleResponse, α::Number) = α * P

# --- Scalar * GridResponse -----------------------------------------------

"""
    α * G::GridResponse -> GridResponse

Scale `data` by `α`. Type-promotes if necessary.
"""
function Base.:*(α::Number, G::GridResponse{T, N}) where {T, N}
    S = promote_type(T, typeof(α))
    new_data = S(α) .* Array{S, N}(G.data)
    return GridResponse{S, N}(new_data, G.ω, G.Eg, G.Γ)
end

"""
    G::GridResponse * α -> GridResponse
"""
Base.:*(G::GridResponse, α::Number) = α * G

# --- Scalar * GreensFunction ---------------------------------------------

"""
    α * GF::GreensFunction -> GreensFunction

Scale each populated channel by `α`.
"""
function Base.:*(α::Number, GF::GreensFunction{T, R}) where {T, R}
    new_addition = isnothing(GF.addition) ? nothing : α * GF.addition
    new_removal  = isnothing(GF.removal)  ? nothing : α * GF.removal
    # Determine the new inner-rep type from whichever channel is populated
    R2 = if !isnothing(new_addition)
        typeof(new_addition)
    else
        typeof(new_removal)
    end
    T2 = eltype(R2)
    return GreensFunction{T2, R2}(
        isnothing(new_addition) ? nothing : new_addition,
        isnothing(new_removal)  ? nothing : new_removal
    )
end

"""
    GF::GreensFunction * α -> GreensFunction
"""
Base.:*(GF::GreensFunction, α::Number) = α * GF

# --- adjoint(PoleResponse) -----------------------------------------------

"""
    adjoint(P::PoleResponse) -> PoleResponse

Take the adjoint of each matrix field:
- `a0  → a0'`
- `residues[n] → residues[n]'`
- `poles` are unchanged (real for Hermitian H)
- `Eg`, `Γ` unchanged
"""
function Base.adjoint(P::PoleResponse{T}) where {T}
    new_a0 = Matrix{T}(adjoint(P.a0))
    new_residues = Vector{Matrix{T}}([Matrix{T}(adjoint(r)) for r in P.residues])
    return PoleResponse{T}(new_a0, P.poles, new_residues, P.Eg, P.Γ)
end

# --- adjoint(LanczosResponse) → PoleResponse (promote-then-adjoint) ------

"""
    adjoint(L::LanczosResponse) -> PoleResponse

Promote `L` to `PoleResponse` via [`to_pole`](@ref), then take its
adjoint. The direct Lanczos-space adjoint is not implemented because `R`
has rectangular shape `(B_active × B_raw)` that is incompatible with the
continued-fraction recurrence.

Supports both `sign = +1` (retarded/neutral) and `sign = -1` (removal)
via the unified pole-conversion formula.
"""
function Base.adjoint(L::LanczosResponse)
    return adjoint(to_pole(L))
end

# --- adjoint(GridResponse) -----------------------------------------------

"""
    adjoint(G::GridResponse) -> GridResponse

Take the element-wise matrix adjoint along the (rows, cols) axes at each
ω-point. The ω-grid, `Eg`, and `Γ` are unchanged.
"""
function Base.adjoint(G::GridResponse{T, N}) where {T, N}
    # Matrix adjoint along the (rows, cols) axes (dims 1 and 2) at each
    # ω-point (and any additional axes). Convention: swap dims 1 ↔ 2 and
    # conjugate, leaving all other axes (dim 3, 4, …, N) in place.
    # Works for any N ≥ 3 (N = 3: standard (rows, cols, nω);
    # N = 4: RIXS (rows, cols, ω_in, ω_out); etc.)
    @assert N >= 3 "GridResponse must have at least 3 dimensions (rows, cols, ω)"
    perm = (2, 1, 3:N...)
    new_data = permutedims(conj.(G.data), perm)
    return GridResponse{T, N}(new_data, G.ω, G.Eg, G.Γ)
end

# --- adjoint(GreensFunction) ---------------------------------------------

"""
    adjoint(GF::GreensFunction) -> GreensFunction

Take the per-channel adjoint.
"""
function Base.adjoint(GF::GreensFunction{T, R}) where {T, R}
    new_addition = isnothing(GF.addition) ? nothing : adjoint(GF.addition)
    new_removal  = isnothing(GF.removal)  ? nothing : adjoint(GF.removal)
    R2 = if !isnothing(new_addition)
        typeof(new_addition)
    else
        typeof(new_removal)
    end
    T2 = eltype(R2)
    return GreensFunction{T2, R2}(
        isnothing(new_addition) ? nothing : new_addition,
        isnothing(new_removal)  ? nothing : new_removal
    )
end

# =====================================================================
# Addition per concrete type
# =====================================================================

# --- PoleResponse + PoleResponse (closed) --------------------------------

"""
    P1::PoleResponse + P2::PoleResponse -> PoleResponse

Add two pole representations. Requires `P1.Eg == P2.Eg` and
`P1.Γ == P2.Γ` (raises `ArgumentError` on mismatch). Poles and residues
are concatenated; residues at coincident poles (within `tol = 1e-12`)
are merged by summation. `a0` fields are summed.

The result uses `promote_type(eltype(P1), eltype(P2))`.
"""
function Base.:+(P1::PoleResponse{T1}, P2::PoleResponse{T2}) where {T1, T2}
    if P1.Eg != P2.Eg || P1.Γ != P2.Γ
        throw(ArgumentError(
            "+: Eg/Γ mismatch — convert one operand to a common frame first " *
            "(P1: Eg=$(P1.Eg), Γ=$(P1.Γ); P2: Eg=$(P2.Eg), Γ=$(P2.Γ))"
        ))
    end

    S = promote_type(T1, T2)
    tol = 1e-12

    # merge coincident poles
    all_poles  = vcat(P1.poles, P2.poles)
    all_res    = vcat(
        Vector{Matrix{S}}([Matrix{S}(r) for r in P1.residues]),
        Vector{Matrix{S}}([Matrix{S}(r) for r in P2.residues])
    )

    merged_poles   = Float64[]
    merged_res     = Matrix{S}[]

    used = fill(false, length(all_poles))
    for i in 1:length(all_poles)
        used[i] && continue
        p   = all_poles[i]
        res = copy(all_res[i])
        used[i] = true
        for j in (i+1):length(all_poles)
            used[j] && continue
            if abs(all_poles[j] - p) < tol
                res .+= all_res[j]
                used[j] = true
            end
        end
        push!(merged_poles, p)
        push!(merged_res, res)
    end

    new_a0 = Matrix{S}(P1.a0) + Matrix{S}(P2.a0)
    return PoleResponse{S}(new_a0, merged_poles, merged_res, P1.Eg, P1.Γ)
end

# --- LanczosResponse + AbstractResponse (lazy-promote via to_pole) -------

"""
    L::LanczosResponse + R::AbstractResponse -> PoleResponse

Convert `L` to `PoleResponse` via [`to_pole`](@ref), then add. Works
for both `L.sign == +1` (addition / neutral) and `L.sign == -1`
(removal) via the unified `to_pole` formula.
"""
function Base.:+(L::LanczosResponse, R::AbstractResponse)
    return to_pole(L) + R
end

"""
    R::AbstractResponse + L::LanczosResponse -> PoleResponse

Convert `L` to `PoleResponse` via [`to_pole`](@ref), then add.
"""
function Base.:+(R::AbstractResponse, L::LanczosResponse)
    return R + to_pole(L)
end

# --- GridResponse + GridResponse -----------------------------------------

"""
    G1::GridResponse + G2::GridResponse -> GridResponse

Pointwise addition of two gridded responses. Requires `G1.ω == G2.ω`,
`G1.Eg == G2.Eg`, and `G1.Γ == G2.Γ` (raises `ArgumentError` on any
mismatch).
"""
function Base.:+(G1::GridResponse{T1, N}, G2::GridResponse{T2, N}) where {T1, T2, N}
    if G1.ω != G2.ω || G1.Eg != G2.Eg || G1.Γ != G2.Γ
        throw(ArgumentError(
            "+: ω/Eg/Γ mismatch between GridResponse operands — " *
            "ensure same grid, energy reference, and broadening before adding"
        ))
    end
    S = promote_type(T1, T2)
    new_data = Array{S, N}(G1.data) .+ Array{S, N}(G2.data)
    return GridResponse{S, N}(new_data, G1.ω, G1.Eg, G1.Γ)
end

# --- GreensFunction + GreensFunction -------------------------------------

"""
    GF1::GreensFunction + GF2::GreensFunction -> GreensFunction

Channel-wise sum. For channels populated in both, the inner responses
are added. For channels populated in only one, that channel is kept.
Eg/Γ compatibility is inherited from the inner representations.

When the addition and removal channels end up with different concrete
representation types (e.g., addition is `PoleResponse` after lazy-promotion
but removal is still `LanczosResponse`), both channels are promoted to a
common representation before constructing the result `GreensFunction`. The
promotion order is: if any channel is `PoleResponse`, all channels are
converted to `PoleResponse` via [`to_pole`](@ref).
"""
function Base.:+(GF1::GreensFunction, GF2::GreensFunction)
    new_addition = _channel_add(GF1.addition, GF2.addition)
    new_removal  = _channel_add(GF1.removal,  GF2.removal)

    # Promote both channels to a common concrete representation type so that
    # GreensFunction{T, R} can hold them in a single parametric R slot.
    add_p, rem_p = _gf_promote_channels(new_addition, new_removal)

    R2 = if !isnothing(add_p)
        typeof(add_p)
    else
        typeof(rem_p)
    end
    T2 = eltype(R2)
    return GreensFunction{T2, R2}(add_p, rem_p)
end

# ---------------------------------------------------------------------------
# Internal helpers for GreensFunction addition
# ---------------------------------------------------------------------------

# Add two optional GreensFunction channels.
#
# Slots that are `nothing` represent absent channels (e.g. an addition-only
# `GreensFunction` has `removal === nothing`); they act as the additive
# identity. The `Nothing + Nothing = nothing` method is required to
# disambiguate dispatch — without it, `_channel_add(nothing, nothing)` hits
# the two single-Nothing methods equally and Julia raises a MethodError.
_channel_add(::Nothing, ::Nothing) = nothing
_channel_add(::Nothing, x)         = x
_channel_add(x, ::Nothing)         = x
_channel_add(x, y)                 = x + y

# Explicit error for the Grid + Pole / Pole + Grid mixed-rep cases that
# would otherwise leak a raw MethodError out of dispatch. Grid → Pole
# is deliberately deferred to v0.5+ (rational-approximation fitting),
# so there is no faithful in-Responses promotion.
function _channel_add(::GridResponse, ::PoleResponse)
    throw(ArgumentError(
        "GreensFunction +: cannot add GridResponse + PoleResponse channel — " *
        "Grid → Pole conversion (rational-approx fitting) deferred to v0.5+; " *
        "convert one operand to a common representation explicitly"
    ))
end
_channel_add(p::PoleResponse, g::GridResponse) = _channel_add(g, p)

# Promote a single channel to PoleResponse (no-op if already Pole, error if Grid).
function _to_pole_channel(ch::LanczosResponse)
    return to_pole(ch)
end
function _to_pole_channel(ch::PoleResponse)
    return ch
end
function _to_pole_channel(ch::GridResponse)
    throw(ArgumentError(
        "GreensFunction +: cannot promote a GridResponse channel to PoleResponse " *
        "without a frequency grid — convert to a common representation manually"
    ))
end

# Given two optional channels (after per-channel addition), ensure they share
# the same concrete type. Returns (add_promoted, rem_promoted).
function _gf_promote_channels(add, rem)
    # If either channel is missing, no conflict is possible.
    isnothing(add) && return (add, rem)
    isnothing(rem) && return (add, rem)
    # Both populated: check if they're already the same base type.
    typeof(add) == typeof(rem) && return (add, rem)
    # Different types: promote both to PoleResponse (the universal target for
    # Lanczos/Pole mixing). GridResponse promotion is not supported.
    add_p = _to_pole_channel(add)
    rem_p = _to_pole_channel(rem)
    return (add_p, rem_p)
end
