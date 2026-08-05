# =====================================================================
# CompiledRestriction — primitive form of a layer-1 Restriction
# =====================================================================
#
# Per §4 of docs/architecture/02_hilbert.md, a restriction is lowered to one
# of three operational shapes:
#
#   POPCOUNT      bitwise mask + popcount: fermion particle counts on a set
#                 of modes; min ≤ Σ bᵢ ≤ max.
#   BYTESUM       per-mode read of multi-bit spans, summed: boson particle
#                 counts.
#   WEIGHTED_SUM  per-mode read times scaled-integer weight, summed:
#                 weighted fermion counts (Sz from up/down splits, sublattice
#                 imbalance, etc.). All min/max/weights stored in scaled
#                 integers so sector boundaries are exact.

@enum RestrictionOp POPCOUNT BYTESUM WEIGHTED_SUM

"""
    CompiledRestriction

Per §4.1 of the layer-2 chapter. Carries a `mask` over `nwords` UInt64 words
(bits set on the modes counted), integer `min`/`max` bounds, a per-mode
`weights` vector (in scaled-integer form), and a global `scale` such that
the user-visible bound is `min/scale ≤ value ≤ max/scale`.
"""
struct CompiledRestriction
    mask::Vector{UInt64}            # length nwords; for POPCOUNT/BYTESUM/WEIGHTED_SUM
    min::Int                        # scaled-integer lower bound
    max::Int                        # scaled-integer upper bound
    scale::Int                      # ≥ 1
    op::RestrictionOp
    # Per-mode info, indexed in the same order as `mode_entries`. Used by
    # BYTESUM and WEIGHTED_SUM only; for POPCOUNT it can be empty.
    weights::Vector{Int}            # scaled-integer weight for each mode entry
    mode_entries::Vector{ModeEntry} # encoding info for each mode in this restriction
end

# --- Compile a layer-1 Restriction → CompiledRestriction --------------------

function compile_restriction(r::Restriction, enc::EncodingMap)::CompiledRestriction
    return _compile_quantity(r.quantity, r.bounds, enc)
end

# --- ParticleCount{Fermionic} → POPCOUNT -----------------------------------

function _compile_quantity(q::ParticleCount{Fermionic},
                           bounds::AbstractRange,
                           enc::EncodingMap)
    mode_entries = ModeEntry[]
    for s in q.sites
        s isa FermionSite ||
            throw(ArgumentError("ParticleCount{Fermionic} on non-fermion site :$(name(s))"))
        for k in 1:_n_modes_fermion(s)
            push!(mode_entries, mode_entry(enc, name(s), (k,)))
        end
    end
    mask = _build_mask(mode_entries, enc.nwords)
    lo, hi = _bounds_int(bounds)
    weights = ones(Int, length(mode_entries))    # unit weights
    return CompiledRestriction(mask, lo, hi, 1, POPCOUNT, weights, mode_entries)
end

# --- ParticleCount{Bosonic} → BYTESUM ---------------------------------------

function _compile_quantity(q::ParticleCount{Bosonic},
                           bounds::AbstractRange,
                           enc::EncodingMap)
    mode_entries = ModeEntry[]
    for s in q.sites
        s isa BosonSite ||
            throw(ArgumentError("ParticleCount{Bosonic} expected BosonSite; got $(typeof(s))"))
        push!(mode_entries, mode_entry(enc, name(s), ()))
    end
    mask = _build_mask(mode_entries, enc.nwords)
    lo, hi = _bounds_int(bounds)
    weights = ones(Int, length(mode_entries))
    return CompiledRestriction(mask, lo, hi, 1, BYTESUM, weights, mode_entries)
end

# --- TotalSz on SpinSites → BYTESUM (with scaled integer for half-integer S)

function _compile_quantity(q::TotalSz,
                           bounds::AbstractRange,
                           enc::EncodingMap)
    mode_entries = ModeEntry[]
    site_S = Rational{Int}[]
    for s in q.sites
        s isa SpinSite ||
            throw(ArgumentError("TotalSz: only SpinSite is supported at this layer; got $(typeof(s))"))
        push!(mode_entries, mode_entry(enc, name(s), ()))
        push!(site_S, _spin_S(s))
    end
    # Encoded value is mz - (-S) ∈ 0:2S. To get Σ mzᵢ in scaled-integer form,
    # use `2 * Σ encoded_i + Σ (-2 S_i)` and a global scale of 2 (covers both
    # integer-S and half-integer-S sites). The per-mode scaled weight is 2;
    # the per-mode constant offset (-2 S_i) is rolled into min/max bounds.
    scale = 2
    weights = fill(scale, length(mode_entries))      # scaled weight on encoded value
    offset_sum = -sum(2 * S for S in site_S; init = 0 // 1)
    @assert isinteger(offset_sum) "TotalSz scale=2 should produce integer offsets"
    offset = Int(offset_sum)
    lo_user, hi_user = _bounds_rational(bounds)
    lo = Int(scale * lo_user) - offset
    hi = Int(scale * hi_user) - offset
    mask = _build_mask(mode_entries, enc.nwords)
    return CompiledRestriction(mask, lo, hi, scale, BYTESUM, weights, mode_entries)
end

# --- WeightedParticleCount → WEIGHTED_SUM ----------------------------------

function _compile_quantity(q::WeightedParticleCount,
                           bounds::AbstractRange,
                           enc::EncodingMap)
    # Collect mode entries in the same order as q.weights.
    mode_entries = ModeEntry[]
    for s in q.sites
        for k in 1:_n_modes_fermion(s)
            push!(mode_entries, mode_entry(enc, name(s), (k,)))
        end
    end
    length(mode_entries) == length(q.weights) ||
        throw(ArgumentError("WeightedParticleCount: mode count $(length(mode_entries)) " *
                            "≠ weight count $(length(q.weights))"))

    # Lowering: scale = LCM of denominators (1 if all Int).
    rats = q.weights isa AbstractVector{Int} ?
                Rational{Int}.(q.weights) : Rational{Int}.(q.weights)
    scale = lcm([denominator(r) for r in rats]...)
    weights_int = Int[Int(numerator(r) * (scale ÷ denominator(r))) for r in rats]

    lo_user, hi_user = _bounds_rational(bounds)
    # Bounds are integer multiples of (1/scale) once scale is the LCM; verify.
    lo_scaled = scale * lo_user
    hi_scaled = scale * hi_user
    isinteger(lo_scaled) && isinteger(hi_scaled) ||
        throw(ArgumentError("WeightedParticleCount: bound $bounds is not " *
                            "representable in scale=$scale (LCM of weight denominators). " *
                            "Use a bound that is a multiple of 1//$scale."))

    mask = _build_mask(mode_entries, enc.nwords)
    return CompiledRestriction(mask, Int(lo_scaled), Int(hi_scaled), scale,
                               WEIGHTED_SUM, weights_int, mode_entries)
end

# --- Bound coercion --------------------------------------------------------

function _bounds_int(bounds::AbstractRange)
    lo, hi = first(bounds), last(bounds)
    (isinteger(lo) && isinteger(hi)) ||
        throw(ArgumentError("Restriction bound $bounds must be integer-valued for this quantity"))
    return Int(lo), Int(hi)
end

function _bounds_rational(bounds::AbstractRange)
    lo, hi = first(bounds), last(bounds)
    return Rational{Int}(lo), Rational{Int}(hi)
end

# --- Helpers ---------------------------------------------------------------

function _build_mask(mode_entries::Vector{ModeEntry}, nwords::Int)
    mask = zeros(UInt64, nwords)
    for e in mode_entries
        nb_mask = (UInt64(1) << e.nbits) - UInt64(1)
        mask[e.word_idx] |= nb_mask << e.bit_offset
    end
    return mask
end

_n_modes_fermion(::FermionSite{N}) where {N} = N

_spin_S(::SpinSite{S}) where {S} = Rational{Int}(S)

# --- Hot-path check --------------------------------------------------------

@inline function _check(c::CompiledRestriction,
                        buf::AbstractVector{UInt64}, off::Int)::Bool
    if c.op === POPCOUNT
        return _check_popcount(c, buf, off)
    elseif c.op === BYTESUM
        return _check_bytesum(c, buf, off)
    else
        return _check_weighted(c, buf, off)
    end
end

@inline function _check_popcount(c::CompiledRestriction,
                                 buf::AbstractVector{UInt64}, off::Int)::Bool
    n = 0
    @inbounds for k in 1:length(c.mask)
        n += count_ones(buf[off + k - 1] & c.mask[k])
    end
    return c.min ≤ n ≤ c.max
end

@inline function _check_bytesum(c::CompiledRestriction,
                                buf::AbstractVector{UInt64}, off::Int)::Bool
    s = 0
    @inbounds for i in eachindex(c.mode_entries)
        e = c.mode_entries[i]
        v = get_span(buf, off, e)
        s += c.weights[i] * v
    end
    return c.min ≤ s ≤ c.max
end

@inline function _check_weighted(c::CompiledRestriction,
                                 buf::AbstractVector{UInt64}, off::Int)::Bool
    # Same kernel as BYTESUM; kept separate for clarity / future divergence
    # (e.g. early termination via bound-tightening in WEIGHTED_SUM is a
    # natural place to add later).
    return _check_bytesum(c, buf, off)
end

# Public projector API `apply_restriction!(v, R, basis)` lives in basis.jl,
# below the AbstractBasis / EagerBasis definitions it dispatches on.
