# =====================================================================
# Builder — convert ParsedOperator + mode_map → MOADyna.Algebra OperatorSum
# =====================================================================
#
# The `mode_map` argument tells the builder how Quanty's integer mode
# index translates to a MOADyna `(site_name::Symbol, label_tuple)`. The
# user supplies this — it encodes their own convention for what each
# Quanty mode represents.
#
# Two forms accepted:
#   - A function `i::Int -> (site_name::Symbol, label)` returning the
#     address (label is whatever the site's normalize_label wants — Int,
#     tuple, etc.)
#   - A `Dict{Int, Tuple{Symbol, Any}}` mapping each integer to the
#     same address tuple.

"""
    build_operator(parsed::ParsedOperator, hilbert::Hilbert, mode_map) -> OperatorSum

Construct a MOADyna `OperatorSum` from a parsed Quanty dump. The chain
order in the parsed dump is preserved literally — the resulting product
goes through MOADyna.Algebra's Tier-2 canonicalization, so the output is
the canonical MOADyna representation regardless of Quanty's storage order.

`mode_map` is either a function `Int -> (Symbol, label)` or a `Dict{Int,
Tuple{Symbol, ...}}` defining which (site, label) each Quanty mode
maps to.

Returns an empty sum if `parsed.terms` is empty.
"""
function build_operator(parsed::ParsedOperator, hilbert::Hilbert, mode_map)
    # Detect whether any term carries a non-zero imaginary part. If all
    # imag-parts are exactly zero (the common case — a real Hamiltonian),
    # we emit an `OperatorSum{Float64}` to keep storage tight. Mixed real-
    # and-complex dumps fall through to `OperatorSum{ComplexF64}`.
    has_imag = any(!iszero(imag(t.coefficient)) for t in parsed.terms)
    T = has_imag ? ComplexF64 : Float64
    out = zero(OperatorSum{T})
    isempty(parsed.terms) && return out

    for term in parsed.terms
        out = out + _build_term(term, hilbert, mode_map, T)
    end
    return out
end

function _build_term(term::ParsedTerm, hilbert::Hilbert, mode_map, ::Type{T}) where {T}
    # Build the chain by multiplying single-entry operators in order.
    # Each lookup goes through MOADyna's canonicalization on the way in
    # via the `*` operator.
    coef = T == Float64 ? real(term.coefficient) : term.coefficient
    chain_op = nothing  # OperatorSum{T} or nothing

    for (kind, qidx) in zip(term.kinds, term.indices)
        site_name, label = _lookup_mode(mode_map, qidx)
        haskey(hilbert, site_name) || throw(ArgumentError(
            "mode_map produced site name :$site_name not in hilbert"))
        site = hilbert[site_name]

        single = if kind === :cdag
            cdag(site, _splat(label)...)
        elseif kind === :c
            c(site, _splat(label)...)
        else
            throw(ArgumentError("unsupported parsed kind: $kind"))
        end

        chain_op = chain_op === nothing ? single : chain_op * single
    end

    if chain_op === nothing
        # Empty chain → length-0 operator (constant). Quanty emits these
        # as `|  <coef>`; we represent them as `coef · 1` in the
        # OperatorSum.
        return coef * one(OperatorSum{T})
    end
    return coef * chain_op
end

# Lookup helpers — accept either Function or Dict for the mode_map
@inline _lookup_mode(f::Function, i::Int) = f(i)
@inline _lookup_mode(d::AbstractDict, i::Int) = d[i]

# Splat helper — accept label as Int, Tuple, or NamedTuple uniformly
@inline _splat(label::Int) = (label,)
@inline _splat(label::Tuple) = label
@inline _splat(label) = (label,)
