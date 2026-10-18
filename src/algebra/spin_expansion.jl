const SXSY_EXPANSION_WARN  = 256
const SXSY_EXPANSION_ERROR = 65_536

# --- Sx/Sy expansion --------------------------------------------------------

# Returns a Vector{Tuple{Complex{Rational}, Chain}} of (coefficient_factor,
# expanded_chain) pairs where every spin entry in the chain is one of
# {:Sz, :Splus, :Sminus}. We do the expansion as a Cartesian product over
# the positions occupied by :Sx / :Sy.
function _expand_spin_xy(chain::Chain)
    # Reject exponential growth before allocating its Cartesian product.
    # Guard the shift separately so long chains cannot overflow Int.
    n_xy = count(e -> e.kind === :Sx || e.kind === :Sy, chain)
    n_xy >= 8 * sizeof(Int) - 1 && throw(ArgumentError(
        "Sx/Sy operator expansion factor 2^$n_xy exceeds " *
        "limit $SXSY_EXPANSION_ERROR. Rewrite the chain in S+/S- form."))
    _check_spin_expansion(1 << n_xy)

    # Find every :Sx/:Sy position and its (factor, replacement_kind) options.
    positions = Tuple{Int, Vector{Tuple{ComplexF64, Symbol}}}[]
    for (i, e) in enumerate(chain)
        if e.kind === :Sx
            push!(positions, (i, [(0.5 + 0im, :Splus), (0.5 + 0im, :Sminus)]))
        elseif e.kind === :Sy
            # Sy = (-i/2)(S+ - S-) = (-i/2)·S+ + (i/2)·S-
            push!(positions, (i, [(-0.5im, :Splus), (0.5im, :Sminus)]))
        end
    end
    isempty(positions) && return [(1.0 + 0im, chain)]

    # Cartesian product of the choice vectors, each producing one (factor, chain).
    out = Tuple{ComplexF64, Chain}[]
    pos_idx = ones(Int, length(positions))
    pos_card = [length(p[2]) for p in positions]
    while true
        factor = 1.0 + 0im
        new_chain = collect(LadderEntry, chain)
        for (j, (chain_pos, options)) in enumerate(positions)
            f, kind = options[pos_idx[j]]
            factor *= f
            old = chain[chain_pos]
            new_chain[chain_pos] = LadderEntry(kind, old.site, old.label)
        end
        push!(out, (factor, Tuple(new_chain)))
        # Increment pos_idx (mixed-radix counter).
        carry = true
        for j in eachindex(pos_idx)
            pos_idx[j] += 1
            if pos_idx[j] > pos_card[j]
                pos_idx[j] = 1
            else
                carry = false
                break
            end
        end
        carry && break
    end
    return out
end

function _check_spin_expansion(factor::Int)
    if factor > SXSY_EXPANSION_ERROR
        throw(ArgumentError("Sx/Sy operator expansion factor $factor exceeds " *
                            "limit $SXSY_EXPANSION_ERROR. Rewrite the chain in S+/S- form."))
    elseif factor > SXSY_EXPANSION_WARN
        @warn "Sx/Sy operator expansion factor $factor exceeds soft limit"
    end
    return nothing
end

function _output_eltype(::Type{T}, H::OperatorSum) where {T}
    has_sy = any(any(e.kind === :Sy for e in term.chain) for term in H)
    return has_sy ? complex(float(T)) : T
end
