_restriction_list(::Nothing) = Restriction[]
_restriction_list(r::Restriction) = Restriction[r]
_restriction_list(rs::AbstractVector{<:Restriction}) = collect(Restriction, rs)

function _selected_basis(template::EagerBasis, states::AbstractVector{UInt64}; restrictions)
    nw = template.nwords
    length(states) % nw == 0 || throw(DimensionMismatch("incomplete packed state"))
    rs = _restriction_list(restrictions)
    compiled = [compile_restriction(r, template.encoding) for r in rs]
    masks = zeros(UInt64, nw)
    entries = collect(values(template.encoding.entries))
    for e in entries
        masks[e.word_idx] |= ((UInt64(1) << e.nbits) - 1) << e.bit_offset
    end
    packed = copy(states)
    count = length(packed) ÷ nw
    for i in 1:count
        offset = (i - 1) * nw + 1
        all(w -> iszero(packed[offset + w - 1] & ~masks[w]), 1:nw) &&
            all(e -> get_span(packed, offset, e) <= e.max_val, entries) ||
            throw(ArgumentError("selected state $i has invalid occupations or padding bits"))
        all(r -> _check(r, packed, offset), compiled) ||
            throw(ArgumentError("selected state $i violates the allowed-space restrictions"))
    end
    packed = _sort_flat_states!(packed, nw, count)
    kept = 0
    for i in 1:count
        src = (i - 1) * nw + 1
        if kept == 0 || any(w -> packed[src + w - 1] != packed[(kept - 1) * nw + w], 1:nw)
            kept += 1
            copyto!(packed, (kept - 1) * nw + 1, packed, src, nw)
        end
    end
    resize!(packed, kept * nw)
    return EagerBasis(template.hilbert, rs, template.encoding, nw, packed, _next_basis_id())
end

function _rebind_action(A::MatrixFreeOperator, basis::EagerBasis)
    _same_encoding(A.ket.encoding, basis.encoding) ||
        throw(ArgumentError("cannot reuse compiled terms with a different encoding"))
    index = _StateIndex(basis)
    rebind(plan) = CompiledHamiltonian(plan.terms, plan.entries, plan.fermion_mask,
        plan.masks_flat, plan.boson_reqs, basis_id(basis), plan.nwords)
    return MatrixFreeOperator(rebind(A.forward), rebind(A.reverse), basis, basis,
                               index, index, A.threaded)
end

function _embed_block(X::AbstractMatrix, from::EagerBasis, to::EagerBasis)
    _same_encoding(from.encoding, to.encoding) ||
        throw(ArgumentError("embedding requires identical encodings"))
    size(X, 1) == length(from) || throw(DimensionMismatch("block and source basis disagree"))
    Y = zeros(eltype(X), length(to), size(X, 2))
    index = _StateIndex(to)
    for i in 1:length(from)
        j = _state_index(index, get_state(from, i))
        j > 0 || throw(ArgumentError("destination basis omits a source state"))
        Y[j, :] = X[i, :]
    end
    return Y
end
