function _linear_combination(coefficients, actions)
    length(coefficients) == length(actions) > 0 ||
        throw(DimensionMismatch("a nonempty action list needs one coefficient per action"))
    if any(A -> A isa MatrixFreeOperator, actions)
        all(A -> A isa MatrixFreeOperator, actions) || throw(ArgumentError(
            "use a single storage backend when combining matrix-free operators"))
        return _matrixfree_combination(coefficients, actions)
    end
    result = coefficients[1] * actions[1]
    for i in 2:length(actions)
        result += coefficients[i] * actions[i]
    end
    return result
end

_plan_scalar_type(::CompiledHamiltonian{T}) where {T} = T

function _combine_plans(coefficients, plans, id)
    T = promote_type(map(typeof, coefficients)..., map(_plan_scalar_type, plans)...)
    terms = CompiledTerm{T}[]
    entries, masks, reqs = CompiledEntry[], UInt64[], Tuple{Int,Int,Int}[]
    for (coefficient, plan) in zip(coefficients, plans)
        iszero(coefficient) && continue
        eo, mo, ro = length(entries), length(masks), length(reqs)
        append!(entries, plan.entries)
        append!(masks, plan.masks_flat)
        append!(reqs, ((i + eo, minimum, maximum) for (i, minimum, maximum) in plan.boson_reqs))
        for t in plan.terms
            push!(terms, CompiledTerm{T}(coefficient * t.coef, t.kind,
                t.chain_offset + eo, t.chain_length,
                t.ann_mask_offset < 0 ? -1 : t.ann_mask_offset + mo,
                t.cre_forbid_mask_offset < 0 ? -1 : t.cre_forbid_mask_offset + mo,
                t.action_mask_offset < 0 ? -1 : t.action_mask_offset + mo,
                t.negative_phase,
                t.boson_req_start + ro, t.boson_req_count))
        end
    end
    first_plan = first(plans)
    return CompiledHamiltonian(terms, entries, first_plan.fermion_mask,
        masks, reqs, id, first_plan.nwords)
end

function _matrixfree_combination(coefficients, actions)
    A = first(actions)
    all(B -> basis_id(B.bra) == basis_id(A.bra) && basis_id(B.ket) == basis_id(A.ket), actions) ||
        throw(ArgumentError("combined matrix-free operators must have identical source and destination bases"))
    forward = _combine_plans(coefficients, map(B -> B.forward, actions), basis_id(A.ket))
    reverse = _combine_plans(conj.(coefficients), map(B -> B.reverse, actions), basis_id(A.bra))
    return MatrixFreeOperator(forward, reverse, A.bra, A.ket,
                               A.bra_index, A.ket_index, A.threaded)
end

Base.:*(c::Number, A::MatrixFreeOperator) = _linear_combination((c,), (A,))
Base.:*(A::MatrixFreeOperator, c::Number) = c * A
Base.:+(A::MatrixFreeOperator, B::MatrixFreeOperator) = _linear_combination((1, 1), (A, B))
Base.:-(A::MatrixFreeOperator, B::MatrixFreeOperator) = _linear_combination((1, -1), (A, B))
Base.:-(A::MatrixFreeOperator) = -1 * A
