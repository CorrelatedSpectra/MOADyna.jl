function _validate_threading(threaded)
    (threaded === :auto || threaded === true || threaded === false) ||
        throw(ArgumentError("threaded must be :auto, true, or false; got $(repr(threaded))"))
    return threaded
end

function _prepare_action(H; nrhs::Int=1, threaded=:auto)
    _validate_threading(threaded)
    return H
end

function _build_action(A::OperatorSum, basis::EagerBasis;
                       backend::Symbol=:sparse, threaded=:auto)
    _validate_threading(threaded)
    backend === :sparse && return assemble(compile(A, basis), basis)
    backend === :matrixfree && return matrixfree(A, basis; threaded)
    throw(ArgumentError("backend must be :sparse or :matrixfree; got $(repr(backend))"))
end

_action_backend(::Any) = :sparse

_transition_action(A::OperatorSum, basis, H; threaded=:auto) =
    _build_action(A, basis; backend=_action_backend(H), threaded)
_transition_action(A::AbstractMatrix, basis, H; threaded=:auto) =
    A

function _validate_action_basis(H, basis)
    size(H, 1) == size(H, 2) == length(basis) || throw(DimensionMismatch(
        "Hamiltonian dimensions $(size(H)) do not match basis size $(length(basis))"))
    _validate_action_coordinates(H, basis)
    return nothing
end

_validate_action_coordinates(H, basis) = nothing

@inline function _foreach_transition(f, plan, basis, i, scratch)
    offset = state_offset(basis, i)
    for term in plan.terms
        _term_precheck_fails(plan, term, basis.states, offset) && continue
        copy_state!(scratch, basis.states, offset, basis.nwords)
        phase = _apply_chain_rtl!(scratch, plan, term)
        iszero(phase) && continue
        f(scratch, term.coef * phase)
    end
    return nothing
end

@inline function _foreach_projected_element(f, plan::CompiledHamiltonian{T}, basis, i,
                                            index, scratch) where {T}
    offset = state_offset(basis, i)
    for group in plan.term_groups
        coefficient::T = zero(T)
        j = 0
        for k in group
            term = plan.terms[k]
            _term_precheck_fails(plan, term, basis.states, offset) && continue
            copy_state!(scratch, basis.states, offset, basis.nwords)
            phase = _apply_chain_rtl!(scratch, plan, term)
            iszero(phase) && continue
            if j == 0
                j = _state_index(index, scratch)
                j == 0 && break
            end
            coefficient += term.coef * phase
        end
        iszero(coefficient) || f(j, coefficient)
    end
    return nothing
end

function _projected_column!(column, plan, basis, i, index, scratch)
    empty!(column)
    _foreach_projected_element(plan, basis, i, index, scratch) do j, value
        column[j] = value
    end
    return column
end
