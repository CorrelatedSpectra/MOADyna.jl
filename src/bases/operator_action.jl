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
