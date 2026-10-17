function _reject_matrixfree_wrapper(A)
    P = parent(A)
    P === A && return nothing
    P isa MatrixFreeOperator && throw(ArgumentError(
        "pass matrix-free operators directly; array wrappers around them are not supported"))
    return _reject_matrixfree_wrapper(P)
end

function _check_affine_backend(terms)
    foreach(_reject_matrixfree_wrapper, terms)
    if any(A -> A isa MatrixFreeOperator, terms)
        all(A -> A isa MatrixFreeOperator, terms) || throw(ArgumentError(
            "use a single storage backend for all matrix-free model terms"))
        b = first(terms).ket
        foreach(A -> _validate_action_basis(A, b), terms)
    end
    return nothing
end

function _groundstate_eigenpairs(H::MatrixFreeOperator, dense_max, gap_tol)
    return eigen(H, H.ket; n=2, tol=min(1e-10, gap_tol))
end

function _require_explicit_spectrum(H::MatrixFreeOperator)
    throw(ArgumentError(
        "full-spectrum analysis requires explicit matrices; convert matrix-free terms " *
        "with sparse or Matrix before constructing the model"))
end
