_to_T_list(T::Union{OperatorSum,AbstractMatrix}) = [T]
function _to_T_list(Ts::AbstractVector)
    all(T -> T isa Union{OperatorSum,AbstractMatrix}, Ts) || throw(ArgumentError(
        "transitions must be symbolic operators or matrix actions"))
    return collect(Ts)
end
_is_T_vector(::Union{OperatorSum,AbstractMatrix}) = false
_is_T_vector(::AbstractVector) = true

function _transition_actions(Ts, basis, H;
                             source_basis=nothing, source_length=nothing,
                             threaded=:auto)
    if source_basis !== nothing
        source_length = length(source_basis)
        if basis_id(source_basis) != basis_id(basis) && any(T -> T isa OperatorSum, Ts)
            throw(ArgumentError(
                "transitions between different bases must be supplied as prebuilt matrix actions"))
        end
    end
    actions = [_transition_action(T, basis, H; threaded) for T in Ts]
    source_id = source_basis === nothing ? nothing : basis_id(source_basis)
    any(T -> T isa OperatorSum, Ts) && (source_id = basis_id(basis))
    for T in actions
        size(T, 1) == length(basis) || throw(DimensionMismatch(
            "transition has $(size(T, 1)) destination states; basis has $(length(basis))"))
        if source_length !== nothing
            size(T, 2) == source_length || throw(DimensionMismatch(
                "transition has $(size(T, 2)) source states; expected $source_length"))
        end
        if T isa MatrixFreeOperator
            basis_id(T.bra) == basis_id(basis) || throw(ArgumentError(
                "transition destination basis does not match the propagation basis"))
            if source_length !== nothing
                source_id === nothing && (source_id = basis_id(T.ket))
                basis_id(T.ket) == source_id || throw(ArgumentError(
                    "transitions must agree on the source basis"))
            end
        end
    end
    return actions
end

function _source_dimension(states)
    n = length(first(states))
    all(ψ -> length(ψ) == n, states) || throw(DimensionMismatch(
        "initial states must have the same dimension"))
    return n
end
