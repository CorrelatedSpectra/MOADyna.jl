function _spectrum_basis(states::Eigenstates, restrictions)
    b = basis(states.basis.hilbert, _restriction_list(restrictions)...)
    length(b) > 0 || throw(ArgumentError("the spectroscopy sector is empty"))
    return b
end

function _spectrum_transitions(T, bra, ket; backend, threaded)
    operators = _to_T_list(T)
    all(t -> t isa OperatorSum, operators) || throw(ArgumentError(
        "the Eigenstates workflow requires symbolic transitions; " *
        "use the explicit-basis interface for prebuilt actions"))
    actions = map(operators) do t
        action = matrixfree(t, bra, ket; threaded)
        backend === :sparse ? sparse(action) : action
    end
    return _is_T_vector(T) ? actions : only(actions)
end

"""
    xas(H, T, states::Eigenstates; restrictions, backend=states.backend, kwargs...)

Compute absorption from basis-carrying eigenstates and symbolic operators.
`restrictions` defines the complete propagation sector, which can differ from
the initial-state sector. Construct its basis, Hamiltonian action and rectangular
transition actions automatically. `backend` selects `:sparse` or `:matrixfree`.

The propagation basis is fully enumerated, even if `states` came from an adaptive
ground-state solve; its selected basis is not reused as a response approximation.
Other keywords, including broadening, grids and `temperature`, follow `xas`.
For repeated calculations with the same actions, use the explicit-basis interface.
"""
function xas(H::OperatorSum, T, states::Eigenstates;
             restrictions, backend::Symbol=states.backend, threaded=:auto, kwargs...)
    b = _spectrum_basis(states, restrictions)
    action = _build_action(H, b; backend, threaded)
    transitions = _spectrum_transitions(T, b, states.basis; backend, threaded)
    return xas(action, b, transitions, states.eigen; threaded, kwargs...)
end

"""
    rixs(H_f, H_i, T_in, T_out, states::Eigenstates;
         restrictions_final, restrictions_intermediate,
         backend=states.backend, kwargs...)

Compute RIXS from symbolic Hamiltonians/transitions and basis-carrying initial
states. Construct the complete intermediate and final sectors from their explicit
restrictions. `T_in` maps initial to intermediate states; `T_out` is written in
absorption orientation, mapping final to intermediate states. Rectangular actions
and their basis coordinates are managed internally.

These response sectors are fully enumerated independently of any adaptive
ground-state selection. Other keywords, including thermal ensembles, follow
`rixs`. Use explicit actions to reuse preparation across calls.
"""
function rixs(H_f::OperatorSum, H_i::OperatorSum, T_in, T_out, states::Eigenstates;
              restrictions_final, restrictions_intermediate,
              backend::Symbol=states.backend, threaded=:auto, kwargs...)
    final = _spectrum_basis(states, restrictions_final)
    intermediate = _spectrum_basis(states, restrictions_intermediate)
    Af = _build_action(H_f, final; backend, threaded)
    Ai = _build_action(H_i, intermediate; backend, threaded)
    incoming = _spectrum_transitions(T_in, intermediate, states.basis; backend, threaded)
    outgoing = _spectrum_transitions(T_out, intermediate, final; backend, threaded)
    return rixs(Af, Ai, final, incoming, outgoing, states.eigen;
                intermediate_basis=intermediate, threaded, kwargs...)
end

"""
    fluorescence_yield(H_i, T_excite, T_decay, states::Eigenstates;
                       restrictions_intermediate, restrictions_final,
                       backend=states.backend, kwargs...)

Compute fluorescence yield with automatically constructed, complete intermediate
and final sectors. `T_decay` uses absorption orientation (final to intermediate);
the final sector includes all allowed decay destinations, not just selected
ground-state determinants. Other keywords follow `fluorescence_yield`.
"""
function fluorescence_yield(H_i::OperatorSum, T_excite, T_decay, states::Eigenstates;
                            restrictions_intermediate, restrictions_final,
                            backend::Symbol=states.backend, threaded=:auto, kwargs...)
    intermediate = _spectrum_basis(states, restrictions_intermediate)
    final = _spectrum_basis(states, restrictions_final)
    action = _build_action(H_i, intermediate; backend, threaded)
    excitation = _spectrum_transitions(T_excite, intermediate, states.basis; backend, threaded)
    decay = _spectrum_transitions(T_decay, intermediate, final; backend, threaded)
    return fluorescence_yield(action, intermediate, excitation, decay, states.eigen;
                              threaded, kwargs...)
end
