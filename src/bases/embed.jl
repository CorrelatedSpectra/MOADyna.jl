# src/bases/embed.jl
#
# Embed a state vector from one basis into another. Pure basis-arithmetic;
# does not require the Shells layer. Used for cross-sector spectroscopy
# where the ground state lives in a smaller basis than the dipole-coupled
# intermediate state.

"""
    embed(psi, basis_a => basis_b) -> Vector

Embed state vector `psi` (defined in `basis_a`) into a typically larger
`basis_b` that contains all the states of `basis_a`. The result is a
vector of length `length(basis_b)` whose entry at each `basis_b` index
matches the corresponding `basis_a` amplitude (or zero if absent —
though by contract every `basis_a` state must be present in `basis_b`).

# Arguments

- `psi::AbstractVector` — state vector in `basis_a`.
- `basis_a => basis_b` — pair of `EagerBasis` objects with `basis_a`'s
  states forming a subset of `basis_b`'s states.

# Errors

- `DimensionMismatch` if `length(psi) != length(basis_a)`.
- `ArgumentError` if any state in `basis_a` is not present in `basis_b`.

# Example
```julia
basis_gs  = EagerBasis(m, n(:Ni_2p) == 6, n(:Ni_3d) + n(:L_3d) == 16)
basis_xas = EagerBasis(m, n(:Ni_2p) in 5:6, total(m) == 24)
psi_xas = embed(psi_gs, basis_gs => basis_xas)
```
"""
function embed(psi::AbstractVector, bases::Pair{<:EagerBasis, <:EagerBasis})
    basis_a, basis_b = bases.first, bases.second
    length(psi) == length(basis_a) || throw(DimensionMismatch(
        "embed: vector length $(length(psi)) does not match basis_a size $(length(basis_a))"))

    psi_b = zeros(eltype(psi), length(basis_b))
    for i in 1:length(basis_a)
        state = get_state(basis_a, i)
        j = get_index(basis_b, state)
        j > 0 || throw(ArgumentError(
            "embed: basis_a state $i (= $(state)) not present in basis_b"))
        psi_b[j] = psi[i]
    end
    return psi_b
end
