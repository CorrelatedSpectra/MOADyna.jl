"""
    MOADyna.Bases

Eager basis enumeration + compiled-Hamiltonian sparse-matrix assembly. Takes a
`Hilbert` and `Restriction`s from [`MOADyna.Algebra`](@ref), produces an
[`EagerBasis`](@ref) and an
`assemble(compile(H, basis), basis) → SparseMatrixCSC`.

This is an internal submodule of `MOADyna`; the names below are
re-exported from the umbrella package, so users typically write `using MOADyna`
instead of reaching for `MOADyna.Bases` directly.

See the **Hilbert spaces & bases** chapter of the manual for the user-facing
contract.
"""
module Bases

using SparseArrays
using LinearAlgebra
using OhMyThreads
using OhMyThreads: @tasks, tcollect

using ..Algebra
using ..Algebra: LadderEntry, OperatorTerm, OperatorSum, Chain
using ..Algebra: name, encoding_bits, statistics
using ..Algebra: AbstractSite, FermionSite, BosonSite, SpinSite
using ..Algebra: Hilbert, Restriction
using ..Algebra: QuantumNumber, ParticleCount, TotalSz, WeightedParticleCount
using ..Algebra: Fermionic, Bosonic

include("encoding.jl")
include("restriction.jl")
include("basis.jl")
include("compile.jl")
include("assemble.jl")
include("embed.jl")

# Basis
export AbstractBasis, EagerBasis, basis
export get_state, get_index, basis_id

# Restrictions (compiled) + public projector API
export apply_restriction!

# Hamiltonian compilation / assembly
export CompiledHamiltonian
export compile, assemble

# Cross-basis embedding
export embed

end # module
