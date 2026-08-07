"""
    MOADyna.Diagnostics

Post-eigenstate analysis utilities. Public API:

- `expectation_table(sys, basis, ops_list)` — pretty-printed table of
  `⟨ψ_i | O_j | ψ_i⟩` per (eigenstate, operator) pair.
- `configuration_weights(psi, basis, m; group_by)` — bucket basis states
  by per-shell fermion occupation and sum `|ψ_i|²` per bucket.
"""
module Diagnostics

using ..Algebra: OperatorSum
using ..Bases: AbstractBasis, compile, assemble, encoding, ModeEntry, get_bit
using ..Shells: ShellModel
using LinearAlgebra: dot

include("expectation_table.jl")
include("configuration_weights.jl")
include("classify_state_fock.jl")

export expectation_table, configuration_weights

end # module
