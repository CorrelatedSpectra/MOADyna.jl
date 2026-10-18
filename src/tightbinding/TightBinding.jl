"""
    MOADyna.TightBinding

Real-space hopping models, Bloch Hamiltonians and finite-cluster construction.
The common workflow is re-exported by `MOADyna`.
"""
module TightBinding

using LinearAlgebra

include("model.jl")

export TightBindingModel, hopping, bloch_hamiltonian, bloch_hamiltonian!

end # module
