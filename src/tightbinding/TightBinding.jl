"""
    MOADyna.TightBinding

Real-space hopping models, Bloch Hamiltonians and finite-cluster construction.
The common workflow is re-exported by `MOADyna`.
"""
module TightBinding

using LinearAlgebra

include("model.jl")
include("bands.jl")

export TightBindingModel, hopping, bloch_hamiltonian, bloch_hamiltonian!
export KPath, BandStructure, kpath, bandstructure

end # module
