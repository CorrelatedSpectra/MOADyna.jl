"""
    MOADyna.TightBinding

Real-space hopping models, Bloch Hamiltonians and finite-cluster construction.
The common workflow is re-exported by `MOADyna`.
"""
module TightBinding

using LinearAlgebra
import ..Algebra: onebody

include("model.jl")
include("bands.jl")
include("clusters.jl")
include("ligand.jl")

export TightBindingModel, hopping, bloch_hamiltonian, bloch_hamiltonian!
export KPath, BandStructure, kpath, bandstructure
export FiniteCluster, finite_cluster, supercell
export ligand_basis

end # module
