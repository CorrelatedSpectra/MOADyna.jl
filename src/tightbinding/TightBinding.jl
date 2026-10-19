"""
    MOADyna.TightBinding

Real-space hopping models, Bloch Hamiltonians and finite-cluster construction.
The common workflow is re-exported by `MOADyna`.
"""
module TightBinding

using LinearAlgebra
using StaticArrays: SVector, SMatrix, MVector
using ..Units: bohr_angstrom, convert_energy
import ..Algebra: onebody

include("model.jl")
include("bands.jl")
include("clusters.jl")
include("cluster_sites.jl")
include("ligand.jl")
include("wannier.jl")
include("wannier_geometry.jl")
include("wigner_seitz.jl")
include("band_io.jl")
include("reference_bands.jl")

export TightBindingModel, hopping, bloch_hamiltonian, bloch_hamiltonian!
export KPath, BandStructure, kpath, bandstructure
export FiniteCluster, finite_cluster, cluster_sites, supercell
export ligand_basis
export read_wannier90
export read_wannier_path, read_wannier_bands
export read_vaspkit_bands, read_wien2k_bands

end # module
