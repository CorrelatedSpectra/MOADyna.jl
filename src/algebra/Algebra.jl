"""
    MOAD.Algebra

Symbolic operator algebra on second-quantized many-body Hilbert spaces. Generic
over fermionic, bosonic, and spin local degrees of freedom. No basis enumeration,
no diagonalization, no domain-specific physics.

This is an internal submodule of `MOAD`; the names below are re-exported
from the umbrella package, so users typically write `using MOAD` instead of
reaching for `MOAD.Algebra` directly.
"""
module Algebra

using DataStructures: OrderedDict
using HDF5

include("sites.jl")
include("hilbert.jl")
include("operators.jl")
include("operator_sum.jl")
include("canonicalize.jl")
include("arithmetic.jl")
include("observables.jl")
include("interactions.jl")
include("rotate.jl")
include("operator_io.jl")

# Statistics
export Statistics, Fermionic, Bosonic, statistics

# Sites
export AbstractSite, FermionSite, BosonSite, SpinSite
export local_dim, mode_labels, mode_labels_canonical, normalize_label
export encoding_bits

# Hilbert space (the type)
export Hilbert, ⊗

# Operators / OperatorSum
export LadderKind, OperatorTerm, OperatorSum
export c, cdag, b, bdag, n, n_b
export Sx, Sy, Sz, Splus, Sminus, S
export chop, add_hc
export rotate
export save_operator, load_operator

# Multi-orbital interaction primitives (FermionSite + orbital-pair list)
export density_density, kanamori, Ssqr

# Observables / Restrictions
export QuantumNumber, QN, ParticleCount, TotalSz, WeightedParticleCount
export n_fermion, n_boson, Sz_total
export Restriction

end # module
