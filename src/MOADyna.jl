"""
    MOADyna — Many-body Operators, Algebra, and Dynamics

A Julia package for symbolic operator algebra, conserved-sector basis
construction, and sparse Hamiltonian assembly on fermionic, bosonic, and spin
many-body Hilbert spaces. Targeted at exact-diagonalization spectroscopy
(XAS, RIXS, multiplet calculations) and as a substrate for tensor-network
work.

# Quick start

```julia
using MOADyna

# Build a 2-orbital Hubbard model on a 4-site chain.
sites = [FermionSite{2}(Symbol("s\$i")) for i in 1:4]
hilbert = Hilbert(s.name => s for s in sites)

t, U = 1.0, 4.0
H_hop = -t * sum(c'(sites[i], σ) * c(sites[i+1], σ) for i = 1:3, σ = 1:2)
H_U   =  U * sum(n(sites[i], 1) * n(sites[i], 2)    for i = 1:4)
H     = H_hop + H_hop' + H_U

# Restrict to the half-filled Sz=0 sector.
weights = repeat([1, -1], 4)         # +1 on up modes, -1 on down modes
basis   = EagerBasis(hilbert,
                     n_fermion(hilbert) == 4,
                     WeightedParticleCount(sites, weights) == 0)

# Compile the symbolic Hamiltonian and assemble the sparse matrix.
H_compiled = compile(H, basis)
H_sparse   = assemble(H_compiled, basis)
```

`H_sparse` is a `SparseMatrixCSC{Float64, Int}` ready to hand to
`KrylovKit.eigsolve` (or any other Krylov library) for ground-state energies,
spectra, and time evolution.

# Submodules

Internal submodules; the names below are re-exported from `MOADyna`, so users
almost never reach for these directly.

- [`MOADyna.Algebra`](@ref) — symbolic operators, conserved quantities, `rotate`
- [`MOADyna.Bases`](@ref) — eager basis enumeration, sparse assembly
- [`MOADyna.ED`](@ref) — sparse eigensolver (`eigen(H, basis; n, which, …)`)
- [`MOADyna.Spectroscopy`](@ref) — `xas`, `rixs`, `fluorescence_yield`, broadening
- [`MOADyna.PointGroups`](@ref) — 42 groups with character tables, `expand_clm`
- [`MOADyna.Shells`](@ref) — shell registry, multiplet physics (L/J/LS, coulomb,
  Akm, hop, dipole, to_real, to_jlmj, onsite_energies)
- [`MOADyna.Diagnostics`](@ref) — post-eigenstate analysis (`expectation_table`,
  `configuration_weights`)
- [`MOADyna.QuantyIO`](@ref) — read Quanty operator dumps into `OperatorSum`s
- [`MOADyna.AtomicParameters`](@ref) — atomic Slater-Condon parameters
  (`atomic_parameters`, static Haverkort dict + optional Cowan)
- [`MOADyna.Units`](@ref) — energy-unit conversions (`convert_energy`, `Units.kB_eV`)
- `MOADyna.Gradients` — differentiable T=0 XAS forward model (`XASGradientModel`,
  `spectrum`, `jacobian`, `spectrum_with_pullback`, `fit_spectrum`). Not
  re-exported by `using MOADyna`; reach it module-qualified as `MOADyna.Gradients`.
"""
module MOADyna

include("units/Units.jl")
include("algebra/Algebra.jl")
include("bases/Bases.jl")
include("shells/Shells.jl")
include("quantyio/QuantyIO.jl")
include("ed/ED.jl")
include("responses/Responses.jl")
include("spectroscopy/Spectroscopy.jl")
include("pointgroups/PointGroups.jl")
include("diagnostics/Diagnostics.jl")
include("atomic_parameters/AtomicParameters.jl")
include("gradients/Gradients.jl")

using .Units
using .Algebra
using .Bases
using .Shells
using .QuantyIO
using .ED
using .Responses
using .Spectroscopy
using .PointGroups
using .Diagnostics
using .AtomicParameters

# `eigen` is added to LinearAlgebra by MOADyna.ED; re-export the symbol so
# `using MOADyna` users can call `eigen(H, basis)` without an explicit
# `using LinearAlgebra`.
import LinearAlgebra: eigen
export eigen

# --- Re-export the user-facing surface ---

# From Algebra
export Statistics, Fermionic, Bosonic, statistics
export AbstractSite, FermionSite, BosonSite, SpinSite
export local_dim, mode_labels, mode_labels_canonical, normalize_label
export encoding_bits
export Hilbert, ⊗
export LadderKind, OperatorTerm, OperatorSum
export c, cdag, b, bdag, n, n_b
export Sx, Sy, Sz, Splus, Sminus, S
export chop, add_hc
export rotate
export save_operator, load_operator
export QuantumNumber, QN, ParticleCount, TotalSz, WeightedParticleCount
export n_fermion, n_boson, Sz_total
export Restriction
export density_density, kanamori, Ssqr

# From Bases
export AbstractBasis, EagerBasis, basis
export get_state, get_index, basis_id
export apply_restriction!
export CompiledHamiltonian
export compile, assemble
export embed

# From Shells
export ShellModel, ell_of, range_of, site_of
export nshells, total
export Lz, Lplus, Lminus, Lx, Ly, Lsqr
export Jx, Jy, Jz, Jplus, Jminus, Jsqr
export LS
export onsite_energies, Akm, hop, dipole, coulomb
export multipole, quadrupole_transition, quadrupole
export nixs, radial_integral
export spin_dipole_T
export to_real, to_jlmj

# From QuantyIO
export ParsedOperator, ParsedTerm
export parse_quanty_operator
export build_operator
export read_quanty_operator, read_quanty_operators
export read_quanty_wavefunction, read_quanty_wavefunctions, read_quanty_eigenvalues

# From ED
export ConvergenceError, max_normres
export save_eigensystem, load_eigensystem

# From Responses (algorithm internals — block_lanczos, cf_block — stay
# at submodule scope; reach via MOADyna.Responses.<name>)
export AbstractResponse
export LanczosResponse, PoleResponse, GridResponse, GreensFunction
export correlator, to_pole, to_grid
export save_response, load_response

# From Spectroscopy (rixs / fluorescence_yield + helpers added in
# later phases). Algorithm internals (LanczosChunk, BlockTriDiagonal,
# Defaults, block_lanczos, cf_block, evaluate_on_grid, pretty_step,
# auto_grid) intentionally not re-exported here; reach via
# MOADyna.Spectroscopy.<name>.
export SpectraTensor
export xas, rixs, fluorescence_yield
export optical_conductivity, dynamical_structure_factor, kubo_response
# SpectraTensor post-processing toolkit (operates on xas/rixs/… results):
# polarisation contraction, pole extraction, re-broadening, ensemble
# combination, windowing, and I/O.
export polarise, poles, re_broaden, re_broaden_table, find_chunk
export average, weighted_sum, restrict_to_window, plot_range
export save_spectra, load_spectra

# From PointGroups (`lift` / `LiftedRep` are module-qualified only;
# the Fock-space apply path needs Layer-1 single-particle rotation
# wiring not yet available — access via `MOADyna.PointGroups.lift`)
export GroupElement, IRrep, PointGroup
export pointgroup
export subduce, project, wignerd
export expand_clm, expand_clm_central, nparams
export classify_subspace, classify_state
export has_reference_labels, reference_label_groups
export character_table, print_character_table, character_table_compare

# From Diagnostics
export expectation_table, configuration_weights

# From AtomicParameters
export atomic_parameters, radial_wavefunction, covered_elements, covered_configurations

# From Units (energy-unit conversions; `Units.kB_eV` stays module-qualified)
export convert_energy

end # module
