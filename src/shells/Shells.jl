"""
    MOADyna.Shells

Shell-registry + multiplet-physics layer over `MOADyna.Algebra` and
`MOADyna.PointGroups`.

**Direct exports** (names you get from `using MOADyna.Shells`):

- `ShellModel` registry and accessors `ell_of`, `range_of`, `site_of`.
- Basis-restriction DSL: `nshells(m, :s, …)`, `total(m)`.
- Angular momentum: `Lx, Ly, Lz, Lplus, Lminus, Lsqr`,
  `Jx, Jy, Jz, Jplus, Jminus, Jsqr`.
- One-body atomic spin-orbit: `LS(m, :s)`.
- Coulomb interactions: `coulomb(m, :s; U, F)` single-shell and
  `coulomb(m, :sA, :sB; U, F, G)` two-shell (Slater-Condon, normal-ordered).
- Group-projected: `Akm(m, :s, :group, coeffs)` (crystal-field) and
  `hop(m, :sA, :sB, :group; irrep)` (irrep-projected hybridization).
- Transition operators (parity/triangle-enforced): `dipole(m, :sA => :sB) ->
  [Tx, Ty, Tz]` (E1); `multipole(m, :sA => :sB, k) -> [T^k_{-k}…T^k_{+k}]`
  (general rank-k spherical); `quadrupole_transition(m, :sA => :sB) ->
  [z², xz, yz, x²−y², xy]` (E2, real tesseral).
- Moment / scattering operators: `quadrupole(m, :s) -> [Qxx,Qyy,Qzz,Qxy,Qxz,Qyz]`
  (charge-quadrupole moment, dimensionless); `nixs(m, :sA => :sB; theta, phi,
  radial_integrals)` (e^{iq·r} nIXS scattering operator); `radial_integral(...)`
  (Bessel / ⟨rᵏ⟩ radial moments).
- Basis-change matrices: `to_real(m, :s)` (real cubic harmonics, m-ordered
  per Questaal) and `to_jlmj(m, :s)` (j-coupled via Clebsch-Gordan).
- Anchor-based onsite-energy solver: `onsite_energies(m; anchors, U, pairs, shells)`.

**Method extensions** (function lives in `MOADyna.Algebra`, this module adds
`ShellModel` dispatch):

- Number `n`, spin `Sx, Sy, Sz, Splus, Sminus, Ssqr`.
- Density-density `density_density` and full `kanamori`.

For users, `using MOADyna` re-exports the public API. The cross-basis state
embedding `embed(psi, basis_a => basis_b)` lives in `MOADyna.Bases`.

See the **Multiplets & standard operators** chapter of the manual for the
user-facing specification.
"""
module Shells

# Types we refer to.
using ..Algebra: FermionSite, Hilbert, AbstractSite, OperatorSum
# Functions we want to extend with ShellModel methods.
import ..Algebra: n, Sx, Sy, Sz, Splus, Sminus, Ssqr, density_density, kanamori
# Functions we use to build operator expressions.
using ..Algebra: c, cdag
# Types needed by the basis-restriction DSL.
using ..Algebra: ParticleCount, Fermionic, n_fermion
# Extend EagerBasis and basis with ShellModel overloads (import, NOT using).
import ..Bases: EagerBasis, basis

using LinearAlgebra: kron

include("parse_tag.jl")
include("mode_layout.jl")
include("shell_model.jl")
include("operators.jl")
include("angular_momentum.jl")
include("interactions.jl")
include("basis_dsl.jl")
include("rotations.jl")
include("group_projected.jl")
include("onsite_energies.jl")
include("dipole.jl")
include("multipole.jl")
include("nixs.jl")
include("slater.jl")

export ShellModel, ell_of, range_of, site_of
export nshells, total
export Lz, Lplus, Lminus, Lx, Ly, Lsqr
export Jx, Jy, Jz, Jplus, Jminus, Jsqr
export LS
export to_real, to_jlmj
export Akm, hop
export onsite_energies
export dipole
export multipole, quadrupole_transition, quadrupole
export nixs, radial_integral
export spin_dipole_T
export coulomb
# Ssqr, density_density, kanamori are exported from Algebra;
# Shells adds ShellModel-keyed dispatch methods via import above.

end # module
