# TightBinding

```@meta
CurrentModule = MOADyna
```

[`TightBindingModel`](@ref) stores one-electron hopping blocks,
``t_{ij}(R)=\langle 0i|h|Rj\rangle``, in eV. A lattice matrix has direct
lattice vectors as columns in Å; orbital centres are fractional-coordinate
columns. Centres retain the cell representatives associated with the hopping
table, which may lie outside the home cell.

For fractional reciprocal coordinates ``\kappa``, the periodic-gauge Hamiltonian is

```math
h_{ij}(\kappa)=\sum_R t_{ij}(R)e^{2\pi i\kappa\cdot R}.
```

The orbital-position gauge includes the extra phase
``e^{2\pi i\kappa\cdot(\tau_j-\tau_i)}``. Both gauges have the same band energies.
Hermiticity requires ``t(-R)=t(R)^\dagger``; the constructor checks this relation.

The [Interfacing with Wannier90](@ref) tutorial connects these objects to
the operator algebra and exact-diagonalization workflow.
For a fragment defined by neighboring atoms, [`cluster_sites`](@ref) selects
periodic site images around specified centres. Map those sites to Wannier
orbitals and pass the ordered selection to [`finite_cluster`](@ref), as shown
in that tutorial.

## API

```@autodocs
Modules = [MOADyna.TightBinding]
Private = false
```
