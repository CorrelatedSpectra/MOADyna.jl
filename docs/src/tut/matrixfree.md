# Matrix-free Calculations

```@meta
CurrentModule = MOADyna
```

**Use sparse matrices for routine spectroscopy.** They are the default and are
usually the best starting point for atomic multiplet and modest cluster or
impurity models. Matrix-free calculations are an advanced option for larger
systems where storing the Hamiltonian becomes limiting. They still store a basis
and solver vectors, and generating entries during multiplication can be slower
than using a sparse matrix.

This small Hubbard example introduces the interface, then explains fixed and
adaptive bases. Its size makes it easy to run; it does not need matrix-free
storage for efficiency. The runnable script is `examples/09_matrixfree.jl`.

## Define the physical problem

Build a half-filled four-site open chain with ``S_z=0`` and ``U/t=4``:

```@example mf
using MOADyna

L, t, U = 4, 1.0, 4.0
sites = [FermionSite{2}(Symbol("s$i")) for i in 1:L]
hilbert = Hilbert(s.name => s for s in sites)
hopping = sum(cdag(sites[i], σ) * c(sites[i+1], σ) for i in 1:L-1, σ in 1:2)
H = -t * (hopping + hopping') + U * sum(n(s, 1) * n(s, 2) for s in sites)
allowed = [n_fermion(hilbert) == L,
           WeightedParticleCount(sites, repeat([1, -1], L)) == 0]
```

`allowed` is an ordinary variable holding physical restrictions. Each restriction
applies to every determinant: here each has four particles and equal up/down
occupations. These are not conditions on the average occupation of a superposition.

## Solve without managing the basis

```@example mf
gs = eigen(H, hilbert; restrictions=allowed, backend=:matrixfree, n=1)
(energy=gs.values[1], states=length(gs.basis))
```

The solver constructs the complete restricted basis internally. The result keeps
that basis with its eigenvectors: `gs.vectors[:, 1]` uses `gs.basis`. A `ShellModel`
can replace `hilbert` for multiplet calculations. Omit `backend=:matrixfree` to
use the recommended sparse default with the same interface.

This is a **fixed-basis** calculation: its determinant list stays unchanged.
The operator acts as ``P H P``, projecting onto that list. Contributions to states
outside it are discarded. Here the basis contains the complete conserved sector,
so the Hamiltonian has no coupling outside it. An incomplete list would instead
omit physical connections; convergence within it would not establish full-space
accuracy. Matrix-free storage alone introduces no coefficient cutoff.

## Adaptive basis growth

When enumerating the complete basis is impractical, enable adaptive selection:

```@example mf
selected = eigen(H, hilbert; restrictions=allowed, backend=:matrixfree,
                 adaptive=true, n=1, tol=1e-8)
(energy=selected.values[1], states=length(selected.basis),
 residual=selected.selection.residual_norm)
```

**Adaptive** means that the determinant list grows during the calculation. The
solver chooses a small initial basis, solves on that basis, evaluates the residual
including omitted allowed determinants, and adds states as needed. Previously
selected states are retained. Each inner eigensolve still uses a fixed linear
operator. Quanty's
[Eigensystem](https://www.quanty.org/documentation/language_reference/functions/eigensystem)
is also adaptive with `ExpandBasis=true`; this term is not specific to MOADyna.
Its usual restriction-based call also generates a random starting wavefunction.
Quanty's `StartRestrictions` apply only to initialization; MOADyna's
`restrictions` defines the physical space allowed throughout selection.

The default initial sample has up to `krylovdim` states and is reproducible.
Automatic sampling avoids materializing the full basis, but complicated
restrictions can make counting costly. `seed_max_work` bounds that work;
`seed_size`, `rng`, or an explicit `seed` provide additional control. See
[Selecting determinants](@ref) for these options and the lower-level
[`selected_eigen`](@ref) interface.

Adaptive selection needs convergence checks beyond an ordinary fixed-sector
solve. A small residual certifies an approximate eigenstate; it cannot prove that
an unseeded, disconnected sector contains no lower state. Check stability under
seed enlargement and choose appropriate physical symmetry sectors. A diagonal
Hamiltonian is a simple counterexample: every determinant already has zero
residual, including excited determinants.

For large calculations, set `max_states` and `max_candidates` to bound retained
and generated states. These are resource limits, not accuracy settings; reaching
a limit raises [`SelectionConvergenceError`](@ref). They are unset by default in
this automatic interface. This small Hubbard sector may need most or all of its
36 determinants, so adaptation need not save time or memory.

## From eigenstates to a spectrum

The absorption interface also evaluates a general probe operator. Use the
staggered charge ``O=\sum_i(-1)^i(n_{i\uparrow}+n_{i\downarrow})`` here:

```@example mf
charge = sum((-1)^i * (n(sites[i], 1) + n(sites[i], 2)) for i in 1:L)
omega = range(0, 12; length=241)
spec = xas(H, charge, selected; restrictions=allowed,
           omega_grid=omega, Gamma=0.3, krylovdim=36)
intensity = -imag.(spec.tensor) / π
length(intensity)
```

No manual embedding or transition assembly is needed. This call constructs the
**complete response sector** from `restrictions`, independently of the selected
ground-state basis. Ground-state convergence does not certify a spectrum on that
selected list. If full response enumeration is impractical, converge a response
space separately with [`selected_response`](@ref).

For core-level XAS, pass dipole operators and the core-hole sector's restrictions.
`rixs` and `fluorescence_yield` similarly accept these eigenstates, with explicit
intermediate and final restrictions; see [Spectroscopy from eigenstates](@ref).
Their polarization tensors and contractions are unchanged.

`Gamma` is the Lorentzian FWHM. This small example can use a Krylov dimension equal
to the full sector. For larger calculations, check convergence at the required
broadening. Re-broadening uses the retained propagation data:

```@example mf
broader = re_broaden(spec; Γ=0.6)
length(broader.tensor)
```

## Explicit control and threading

The original explicit-basis interface remains available for reusing actions:

```julia
b = basis(hilbert, allowed...)
H_action = matrixfree(H, b)
gs_explicit = eigen(H_action, b; n=1)
```

Here `gs_explicit` is a standard `LinearAlgebra.Eigen`; the caller keeps `b`.
Rectangular actions use `matrixfree(T, destination_basis, source_basis)`.
Use the same basis objects when building and solving with prepared actions.

Start Julia with several worker threads, for example:

```sh
julia --project=. --threads=4,0 examples/09_matrixfree.jl
```

Matrix-free products use the available Julia threads by default. Set
`threaded=false` in the constructor or solver to use serial products, or
`threaded=true` to request parallel products. These options do not create threads
or change BLAS thread counts. Tiny calculations such as this tutorial can be
faster in serial; see [Multiplication threads](@ref).
