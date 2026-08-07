# Introduction

**MOADyna** (Many-body Operators, Algebra, and Dynamics) is a Julia framework for exact
diagonalization and core-level spectroscopy of small, strongly-correlated quantum systems —
atomic multiplets, impurity and cluster models, and small lattices of fermions, bosons, and
spins. It is designed as a theorist-first tool: a programmable, inspectable environment in
which a Hamiltonian, a basis, a matrix, and a spectrum are all ordinary Julia values you can
build, examine, and reuse.

## Design

The guiding idea is that **the code should read like the operators**. `c'(s, 1) * c(s, 2)`
*is* the operator ``\hat{c}^\dagger_{s\uparrow}\,\hat{c}_{s\downarrow}``; fermionic anticommutation is tracked symbolically and
exactly; and the path from a symbolic Hamiltonian to a sparse matrix to a spectrum is
transparent at every step — no hidden basis choices, no opaque solver. You can stop at any
layer and inspect what you have.

## Two registers

It helps to know which of two registers you are reading. The **core** — operator algebra,
bases, exact diagonalization, response functions, and the point-group engine — speaks
standard many-body theory: its types and functions are named for the mathematical objects
they are (an [`OperatorSum`](@ref), a [`QuantumNumber`](@ref MOADyna.Algebra.QuantumNumber), a
[`correlator`](@ref)), and its docstrings give the precise definition. The **spectroscopy
layer** on top — [`xas`](@ref), [`rixs`](@ref), the multiplet builders, the
[NiO: XAS, RIXS, and nIXS](@ref) tutorial — speaks the experimentalist's language of
edges, polarisations, and spectra, and lets you compute a spectrum without touching the core
abstractions. The dependency runs one way: the spectroscopy layer is a thin skin over the
core, never the reverse.

## How the package is organised

MOADyna is one package of submodules, each exercised by the test suite and each the subject of a
Manual chapter. They fall into two groups: a **calculation pipeline** that carries a model from
operators to a spectrum, and a **multiplet toolkit** that builds the physical operators and
parameters the pipeline consumes.

**The calculation pipeline** — each stage builds directly on the one before it:

1. **[Operator Algebra](@ref)** (`Algebra`) — sites, modes, ladder operators, the
   symbolic [`OperatorSum`](@ref), quantum numbers.
2. **[Hilbert Spaces & Bases](@ref)** (`Bases`) — conserved-sector enumeration with
   [`EagerBasis`](@ref); [`compile`](@ref) / [`assemble`](@ref) to a sparse matrix.
3. **[Exact Diagonalization](@ref)** (`ED`) — the `eigen(H, basis)` extension with
   automatic dense ↔ Krylov dispatch.
4. **[Responses](@ref)** (`Responses`) — the matrix-of-ω [`correlator`](@ref) and its
   Lanczos / Pole / Grid representations.
5. **[Spectroscopy](@ref)** (`Spectroscopy`) — [`xas`](@ref), [`rixs`](@ref),
   [`fluorescence_yield`](@ref), [`optical_conductivity`](@ref),
   [`dynamical_structure_factor`](@ref).

**The multiplet toolkit** — these sit *beside* the pipeline (not above it), producing the
operators and parameters it runs on:

- **[Multiplets & Standard Operators](@ref)** (`Shells`) — the `ShellModel` registry and every
  standard multiplet / transition operator (Coulomb, spin–orbit, crystal field, hybridisation,
  and the dipole / quadrupole / nIXS transition operators). This is where most spectroscopy
  Hamiltonians are written; it draws on Point Groups for crystal fields and is commonly paired
  with Atomic Parameters for the radial integrals.
- **[Point Groups](@ref)** (`PointGroups`) — 32 crystallographic + 10 molecular groups with
  reference-verified character tables, supplying the irrep-projected crystal fields.
- **[Atomic Parameters](@ref)** (`AtomicParameters`) — Slater–Condon ``F^{k}``/``G^{k}``,
  ``\zeta``, ``\langle r^{k}\rangle`` for transition-metal and lanthanide ions.

## Where to start

- New to MOADyna? The [Getting Started](@ref) tutorial builds and solves a Hubbard chain end
  to end.
- Here for spectroscopy? Jump to [Multiplets & Standard Operators](@ref) and the
  [NiO: XAS, RIXS, and nIXS](@ref) tutorial.

The reference docstrings for every exported symbol are in the **Library** section; the
**Appendix** collects conventions and the point-group theory primer.
