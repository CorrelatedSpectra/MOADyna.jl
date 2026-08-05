# Operator Algebra

```@meta
CurrentModule = MOAD
DocTestSetup  = quote
    using MOAD
end
```

`MOAD.Algebra` is the symbolic second-quantized layer: it defines the sites and their
modes, the ladder operators, and the [`OperatorSum`](@ref) algebra in which every MOAD
Hamiltonian and observable is written. Expressions are built to read like the physics —
`c'(s, 1) * c(s, 2)` *is* the operator ``\hat{c}^\dagger_{s\uparrow}\,\hat{c}_{s\downarrow}`` — with fermionic anticommutation
tracked exactly and symbolically; nothing is materialized as a matrix until the
[Hilbert Spaces & Bases](@ref) layer.

## Sites and modes

A site is a local Hilbert space with a fixed set of modes. Three kinds:
[`FermionSite`](@ref MOAD.Algebra.FermionSite)`{N}` (N fermionic modes),
[`BosonSite`](@ref MOAD.Algebra.BosonSite) (a truncated boson), and
[`SpinSite`](@ref MOAD.Algebra.SpinSite) (a spin-`S`). Sites are named; modes are addressed by 1-based index.

```@example alg
using MOAD

s = FermionSite{2}(:imp)        # a 2-mode fermion site (e.g. ↑, ↓)
(local_dim(s), statistics(s))
```

## Ladder operators

The fermionic ladder operators are [`c`](@ref) / [`cdag`](@ref) (also written `c'`), the
bosonic ones [`b`](@ref) / [`bdag`](@ref), the number operator [`n`](@ref) (and `n_b`), and
the spin operators [`Sx`](@ref MOAD.Algebra.Sx)/`Sy`/`Sz`/`Splus`/`Sminus`. A single ladder operator and any
sum/product of them is an [`OperatorSum`](@ref); arithmetic (`+`, `-`, `*`, scalar `*`) and
the adjoint `'` are defined.

```@example alg
n_up = cdag(s, 1) * c(s, 1)     # number operator for mode 1
(n_up == n(s, 1), (cdag(s, 1) * c(s, 2))' isa OperatorSum)
```

Fermionic anticommutation is enforced symbolically: reordering ladder operators carries the
Jordan–Wigner sign automatically, so the algebra is exact regardless of the order in which
terms are written.

## OperatorSum utilities

[`chop`](@ref) drops terms below a tolerance (cleaning up numerical roundoff after
arithmetic), and [`add_hc`](@ref) adds the Hermitian conjugate. [`rotate`](@ref) applies a
single-particle basis rotation. Operators serialize with [`save_operator`](@ref) /
[`load_operator`](@ref).

```@example alg
H = -1.0 * (cdag(s, 1) * c(s, 2))
H = add_hc(H)                   # H + H†  (Hermitian hopping)
length(H)
```

## Conserved quantities

Particle-number and spin observables — [`n_fermion`](@ref), [`n_boson`](@ref),
[`Sz_total`](@ref), and the general [`WeightedParticleCount`](@ref) — are
[`QuantumNumber`](@ref MOAD.Algebra.QuantumNumber) objects. Compared against a value with `==` (or `∈` a range)
they produce a [`Restriction`](@ref) used to carve out a conserved sector; this is the
subject of the [Hilbert Spaces & Bases](@ref) chapter.
