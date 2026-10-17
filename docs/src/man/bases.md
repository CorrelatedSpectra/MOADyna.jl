# Hilbert Spaces & Bases

```@meta
CurrentModule = MOADyna
DocTestSetup  = quote
    using MOADyna
end
```

`MOADyna.Bases` turns the symbolic [Operator Algebra](@ref) into numbers. It assembles named
sites into a [`Hilbert`](@ref) space and enumerates a conserved sector with
[`basis`](@ref). A symbolic [`OperatorSum`](@ref) can then be assembled as a
sparse matrix with [`compile`](@ref) / [`assemble`](@ref), or applied without
storing matrix elements using [`matrixfree`](@ref).

## Hilbert spaces

A [`Hilbert`](@ref) collects sites under their names; the tensor product `⊗` and the
`name => site` constructor both build one.

```@example bas
using MOADyna

sites = [FermionSite{2}(Symbol("s$i")) for i in 1:4]
h = Hilbert(s.name => s for s in sites)
h isa Hilbert
```

## Conserved-sector restrictions

Enumerating the full Fock space is exponential and almost never needed: physical
Hamiltonians conserve particle number, `Sz`, etc., so work in one sector. A
[`QuantumNumber`](@ref MOADyna.Algebra.QuantumNumber) ([`n_fermion`](@ref), [`Sz_total`](@ref),
[`WeightedParticleCount`](@ref)) compared with `==` (or `∈` a range) yields a
[`Restriction`](@ref). [`basis`](@ref) enumerates exactly the states satisfying all
restrictions.

```@example bas
sz_w  = repeat([1, -1], 4)                         # +1 on ↑, −1 on ↓
b = basis(h,
                   n_fermion(h) == 4,              # half-filling
                   WeightedParticleCount(sites, sz_w) == 0)   # Sz = 0
length(b)
```

`EagerBasis` materializes the sector's state list eagerly (the bit-packed Fock states), so
[`get_state`](@ref) / [`get_index`](@ref) map between states and indices in ``O(1)``–``O(\log n)``.

## Compile and assemble

[`compile`](@ref) resolves a symbolic operator against a basis (matching its sites and modes
to their bit positions, and expanding `Sx`/`Sy` into `S+`/`S−`); [`assemble`](@ref) then
builds the `SparseMatrixCSC`. Splitting the two lets you compile several operators (the
Hamiltonian, observables, …) against one shared basis and assemble each separately, and
supports the parallel sparse-assembly path.

```@example bas
H  = -1.0 * sum(add_hc(cdag(sites[i], σ) * c(sites[i+1], σ)) for i in 1:3, σ in 1:2)
Hc = compile(H, b)
Hs = assemble(Hc, b)
size(Hs)
```

## Matrix-free actions

[`matrixfree`](@ref) retains compiled operator terms and determinant lookup
tables, and generates matrix elements during multiplication. It supports
vectors, blocks of vectors, `mul!`, adjoints and the same fixed-basis solvers
as sparse storage.

```@example bas
A = matrixfree(H, b; threaded=false)
v = ones(length(b))
A * v ≈ Hs * v
```

This construction keeps every state in `b` and applies the projected operator
exactly, up to floating-point arithmetic. Keep the basis data unchanged while
using the action. `threaded=:auto` uses the available Julia threads; `true` and
`false` explicitly select parallel or serial products. `sparse(A)` and `Matrix(A)`
are explicit conversions when stored matrix elements are needed.

For transitions between different sectors, use `matrixfree(T, bra, ket)`.
Its output coordinates belong to `bra` and input coordinates to `ket`; both
bases must use the same site and mode encoding. The adjoint reverses these roles.

## Embedding between sectors

[`embed`](@ref) lifts a vector from one basis into a larger one (e.g. a ground state into
the bigger basis a core-hole spectrum lives in) — the mechanism the spectroscopy wrappers
use to connect the initial- and final-state sectors. [`apply_restriction!`](@ref) projects a
vector into a sector in place (used by the restriction-resolved partial-spectrum path).
