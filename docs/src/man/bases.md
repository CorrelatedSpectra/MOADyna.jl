# Hilbert Spaces & Bases

```@meta
CurrentModule = MOADyna
DocTestSetup  = quote
    using MOADyna
end
```

`MOADyna.Bases` turns the symbolic [Operator Algebra](@ref) into numbers. It assembles named
sites into a [`Hilbert`](@ref) space, enumerates the basis of a conserved sector with
[`EagerBasis`](@ref), and compiles a symbolic [`OperatorSum`](@ref) into a sparse matrix on
that basis with [`compile`](@ref) / [`assemble`](@ref).

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
[`Restriction`](@ref). [`EagerBasis`](@ref) enumerates exactly the states satisfying all
restrictions.

```@example bas
sz_w  = repeat([1, -1], 4)                         # +1 on ↑, −1 on ↓
basis = EagerBasis(h,
                   n_fermion(h) == 4,              # half-filling
                   WeightedParticleCount(sites, sz_w) == 0)   # Sz = 0
length(basis)
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
Hc = compile(H, basis)
Hs = assemble(Hc, basis)
size(Hs)
```

## Embedding between sectors

[`embed`](@ref) lifts a vector from one basis into a larger one (e.g. a ground state into
the bigger basis a core-hole spectrum lives in) — the mechanism the spectroscopy wrappers
use to connect the initial- and final-state sectors. [`apply_restriction!`](@ref) projects a
vector into a sector in place (used by the restriction-resolved partial-spectrum path).
