# Getting Started

```@meta
CurrentModule = MOAD
DocTestSetup  = quote
    using MOAD
end
```

This walkthrough builds a small many-body model from scratch and diagonalizes it, touching
each layer of MOAD in turn: define the sites and Hilbert space, write the Hamiltonian with
the operator DSL, restrict to a conserved sector, assemble the sparse matrix, and solve.
The running example is the half-filled 4-site spinful Hubbard chain.

## Sites and the Hilbert space

A model is a collection of named [`FermionSite`](@ref MOAD.Algebra.FermionSite)/[`BosonSite`](@ref MOAD.Algebra.BosonSite)/[`SpinSite`](@ref MOAD.Algebra.SpinSite)
sites gathered into a [`Hilbert`](@ref) space. A `FermionSite{N}` carries `N` fermionic
modes; here each site has two modes, read as spin ↑ (mode 1) and ↓ (mode 2).

```@example gs
using MOAD

const L = 4
const UP, DN = 1, 2

sites   = [FermionSite{2}(Symbol("s$i")) for i in 1:L]
hilbert = Hilbert(s.name => s for s in sites)
(length(sites), local_dim(sites[1]))   # 4 sites; each FermionSite{2} has dim 4
```

## The Hamiltonian (operator DSL)

Operators are built symbolically with [`c`](@ref)/`c'` (annihilation/creation), [`n`](@ref)
(number), etc., and combine with `+`, `*`, and scalar multiplication into an
[`OperatorSum`](@ref). The Hubbard Hamiltonian is nearest-neighbor hopping plus on-site
Coulomb repulsion:

```@example gs
t, U = 1.0, 4.0

H_hop = sum(c'(sites[i], σ) * c(sites[i+1], σ) for i in 1:L-1, σ in (UP, DN))
H_hop = -t * (H_hop + H_hop')                       # add the h.c.
H_U   =  U * sum(n(sites[i], UP) * n(sites[i], DN) for i in 1:L)
H     = H_hop + H_U
H isa OperatorSum
```

## Conserved-sector basis

Rather than enumerate the full ``2^8`` Fock space, restrict to the physical sector with
conserved-quantity predicates. Here: half-filling (`L` electrons) and total `Sz = 0`. A bare
`FermionSite` carries no spin convention, so `Sz` is expressed by handing the basis explicit
mode weights `+1` on ↑ and `−1` on ↓ (so ``\sum_i w_i n_i = 2 S_z``).

```@example gs
sz_weights = repeat([1, -1], L)        # (+1, −1) per site

basis = EagerBasis(hilbert,
                   n_fermion(hilbert) == L,                       # half-filling
                   WeightedParticleCount(sites, sz_weights) == 0) # Sz = 0
length(basis)
```

## Assemble and diagonalize

[`compile`](@ref) resolves the symbolic operator against the basis, [`assemble`](@ref)
builds the sparse `SparseMatrixCSC`, and `eigen` (the MOAD extension, dispatching to dense or
`KrylovKit` automatically; see [Exact Diagonalization](@ref)) returns the eigensystem.

```@example gs
H_compiled = compile(H, basis)
H_sparse   = assemble(H_compiled, basis)
result     = eigen(H_sparse, basis; n = 4)
round.(result.values; digits = 4)
```

The ground-state energy of the half-filled `Sz = 0` 4-site Hubbard chain at `U/t = 4` is
the value printed above. From here the same `H_sparse`/`basis` feed the [Responses](@ref)
and [Spectroscopy](@ref) layers; the [NiO: XAS, RIXS, and nIXS](@ref) tutorial scales this
workflow up to a full multiplet calculation.
