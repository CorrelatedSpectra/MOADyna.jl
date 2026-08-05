# Exact Diagonalization

```@meta
CurrentModule = MOAD
DocTestSetup  = quote
    using MOAD
end
```

`MOAD.ED` extends `LinearAlgebra.eigen` with two methods — one taking a symbolic
[`OperatorSum`](@ref), one taking an already-assembled `SparseMatrixCSC` — each with an
[`EagerBasis`](@ref). It diagonalizes the Hamiltonian in that conserved sector and returns a
standard `LinearAlgebra.Eigen` (so `.values` and `.vectors` work as usual). `eigen` is re-exported
from `MOAD`, so `using MOAD` is enough — no explicit `using LinearAlgebra`.

## Diagonalizing

`eigen` accepts either the symbolic [`OperatorSum`](@ref) plus a basis, or an
already-assembled sparse matrix plus the basis it was built on.

```@example ed
using MOAD

sites = [FermionSite{2}(Symbol("s$i")) for i in 1:4]
h = Hilbert(s.name => s for s in sites)
H = -1.0 * sum(add_hc(cdag(sites[i], σ) * c(sites[i+1], σ)) for i in 1:3, σ in 1:2) +
     4.0 * sum(n(sites[i], 1) * n(sites[i], 2) for i in 1:4)
b = EagerBasis(h, n_fermion(h) == 4, WeightedParticleCount(sites, repeat([1, -1], 4)) == 0)

sys = eigen(H, b; n = 4)        # lowest 4 eigenpairs
round.(sys.values; digits = 4)
```

## Dense vs Krylov dispatch

The solver dispatches automatically on sector size: small sectors are diagonalized densely
(`eigen(Hermitian(...))`), large sectors via `KrylovKit.eigsolve`. Key keywords:

| kwarg | role |
|---|---|
| `n` | number of eigenpairs to return |
| `which` | which end of the spectrum (`:SR` smallest-real by default) |
| `krylovdim`, `tol`, `maxiter` | forwarded to the Krylov path for large sectors |

A [`ConvergenceError`](@ref) is thrown if the Krylov path fails to converge;
[`max_normres`](@ref) reports the residual.

## Degeneracy

The returned eigenvalues respect degeneracy — the solver does not split a degenerate
multiplet across the `n` cut arbitrarily; request enough states (`n`) to cover a degenerate
ground manifold when you need the full multiplet (e.g. for a thermal average or a
degeneracy-complete spectroscopy calculation).

## Saving eigensystems

[`save_eigensystem`](@ref) / [`load_eigensystem`](@ref) round-trip an eigensystem to HDF5,
so an expensive diagonalization can be reused across spectroscopy runs (diagonalise once,
reload everywhere):

```@example ed
f = tempname() * ".h5"
save_eigensystem(f, sys)
sys2 = load_eigensystem(f)
rm(f; force = true)                       # (cleanup; the file persists in real use)
sys2.values ≈ sys.values                  # exact round-trip
```

The companion [`save_response`](@ref)/[`load_response`](@ref),
[`save_operator`](@ref)/[`load_operator`](@ref), and
[`save_spectra`](@ref)/[`load_spectra`](@ref) do the same for response functions
(representation + metadata preserved), symbolic operators, and computed spectra.
