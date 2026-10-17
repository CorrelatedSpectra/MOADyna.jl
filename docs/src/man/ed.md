# Exact Diagonalization

```@meta
CurrentModule = MOADyna
DocTestSetup  = quote
    using MOADyna
end
```

`MOADyna.ED` extends `LinearAlgebra.eigen` for symbolic
[`OperatorSum`](@ref), sparse matrices and [`MatrixFreeOperator`](@ref), each with a
basis constructed with [`basis`](@ref). It diagonalizes the Hamiltonian in that conserved sector and returns a
standard `LinearAlgebra.Eigen` (so `.values` and `.vectors` work as usual). `eigen` is re-exported
from `MOADyna`, so `using MOADyna` is enough — no explicit `using LinearAlgebra`.

## Diagonalizing

`eigen` accepts a symbolic [`OperatorSum`](@ref) or a prepared operator, together
with the basis it was built on.

```@example ed
using MOADyna

sites = [FermionSite{2}(Symbol("s$i")) for i in 1:4]
h = Hilbert(s.name => s for s in sites)
H = -1.0 * sum(add_hc(cdag(sites[i], σ) * c(sites[i+1], σ)) for i in 1:3, σ in 1:2) +
     4.0 * sum(n(sites[i], 1) * n(sites[i], 2) for i in 1:4)
b = basis(h, n_fermion(h) == 4, WeightedParticleCount(sites, repeat([1, -1], 4)) == 0)

sys = eigen(H, b; n = 4)        # lowest 4 eigenpairs
round.(sys.values; digits = 4)
```

## Dense vs Krylov dispatch

For sparse storage, small sectors are diagonalized densely
(`eigen(Hermitian(...))`) and large sectors use `KrylovKit.eigsolve`.
Matrix-free operators always use the iterative solver. Key keywords:

| kwarg | role |
|---|---|
| `n` | number of eigenpairs to return |
| `backend` | for symbolic input: `:sparse` (default) or `:matrixfree` |
| `which` | which end of the spectrum (`:SR` smallest-real by default) |
| `krylovdim`, `tol`, `maxiter` | forwarded to the Krylov path for large sectors |
| `threaded` | Hamiltonian multiplication: `:auto` (default), `true`, or `false` |
| `x0` | initial vector or block with at most `n` columns; copied before use |

A [`ConvergenceError`](@ref) is thrown if the Krylov path fails to converge;
[`max_normres`](@ref) reports the residual.

To avoid storing Hamiltonian matrix elements, use [`matrixfree`](@ref):

```@example ed
H_action = matrixfree(H, b; threaded=false)
sys_matrixfree = eigen(H_action, b; n=4)
sys_matrixfree.values ≈ sys.values
```

Equivalently, call `eigen(H, b; backend=:matrixfree, n=4)`. Both forms use
the same fixed basis as sparse assembly and introduce no determinant cutoff.
`dense_below` applies only to sparse storage; it never makes a matrix-free
calculation assemble a dense matrix implicitly.

## Multiplication threads

Start Julia with multiple threads to make workers available. For a batch
calculation on Julia 1.12 or later, use `julia --threads=4,0 --project`: `4,0`
requests four workers and no separate interactive pool. On Julia 1.10, use
`julia --threads=4 --project`, which has no interactive pool by default.
Keeping a batch calculation on the
worker pool can reduce task handoff overhead; interactive sessions can still
use a separate interactive pool. `threaded=:auto` uses workers for sufficiently large sparse
Hamiltonian products; small products remain serial to avoid scheduling overhead.
Use `threaded=true` to force the threaded kernel or `threaded=false` to force the
serial kernel for a calculation:

```@example ed
sys_serial = eigen(H, b; n = 4, dense_below = 0, threaded = false)
sys_threaded = eigen(H, b; n = 4, dense_below = 0, threaded = true)
sys_threaded.values ≈ sys_serial.values
```

For matrix-free operators, `:auto` preserves the threading choice made by
`matrixfree`; `true` and `false` override it for the current solve. The default
constructor uses the available Julia threads.

This keyword controls operator multiplication in the Krylov solver.
It does not change basis construction, Hamiltonian assembly, dense solver
selection, or BLAS thread settings. `true` uses the available Julia threads and
still runs on one worker when Julia was started with one thread. The Hamiltonian
must remain fixed during a solve; the solver reuses its prepared sparse action.
The threaded kernel supports CSC matrices with `Float32`, `Float64`,
`ComplexF32`, or `ComplexF64` entries. Other matrix types retain their existing
multiplication methods.

## Degeneracy

The iterative solver uses scalar Lanczos for one eigenvector and block Lanczos
for several, allowing independent eigenvectors with the same eigenvalue.
`n` is the number returned and can cut through a multiplet. Request enough states
to cover the complete manifold when needed for thermal averages or spectroscopy.

## Selecting determinants

[`selected_eigen`](@ref) starts from a small basis and discovers connected states
without enumerating the full allowed space. Each inner eigensolve uses a fixed
linear operator. Between solves, the basis grows according to the residual on
omitted states; necessary small coefficients are retained.

```@example selected
using MOADyna
a, b = FermionSite{1}(:a), FermionSite{1}(:b)
h = Hilbert(:a => a, :b => b)
allowed = n_fermion(h) == 1
seed = basis(h, allowed, n_fermion([a]) == 1)
H = n(b, 1) + 0.01add_hc(cdag(b, 1) * c(a, 1))
result = selected_eigen(H, seed; restrictions=allowed, max_states=2,
                         residual_tol=1e-10)
result.eigen.values
```

The returned vectors belong to `result.basis`, which can be used with ordinary
sparse assembly or matrix-free actions. `restrictions` explicitly defines the
allowed physical space; the extra restriction used to create this seed does
not constrain discovery.

Convergence requires the full residual block
`H*Ψ - Ψ*Diagonal(E)` to have Frobenius norm at most `residual_tol`.
`result.residuals` reports the norm for each eigenstate. The check includes
omitted determinants, so convergence of the inner projected problem is
insufficient. Complete degenerate clusters of the selected Hamiltonian crossing
the requested `n` states are included. Residual convergence alone cannot certify
the full allowed-space multiplicity, which matters for thermal averages.
The previous eigenvector block is reused after enlargement.

`max_states` limits retained states and `max_candidates` limits unique generated
external states before all cancellations, defaulting to `max_states`. Each
discovery worker holds a single-column and a target-block accumulator, each
bounded by this limit, in addition to merge and output storage.
A [`SelectionConvergenceError`](@ref) preserves the latest result when a resource
or iteration limit prevents convergence. An infinite residual in that result
means the candidate limit prevented a complete residual check.

The allowed-space Hamiltonian must be Hermitian. A small residual identifies
accurate eigenstates within the reachable sector; it does not prove that an
unseeded, disconnected sector has no lower state. Ground-state selection also
does not establish spectroscopy convergence. Transition sources and response
spaces must be checked separately.

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
