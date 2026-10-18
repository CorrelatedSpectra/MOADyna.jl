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
Alternatively, pass a `Hilbert` or `ShellModel` and physical `restrictions` to
construct the basis automatically and return [`Eigenstates`](@ref), which keeps
the eigenvectors and their basis together.

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

To let the solver manage the basis:

```@example ed
allowed = [n_fermion(h) == 4, WeightedParticleCount(sites, repeat([1, -1], 4)) == 0]
states = eigen(H, h; restrictions=allowed, n=4)
states.values
```

`states.vectors` uses `states.basis`. This result can be passed directly to
`xas`, `rixs`, and `fluorescence_yield`; see [Spectroscopy from eigenstates](@ref).
The explicit-basis form remains useful for reusing prepared operators.

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

## Choosing a representation

Sparse storage is recommended for routine spectroscopy, including typical atomic
multiplet and modest cluster or impurity calculations. Stored matrix entries
make repeated products efficient. Matrix-free storage is useful for large
systems where storing the Hamiltonian becomes limiting; the basis and solver
vectors still need to fit in memory.

| Situation | Starting point |
|---|---|
| Routine spectroscopy; the Hamiltonian fits in memory | The default sparse backend |
| Hamiltonian storage is limiting | `backend=:matrixfree` |
| Full basis enumeration is impractical | `adaptive=true`, with seed and residual convergence checks |

A fixed-basis operator represents ``P H P``, where ``P`` projects onto the
supplied determinant list. Matrix elements connecting to determinants outside
that list are omitted. A complete invariant sector contains every state coupled
by the Hamiltonian within that sector.

Adaptive selection grows the determinant basis during the calculation. Each
iteration solves on a fixed working basis, evaluates the residual including
omitted allowed states, and adds determinants as needed. Previously selected
states are retained. This is useful when the eigenstates can be represented
accurately in a small part of the allowed space.

See [Matrix-free Calculations](@ref) for a worked example and
[Selecting determinants](@ref) for selection controls.

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

The automatic interface manages the seed and basis growth:

```julia
gs = eigen(H, h; restrictions=allowed, backend=:matrixfree,
           adaptive=true, n=1, tol=1e-8)
gs.values
gs.selection.residual_norm
```

`restrictions` defines the physical space for the entire solve: each selected
determinant must satisfy these conditions. The initial basis samples up to
`krylovdim` allowed determinants. `seed_size` changes its size, and `rng` supplies
a local random-number generator. To start from a chosen configuration, pass an
explicit `seed` basis.

Check convergence by tightening `tol` and enlarging the seed. Selection explores
Hamiltonian connections from the seed; disconnected symmetry sectors need
appropriate initial coverage. Include enough eigenstates to resolve degeneracy
when calculating thermal averages.

| Keyword | Role |
|---|---|
| `tol` | Full residual tolerance, including omitted allowed states |
| `maxiter` | Maximum inner eigensolver iterations |
| `selection_maxiter` | Maximum selection iterations; default `50` |
| `max_states` | Limit on retained determinants |
| `max_candidates` | Limit on generated external states; defaults to `max_states` |
| `seed_max_work` | Work budget for automatic seed construction; default `100_000` |

State and candidate limits are unset by default in this interface. Reaching a
selection limit raises [`SelectionConvergenceError`](@ref) with the latest result.
An inner eigensolver failure raises [`ConvergenceError`](@ref). Exceeding
`seed_max_work` raises `ArgumentError`; supply an explicit seed to bypass
automatic seed construction, or increase that work budget.

For direct control over the starting basis, use [`selected_eigen`](@ref):

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

The returned vectors use `result.basis`. The physical `restrictions` apply during
basis growth; the additional condition defining this seed only selects its
starting configuration.

`selected_eigen` uses `residual_tol` for the full residual block
`H*Ψ - Ψ*Diagonal(E)`. Its Frobenius norm bounds the residual of every normalized
combination of the returned states. `result.residuals` gives individual norms.
The lowest `n` states include any degenerate cluster crossing that boundary in
the selected Hamiltonian. Resolving the multiplicity of the full allowed space
also requires adequate seed coverage.

`max_states` is required by this lower-level interface. `max_candidates` bounds
unique generated states before cancellation; discovery uses per-thread
workspaces, so temporary memory also depends on the thread count. An infinite
residual in a failed result indicates that the candidate limit prevented the
residual check from finishing.

For spectroscopy on a selected response basis, use [`selected_response`](@ref)
to check convergence at the frequencies and broadening of interest.

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
