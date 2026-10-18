# MOADyna.jl

**MOADyna** — *Many-body Operators, Algebra, and Dynamics* — is a Julia package for the
exact diagonalization and core-level spectroscopy of small, strongly correlated
quantum systems: atomic multiplets, Anderson impurity and cluster models, and small
lattices built from fermions, bosons, and spins. You write a Hamiltonian as a symbolic
operator expression, restrict it to a conserved-quantum-number sector, solve with
sparse matrices or matrix-free operator actions, and compute dynamical response functions —
XAS, RIXS, fluorescence yield, optical conductivity, the dynamical structure factor,
and single-particle Green's functions — all within a single package.

!!! note "Installation"
    ```julia
    using Pkg
    Pkg.add("MOADyna")
    ```
    Development and CI use the latest stable Julia, currently 1.13. Julia
    1.10–1.12 are expected to work, but are not covered by CI.
    See the [Changelog](@ref) for what's new.

!!! tip "Precompilation and first-use time"
    Julia precompiles packages after installation or updates and caches compiled
    code for later sessions. You can also run `using Pkg; Pkg.precompile()` in the
    environment where you installed MOADyna. The first calculation can take
    longer while Julia compiles additional methods.

The organizing principle is that *the code should read like the operators*. The
expression `c'(s, 1) * c(s, 2)` is literally the operator
``\hat{c}^{\dagger}_{s\uparrow}\,\hat{c}_{s\downarrow}``; fermionic anticommutation is
tracked symbolically and exactly, and the path from a symbolic Hamiltonian through a
sparse matrix or matrix-free action to a spectrum is explicit at every step — no hidden basis choices and no
opaque solver. Every intermediate object — the [`OperatorSum`](@ref), the
basis, the `SparseMatrixCSC` or [`MatrixFreeOperator`](@ref), the eigensystem, the response
function — is an ordinary Julia value you can inspect, store, and reuse.

MOADyna ships as one package of submodules, each exercised by the test suite:
symbolic [operator algebra](@ref "Operator Algebra"), [conserved-sector bases](@ref "Hilbert Spaces & Bases"),
[exact diagonalization](@ref "Exact Diagonalization"), dynamical [responses](@ref "Responses"),
and [spectroscopy](@ref "Spectroscopy"), together with the multiplet-physics machinery —
Slater–Condon ``F^{k}``/``G^{k}`` Coulomb interactions, spin–orbit coupling,
point-group-projected crystal fields, and tabulated atomic parameters for
transition-metal and lanthanide ions. The quick start below builds and solves a
Hubbard chain; the tutorials cover [Matrix-free Calculations](@ref) and carry a full
``L``-edge multiplet calculation end to end.

```@meta
CurrentModule = MOADyna
DocTestSetup  = quote
    using MOADyna
end
```

## Quick start

```@example quickstart
using MOADyna

# ── Sites and Hilbert space ─────────────────────────────────────────────────
# 4-site spinful Hubbard chain; each FermionSite{2} carries 2 modes (↑, ↓).
const L  = 4
const UP = 1
const DN = 2

sites   = [FermionSite{2}(Symbol("s$i")) for i = 1:L]
hilbert = Hilbert(s.name => s for s in sites)

# ── Hamiltonian (symbolic) ──────────────────────────────────────────────────
t_hop, U = 1.0, 4.0

H_hop = sum(c'(sites[i], σ) * c(sites[i+1], σ)
            for i = 1:L-1, σ in (UP, DN))
H_hop = -t_hop * (H_hop + H_hop')

H_U   = U * sum(n(sites[i], UP) * n(sites[i], DN) for i = 1:L)

H     = H_hop + H_U

# ── Conserved-sector basis ──────────────────────────────────────────────────
# Half-filling (L electrons) and Sz = 0 (equal ↑ and ↓ count).
sz_weights   = repeat([1, -1], L)          # weight +1 on UP modes, −1 on DN modes

restrictions = (
    n_fermion(hilbert) == L,               # half-filling
    WeightedParticleCount(sites, sz_weights) == 0,  # Sz = 0
)

hub_basis = basis(hilbert, restrictions...)
println("Sector dimension: ", length(hub_basis))

# ── Sparse assembly ─────────────────────────────────────────────────────────
H_compiled = compile(H, hub_basis)
H_sparse   = assemble(H_compiled, hub_basis)
println("Sparse size: ", size(H_sparse), "  nnz = ", count(!iszero, H_sparse))

# ── Exact diagonalization ───────────────────────────────────────────────────
# `eigen` accepts either the symbolic `OperatorSum` or, as here, the
# already-assembled sparse matrix together with the basis it was built in.
result = eigen(H_sparse, hub_basis)
println("Ground-state energy: ", round(result.values[1], digits = 6))
nothing  # hide
```

The assembled `H_sparse` is an ordinary sparse matrix you can hand to
`KrylovKit.eigsolve` (or any other Krylov library) for ground-state energies,
spectra, and time evolution.

## Manual

```@contents
Pages = [
    "man/intro.md",
    "man/algebra.md",
    "man/bases.md",
    "man/ed.md",
    "man/responses.md",
    "man/spectroscopy.md",
    "man/multiplets.md",
    "man/pointgroups.md",
    "man/atomic.md",
]
Depth = 2
```

## Tutorials

```@contents
Pages = [
    "tut/getting_started.md",
    "tut/matrixfree.md",
    "tut/nio.md",
]
Depth = 2
```

## Library (API reference)

```@contents
Pages = [
    "lib/algebra.md",
    "lib/bases.md",
    "lib/ed.md",
    "lib/responses.md",
    "lib/spectroscopy.md",
    "lib/shells.md",
    "lib/atomic.md",
    "lib/pointgroups.md",
    "lib/quantyio.md",
    "lib/diagnostics.md",
    "lib/units.md",
    "lib/index_page.md",
]
Depth = 1
```

## Appendix

```@contents
Pages = [
    "app/conventions.md",
    "app/point_groups.md",
    "app/importing_data.md",
    "app/acknowledgments.md",
]
Depth = 2
```

## How the package is organised

MOADyna is one package of submodules in two groups: a **calculation pipeline** that
carries a model from operators to a spectrum, and a **multiplet toolkit** that
builds the operators and parameters the pipeline consumes (plus small utilities).

- **Pipeline:** [Operator Algebra](@ref) → [Hilbert Spaces & Bases](@ref) →
  [Exact Diagonalization](@ref) → [Responses](@ref) → [Spectroscopy](@ref).
- **Toolkit:** [Multiplets & Standard Operators](@ref), [Point Groups](@ref),
  [Atomic Parameters](@ref), and [energy-unit conversions](@ref MOADyna.Units).
- **Utilities:** diagnostics for ground-state inspection and shared HDF5 I/O.

The [Introduction](@ref) walks through this organisation in full.

## Status

MOADyna is tested with unit and end-to-end validation suites — covering operator
algebra, bases, ED, response functions, spectroscopy, and multiplet workflows
against analytic and independent references — that run on every push.
Contributions are welcome — see
[`CONTRIBUTING.md`](https://github.com/CorrelatedSpectra/MOADyna.jl/blob/main/CONTRIBUTING.md).
