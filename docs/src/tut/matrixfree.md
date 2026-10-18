# Matrix-free Calculations

```@meta
CurrentModule = MOADyna
```

Matrix-free operators avoid storing Hamiltonian entries. They use the same
symbolic Hamiltonian, basis, and solver interfaces as the sparse backend. The
basis and solver vectors still occupy memory, and generating entries during
every multiplication can take more time than using a stored sparse matrix.
Use this backend when matrix storage is limiting; keep the sparse default when
it fits and performs well.

This tutorial solves a small Hubbard chain, computes a dynamical response, and
then demonstrates adaptive basis selection. The small model lets us compare
against a complete sparse calculation. The runnable script is
`examples/09_matrixfree.jl` in the repository.

## A fixed physical basis

Build a half-filled four-site open chain with ``S_z=0`` and ``U/t=4``:

```@example mf
using MOADyna
using LinearAlgebra: norm

L, t, U = 4, 1.0, 4.0
sites = [FermionSite{2}(Symbol("s$i")) for i in 1:L]
hilbert = Hilbert(s.name => s for s in sites)
hopping = sum(cdag(sites[i], σ) * c(sites[i+1], σ) for i in 1:L-1, σ in 1:2)
H = -t * (hopping + hopping') + U * sum(n(s, 1) * n(s, 2) for s in sites)
allowed = [n_fermion(hilbert) == L,
           WeightedParticleCount(sites, repeat([1, -1], L)) == 0]
full_basis = basis(hilbert, allowed...)
length(full_basis)
```

Prepare the action and solve for the ground state:

```@example mf
H_action = matrixfree(H, full_basis)
gs = eigen(H_action, full_basis; n=1, tol=1e-10)
psi0, Eg = gs.vectors[:, 1], gs.values[1]
(energy=Eg, residual=norm(H_action * psi0 - Eg * psi0))
```

The equivalent one-call form is
`eigen(H, full_basis; backend=:matrixfree, n=1, tol=1e-10)`.
Preparing `H_action` explicitly lets subsequent calculations reuse it.
`matrixfree` retains every state in the supplied basis and introduces no
coefficient cutoff. The iterative solver's `tol` still controls convergence.

For this small example, compare with the sparse backend:

```@example mf
gs_sparse = eigen(H, full_basis; n=1)
isapprox(Eg, gs_sparse.values[1]; atol=1e-9, rtol=0)
```

Use the same basis object when constructing and solving with an action, and keep
its state data unchanged. For operators between different sectors,
`matrixfree(T, destination_basis, source_basis)` supplies a rectangular action;
see [Matrix-free operators and separate sectors](@ref) for XAS and RIXS.

## Dynamical response

Compute the response of the staggered charge operator
``O=\sum_i(-1)^i(n_{i\uparrow}+n_{i\downarrow})``:

```@example mf
charge = sum((-1)^i * (n(sites[i], 1) + n(sites[i], 2)) for i in 1:L)
ops = [charge]
response = correlator(H_action, full_basis, ops, ops;
                      state=psi0, Eg, Γ=0.3, krylovdim=length(full_basis))
omega = range(0, 12; length=241)
grid = to_grid(response, omega)
intensity = -imag.(grid.data[1, 1, :]) / π
length(intensity)
```

Here `Γ` is the Lorentzian FWHM. The small example allows a Krylov space as large
as the full sector; larger calculations should check convergence with Krylov
dimension at the required broadening. The retained response can be evaluated on
another grid or re-broadened without repeating propagation:

```@example mf
broader = to_grid(response, omega; Γ=0.6)
broader.Γ
```

`xas`, `rixs`, and `fluorescence_yield` also accept matrix-free Hamiltonians. Their
tensor outputs and polarization contractions are unchanged; see [Spectroscopy](@ref).

## Selecting determinants

Matrix-free multiplication alone does not avoid enumerating a basis. When full
enumeration is impractical, [`selected_eigen`](@ref) can start from a small seed
and add determinants according to their contributions to the full residual.

For this model, a Néel configuration supplies a one-determinant seed:

```@example mf
neel_weights = vcat([isodd(i) ? [1, -1] : [-1, 1] for i in 1:L]...)
seed = basis(hilbert, allowed..., WeightedParticleCount(sites, neel_weights) == L)
length(seed)
```

The `restrictions` passed to selection define the allowed physical space. They
omit the extra Néel restriction, so other determinants can be discovered:

```@example mf
selected = selected_eigen(H, seed; restrictions=allowed,
                          max_states=length(full_basis), residual_tol=1e-8)
(energy=selected.eigen.values[1], states=length(selected.basis),
 residual=selected.residual_norm)
```

`max_states` is a resource limit, not a target dimension. We allow the complete
36-state sector here to compare with the reference; a larger calculation can
set this limit without constructing the full basis. This model may need most or
all determinants, so selection need not save time or memory.

Each inner solve uses a fixed linear operator. The outer loop retains selected
states and adds more until the residual, including omitted states, meets
`residual_tol`. The returned vectors use **`selected.basis`**, not `seed`.

```@example mf
isapprox(selected.eigen.values[1], Eg; atol=1e-8, rtol=0)
```

A small residual certifies an approximate eigenstate, but cannot discover an
unseeded disconnected sector or prove that no lower state exists there. Choose
seeds that cover the relevant sectors and check stability when enlarging them.
Reaching a state, candidate, or iteration limit raises
[`SelectionConvergenceError`](@ref) with the latest result.

Ground-state convergence does not certify a spectrum on that selected basis.
Transitions may require other determinants. Use [`selected_response`](@ref) to
converge the response space separately at the chosen frequencies and broadening;
see [Selecting a response space](@ref). The response above deliberately uses the
complete physical sector.

## Threading

Start Julia with several worker threads, for example:

```sh
julia --project=. --threads=4,0 examples/09_matrixfree.jl
```

The default `matrixfree(H, full_basis; threaded=:auto)` uses the available Julia
threads. Set `threaded=false` for serial products or `threaded=true` to request
parallel products. A solver or spectroscopy call can override the constructor's
choice for that calculation:

```julia
gs_serial = eigen(H_action, full_basis; n=1, threaded=false)
gs_parallel = eigen(H_action, full_basis; n=1, threaded=true)
```

These settings control operator multiplication; they do not create Julia
threads or change BLAS thread counts. Tiny calculations such as this tutorial
can be faster in serial. See [Multiplication threads](@ref) for details.
