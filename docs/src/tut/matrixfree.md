# Matrix-free Calculations

```@meta
CurrentModule = MOADyna
```

This tutorial calculates the ground state and charge spectrum of a Hubbard chain
using the matrix-free backend, then introduces adaptive basis selection. The
complete script is `examples/09_matrixfree.jl`.

For routine spectroscopy, the default sparse backend is usually the best choice.
Matrix-free storage is useful for larger systems when storing the Hamiltonian
becomes limiting; it generates matrix elements during multiplication.

## Hubbard chain

We use a four-site open chain at half filling, with equal numbers of spin-up and
spin-down electrons and an interaction strength of ``U/t=4``:

```@example mf
using MOADyna

L, t, U = 4, 1.0, 4.0
sites = [FermionSite{2}(Symbol("s$i")) for i in 1:L]
hilbert = Hilbert(s.name => s for s in sites)
hopping = sum(cdag(sites[i], σ) * c(sites[i+1], σ) for i in 1:L-1, σ in 1:2)
H = -t * (hopping + hopping') + U * sum(n(s, 1) * n(s, 2) for s in sites)
sector = [n_fermion(hilbert) == L,
          WeightedParticleCount(sites, repeat([1, -1], L)) == 0]
```

The restrictions in `sector` select determinants with four electrons and
``S_z=0``. This sector contains 36 determinants.

## Ground state

Pass the Hamiltonian, Hilbert space, and sector restrictions to `eigen`:

```@example mf
gs = eigen(H, hilbert; restrictions=sector, backend=:matrixfree, n=1)
(energy=gs.values[1], states=length(gs.basis))
```

`eigen` constructs the sector basis and finds its lowest-energy state. The energy
is approximately ``-1.95315t``. The eigenvector is `gs.vectors[:, 1]`, with
coordinates given by `gs.basis`.

The basis stays fixed throughout this calculation. Both sparse and matrix-free
operators act within the supplied basis; see [Choosing a representation](@ref)
for details. Omit `backend=:matrixfree` to use sparse storage. For a multiplet
model, a `ShellModel` can be passed in place of `hilbert`.

## Charge spectrum

The staggered charge operator probes charge fluctuations between alternating
sites:

```@example mf
charge = sum((-1)^i * (n(sites[i], 1) + n(sites[i], 2)) for i in 1:L)
omega = range(0, 12; length=241)
spec = xas(H, charge, gs; restrictions=sector,
           omega_grid=omega, Gamma=0.3, krylovdim=36)
intensity = -imag.(spec.tensor) / π
length(intensity)
```

The `xas` interface evaluates the absorption response of the supplied operator.
It constructs the propagation basis and transition action from `sector` and
`gs`. Here we use all 36 sector states as the maximum Krylov dimension;
`Gamma=0.3` sets the Lorentzian FWHM.

The result retains its Lanczos data. To broaden the spectrum further:

```@example mf
broader = re_broaden(spec; Γ=0.6)
length(broader.tensor)
```

For core-level absorption, use dipole operators and the core-hole sector's
restrictions. The same workflow extends to `rixs` and `fluorescence_yield`, with
separate intermediate and final sectors; see [Spectroscopy from eigenstates](@ref).

## Adaptive basis selection

For a larger system, the ground state can be found by growing a basis from a
small initial sample of determinants. Enable this with `adaptive=true`:

```@example mf
selected = eigen(H, hilbert; restrictions=sector, backend=:matrixfree,
                 adaptive=true, n=1, tol=1e-8)
(energy=selected.values[1], states=length(selected.basis),
 residual=selected.selection.residual_norm)
```

The solver samples an initial basis within `sector`, solves for its lowest-energy
state, and evaluates the residual including omitted determinants. It then adds
states until the full residual reaches `tol`. This growth of the determinant
basis is what *adaptive* means. The resulting eigenvector uses `selected.basis`.

For this chain, selection recovers the complete 36-state sector and the same
ground-state energy. In larger calculations, check convergence by tightening
`tol` and increasing `seed_size`. The [Selecting determinants](@ref) section
covers seed choice, symmetry sectors, and resource limits.

To calculate a spectrum from `selected`, pass it in place of `gs` above. The
spectroscopy call constructs the complete response sector from its
`restrictions`. Response basis selection is available separately through
[`selected_response`](@ref).

## Threading

Start Julia with worker threads to use parallel matrix-free products:

```sh
julia --project=. --threads=4,0 examples/09_matrixfree.jl
```

`threaded=false` selects serial products, while `threaded=true` requests parallel
products using the available workers. The default is `threaded=:auto`.
See [Multiplication threads](@ref) for the threading options.
