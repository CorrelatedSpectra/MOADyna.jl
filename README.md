<p align="center">
  <img src="assets/logo.svg" width="420" alt="MOADyna — Many-body Operators, Algebra, and Dynamics">
</p>

# MOADyna.jl

[![CI](https://github.com/CorrelatedSpectra/MOADyna.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CorrelatedSpectra/MOADyna.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://correlatedspectra.github.io/MOADyna.jl/stable)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21808796.svg)](https://doi.org/10.5281/zenodo.21808796)
![Julia: 1.10+](https://img.shields.io/badge/julia-1.10%2B-9558B2)

**Many-body Operators, Algebra, and Dynamics** — a Julia framework for
exact diagonalization of finite quantum many-body systems and the dynamical
response functions built on top of it. You write a Hamiltonian as a symbolic
operator expression; MOADyna enumerates the conserved-sector Hilbert space, assembles
a sparse matrix, diagonalizes it, and evaluates correlation functions and
spectra — lattice models and atomic multiplet problems through one operator
language.

**Background.** MOADyna grew out of many years of working with
[Quanty](https://www.quanty.org/), and owes a real intellectual debt to Maurits
Haverkort — his code and ideas shaped how the author thinks about multiplet and
core-level spectroscopy. MOADyna is an independent Julia project with its own design
goals: explicit, inspectable operator algebra; programmable exact-diagonalization
workflows; and a path toward parameter estimation / inverse problems, lattice
methods, and time-dependent dynamics. `MOADyna.QuantyIO` and the [Coming from
Quanty](https://correlatedspectra.github.io/MOADyna.jl/stable/app/from_quanty/) appendix keep practical interoperability
available where it helps.

## What it does

- **Symbolic second-quantized operators.** Fermions, bosons, and spins build by
  arithmetic: `cdag(s,1)*c(s,2)`, `n(s,1)*n(s,2)`, `S(s1)⋅S(s2)`. Operators are
  inspectable Julia values, compiled and assembled onto a basis on demand.
- **Conserved-sector bases.** Particle number, total `Sz`, and weighted particle
  counts are integer/rational valued, so sector enumeration is exact — no state
  is missed to floating-point rounding. Bit-packed states with parallel
  enumeration and restriction-aware (suffix-pruned) walks.
- **Exact diagonalization.** `eigen(H, basis; n, which)` dispatches between dense
  LAPACK and Krylov (KrylovKit) automatically; the assembled `SparseMatrixCSC` is
  also usable directly with your own solver.
- **Dynamical correlation functions.** A `correlator` builds matrix-of-ω response
  objects `⟨ψ|A†G(ω)B|ψ⟩` in interchangeable Lanczos / pole / grid
  representations, including single-particle Green's functions (`:addition` /
  `:removal` / `:both`) and finite-temperature thermal averages.
- **Core-level spectroscopy.** `xas`, `rixs`, and `fluorescence_yield` return
  `SpectraTensor` objects that retain their Lanczos data — re-broaden, slice,
  contract to any polarization, save/load HDF5, plot via an optional `Plots.jl`
  recipe. Optical conductivity, dynamical structure factor, and the full Kubo
  susceptibility (`kubo_response`) are also provided.
- **Atomic multiplets.** Slater–Condon Coulomb (single- and cross-shell),
  one-body spin–orbit (`LS`), irrep-projected crystal field (`Akm`) and
  hybridization (`hop`), and Wigner–Eckart transition operators (`dipole`,
  `multipole`, `quadrupole`, `nixs`) — built from the rank-k engine.
- **Point groups.** 42 groups with character tables, subduction, crystal-field
  expansion, and classification of many-body eigenstates by spatial irrep.
- **Differentiable forward model.** `MOADyna.Gradients` makes the map from physical
  parameters (Slater reductions, Δ, 10Dq, hybridization, ζ) to a T=0 XAS spectrum
  differentiable — forward-mode with an analytic resolvent, exposing a
  VJP/pullback over a degeneracy-clustered spectral measure, plus a deterministic
  direct-fit baseline.

## Installation

```julia
using Pkg
Pkg.add("MOADyna")
```

Requires Julia 1.10 or newer. See [`CHANGELOG.md`](CHANGELOG.md) for what each
release changed.

## Quick start

A half-filled 4-site Hubbard chain: build the Hamiltonian, restrict to a
conserved sector, diagonalize.

```julia
using MOADyna

L, t, U = 4, 1.0, 4.0
sites   = [FermionSite{2}(Symbol("s$i")) for i in 1:L]   # modes 1,2 = ↑,↓
hilbert = Hilbert(s.name => s for s in sites)

hop_term = sum(c'(sites[i], σ) * c(sites[i+1], σ) for i in 1:L-1, σ in (1, 2))
H        = -t * (hop_term + hop_term') + U * sum(n(sites[i], 1) * n(sites[i], 2) for i in 1:L)

# Half-filling, Sz = 0 (weights +1/−1 on the ↑/↓ modes give 2·Sz_total).
hub_basis = EagerBasis(hilbert,
                       n_fermion(hilbert) == L,
                       WeightedParticleCount(sites, repeat([1, -1], L)) == 0)

E = eigen(H, hub_basis; n = 4)        # 4 lowest states; or assemble() for your own solver
```

## A complete NiO L₂,₃ XAS calculation

Full d-shell + 2p core + ligand band, Slater–Condon Coulomb, atomic spin–orbit,
octahedral crystal field, eg/t2g hybridization, ground state, and the L₂,₃ XAS
spectrum:

```julia
using MOADyna

m     = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
p_gs  = atomic_parameters(:Ni, "3d8";     scaling = :scaled_80) # Ni²⁺ ground
p_xas = atomic_parameters(:Ni, "2p5 3d9"; scaling = :scaled_80) # 2p core hole

# ZSA onsite energies — solved from configuration anchors (Udd=7.3, Upd=8.5, Δ=4.7).
es_gs  = onsite_energies(m; shells = (:Ni_3d, :L_3d), U = (Ni_3d = 7.3,),
    anchors = [(Ni_3d=8, L_3d=10) => 0.0, (Ni_3d=9, L_3d=9) => 4.7])
es_xas = onsite_energies(m; U = (Ni_3d = 7.3,), pairs = ((:Ni_2p, :Ni_3d) => 8.5,),
    anchors = [(Ni_2p=6, Ni_3d=8, L_3d=10) => 0.0,
               (Ni_2p=6, Ni_3d=9, L_3d=9)  => 4.7,
               (Ni_2p=5, Ni_3d=9, L_3d=10) => 0.0])

TenDq_Ni, TenDq_L = 0.56, 1.44      # 10Dq cubic crystal-field splittings (eV)
V_eg, V_t2g       = 2.06, 1.21      # Ni 3d–O 2p hybridization, per irrep (eV)

H_common = coulomb(m, :Ni_3d; U=7.3, F=p_gs.Fdd) + p_gs.zeta.d * LS(m, :Ni_3d) +
           TenDq_Ni * Akm(m, :Ni_3d, :Oh, [0.6, -0.4]) +
           TenDq_L  * Akm(m, :L_3d,  :Oh, [0.6, -0.4]) +
           V_eg  * hop(m, :Ni_3d, :L_3d, :Oh; irrep=:Eg) +
           V_t2g * hop(m, :Ni_3d, :L_3d, :Oh; irrep=:T2g)

H_GS  = H_common + es_gs.Ni_3d * n(m, :Ni_3d) + es_gs.L_3d * n(m, :L_3d)
H_XAS = H_common + coulomb(m, :Ni_2p, :Ni_3d; U=8.5, F=p_xas.Fpd, G=p_xas.Gpd) +
        p_xas.zeta.p * LS(m, :Ni_2p) +
        es_xas.Ni_2p * n(m, :Ni_2p) + es_xas.Ni_3d * n(m, :Ni_3d) + es_xas.L_3d * n(m, :L_3d)

basis_gs  = basis(m, nshells(m, :Ni_2p) == 6,
                     nshells(m, :Ni_3d) + nshells(m, :L_3d) == 18)         # 190 states
basis_xas = basis(m, nshells(m, :Ni_2p) in 5:6, total(m) == 24)            # 310 states

gs_eig = eigen(H_GS, basis_gs; n = 1)
psi0   = embed(gs_eig.vectors[:, 1], basis_gs => basis_xas)

spec_x = xas(assemble(compile(H_XAS, basis_xas), basis_xas), basis_xas,
             dipole(m, :Ni_2p => :Ni_3d)[1], psi0;
             omega_grid = range(-15.0, 25.0; length = 801),
             Gamma = 0.6, Eg = gs_eig.values[1])
```

A compact block of MOADyna primitives produces a tensor-valued spectrum ready to
broaden, polarize, or write to disk. The full three-polarization version, with the
canonical NiO parameter set, is in
[`examples/05_nio_xas.jl`](examples/05_nio_xas.jl), and the same physics is walked
through step by step in the [NiO tutorial](https://correlatedspectra.github.io/MOADyna.jl/stable/tut/nio/).

## More examples

The [`examples/`](examples/) directory contains runnable scripts:

- 4-site Hubbard chain (operator algebra)
- Spin-S Heisenberg chain (any S, same syntax)
- Bose–Hubbard with occupation cutoff
- Hubbard–Holstein (electron–phonon coupling)
- **NiO L₂,₃ XAS** — multiplet calculation, three polarizations
- **NiO L₃ RIXS** — full (ω_in, ω_out) intensity map
- **NiO L₂,₃ XAS (compact)** — same calculation on a `n_2p = 5`-only basis
- **NiO 3d nIXS** — non-resonant inelastic X-ray scattering multipoles

## Documentation

**📖 [Read the documentation](https://correlatedspectra.github.io/MOADyna.jl/stable)** — manual, tutorials, and the full API
reference.

Useful entry points:

- [Getting started](https://correlatedspectra.github.io/MOADyna.jl/stable/tut/getting_started/) — installation and first calculation
- [NiO: XAS, RIXS, and nIXS](https://correlatedspectra.github.io/MOADyna.jl/stable/tut/nio/) — a complete worked multiplet calculation
- [Manual](https://correlatedspectra.github.io/MOADyna.jl/stable/man/intro/) — operator algebra, bases, ED, responses, spectroscopy
- [Library reference](https://correlatedspectra.github.io/MOADyna.jl/stable/lib/algebra/) — every exported function
- [Coming from Quanty](https://correlatedspectra.github.io/MOADyna.jl/stable/app/from_quanty/) — command mapping and interop notes

The docs are built from [`docs/src/`](docs/src/) with Documenter.jl. Cross-code
validation reports and reference data live under
[`docs/dev/validation/`](docs/dev/validation/).

## Interoperability

`MOADyna.QuantyIO` reads [Quanty](https://www.quanty.org/) text dumps (operators and
related data), so an external operator can be imported as a native
[`OperatorSum`](https://correlatedspectra.github.io/MOADyna.jl/stable/lib/algebra/):

```julia
hilbert  = Hilbert(:s => FermionSite{10}(:s))
mode_map = i -> (:s, i + 1)           # external 0-indexed → MOADyna 1-indexed
H        = read_quanty_operator("hamiltonian_dump.txt", hilbert, mode_map)
```

For Quanty-to-MOADyna command mapping and interop notes, see [Coming from
Quanty](https://correlatedspectra.github.io/MOADyna.jl/stable/app/from_quanty/).

## Citing

**Citing the software.** Please cite the version you used. Machine-readable
metadata is in [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this
repository" button from it).

Each tagged release is archived on Zenodo with its own DOI:

| | DOI |
|---|---|
| All versions (resolves to the latest) | [10.5281/zenodo.21808796](https://doi.org/10.5281/zenodo.21808796) |
| v0.3.1 | [10.5281/zenodo.21881201](https://doi.org/10.5281/zenodo.21881201) |
| v0.3.0 (as `MOAD`) | [10.5281/zenodo.21808797](https://doi.org/10.5281/zenodo.21808797) |

Cite the **version-specific** DOI for reproducibility — it pins the exact code
a calculation used.

**Methods paper.** A separate paper describing the methods is planned. It is
not the same citation as the software archive, and this section will be updated
when it appears.

## AI assistance

MOADyna was developed with substantial assistance from a blend of LLM-based
tools, used for both implementation and code review. I directed the design,
reviewed the resulting code by hand, and am responsible for the package.
Correctness is checked by the test suite in CI and by cross-code
numerical validation against [Quanty](https://www.quanty.org/) and
[QuSpin](https://quspin.github.io/QuSpin/) — see
[`docs/dev/validation/`](docs/dev/validation/).

## Acknowledgments

MOADyna builds on the Julia numerical ecosystem — notably KrylovKit.jl (eigensolvers)
and WignerSymbols.jl (angular-momentum coupling), plus HDF5.jl, OhMyThreads.jl,
StaticArrays.jl, and DataStructures.jl. See
[Acknowledgments](https://correlatedspectra.github.io/MOADyna.jl/stable/app/acknowledgments/) for the full list and citations.

## License

MIT — see [`LICENSE`](LICENSE).

A small amount of third-party material is redistributed here under its own
terms — a Quanty tutorial script and the NiO reference spectra, both CC-BY.
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), which also records the
typeface credit for the logo and the sources of the atomic parameter tables.

## Contributing

Issues and pull requests are welcome. For anything larger than a bug
fix, please open an issue first to discuss the design.

## Roadmap

MOADyna is under active development toward a stable v1.0 API; pre-1.0
minor releases may contain breaking changes (semver 0.x). See
[`CHANGELOG.md`](CHANGELOG.md) for what each release added. Planned
directions — matrix-free operator application, symmetry-adapted bases,
lattice-cluster methods — land as they mature, without version or
date commitments.
