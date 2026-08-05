# Changelog

A high-level summary of what each release adds. Dates are release (tag)
dates.

## v0.3.0 — 2026-08-05

**Added**

- **`MOAD.Gradients`** — the T=0 differentiable XAS forward model: affine
  Hamiltonian models, ground-state derivatives, resolvent JVP, spectrum
  VJP/pullback over a degeneracy-clustered spectral measure, and a
  deterministic direct-fit baseline (`Optim` weakdep extension). See the
  full changelog in the repository for details.

**Fixed**

- **Crystal-field expansion now represents the true CF subspace per irrep.**
  [`expand_clm`](@ref MOAD.PointGroups.expand_clm) previously parametrized each multiplicity block as a
  real-symmetric `m_Γ × m_Γ` matrix embedded as `H_Γ ⊗ I_{d_Γ}`. That silently
  mis-represented every point group whose multiplicity block carries an imaginary
  (e.g. trigonal `A_{4,±3}`) coupling — D3d, C3, C3v, D3, S6 (ℓ=2,3), C5, C5v, D5,
  D5d (ℓ=3), and C1/Ci — and the block overload could emit symmetry-forbidden
  `A_{km}` behind a non-fatal warning. Each irrep block is now parametrized on its
  **CF-reachable Hermitian basis** over the full isotypic block (built from the
  group-symmetrised `C^k_m`), the residual check is a fatal scale-normalized error,
  and non-CF blocks are rejected. D3d ℓ=2 is validated bit-exact against Quanty
  D3dB/C. Real-coupling groups (C4v, D4h, D2h, O, Oh, Td) are byte-identical.
  **Breaking:** the flat-vector parameter *meaning* changed for the affected groups
  (the parameter *count* is unchanged). Results are at MOAD's single canonical
  orientation; multi-setting support is deferred.

## v0.2.1 — 2026-05-31

**Finite-temperature closeout + energy-unit helper.**

- **Finite temperature across the spectroscopy layer** (`k_B = 1`).
  [`xas`](@ref), [`rixs`](@ref) and [`fluorescence_yield`](@ref) take a
  `temperature` kwarg alongside an `Eigen` of the initial multiplet and return
  the Boltzmann ensemble average over the thermally populated initial states,
  `A_T(ω) = Σ_m ρ_m A_m(ω)` (worked end to end in the NiO tutorial).
  [`optical_conductivity`](@ref) / [`dynamical_structure_factor`](@ref) take the
  same `temperature` kwarg (`T` accepted as an alias). [`correlator`](@ref)
  computes the finite-T single-particle Green's function
  (`:addition`/`:removal`/`:both`) from a caller-supplied initial ensemble
  (`ensemble_states` + `ensemble_energies`).
- **[`kubo_response`](@ref)** — the full two-sided Kubo retarded susceptibility
  ``\chi_{AB}(\omega,T) = \sum_{m,n}(w_m-w_n)\,\langle m|A|n\rangle\langle n|B|m\rangle/(\omega+i\Gamma/2-(E_n-E_m))``
  from a direct eigenstate-pair Lehmann sum (cross-correlators allowed).
- **[`MOAD.Units`](@ref)** — a dependency-free energy-unit converter
  [`convert_energy`](@ref)`(x, :from => :to)` over `:J`/`:eV`/`:meV`/`:K`/
  `:invcm`/`:THz`/`:Ry`/`:Ha` (exact post-2019-SI factors; Rydberg CODATA-2018),
  plus `Units.kB_eV`. Turns a Kelvin temperature into the eV-based `temperature`
  kwarg without hard-coding a Boltzmann constant.

## v0.2.0 — 2026-05-30

**Bosons, the Responses layer, and spectroscopy completion.**

- **Bosonic infrastructure** — `BosonSite`, bosonic occupation encoding, and
  Bose–Hubbard / Hubbard–Holstein validation.
- **`MOAD.Responses` layer** — a matrix-of-ω response object
  `C_{ij}(\omega) = \langle\psi_0| A_i^\dagger\, G(\omega)\, B_j |\psi_0\rangle`
  with the `correlator(...)` constructor and three interchangeable
  representations (`LanczosResponse`, `PoleResponse`, `GridResponse`) plus the
  parametric `GreensFunction{T,R}`. Spectroscopy is now a consumer of this
  layer.
- **Single-particle Green's functions** — `correlator(...; channel =
  :addition / :removal / :both)` for the `N±1` photoemission / inverse-PES
  spectral functions.
- **Finite temperature (neutral, same-sector)** — a built-in `T` keyword on
  [`optical_conductivity`](@ref) and [`dynamical_structure_factor`](@ref) gives
  the `:neutral` same-sector Boltzmann average; [`correlator`](@ref) computes the
  finite-T neutral thermal correlator. (The finite-T closeout — core-level
  spectra, charged channels, and full Kubo χ — landed in v0.2.1.)
- **Response functions** — [`optical_conductivity`](@ref) (regular
  ``\mathrm{Re}\,\sigma_{\alpha\beta}(\omega>0)``) and
  [`dynamical_structure_factor`](@ref) (``S(\mathbf q,\omega)``), thin
  bring-your-own-operator wrappers over `correlator`.
- **Multipole / transition operators** — [`multipole`](@ref) (rank-``k``
  Wigner–Eckart engine), [`quadrupole_transition`](@ref) (E2),
  [`quadrupole`](@ref) (charge moment), [`nixs`](@ref) (``e^{i\mathbf
  q\cdot\mathbf r}`` scattering) with [`radial_integral`](@ref), and
  [`spin_dipole_T`](@ref); [`dipole`](@ref) refactored onto the engine.
- **Polarisation** — `xas`/`rixs` return the full Cartesian response tensor;
  [`polarise`](@ref) contracts it for any incoming/outgoing polarisation
  (linear, circular, XMLD), demonstrated end to end in the NiO tutorial.
- **Spectrum post-processing toolkit** exposed: [`polarise`](@ref),
  [`re_broaden`](@ref) / [`re_broaden_table`](@ref) (re-broaden a stored
  Lanczos spectrum with no recomputation, incl. energy-dependent widths),
  [`average`](@ref) / [`weighted_sum`](@ref), [`restrict_to_window`](@ref),
  and HDF5 [`save_spectra`](@ref) / [`load_spectra`](@ref).
- **Atomic radial functions from Cowan** — [`radial_wavefunction`](@ref)
  extracts Hartree–Fock ``P_{n\ell}(r)`` from a live Cowan RCN run, so the
  nIXS radial Bessel moments need no external data file.
- **Point-group classification of many-body states** —
  [`classify_state`](@ref)`(ψ, basis, m, G)` returns the spatial irrep of an
  ED eigenstate (e.g. the NiO ``d^8`` ground state as ``{}^3A_{2g}``).
- **`ConservedQuantity` renamed to [`QuantumNumber`](@ref
  MOAD.Algebra.QuantumNumber)** (alias `QN`).
- **Documentation** — full manual + tutorials site: a worked NiO XAS / RIXS /
  XMLD / nIXS tutorial, a Manual point-group chapter, and per-symbol Library
  reference. Every numerical result in the docs is computed live at build.

## v0.1.0 — 2026-05-08

**First release: atomic multiplets + spectroscopy.** Exact diagonalization,
the symbolic operator algebra and conserved-sector bases, multiplet operators,
the point-group engine (character tables, subduction, crystal fields), and
NiO ``L_{2,3}`` XAS / RIXS validated against reference calculations.
