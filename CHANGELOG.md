# Changelog

All notable changes to MOAD are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-08-05

Differentiable forward model; crystal-field expansion correctness fix.
First public release.

### Added

- **`MOAD.Gradients` — T=0 differentiable XAS forward model.**
  The `(physical parameters → spectrum)` map made
  differentiable in θ, forward-mode + analytic-resolvent (no reverse-AD through the
  eigensolver). Built and finite-difference-validated in seven steps:
  - **Affine Hamiltonian model** `H(θ) = Σ cᵢ(θ)·Mᵢ` with preassembled term matrices
    and a coefficient map owning `∂cᵢ/∂θ` (`AffineModel`/`AffineMap`); a ZSA onsite
    solve (`zsa_onsite`). The symbolic builders are never re-run inside the
    differentiated path.
  - **Ground-state derivatives** — Hellmann–Feynman `∂E0/∂θ` (`grad_E0`) and the
    projected Sternheimer response `dψ0` (`sternheimer_dψ0`), with a non-degeneracy
    gate (`groundstate`).
  - **Resolvent JVP** for `C(ω)=X†G X` at fixed source, and the **full coupled XAS
    JVP** (`xas_response_C`/`xas_response_jvp`) threading `dE0`, `dψ0→dX` and `dH_f`
    via a two-solve (conjugate-shift left solve) scheme.
  - **Degeneracy gate + sector-aware ground state** (`groundstate_manifold`,
    `manifold_gate`, `classify_groundstate`) — a manifold/splitting gate on the bare
    matrices plus an opt-in spatial-irrep + ⟨S²⟩ labeling of the ground and competing
    low-lying sectors.
  - **Spectrum VJP/pullback** (`XASGradientModel`, `spectrum`, `spectrum_and_jvp`,
    `jacobian`, `spectrum_with_pullback`) for `S = −Im C/π` (cotangent `W = −iλ/π`),
    plus a **degeneracy-clustered spectral measure** over frozen windows
    (`freeze_windows`, `spectral_clusters`).
  - **Deterministic direct-fit baseline** `fit_spectrum` (an `Optim` weakdep
    extension, `MOADOptimExt`), validated by synthetic-truth θ recovery and a
    forward-fidelity check that reproduces the physical `xas`/Lanczos path (including
    a Ni d⁸→2p⁵d⁹ L-edge multiplet through a real ground→core-hole embedding).
  Parameter inference (the observation model, NPE/HMC/SBC/OOD) lives in a separate
  sibling package that depends on MOAD; MOAD never hard-depends on the ML stack.

### Fixed

- **Crystal-field expansion now spans the true CF subspace per irrep.**
  `expand_clm` previously parametrized each multiplicity block as a
  real-symmetric matrix embedded as `H_Γ ⊗ I_{d_Γ}`, which cannot represent
  the imaginary (e.g. trigonal `A_{4,±3}`) couplings of D3d, C3, C3v, D3, S6
  (ℓ=2,3), the pentagonal groups (ℓ=3), and C1/Ci; the block overload could
  emit symmetry-forbidden `A_km` behind a non-fatal warning. Each irrep block
  is now parametrized on its CF-reachable Hermitian basis over the full
  isotypic block, the residual check is a fatal error, and non-CF blocks are
  rejected. D3d ℓ=2 is validated bit-exact against Quanty; real-coupling
  groups (C4v, D4h, D2h, O, Oh, Td) are byte-identical. **Breaking:** the
  flat-vector parameter *meaning* changed for the affected groups (the
  parameter *count* is unchanged).

## [0.2.1] — 2026-05-31

Finite-temperature closeout and units.

### Added

- **Finite-T spectroscopy.** A `temperature=` kwarg on `xas`, `rixs`, and
  `fluorescence_yield` — Boltzmann ensemble over the thermally populated
  initial states.
- **Finite-T charged-channel Green's function** via `correlator` with a
  caller-supplied initial ensemble (`ensemble_states` / `ensemble_energies`).
- **`kubo_response`** — the full two-sided Kubo retarded χ from a dense
  eigenstate-pair Lehmann sum.
- **`MOAD.Units`** (`convert_energy`, `Units.kB_eV`) — dependency-free
  energy-unit conversion (exact post-2019 SI constants).

### Changed

- `optical_conductivity` / `dynamical_structure_factor` adopt `temperature=`
  as the canonical kwarg, with `T=` accepted as an alias.

## [0.2.0] — 2026-05-30

Bosons, a foundational responses layer, spectroscopy completion, and the
standard-operator surface — plus the first documentation site. MOAD now
covers the multiplet-spectroscopy bar end to end.

### Added

- **Bosons & electron-phonon (2a)** — `BosonSite`, bosonic ladder operators,
  Bose-Hubbard and Hubbard-Holstein models validated against QuSpin.
- **`MOAD.Responses` layer (2b/2c)** — a matrix-of-ω response abstraction
  (`AbstractResponse` with `LanczosResponse` / `PoleResponse` / `GridResponse`
  representations and a `GreensFunction` wrapper) computed by a block-Lanczos /
  continued-fraction kernel, with the single `correlator(H, basis, As, Bs; …)`
  constructor underlying every spectrum. Channels `:neutral` (default),
  `:addition` / `:removal` / `:both` single-particle Green's functions
  (Hubbard-dimer validation to 1e-10); `to_pole` / `to_grid` conversions; HDF5
  round-trip.
- **Finite-T neutral response (2d)** — Boltzmann-weighted, degeneracy-complete
  thermal average for the same-sector neutral correlator.
- **σ(ω) and S(q,ω) wrappers (2e)** — `optical_conductivity` (regular
  `Re σ_αβ(ω>0)`) and `dynamical_structure_factor` (`S(q,ω)`), one-sided
  bring-your-own-operator wrappers over `correlator(:neutral)`; dense-Lehmann
  validated to ~1e-14.
- **Standard / multipole transition operators (2h)** — generalized the rank-1
  Wigner-Eckart `dipole` to a rank-k engine `multipole`, plus
  `quadrupole_transition` (E2), `quadrupole` (charge-quadrupole moment),
  `nixs` (the `e^{iq·r}` non-resonant scattering operator) with `radial_integral`
  (Bessel / `⟨rᵏ⟩` radial moments), and `spin_dipole_T` (the XMCD spin-dipole
  tensor). Coefficients validated bit-for-bit against Quanty `SlaterCoefficientC`
  / `CreateOperatorQ*` / `CreateOperatorT*` (and the TXAS dumps); the nIXS
  conjugation is locked by the Legendre addition theorem. Worked example
  `examples/08_nio_nixs.jl`.
- **Documentation site** — a Documenter.jl site (Manual + Tutorials +
  auto-generated Library + Appendix) covering the operator algebra, bases, ED,
  responses, spectroscopy, multiplets and atomic parameters, with a Getting
  Started walkthrough and the flagship NiO L-edge XAS / RIXS / nIXS tutorial.
  Every example is executed at build time; built locally via
  `julia --project=docs docs/make.jl`.

### Changed

- `dipole` is now the `k = 1` special case of the `multipole` engine
  (bit-for-bit identical; the Quanty TXAS regression matches to `err = 0`).

### Notes

- Restriction-resolved partial excitations (2f) are subsumed by the existing
  `restrictions` machinery — no separate API. Quantum-information utilities are
  deferred to a future release. The full Kubo retarded χ/σ, finite-T charged-channel GF,
  and finite-T XAS/RIXS remain deferred to a follow-on sub-phase.

## [0.1.0] — 2026-05-08

First feature-complete release. MOAD covers operator algebra → Hilbert
construction → exact diagonalization → core-level spectroscopy → point
groups → atomic multiplet primitives → atomic Slater-Condon parameters,
in seven internal layers. The NiO L_{2,3} XAS native acceptance test
reproduces the PyQuanty reference at 1.5×10⁻¹² of peak across three
Cartesian polarisations.

### Added — by layer

- **`MOAD.Algebra`** — symbolic `OperatorSum` with id-keyed canonical
  storage; fermion / boson / spin sites; Tier-2 canonicalization on
  construction (normal-ordering, Pauli-zero collapse, anticommutator
  constants); conserved-quantity observables (`ParticleCount`,
  `TotalSz`, `WeightedParticleCount`) and bound `Restriction`s;
  `rotate(op, h_in, U; h_out, project=false)` Sakurai-style basis
  change (square unitary) / projection (rectangular isometry);
  HDF5 operator I/O (`save_operator` / `load_operator`).
- **`MOAD.Bases`** — `EagerBasis` with bit-packed states, partitioned
  mixed-radix enumeration, k-combinations + suffix-pruned weighted-sum
  walks; `compile` + `assemble` produce
  `SparseMatrixCSC{T, Int32}` directly (auto-fallback to `Int`);
  `apply_restriction!(v, R, basis)` projector; cross-basis
  `embed(psi, basis_a => basis_b)`.
- **`MOAD.ED`** — single `eigen(H, basis; n, which, …)` extending
  `Base.eigen`; auto-switches dense LAPACK vs `KrylovKit.eigsolve` at
  `dense_below = 1024`; cluster-scoped QR for degenerate Krylov blocks;
  `ConvergenceError` on non-convergence; Hermiticity check upfront;
  HDF5 eigensystem I/O (`save_eigensystem` / `load_eigensystem`).
- **`MOAD.Spectroscopy`** — `xas`, `rixs`, `fluorescence_yield`;
  `SpectraTensor{S, T, N, F}` result type with stored Lanczos data;
  hand-rolled block Lanczos with rectangular `R` from rank-revealing
  QR + ragged-block deflation; helpers `re_broaden`,
  `re_broaden_table`, `polarise`, `poles`, `find_chunk`, `tridiagonal`,
  `restrict_to_window`, `average`, `weighted_sum`, `plot_range`,
  spectrum algebra; ASCII + HDF5 round-trip via `save_spectra` /
  `load_spectra`; `DEFAULTS` mutable struct with Greek↔ASCII alias
  support; opt-in Plots.jl recipe via `ext/MOADPlotsExt.jl`
  (RecipesBase weakdep).
- **`MOAD.PointGroups`** — 42 supported groups (32 crystallographic +
  10 molecular incl. `I, Ih`); Mulliken-label character tables for
  every group; setting variant `:y` for D3h/D3d/C3v/D6h via 30°/15°
  rotation around z; `expand_clm` / `expand_clm_central` /
  `nparams` produce the multiplicity-aware Akm expansion via
  matrix-unit intertwiner; production-mode strict gate via
  `IRrep.provenance`; `character_table`, `print_character_table`,
  `character_table_compare`, `Base.show MIME"text/plain"` pretty-print.
- **`MOAD.Shells`** — `ShellModel(tags)` registry parses
  `<atom>_<n><orbital>` to allocate `2(2ℓ+1)` fermionic modes per shell
  in m-major + dn-then-up order (one `FermionSite` per shell);
  accessors `ell_of`, `range_of`, `site_of`; shell-keyed dispatch
  wrappers for `n`, `Sx`, `Sy`, `Sz`, `Splus`, `Sminus`, `Ssqr`;
  multiplet primitives `Lx, Ly, Lz, Lplus, Lminus, Lsqr,
  Jx, Jy, Jz, Jplus, Jminus, Jsqr, LS` (all one-body — `LS` built
  from single-particle l·s, not the product of totals); `coulomb`
  single-shell (`coulomb(m, :s; U, F)`, prefactor `+1/2`) and two-shell
  (`coulomb(m, :sA, :sB; U, F, G)`, direct `+1`, exchange `−1`,
  F⁰ derived from U); `Akm(m, :s, :group, coeffs)`,
  `hop(m, :sA, :sB, :group; irrep)`, `dipole(m, :sA => :sB)`
  (Wigner-Eckart E1, parity rule enforced); `to_real` (Questaal
  m-ordering, Condon-Shortley tesseral) and `to_jlmj` (Clebsch-Gordan
  |l,m,σ⟩ → |j,mⱼ⟩); anchor-based `onsite_energies(m; anchors, U,
  pairs, shells)` (NamedTuple keyed by shell symbol); legacy
  `density_density` and `kanamori` shell-keyed sugar; basis-restriction
  DSL (`nshells`, `total`, `basis(m, restrictions…)` factory).
- **`MOAD.AtomicParameters`** — primary API
  `atomic_parameters(:elem, "config")` keyed by element +
  configuration string; ground-state sugar
  `atomic_parameters(:elem; charge=...)`. Static Haverkort thesis
  dictionary (184 entries: 3d / 4d block, ground + L_{2,3}
  intermediate). Optional live Cowan runner via the `MOAD_COWAN`
  environment variable. Provenance + scaling tags carried with each
  result.
- **`MOAD.Diagnostics`** — `expectation_table(eigensystem, basis,
  ops_list)`, `configuration_weights(psi, basis; group_by)`.
- **`MOAD.QuantyIO`** — `read_quanty_operator(path, hilbert,
  mode_map)` parses real (`QComplex=0`) and complex (`QComplex=1`)
  operator dumps; `read_quanty_wavefunction`,
  `read_quanty_eigenvalues`.

### Validation

- **NiO L_{2,3} XAS (native).** MOAD reproduces the PyQuanty
  `XAS_lanczos_cont_frac.txt` reference at **1.5×10⁻¹² of peak** across
  three Cartesian polarisations on 801 ω points
  (`test/shells/validation/test_nio_xas_native.jl`, validation tier).
  Far below the originally-spec'd 1×10⁻⁴ target — the native build
  agrees with Quanty to LAPACK roundoff.
- **NiO L_3 RIXS.** Three-code agreement (MOAD / Quanty / PyQuanty)
  at 10⁻⁷ to 10⁻⁴ across 11 incident energies × 851 emission
  energies. Plots in
  `docs/dev/validation/spectroscopy/nio_xas/plots/`.
- **Double-impurity cluster benchmark.** MOAD is 12× faster than Quanty and
  1.7× faster than QuSpin wall-clock on the 2.6M-state GS sector;
  eigenvalues match QuSpin to 4×10⁻¹³.
- **PointGroups Akm vs Quanty.** 31/31 audited (group, ℓ) cells produce
  bit-exact Akm output against Quanty's hardcoded closed forms in
  `BasicMath_StandardFunctions.cpp`. The Td ℓ=3 case documents a
  Cotton-vs-Quanty Mulliken T1↔T2 label-swap convention difference,
  handled at fixture level.
- **PointGroups characters vs Bilbao.** All 32 crystallographic groups
  cross-checked at character level against the Bilbao Crystallographic
  Server (Aroyo et al., Acta Cryst. A62, 115-128 (2006)). Zero
  numerical discrepancies modulo class-permutation / notation
  conventions.
- **AtomicParameters.** Pr 4f² Cowan run matches user reference to
  0.01 eV; Ni²⁺ 3d⁸ matches the Haverkort dict bit-exact.

### Locked design decisions

- Single `MOAD` package with internal submodules — no monorepo split.
- Tier-2 canonicalization on `OperatorSum` construction.
- Eager-basis + sparse-assembly for `MOAD.Bases`; sorted-vector +
  binary search for lookup.
- `SparseMatrixCSC{T, Int32}` is the only product `MOAD.Bases` hands
  `MOAD.ED`.
- Right-to-left chain application; per-column dedup via Dict in
  `_assemble_chunk`; direct-CSC build with per-chunk mini-CSCs
  stitched in pass-2.
- `OhMyThreads.jl` for parallelism. `KrylovKit` is wrapped behind
  `MOAD.ED`; no other module imports it.
- Layer 4: `G(ω) = ((ω + Eg + iΓ/2)·I − H)⁻¹`, `Γ` is FWHM. Three
  user-facing functions (`xas`, `rixs`, `fluorescence_yield`). FY
  uses the analytic ω_out integral. Hand-rolled block Lanczos with
  rectangular `R`. Greek kwargs accept ASCII Latin aliases; mixing
  both raises `ArgumentError`.
- Layer 5: matrix-unit intertwiner produces canonical chemistry gauge
  by construction; Bilbao + Quanty are validation oracles, not the
  defining convention. `expand_clm` strict-mode requires `:reference`
  provenance + multiplicity-frame alignment; `experimental=true` opts
  into MOAD's internal convention.
- Layer 6: primary form of `density_density`, `kanamori`, and all spin
  operators is the Algebra-level primitive
  `(site::FermionSite, orbital_pairs; …)`; shell-keyed forms are
  sugar that delegates.
- Plan 2 (Shells multiplet extension):
  `rotate(op, h, U; project=Bool)` is two-regime via a single
  function; `project=false` requires square unitary, `project=true`
  requires rectangular isometry; Sakurai
  `U[j, i] := ⟨e_old_j | e_new_i⟩` (Quanty differs by adjoint, real
  matrices agree). `LS` is one-body, NOT the product of total
  operators. Slater-Condon Coulomb prefactors: single-shell `+1/2`;
  two-shell direct `+1`, exchange `−1` (the asymmetry is a real
  consequence of Quanty's c-orderings; canonicalization inserts the
  direct-only sign flip). F⁰ derived from U via `WignerSymbols.jl`
  3j products. Mulliken irrep labels are initial-caps (`:Eg`, `:T2g`);
  lowercase rejected.
- Layer 7: configuration string is the canonical lookup key (charge
  is sugar for ground states only). Configurations normalised
  internally — `"3d^8"`, `"3D8"`, `"3d_8"`, `" 3d8 "` → `"3d8"`. The
  Cowan runner refuses to silently default missing required Slater
  integrals — `_get_required` throws `KeyError`. Scratch directories
  use explicit `try/finally` cleanup (not `mktempdir(cleanup=true)`).

### Test counts

- 2154 unit + 36 validation = 2190 total. Counts re-baselined when
  `for ... @test` patterns were collapsed to `@test all(...)` /
  `@test maximum(...) < tol` per logical property; coverage unchanged.

### Breaking changes

- `slater_integrals` → `atomic_parameters`. The function was renamed
  because ζ and the radial r-moments it returns are not Slater
  integrals — the new name is honest.
- `AtomicParameters` lookup key restructured: primary form is now
  `(element, config_string)`; the charge-keyed form is sugar for
  ground states only.

### Citations

- Cowan, *The Theory of Atomic Structure and Spectra* (1981), ch. 6
  (Slater-Condon angular tensor).
- Sobelman, *Atomic Spectra and Radiative Transitions* (1992), ch. 4.
- Haverkort, PhD thesis (2005), atomic Slater-Condon tables for
  3d / 4d transition-metal ions and their L_{2,3} core-hole
  intermediates.
- Aroyo et al., *Acta Cryst.* **A62**, 115-128 (2006), Bilbao
  Crystallographic Server character-table reference.
- Haverkort et al., *J. Phys.: Conf. Ser.* **712**, 012001 (2016) /
  Lu et al., *Phys. Rev. B* **90**, 085102 (2014), Quanty reference
  papers.

## [pre-0.1.0] — internal milestones

The pre-release work (originally three monorepo packages
`MOADAlgebra`, `MOADHilbert`, `MOADQuanty`) was consolidated into a
single `MOAD` package with internal submodules `MOAD.Algebra`,
`MOAD.Bases`, `MOAD.ED`, `MOAD.QuantyIO` before the v0.1.0 release.
User-facing surface unchanged at consolidation time; `using MOAD`
brings in every public name. Internal design notes moved out of the
user-facing tree; Quanty regression data lives under
`docs/dev/validation/`.

<!-- Releases up to and including v0.2.1 predate the public repository;
     their headings are internal history and intentionally unlinked.
     Compare/release links resume with the first public tag. -->
