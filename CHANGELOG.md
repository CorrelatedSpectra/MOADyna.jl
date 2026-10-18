# Changelog

Notable user-facing changes, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.4] — 2026-10-18

### Added

- Matrix-free operator actions for diagonalization, spectroscopy and fixed-basis gradients.
- Residual-controlled determinant selection for eigenstates and response tensors.
- Automatic basis management for eigensolves and spectroscopy.

### Fixed

- Adjoint operations for spin operators resolved by magnetic quantum number.

## [0.3.3] — 2026-10-17

### Changed

- Improved efficiency in basis construction, diagonalization, XAS/RIXS calculations
  and spectrum derivatives.
- Updated README examples and software citation guidance.

### Added

- Automatic threading of sparse Hamiltonian products, with a per-call `threaded` override.
- Precompilation of common diagonalization and spectroscopy workflows.

### Fixed

- Incorrect fermion signs in mixed fermion–boson/spin bases.
- Incomplete RIXS convergence reporting and incorrect spectrum version metadata.
- Spin-operator expansion guards and README validation on Windows.

## [0.3.2] — 2026-09-07

### Fixed

- README quick-start and NiO XAS examples.

### Changed

- Coulomb builders accept atomic-parameter `NamedTuple` values for `F` and `G`,
  with integral-rank validation.

## [0.3.1] — 2026-08-07

### Breaking changes

- Renamed the package and module from `MOAD` to `MOADyna`, without numerical changes.
- Renamed environment variables: `MOAD_COWAN` → `MOADYNA_COWAN`,
  `MOAD_QUANTY_ROOT` → `MOADYNA_QUANTY_ROOT`, and
  `MOAD_ROBOTO_BOLD` → `MOADYNA_ROBOTO_BOLD`.

The package UUID is unchanged. Remove the old `MOAD` package entry before adding
`MOADyna`, then replace `using MOAD` with `using MOADyna`.

## [0.3.0] — 2026-08-05

First public release.

### Added

- Differentiable zero-temperature XAS models, spectrum derivatives, Jacobians
  and parameter fitting through `MOADyna.Gradients`.

### Fixed

- Crystal-field expansions for D3d, C3, C3v, D3, S6, C1/Ci and pentagonal groups.

### Breaking changes

- Crystal-field parameter definitions changed for the affected expansions;
  parameter counts are unchanged.

## [0.2.1] — 2026-05-31

### Added

- Finite-temperature XAS, RIXS, fluorescence yield and charged-channel Green's functions.
- Kubo response functions and energy-unit conversion utilities.

### Changed

- Standardized `temperature` as the temperature keyword, retaining `T` as an alias
  for optical conductivity and dynamical structure factors.

## [0.2.0] — 2026-05-30

### Added

- Bosons and electron–phonon models, including Bose–Hubbard and Hubbard–Holstein models.
- A shared response API for correlation functions, Green's functions and finite-temperature
  neutral responses.
- Optical conductivity and dynamical structure factors.
- Multipole transitions, quadrupoles, non-resonant inelastic X-ray scattering
  and spin-dipole operators.
- Documentation site with manuals and spectroscopy tutorials.

## [0.1.0] — 2026-05-08

Initial feature-complete version.

### Added

- Operator algebra, Hilbert-space construction and exact diagonalization.
- XAS/RIXS spectrum tensors, fluorescence yield, polarization contraction and rebroadening.
- Point groups, crystal fields, atomic multiplets and atomic parameters.
- Spectrum and operator I/O, text import, diagnostics and plotting support.

### Breaking changes

- Renamed `slater_integrals` to `atomic_parameters`.
- Atomic-parameter lookup uses `(element, configuration)`; charge-based lookup
  remains available for ground states.

## [pre-0.1.0] — internal milestones

Consolidated the experimental algebra, basis and import components into a single package.

[0.3.4]: https://github.com/CorrelatedSpectra/MOADyna.jl/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/CorrelatedSpectra/MOADyna.jl/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/CorrelatedSpectra/MOADyna.jl/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/CorrelatedSpectra/MOADyna.jl/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/CorrelatedSpectra/MOADyna.jl/releases/tag/v0.3.0
