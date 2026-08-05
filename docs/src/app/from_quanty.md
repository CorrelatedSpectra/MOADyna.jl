# Coming from Quanty

This appendix maps Quanty's scripting idioms to the equivalent MOAD API. The biggest
difference is that MOAD operators are symbolic Julia values you build and inspect directly,
rather than strings evaluated by an interpreter — but the multiplet-physics vocabulary lines
up closely.

## Command mapping

| Quanty | MOAD | notes |
|---|---|---|
| `NewOperator("Number", …)` etc. | `n`, `c`, `cdag`, `b`, `bdag` | symbolic ladder/number operators |
| `OppF[k]` Slater Coulomb | `coulomb(m, shell; U, F)` | `F` is the required Slater tuple ``(F^2, F^4, \ldots)``; two-shell form takes `G` for exchange |
| `OppSO` spin–orbit | `LS(m, shell)` (×ζ) | one-body ``L\cdot S`` |
| `NewOperator("CF", …, Akm)` | `Akm(m, shell, group, coeffs)` | irrep-projected crystal field |
| ligand hybridization | `hop(m, shellA, shellB, group; irrep)` | irrep-projected, unit amplitude |
| `OppSx/Sy/Sz`, `OppLx…`, `OppJz…` | `Sx`/`Sz`/…, `Lz`/…, `Jz`/… | shell-keyed |
| transition operator `OppTXAS` | `dipole(m, :A => :B)` → `[Tx,Ty,Tz]` | E1; general rank: `multipole(m, :A=>:B, k)` |
| `Qxx…` quadrupole moment | `quadrupole(m, shell)` | six dimensionless Cartesian components |
| `Tx/Ty/Tz` spin-dipole | `spin_dipole_T(m, shell)` | XMCD sum-rule operator |
| nIXS `e^{iq·r}` | `nixs(m, :A=>:B; theta, phi, radial_integrals)` | + `radial_integral` for `Rj_k(q)` |
| `Eigensystem(H, …)` | `eigen(H, basis; n, which)` | dense ↔ Krylov auto-dispatch |
| `CreateSpectra(psi, H, …)` | `xas(H, basis, T, ψ; …)` | also `rixs`, `fluorescence_yield` |
| `CreateResonantSpectra` | `rixs(H_f, H_i, basis, T_in, T_out, ψ; …)` | |
| — | `optical_conductivity`, `dynamical_structure_factor` | response-function extras |

MOAD uses the same Racah ``C^{k}_{q}`` convention as Quanty's `SlaterCoefficientC`, so
equivalent operators carry equivalent matrix elements. The `dipole` operator is regressed
against Quanty operator dumps, and the remaining standard-operator coefficients are locked
against Quanty source formulas and independent in-tree oracles (see
[Validation & benchmarks](@ref) below).

## Validation & benchmarks

MOAD is checked against independent references on every push — analytic results, other
exact-diagonalization codes, and Quanty/PyQuanty — so the physics is verified, not just the
code paths. The suite lives under `test/`, with the cross-code checks collected under
`test/**/validation/`:

- **Operator level.** The rank-``k`` Wigner–Eckart coefficient reproduces Quanty's
  `SlaterCoefficientC`, and irrep-projected `Akm` crystal fields match a closed-form oracle
  transcribed from Quanty source. The `dipole` operator is regressed against Quanty `TXAS`
  operator dumps; `multipole` / `quadrupole` / `spin_dipole_T` are locked against Quanty
  source formulas (e.g. the `Qxx`/`Qyy`/`Qzz` combinations) and independent in-tree oracles
  (the same-shell `Bkm_matrix` and the ``k=1`` `dipole` cross-shell oracle).
- **Spectroscopy.** The NiO ``L_{2,3}`` XAS multiplet calculation is regressed against a
  PyQuanty reference spectrum (the residual is limited by Krylov truncation). The
  Hubbard-dimer single-particle Green's function is checked against its analytic poles, and
  `optical_conductivity` / `dynamical_structure_factor` against independent dense-Lehmann
  sums, with the ``S(q,\omega)`` zeroth-moment sum rule checked.
- **Models.** Bose–Hubbard and Hubbard–Holstein energies and observables are cross-checked
  against QuSpin.

## Reading a Quanty operator dump

`MOAD.QuantyIO` parses the text format produced by Quanty's `print(operator)` and builds the
equivalent [`OperatorSum`](@ref), so you can cross-check or import a Quanty operator directly:

```julia
hilbert  = Hilbert(:s => FermionSite{10}(:s))
mode_map = i -> (:s, i + 1)           # Quanty 0-indexed → MOAD 1-indexed
H        = read_quanty_operator("hamiltonian_dump.txt", hilbert, mode_map)
```

See [`read_quanty_operator`](@ref) / [`read_quanty_operators`](@ref) and the
wavefunction/eigenvalue readers in `MOAD.QuantyIO`.

## Scope

MOAD currently builds an explicit sparse Hamiltonian for each conserved sector and diagonalises
it, so the tractable Hilbert-space size is bounded by what fits in memory as a sparse matrix.
The principal difference from Quanty is the **matrix-free** (on-the-fly) operator
application — applying ``H`` to a vector directly on the bit-packed basis without
materialising a sparse matrix, which reaches much larger Hilbert spaces. In MOAD this
is planned for a future *matrix-free + symmetry* release.
