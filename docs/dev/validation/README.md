# Validation

Cross-checks against external codes (Quanty, PyQuanty, …) and standalone
correctness scripts that produce reference data, plots, or numerical
fingerprints. Organised by MOADyna module:

| Folder | What lives here |
|---|---|
| [`algebra/`](algebra/) | `MOADyna.Algebra` term-by-term comparisons against Quanty's normal-ordered operator dumps. Original `z_only` round-trip from the early bring-up phase, plus the small fixture used to scaffold `read_quanty_operator`. |
| [`ed/`](ed/) | `MOADyna.ED` numerical cross-checks. Phase-0 `z_only` eigensystem comparison vs Quanty's `Eigensystem` (eigenvalues + eigenvectors). Bose–Hubbard / Holstein reference data for the boson validation tests. |
| [`spectroscopy/`](spectroscopy/) | `MOADyna.Spectroscopy` reference-data + comparison plots. NiO L_{2,3} XAS validation (operator dumps, PyQuanty / Quanty reference outputs, regen scripts, MOADyna/PyQuanty/Quanty comparison plots). |

Tests in `test/` import or reference these folders; the test files
themselves stay under `test/`, while the heavyweight artifacts (operator
dumps, reference spectra, generated plots) live here under `docs/dev/`
so they don't bloat the package's runtime directory.
