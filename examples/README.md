# MOAD examples

Runnable, self-contained scripts. Each builds a model with the public API and
prints (or writes) a result. Run from the repo root, e.g.:

```bash
julia --project=. examples/01_hubbard_chain.jl
```

## Current examples

| Script | What it shows |
|---|---|
| `01_hubbard_chain.jl` | Hubbard chain — 4-site spinful Hubbard model (ED, ground state) |
| `02_spin_chain.jl` | Spin-S Heisenberg chain — same code for any S |
| `03_bose_hubbard.jl` | Bose-Hubbard chain — bosonic on-site interaction |
| `04_holstein.jl` | Hubbard-Holstein — electron-phonon coupling on a chain (mixed fermion+boson basis) |
| `05_nio_xas.jl` | NiO L₂,₃ XAS — native multiplet calculation |
| `06_nio_rixs.jl` | NiO L₃ RIXS — native multiplet calculation |
| `07_nio_xas_compact.jl` | NiO L₂,₃ XAS — compact-basis (n_2p = 5 only) workflow |
| `08_nio_nixs.jl` | NiO 3d d-d nIXS — `nixs` multipole scattering operator → `S(q,ω)` |

## Future example ideas

Demonstrations worth writing later (capability already present — these are
worked examples, **not** new implementation targets):

- **Electron–phonon (Holstein) RIXS / dynamical response.** Combines the
  mixed fermion+boson Hamiltonian of `04_holstein.jl` with the spectroscopy of
  `06_nio_rixs.jl`. Supported *by construction*: the e-ph Hamiltonian is
  validated (`test/ed/test_hubbard_holstein.jl`, QuSpin cross-checks), and the
  spectroscopy stack (`eigen`, `block_lanczos`, `cf_block`, `correlator`,
  `xas`/`rixs`) is statistics-agnostic — it operates on the assembled
  `SparseMatrixCSC` + `OperatorSum`, with Jordan-Wigner signs confined to the
  fermionic branch of the assembly layer. So a RIXS / dynamical-correlator
  example on a Holstein-coupled basis (phonon sidebands in the spectrum) only
  needs writing, not new code. A small validation against a dense Lehmann
  reference would also close the one untested path (no current test exercises
  the Lanczos/spectroscopy route on a boson-containing basis).
