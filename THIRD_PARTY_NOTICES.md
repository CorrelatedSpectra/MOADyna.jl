# Third-party notices

MOAD.jl is released under the MIT License (see [`LICENSE`](LICENSE)). That
license covers the MOAD-authored material in this repository; it does **not**
relicense the third-party material listed below, which remains under its own
terms.

Julia package dependencies are not listed here — they are declared in
`Project.toml`, installed by the package manager, and carry their own licenses
in their own repositories. This file covers only third-party material
*redistributed inside this repository*.

---

## 1. Quanty tutorial script

**File:** `docs/dev/validation/spectroscopy/nio_xas/scripts/generate_quanty_operators.lua`

Derived from the NiO ligand-field XAS L₂,₃ tutorial on the Quanty website:

- Source: <https://www.quanty.org/documentation/tutorials/nio_ligand_field/xas_l23>
- Copyright: the Quanty authors (M. W. Haverkort and contributors)
- License: Creative Commons Attribution (CC-BY) —
  <https://www.quanty.org/copyright>. Full text in
  [`licenses/CC-BY-4.0.txt`](licenses/CC-BY-4.0.txt).

**Changes made:** adapted to emit the operator dumps that MOAD's validation
suite reads — operator definitions retained, with added `Print`/write calls for
the Hamiltonian, XAS Hamiltonian, and the three XAS transition operators, and
verbosity and basis-index bookkeeping adjusted for that purpose.

This script is a *regeneration* tool for reference data. It is not part of the
MOAD package: it runs under Quanty, not Julia, and nothing in `src/` loads it.

## 2. Logo — typeface credit (courtesy, not a license obligation)

**Files:** `assets/logo.svg`, `docs/src/assets/logo.svg`
(generator: `dev_scripts/build_logo.py`)

The MOAD logo is an original work of this project, © Yi Lu, under the same MIT
license as the rest of the repository. Its wordmark letterforms were set in
**Roboto Bold** (Christian Robertson and the Roboto Project Authors,
<https://github.com/googlefonts/roboto-2>, licensed Apache-2.0), and we credit
that here as a courtesy.

No obligation arises from the font license: the font software is not
redistributed by this repository (`build_logo.py` reads it from the local
system), and the SVG contains only static outline paths — rendered output of
the font program, not the program or any part of it. Apache-2.0 governs the
font software itself, and typeface designs are in any case not subject to
copyright in the United States.

## 3. NiO experimental reference spectra

**Files:** `docs/src/assets/nio/nio_xas_L23_exp.dat`,
`docs/src/assets/nio/nio_nixs_dd_exp.dat`

Obtained from the Quanty website's NiO tutorial material, which digitized the
published measurements. Redistributed with credit under the Quanty site's
CC-BY terms (<https://www.quanty.org/copyright>; full text in
[`licenses/CC-BY-4.0.txt`](licenses/CC-BY-4.0.txt)). Values are unmodified;
each file carries a provenance header.

Underlying measurements:

- XAS: D. Alders *et al.*, *Phys. Rev. B* **57**, 11623 (1998).
- nIXS: R. Verbeni *et al.*, *J. Synchrotron Rad.* **16**, 469 (2009).

Please cite the original publications when using these data.

## 4. Atomic parameter tables

**File:** `src/atomic_parameters/haverkort_data.jl`

Tabulated Hartree–Fock Slater integrals and spin–orbit parameters transcribed
from:

- M. W. Haverkort, PhD thesis, Universität zu Köln (2005),
  [arXiv:cond-mat/0505214](https://arxiv.org/abs/cond-mat/0505214),
  appendix "Slater integrals for 3d and 4d elements",

computed with R. D. Cowan's RCN atomic-structure code (R. D. Cowan, *The Theory
of Atomic Structure and Spectra*, University of California Press, 1981).

These are measured/computed physical constants reproduced for scientific use.
Please cite the thesis when using them.
