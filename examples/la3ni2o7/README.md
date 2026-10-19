# La₃Ni₂O₇: Wannier bands and double-cluster XAS

The [tutorial](../../docs/src/tut/wannier.md) describes the orbital mapping,
ligand reduction and interaction convention.
[Download the input archive](https://github.com/CorrelatedSpectra/MOADyna.jl/releases/download/v0.3.5/la3ni2o7-wannier.tar.gz)
and unpack it outside the source tree, or in a local working directory.

For cluster construction and XAS, the scripts read only `model_hr.dat`,
`model.win`, and `geometry.toml`. The band comparison also reads `model_band.dat`,
`dft.spaghetti_ene`, `dft.klist_band`, and `model.struct`. The tutorial describes
each input and shows the underlying package calls with their outputs.

From the MOADyna checkout (Plots is needed only by the plotting scripts):

```sh
tar -xzf /path/to/la3ni2o7-wannier.tar.gz -C /path/to/data
julia --project=. --threads=4 examples/la3ni2o7/bands.jl /path/to/data/la3ni2o7-wannier /path/to/results
julia --project=. --threads=8 examples/la3ni2o7/xas.jl /path/to/data/la3ni2o7-wannier /path/to/results
julia --project=. examples/la3ni2o7/plot_xas.jl /path/to/results
```

`model.jl` defines example helpers: `load_wannier` calls the package importer,
`bilayer_cluster` selects Ni and O images with `cluster_sites` and extracts their
orbitals with `finite_cluster`, `reduced_cluster` rotates and selects the direct
ligands with `ligand_basis`, and `interacting_cluster` adds the atomic terms
using MOADyna's shell and operator functions. These helpers are specific to the
example; their essential lines are shown on the tutorial page.

`bands.jl` compares the
original DFT and Wannier bands with MOADyna interpolation. `xas.jl` solves the
full reduced cluster and writes its ground eigenstates, absorption tensor,
Cartesian intensities and calculation settings. `plot_xas.jl` produces the
in-plane and out-of-plane spectra without repeating the calculation.
The columns of `xas.txt` are relative photon energy (eV), `I_x`, `I_y` and
`I_z`; `xas.h5` contains the full absorption tensor.

Both figures are exported as SVG and PDF. The plotting scripts use colours
from [Paul Tol's muted scheme](https://sronpersonalpages.nl/~pault/).
Raw DFT inputs and generated results are not part of the source distribution.
