# La₃Ni₂O₇: Wannier bands and double-cluster XAS

The [tutorial](../../docs/src/tut/la3ni2o7.md) describes the orbital mapping,
ligand reduction and interaction convention.
[Download the input archive](https://github.com/CorrelatedSpectra/MOADyna.jl/releases/download/v0.3.5/la3ni2o7-wannier.tar.gz)
and unpack it outside the source tree, or in a local working directory.

From the MOADyna checkout, with Plots available in the active environment:

```sh
tar -xzf /path/to/la3ni2o7-wannier.tar.gz -C /path/to/data
julia --project=. --threads=4 examples/la3ni2o7/bands.jl /path/to/data/la3ni2o7-wannier /path/to/results
julia --project=. --threads=8 examples/la3ni2o7/xas.jl /path/to/data/la3ni2o7-wannier /path/to/results
julia --project=. examples/la3ni2o7/plot_xas.jl /path/to/results
```

`model.jl` contains the shared cluster construction. `bands.jl` compares the
original DFT and Wannier bands with MOADyna interpolation. `xas.jl` solves the
full reduced cluster and writes its ground eigenstates, absorption tensor,
Cartesian intensities and calculation settings. `plot_xas.jl` produces the
in-plane and out-of-plane spectra without repeating the calculation.
The columns of `xas.txt` are relative photon energy (eV), `I_x`, `I_y` and
`I_z`; `xas.h5` contains the full absorption tensor.

Both figures are exported as SVG and PDF. The plotting scripts use colours
from [Paul Tol's muted scheme](https://sronpersonalpages.nl/~pault/).
Raw DFT inputs and generated results are not part of the source distribution.
