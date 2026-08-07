# Acknowledgments

MOADyna builds on the Julia numerical ecosystem. The packages below are runtime
dependencies — code that ships and runs as part of MOADyna — and we are grateful to
their authors. All are permissively licensed (MIT, or BSD-family for the HDF5 C
library that HDF5.jl wraps).

- **[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl)** (Jutho Haegeman and
  contributors) — the Lanczos/Arnoldi eigensolvers and iterative linear-algebra
  primitives behind MOADyna's exact diagonalization and the block-Lanczos
  correlator. Canonical reference:
  [doi:10.5281/zenodo.10622234](https://doi.org/10.5281/zenodo.10622234).
- **[WignerSymbols.jl](https://github.com/Jutho/WignerSymbols.jl)** (Jutho
  Haegeman and contributors) — the 3-``j``/6-``j``/9-``j`` symbols used
  throughout the multiplet / Wigner–Eckart engine (Slater–Condon integrals,
  `dipole`, `multipole`), implementing the algorithm of H. T. Johansson and
  C. Forssén, *SIAM J. Sci. Comput.* **38**, 376–384 (2016),
  [doi:10.1137/15M1021908](https://doi.org/10.1137/15M1021908).
- **[HDF5.jl](https://github.com/JuliaIO/HDF5.jl)** — the `save_spectra` /
  `load_spectra` and eigensystem/response I/O formats (wraps the HDF Group's HDF5
  C library).
- **[OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl)** — the
  threaded basis enumeration and sparse-assembly loops.
- **[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)** — small
  fixed-size arrays in the hot paths.
- **[DataStructures.jl](https://github.com/JuliaCollections/DataStructures.jl)** —
  ordered dictionaries and heaps in the basis / operator bookkeeping.

MOADyna also ships an optional plot recipe for
**[Plots.jl](https://github.com/JuliaPlots/Plots.jl)**, wired through a package
extension so it loads only when the user has Plots installed (MOADyna itself does
not depend on Plots). And it relies on Julia standard libraries
(`LinearAlgebra`, `SparseArrays`, `Random`, `Printf`, `Dates`) that ship with the
language itself.

For arbitrary physical constants beyond MOADyna's small energy-unit helper
([`MOADyna.Units`](@ref)), the idiomatic source is
[PhysicalConstants.jl](https://github.com/JuliaPhysics/PhysicalConstants.jl)
(CODATA, `Unitful`-typed); MOADyna does not depend on it.

## Citing

**Citing the software.** Please cite the version you used; machine-readable
metadata is in
[`CITATION.cff`](https://github.com/CorrelatedSpectra/MOADyna.jl/blob/main/CITATION.cff).
Each tagged release is archived with its own DOI, which identifies that
software archive; release DOIs will be listed here once available.

**Methods paper.** A separate paper describing the methods is planned — a
distinct citation from the software archive. This page will be updated when it
appears.

## Lineage

MOADyna grew out of many years of working with
[Quanty](https://www.quanty.org/) and owes a real intellectual debt to **Maurits
Haverkort** — his code and ideas shaped how the author thinks about multiplet and
core-level spectroscopy (the author is a Quanty coauthor). MOADyna is an independent
Julia project with its own API, data structures, and development direction; the
[Coming from Quanty](@ref) appendix documents the interoperability and the
multiplet-physics correspondence.
