"""
    MOAD.Spectroscopy

Multiplet spectroscopy on Hamiltonians built by `MOAD.Bases`. Provides
three user-facing entry points

- [`xas`](@ref MOAD.Spectroscopy.xas) — X-ray absorption spectrum,
  ``\\chi_{ab}(\\omega) = \\langle\\psi| T_a^\\dagger \\, G(\\omega) \\, T_b |\\psi\\rangle``;
- [`rixs`](@ref) — Resonant Inelastic X-ray Scattering map,
  ``\\chi_{ijkl}(\\omega_\\mathrm{in},\\omega) = \\langle\\psi| T^\\dagger_\\mathrm{in,i} G_\\mathrm{int}^\\dagger \\, T_\\mathrm{out,j} \\, G(\\omega) \\, T_\\mathrm{out,k}^\\dagger \\, G_\\mathrm{int} \\, T_\\mathrm{in,l} |\\psi\\rangle``;
- [`fluorescence_yield`](@ref) — total emission yield via the analytic
  ``\\omega_\\mathrm{out}`` integral.

…plus the [`SpectraTensor`](@ref) result type and a set of
post-processing helpers for re-broadening, polarisation contraction,
pole extraction, spectrum algebra, and disk I/O.

The pipeline is hand-rolled block Lanczos with continued-fraction
evaluation (rectangular `R` from rank-revealing initial QR + ragged-
block deflation). Projected dynamics for symmetry- or sector-restricted
calculations route through `MOAD.Bases.apply_restriction!` after every
matvec. See the docstrings on individual entry points for usage.

The submodule re-exports `xas`, `rixs`, `fluorescence_yield`, and
`SpectraTensor` to the umbrella `MOAD` namespace; the post-processing
helpers (`re_broaden`, `polarise`, `poles`, `find_chunk`,
`save_spectra`, `load_spectra`, …) are reached as
`MOAD.Spectroscopy.<name>`.
"""
module Spectroscopy

using SparseArrays
using LinearAlgebra
using Base: @kwdef
using Dates
using Printf
using HDF5

using ..Algebra: OperatorSum, Restriction
using ..Bases: AbstractBasis, EagerBasis, basis_id, apply_restriction!,
                compile, assemble
import ..Responses

include("types.jl")
include("defaults.jl")
include("kwarg_aliases.jl")
include("pretty_step.jl")
include("cf.jl")
include("xas.jl")
include("rixs.jl")
include("fluorescence_yield.jl")
include("conductivity.jl")
include("structure_factor.jl")
include("kubo.jl")
include("helpers.jl")
include("thermal.jl")
include("re_broaden_table.jl")
include("io.jl")

# --- Exports -----------------------------------------------------------------
#
# Top-level `MOAD` re-exports the three entry points + result type. Helpers
# stay at submodule scope (`MOAD.Spectroscopy.<name>`) to keep the umbrella
# namespace focused.
#
# Algorithm internals (LanczosChunk, BlockTriDiagonal, block_lanczos, cf_block,
# evaluate_on_grid, Defaults, pretty_step, auto_grid) are intentionally NOT
# exported — accessible via MOAD.Spectroscopy.<name> for power users / tests.

export SpectraTensor
export DEFAULTS
export xas, rixs, fluorescence_yield
export optical_conductivity, dynamical_structure_factor, kubo_response
export find_chunk
export poles, re_broaden, re_broaden_table, polarise
export average, weighted_sum, restrict_to_window, plot_range
export save_spectra, load_spectra

# ---------------------------------------------------------------------------
# Backward-compat shims (v0.2 → v0.7): kernel functions moved to
# MOAD.Responses. These const aliases preserve MOAD.Spectroscopy.<name>
# call sites without requiring migration. Deprecation removed at v0.7 freeze.
# ---------------------------------------------------------------------------
const block_lanczos = Responses.block_lanczos
const cf_block      = Responses.cf_block

end # module
