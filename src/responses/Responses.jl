"""
    MOADyna.Responses

Foundational layer for matrix-of-ω response objects.

Provides:
- [`AbstractResponse{T}`](@ref) — abstract supertype
- [`LanczosResponse{T}`](@ref) — block-Lanczos tridiagonal representation
- [`PoleResponse{T}`](@ref)    — pole + residue (spectral) representation
- [`GridResponse{T, N}`](@ref) — sampled on an ω-grid
- [`GreensFunction{T, R}`](@ref) — parametric single-particle Green's function

In `MOADyna.Responses`, all matrix-of-ω response objects live here. The
spectroscopy-domain wrappers and post-processing (polarisation, dipole
bookkeeping, `re_broaden`, spectrum algebra, Plots ext) remain in
`MOADyna.Spectroscopy`.

See the **Responses** chapter of the manual for the user-facing contract.
"""
module Responses

using LinearAlgebra
using SparseArrays
using HDF5
using OhMyThreads: @tasks

using ..Algebra
using ..Bases
using ..Bases: _prepare_action, _validate_threading
using ..Bases: _transition_action, _validate_action_basis
using ..Bases: SelectionConvergenceError, _selected_basis, _rebind_action, _embed_block
using ..Bases: _external_action, _discover_action, _enrichment_rows, _CandidateLimit
using ..Bases: _StateAccumulator, _accumulator_row!, _finish_accumulator

include("types.jl")
include("scalar_lanczos.jl")
include("block_reorth.jl")
include("block_lanczos.jl")
include("block_tridiagonal.jl")
include("cf.jl")
include("conversions.jl")
include("selection.jl")
include("arithmetic.jl")
include("correlator.jl")
include("io.jl")

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export AbstractResponse
export LanczosResponse
export PoleResponse
export GridResponse
export GreensFunction
export to_pole
export to_grid
export correlator
export selected_response, SelectedResponse, resolvent
export save_response
export load_response

end # module Responses
