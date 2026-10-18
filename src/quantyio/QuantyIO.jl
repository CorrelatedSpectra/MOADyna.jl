"""
    MOADyna.QuantyIO

Readers for operator and wavefunction text dumps. Operator files are converted
to symbolic values in [`MOADyna.Algebra`](@ref), using a supplied mapping from
zero-based file indices to `(site, label)` addresses.

This is an internal submodule of `MOADyna`; the user-facing function
[`read_quanty_operator`](@ref) is re-exported from the umbrella package.
"""
module QuantyIO

using ..Algebra
using ..Bases: get_index, mode_entry

include("parse.jl")
include("build.jl")
include("io.jl")
include("wavefunction.jl")

export ParsedOperator, ParsedTerm
export parse_quanty_operator
export build_operator
export read_quanty_operator, read_quanty_operators
export read_quanty_wavefunction, read_quanty_wavefunctions, read_quanty_eigenvalues

end # module
