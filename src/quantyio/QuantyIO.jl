"""
    MOADyna.QuantyIO

I/O bridge for the Quanty multiplet code. Reads the text format produced by
Quanty's `print(operator)` and constructs the equivalent symbolic operator in
[`MOADyna.Algebra`](@ref), applying a user-supplied mapping from Quanty's
0-indexed mode integers to MOADyna `(site, label)` addresses.

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
