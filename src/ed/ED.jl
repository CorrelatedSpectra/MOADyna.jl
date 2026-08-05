"""
    MOAD.ED

Exact diagonalization for `MOAD.Bases`-assembled sparse matrices.

A thin wrapper around `KrylovKit.eigsolve` and dense LAPACK exposed as a
single extended `Base.eigen` method. See the **Exact diagonalization**
chapter of the manual for the user-facing contract.

This is an internal submodule of `MOAD`; the names below are
re-exported from the umbrella package, so users typically write `using
MOAD` and call `eigen(H, basis)` directly.
"""
module ED

using LinearAlgebra
using SparseArrays
using Random: MersenneTwister
using KrylovKit
using HDF5

using ..Algebra: OperatorSum
using ..Bases: EagerBasis, compile, assemble

include("errors.jl")
include("validation.jl")
include("utils.jl")
include("eigen.jl")
include("eigensystem_io.jl")

# `eigen` is exported by LinearAlgebra; we add methods to it. Re-exporting
# the symbol from MOAD.ED (and from MOAD itself) lets `using MOAD` provide
# `eigen` without forcing the user to also `using LinearAlgebra`.

export ConvergenceError, max_normres
export save_eigensystem, load_eigensystem
# Note: `eigen` is intentionally NOT exported here — it lives in
# LinearAlgebra and our methods extend the existing function. The umbrella
# `MOAD.jl` module re-exports it explicitly.

end # module
