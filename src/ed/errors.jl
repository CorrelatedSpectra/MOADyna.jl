# =====================================================================
# ConvergenceError — raised by `eigen` on Krylov non-convergence
# =====================================================================
#
# Silent partial-convergence is a foot-gun. Users running batched
# parameter scans wrap the call in `try`/`catch` and decide whether
# to retry with larger `maxiter` / `krylovdim`.
#
# `normres` carries KrylovKit's `info.normres` — a `Vector{Float64}` of
# per-eigenpair residual norms. NOT KrylovKit's `info.residual`, which is
# a Vector of residual VECTORS (not numbers); that is the wrong field for
# this purpose.

"""
    ConvergenceError <: Exception

Thrown when `eigen(...)` on the Krylov path doesn't converge to the
requested number of eigenpairs.

# Fields
- `converged::Int`              — number of eigenpairs that did converge
- `normres::Vector{Float64}`    — per-eigenpair residual norms (`info.normres`)
- `info`                        — full KrylovKit `ConvergenceInfo` for diagnostics

Use [`max_normres`](@ref) for the worst-case scalar diagnostic.
"""
struct ConvergenceError <: Exception
    converged::Int
    normres::Vector{Float64}
    info::Any
end

"""
    max_normres(e::ConvergenceError) -> Float64

Largest per-eigenpair residual norm at termination, or `NaN` if the
residual vector was empty (shouldn't happen for any reasonable
KrylovKit run, but defensive).
"""
max_normres(e::ConvergenceError) = isempty(e.normres) ? NaN : maximum(e.normres)

function Base.showerror(io::IO, e::ConvergenceError)
    print(io, "ConvergenceError: only $(e.converged) eigenpairs converged")
    if !isempty(e.normres)
        print(io, " (max residual norm = ", max_normres(e), ")")
    end
end
