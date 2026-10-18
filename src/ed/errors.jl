"""
    ConvergenceError <: Exception

Thrown when `eigen(...)` on the Krylov path doesn't converge to the
requested number of eigenpairs.

# Fields
- `converged::Int`              — number of eigenpairs that did converge
- `normres::Vector{Float64}`    — per-eigenpair residual norms; independently
  measured when the final check fails, otherwise reported by KrylovKit
- `info`                        — original KrylovKit `ConvergenceInfo` for diagnostics

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
