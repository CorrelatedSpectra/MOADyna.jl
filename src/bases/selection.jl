"""
    SelectionConvergenceError <: Exception

Basis selection stopped before satisfying its requested tolerance. `reason`
identifies the stopping condition and `result` retains the latest solution and
diagnostics. Infinite residuals denote incomplete checks, not measured errors.
`result === nothing` means a transition source exceeded a resource cap before
the first solve.
"""
struct SelectionConvergenceError{R} <: Exception
    reason::Symbol
    result::R
end

function Base.showerror(io::IO, e::SelectionConvergenceError)
    print(io, "basis selection stopped at ", e.reason)
    if e.result === nothing
        print(io, " before source construction completed")
    else
        print(io, " with ", length(e.result.basis), " states")
    end
end

function _enrichment_rows(values, budget::Real, available)
    weights = reshape([norm(row) for row in eachrow(values)], :, 1)
    return _enrichment_rows(weights, [budget], available)
end

# Rank jointly, then retain enough of that ordering to meet every tail budget.
function _enrichment_rows(weights, budgets::AbstractVector, available)
    order = sortperm([norm(row) for row in eachrow(weights)]; rev=true)
    tail = zeros(Float64, length(budgets))
    count = length(order)
    for k in length(order):-1:1
        row = order[k]
        all(j -> hypot(tail[j], weights[row, j]) <= budgets[j], eachindex(tail)) || break
        for j in eachindex(tail)
            tail[j] = hypot(tail[j], weights[row, j])
        end
        count -= 1
    end
    return order[1:min(count, available)]
end
