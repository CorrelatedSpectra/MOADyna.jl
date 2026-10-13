# Dense ragged-block assembly shared by response and spectroscopy conversions.
# A trailing coupling lies outside the represented finite chain.
function _build_T_K(α::Vector{<:AbstractMatrix}, β::Vector{<:AbstractMatrix},
                    working_type=nothing)
    K = length(α)
    K >= 1 || throw(ArgumentError("_build_T_K: empty α stack"))
    length(β) in (K-1, K) ||
        throw(ArgumentError("_build_T_K: β stack length inconsistent with α"))
    sizes = size.(α, 1)
    all(size(a, 1) == size(a, 2) for a in α) ||
        throw(DimensionMismatch("Block diagonal matrices must be square"))
    all(size(β[k]) == (sizes[k+1], sizes[k]) for k in 1:K-1) ||
        throw(DimensionMismatch("Coupling block dimensions"))
    offsets = cumsum(vcat(0, sizes))
    T = isnothing(working_type) ? eltype(α[1]) : working_type
    out = zeros(T, offsets[end], offsets[end])
    for k in 1:K
        rows = offsets[k]+1:offsets[k+1]
        out[rows, rows] .= α[k]
        if k < K
            next_rows = offsets[k+1]+1:offsets[k+2]
            out[next_rows, rows] .= β[k]
            out[rows, next_rows] .= adjoint(β[k])
        end
    end
    return out
end
