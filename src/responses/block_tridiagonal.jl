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

# Apply (zI − T_K)^(-1) to a source confined to the first Krylov block.
# Tail elimination keeps the factors block-sized, including ragged couplings.
function _block_resolvent_action(α::Vector{<:AbstractMatrix},
                                 β::Vector{<:AbstractMatrix}, z::Number,
                                 R::AbstractMatrix)
    K = length(α)
    K >= 1 || throw(ArgumentError("_block_resolvent_action: empty α stack"))
    length(β) in (K - 1, K) || throw(ArgumentError(
        "_block_resolvent_action: β stack does not match α"))
    size(R, 1) == size(α[1], 1) || throw(DimensionMismatch(
        "_block_resolvent_action: source does not match first block"))
    Tc = promote_type(typeof(z), eltype(α[1]), eltype(R))
    zc = Tc(z)
    last_factor = lu(zc*I - Matrix{Tc}(α[K]))
    factors = Vector{typeof(last_factor)}(undef, K)
    factors[K] = last_factor
    for k in (K-1):-1:1
        coupling = β[k]
        schur = zc*I - Matrix{Tc}(α[k]) -
                adjoint(coupling)*(factors[k+1] \ coupling)
        factors[k] = lu(schur)
    end
    offsets = cumsum(vcat(0, [size(a, 1) for a in α]))
    W = Matrix{Tc}(undef, offsets[end], size(R, 2))
    W[1:offsets[2], :] .= factors[1] \ R
    for k in 2:K
        previous = (offsets[k-1]+1):offsets[k]
        current = (offsets[k]+1):offsets[k+1]
        W[current, :] .= factors[k] \ (β[k-1]*view(W, previous, :))
    end
    return W
end

function _resolvent_from_lanczos(L, z)
    W = _block_resolvent_action(L.α, L.β, z, L.R)
    X = zeros(eltype(W), size(first(L.V_basis), 1), size(W, 2))
    offset = 0
    for k in eachindex(L.α)
        V = L.V_basis[k]
        rows = offset + 1:offset + size(V, 2)
        mul!(X, V, view(W, rows, :), true, true)
        offset += size(V, 2)
    end
    return X
end
