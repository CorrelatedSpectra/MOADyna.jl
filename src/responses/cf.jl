# =====================================================================
# Continued-fraction evaluation of the block resolvent
# =====================================================================
#
# Given the block
# tridiagonal `T_K` produced by `block_lanczos` (stored as α/β stacks
# with possibly ragged blocks), computes the upper-left `B_active ×
# B_active` block of `(z·I − T_K)⁻¹` for any complex `z`. The user-
# basis correlator is recovered by the `R† · G₁(z) · R` sandwich
# applied at the call site (e.g. `evaluate_on_grid` below).

"""
    cf_block(α, β, z) -> Matrix{Complex}

Compute the upper-left active-sized block of `(z·I − T_K)⁻¹` for the
block tridiagonal `T_K` defined by the α/β stacks. The recurrence
runs from the bottom up:

```
G_K = (z·I − α_K)⁻¹                                 # B_K × B_K
G_k = (z·I − α_k − β_k† · G_{k+1} · β_k)⁻¹           for k = K-1, …, 1
```

Each `β_k` is `B_{k+1} × B_k`, so `β_k† · G_{k+1} · β_k` is
`B_k × B_k` and the subtraction inside `G_k` is well-formed even
under ragged-block deflation. Returns `G_1`, of shape
`B_active × B_active`.

The eltype is `Complex{T}` where `T` is the `α/β` precision; `z` may
be `Real` or `Complex` and is promoted to `Complex` internally.

This is a pure post-processing primitive — no matvecs, no Lanczos.
The arithmetic touches only small block matrices (`B × B` and
`B × B_prev`). Cost per call: `O(K · B³)` flops.
"""
function cf_block(α::Vector{<:AbstractMatrix},
                  β::Vector{<:AbstractMatrix},
                  z::Number)
    K = length(α)
    K ≥ 1 || throw(ArgumentError("cf_block: α stack is empty"))
    length(β) == K || length(β) == K - 1 ||
        throw(ArgumentError("cf_block: β stack length $(length(β)) inconsistent with α length $K"))
    T = promote_type(eltype(eltype(α)), eltype(eltype(β)), typeof(z))
    Tc = T <: Complex ? T : Complex{T}

    zc = Tc(z)

    # Start from the bottom: G_K = (z·I - α_K)^{-1}
    G = Tc.(α[K])
    for i in axes(G, 1)
        @inbounds G[i, i] = zc - G[i, i]
        @inbounds for j in axes(G, 2)
            i == j && continue
            G[i, j] = -G[i, j]
        end
    end
    G = inv(G)

    # Recurse upward.
    for k in (K-1):-1:1
        # β_k connects α_k (B_k × B_k) to α_{k+1} (B_{k+1} × B_{k+1}).
        # β[k] has shape (B_{k+1} × B_k). Hence β'·G·β has shape (B_k × B_k).
        βk = β[k]
        αk = α[k]
        # (z·I - αk - β'·G·β)^{-1}
        M = Tc.(αk)
        for i in axes(M, 1)
            @inbounds M[i, i] = zc - M[i, i]
            @inbounds for j in axes(M, 2)
                i == j && continue
                M[i, j] = -M[i, j]
            end
        end
        M -= adjoint(βk) * G * βk
        G = inv(M)
    end

    return G
end

# evaluate_on_grid depends on LanczosChunk (a Spectroscopy-domain type) and
# therefore lives in src/spectroscopy/cf.jl to avoid a circular dependency.
# (Responses cannot import from Spectroscopy.)
