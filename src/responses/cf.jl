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
When every active block has size one, scalar arithmetic avoids temporary
matrices inside the recurrence while preserving the matrix return shape.
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

    # Scalar spectra share the response representation with block spectra,
    # but do not need matrix products or inverses at every Krylov step.
    # Check the complete active chain: ragged blocks can grow beyond size
    # one even when its first and last blocks are scalar. A trailing β_K
    # lies outside the represented tridiagonal and remains unused.
    if all(a -> size(a) == (1, 1), α) &&
       all(k -> size(β[k]) == (1, 1), 1:(K-1))
        g = inv(zc - Tc(α[K][1, 1]))
        for k in (K-1):-1:1
            b = Tc(β[k][1, 1])
            g = inv(zc - Tc(α[k][1, 1]) - conj(b) * g * b)
        end
        return fill(g, 1, 1)
    end

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

# Evaluate a complete grid with call-local block-solve workspaces. The caller
# supplies the shift expression and its type to retain its arithmetic convention.
# cf_block remains the scalar/generic fallback and the public single-point API.
function _cf_grid(α::Vector{<:AbstractMatrix}, β::Vector{<:AbstractMatrix},
                  R::AbstractMatrix, ω, z_at::F, ::Type{Z};
                  sign::Integer=1, prefactor=nothing,
                  raw_block_size::Int=size(R, 2)) where {F,Z}
    T = eltype(R)
    Tc = T <: Complex ? T : Complex{T}
    out = Array{Tc,3}(undef, raw_block_size, raw_block_size, length(ω))
    length(ω) == 0 && return out

    K = length(α)
    K >= 1 || throw(ArgumentError("cf_block: α stack is empty"))
    length(β) == K || length(β) == K - 1 ||
        throw(ArgumentError("cf_block: β stack length $(length(β)) inconsistent with α length $K"))
    widths = size.(α, 1)
    all(size(a, 1) == size(a, 2) for a in α) ||
        throw(DimensionMismatch("CF diagonal blocks must be square"))
    all(size(β[k]) == (widths[k+1], widths[k]) for k in 1:K-1) ||
        throw(DimensionMismatch("CF coupling block dimensions"))
    size(R, 1) == widths[1] && size(R, 2) == raw_block_size ||
        throw(DimensionMismatch("CF source factor dimensions"))

    # Apply the response prefactor before promotion, as in single-point calls.
    # Spectroscopy uses the unscaled adjoint directly.
    left = isnothing(prefactor) ? adjoint(R) : sign * prefactor * adjoint(R)
    W = promote_type(eltype(eltype(α)), eltype(eltype(β)), Z)
    scalar = all(==(1), widths) && all(size(β[k]) == (1, 1) for k in 1:K-1)
    if scalar || !(W <: Union{ComplexF32,ComplexF64}) || any(==(0), widths)
        for (i, w) in enumerate(ω)
            out[:, :, i] = left * cf_block(α, β, z_at(w)) * R
        end
        return out
    end

    right = Matrix{W}(R)
    left_work = Matrix{W}(left)
    shifts = nothing
    M = sum(widths)
    # Amortize dense setup only for sufficiently long, small Hermitian chains.
    if W === ComplexF64 && ω isa AbstractVector && M <= 256 && length(ω) >= max(_HERMITIAN_GRID_MIN_POINTS, M) &&
       Float64(M)*Float64(raw_block_size)^2 <= 2sum(Float64(b)^3 for b in widths) &&
       all(a -> a isa Matrix, α) && all(k -> β[k] isa Matrix, 1:K-1) &&
       all(a -> all(isfinite, a) && ishermitian(a), α) &&
       all(k -> all(isfinite, β[k]), 1:K-1) && all(isfinite, R) && all(isfinite, left)
        shifts = W[z_at(w) for w in ω]
        _hermitian_cf_grid!(out, α, β, right, left_work, shifts, widths) && return out
    end

    # D_k = zI - α_k - β_k† (D_(k+1) \ β_k). Consume each factor
    # before reusing its matrix storage. No Hermitian projection is applied.
    nmax = maximum(widths)
    factors = Matrix{W}(undef, nmax, nmax)
    coupling = Matrix{W}(undef, nmax, nmax)
    solved_R = Matrix{W}(undef, widths[1], raw_block_size)
    χ = Matrix{W}(undef, raw_block_size, raw_block_size)
    # Convert mixed-precision couplings once per grid; ignore a trailing β_K.
    betas = [eltype(β[k]) === W ? β[k] : Matrix{W}(β[k]) for k in 1:K-1]

    # Reuse materialized shifts after a failed attempt; do not consume a grid twice.
    for (i, w) in enumerate(isnothing(shifts) ? ω : shifts)
        z = isnothing(shifts) ? W(z_at(w)) : w
        last = @view factors[1:widths[K], 1:widths[K]]
        _cf_shifted!(last, α[K], z)
        factor = lu!(last)
        for k in K-1:-1:1
            rhs = @view coupling[1:widths[k+1], 1:widths[k]]
            copyto!(rhs, betas[k])
            ldiv!(factor, rhs)
            current = @view factors[1:widths[k], 1:widths[k]]
            _cf_shifted!(current, α[k], z)
            mul!(current, adjoint(betas[k]), rhs, -one(W), one(W))
            factor = lu!(current)
        end
        copyto!(solved_R, right)
        ldiv!(factor, solved_R)
        mul!(χ, left_work, solved_R)
        @views out[:, :, i] .= χ
    end
    return out
end

function _cf_shifted!(target, α, z)
    @inbounds for j in axes(target, 2), i in axes(target, 1)
        target[i, j] = i == j ? z - α[i, j] : -α[i, j]
    end
    return target
end

const _HERMITIAN_GRID_MIN_POINTS = 192

# Avoid extreme intermediate products; this is not a conditioning certificate.
function _moderate_pole_range(A, M)
    low = 4sqrt(floatmin(Float64))
    high = sqrt(floatmax(Float64))/(4max(1, M))
    all(A) do value
        magnitude = abs(value)
        iszero(value) || (isfinite(magnitude) && low <= magnitude <= high)
    end
end

# Evaluate a small Hermitian chain, returning false to use continued fractions.
# The caller owns all buffers and overwrites out completely after a fallback.
function _hermitian_cf_grid!(out, α, β, right_work, left_work, shifts, widths)
    all(isfinite, shifts) && all(z -> !iszero(imag(z)), shifts) || return false
    M = sum(widths)
    T = _build_T_K(α, β, ComplexF64)
    all(isfinite, T) && ishermitian(T) || return false
    _moderate_pole_range(right_work, M) && _moderate_pole_range(left_work, M) || return false
    scale = max(1.0, opnorm(T, Inf), maximum(abs(real(z)) for z in shifts))
    minimum(abs(imag(z)) for z in shifts) > M*sqrt(eps(Float64))*scale || return false
    F = eigen(Hermitian(T))
    all(isfinite, F.values) && all(isfinite, F.vectors) || return false
    top = view(F.vectors, 1:first(widths), :)
    right = adjoint(top)*right_work
    left = left_work*top
    _moderate_pole_range(right, M) && _moderate_pole_range(left, M) || return false
    b = size(out, 1)
    weights = Matrix{ComplexF64}(undef, b*b, M)
    for n in 1:M, j in 1:b, i in 1:b
        weights[i+(j-1)*b, n] = left[i,n]*right[n,j]
    end
    _moderate_pole_range(weights, M) || return false
    denominators = Vector{ComplexF64}(undef, M)
    values = Vector{ComplexF64}(undef, b*b)
    for (index, z) in enumerate(shifts)
        for n in 1:M
            denominators[n] = inv(z-F.values[n])
        end
        _moderate_pole_range(denominators, M) || return false
        mul!(values, weights, denominators)
        for j in 1:b, i in 1:b
            out[i,j,index] = values[i+(j-1)*b]
            isfinite(out[i,j,index]) || return false
        end
    end
    return true
end
