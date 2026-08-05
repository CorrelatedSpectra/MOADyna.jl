# =====================================================================
# Helpers: index selection by `which`, default x0, cluster-scoped QR
# =====================================================================

# --- _select_indices --------------------------------------------------
#
# Translate KrylovKit's `which` symbol into a sort permutation over a
# vector of (real) eigenvalues, and return the top-`n` indices. Used by
# the dense path; the Krylov path lets KrylovKit do the selection itself.
#
# Hermitian-only vocabulary: :SR, :LR, :LM.

function _select_indices(λ::AbstractVector{<:Real}, n::Int, which::Symbol)
    if which === :SR
        return partialsortperm(λ, 1:n; by = identity)            # ascending
    elseif which === :LR
        return partialsortperm(λ, 1:n; by = identity, rev = true)
    elseif which === :LM
        return partialsortperm(λ, 1:n; by = abs, rev = true)
    else
        # Validation should have caught this earlier; defensive anyway.
        throw(ArgumentError("which=$(repr(which)) not supported."))
    end
end

# --- Default initial Krylov vector ------------------------------------
#
# Reproducibility: every call uses a fresh COPY of a fixed-seed RNG, so
# repeated `eigen(H, basis)` calls on the same problem walk the same
# Krylov trajectory and produce identical numeric output.

const _DEFAULT_X0_RNG = MersenneTwister(0)

function _default_x0(N::Int, ::Type{T}) where {T}
    rng = copy(_DEFAULT_X0_RNG)        # fresh copy each call
    v = if T <: Complex
        randn(rng, T, N)
    else
        randn(rng, real(T), N)
    end
    return v ./ norm(v)
end

# --- Cluster-scoped Gram-Schmidt via QR -------------------------------
#
# Per §4 (Krylov path) of the chapter. KrylovKit's eigenvectors are
# individually normalized; across distinct eigenvalues the Lanczos
# construction makes them orthogonal to within numerical tolerance, so
# we don't touch them. Within a degenerate cluster (eigvals equal to
# within `degen_tol`), the cluster spans the right eigenspace but the
# columns aren't necessarily orthonormal — QR them.
#
# Eigenvalues are assumed sorted in the same order as the eigvec columns
# (KrylovKit returns them that way for a given `which`).

function _orthonormalize_clusters!(V::AbstractMatrix, λ::AbstractVector,
                                   degen_tol::Real)
    n = size(V, 2)
    i = 1
    while i ≤ n
        # Find contiguous cluster of eigenvalues equal to λ[i] within tol.
        j = i
        while j < n && abs(λ[j + 1] - λ[i]) ≤ degen_tol
            j += 1
        end
        if j > i
            # ≥ 2 columns in a degenerate cluster — QR them.
            block = V[:, i:j]
            Q, _ = qr(block)
            V[:, i:j] = Matrix(Q)[:, 1:(j - i + 1)]
        end
        # Single-column cluster: leave KrylovKit's vector alone.
        i = j + 1
    end
    return V
end
