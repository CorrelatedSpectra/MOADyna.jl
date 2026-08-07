# =====================================================================
# eigen(H, basis; ...)  — single user-facing method
# =====================================================================
#
# Two methods on Base.LinearAlgebra.eigen:
#   - eigen(H::OperatorSum,     basis::EagerBasis; kwargs...)
#   - eigen(H::SparseMatrixCSC, basis::EagerBasis; kwargs...)
#
# Both return LinearAlgebra.Eigen{Float64, T} with `.values::Vector{Float64}`
# (sorted by `which`) and `.vectors::Matrix{T}` (orthonormal columns,
# T = eltype(H)).

# --- Convenience method on OperatorSum --------------------------------
#
# The user gives us a symbolic H; we compile + assemble + dispatch. Power
# users with parameter-scan workflows hold the `SparseMatrixCSC`
# themselves and call the second method directly to skip recompilation.

function LinearAlgebra.eigen(H::OperatorSum, basis::EagerBasis;
                             kwargs...)
    H_compiled = compile(H, basis)
    H_sparse = assemble(H_compiled, basis)
    return LinearAlgebra.eigen(H_sparse, basis; kwargs...)
end

# --- Main method on SparseMatrixCSC -----------------------------------

"""
    eigen(H, basis; n=1, which=:SR, kwargs...) -> LinearAlgebra.Eigen

Compute `n` eigenpairs of `H` in `basis`. Auto-selects between dense
`eigen(Hermitian(...))` (small bases) and `KrylovKit.eigsolve` (large).
The Hamiltonian is assumed Hermitian; eigenvalues are returned as
`Vector{Float64}`, eigenvectors keep `eltype(H)`.

# Arguments
- `H`     :: `OperatorSum` (compiled+assembled internally) or pre-assembled
            `SparseMatrixCSC`.
- `basis` :: `EagerBasis`.

# Keyword arguments
| kwarg          | default                  | meaning                                            |
|----------------|--------------------------|----------------------------------------------------|
| `n`            | `1`                      | number of eigenpairs                               |
| `which`        | `:SR`                    | selection: `:SR`, `:LR`, `:LM` (Hermitian-only)    |
| `tol`          | `1e-10`                  | Krylov convergence tolerance                       |
| `degen_tol`    | `1e-8`                   | tolerance for grouping degenerate eigvals (QR)     |
| `maxiter`      | `300`                    | Krylov max iterations                              |
| `krylovdim`    | `max(20, 2n + 10)`       | Krylov subspace dimension                          |
| `dense_below`  | `1024`                   | dense path when `length(basis) ≤ this`             |
| `x0`           | `nothing`                | initial Krylov vector (seeded random if `nothing`) |

# Returns
A `LinearAlgebra.Eigen{Float64, T, Matrix{T}}` with:
- `E.values  :: Vector{Float64}`   — length `n`, sorted by `which`
- `E.vectors :: Matrix{T}`         — `length(basis) × n`, orthonormal columns

# Throws
- `ArgumentError`     — bad `which`, `n` out of range, `x0` shape/eltype mismatch
- `DimensionMismatch` — H/basis size disagree, x0 wrong length
- `ConvergenceError`  — Krylov path failed to converge `n` eigenpairs
"""
function LinearAlgebra.eigen(H_sparse::SparseMatrixCSC, basis::EagerBasis;
                             n::Int = 1,
                             which::Symbol = :SR,
                             tol::Real = 1e-10,
                             degen_tol::Real = 1e-8,
                             maxiter::Int = 300,
                             krylovdim::Int = max(20, 2n + 10),
                             dense_below::Int = 1024,
                             x0::Union{Nothing, AbstractVector} = nothing)
    _validate_eigen_inputs(H_sparse, basis;
                           n, which, krylovdim, x0)

    # The public API accepts SparseMatrixCSC; check Hermiticity upfront.
    # `Hermitian(M)` would silently mirror one triangle on non-Hermitian
    # input, producing wrong eigenvalues for a different operator than
    # the caller passed — better to fail loudly than guess.
    # `ishermitian` on a sparse matrix is O(nnz); cheap.
    ishermitian(H_sparse) || throw(ArgumentError(
        "H is not Hermitian. MOADyna's eigen contract requires Hermitian " *
        "input (the assembled Hamiltonian from MOADyna.Bases is always " *
        "Hermitian by construction). If you have a non-Hermitian " *
        "operator, use a different solver."))
    if length(basis) ≤ dense_below
        return _eigen_dense(H_sparse; n, which)
    else
        return _eigen_krylov(H_sparse; n, which, tol, degen_tol,
                             maxiter, krylovdim, x0)
    end
end

# --- Dense backend ----------------------------------------------------
#
# `eigen(Hermitian(M))` calls LAPACK's symmetric/Hermitian eigensolver
# (SYEVR / HEEVR). Eigenvectors are orthonormal by LAPACK guarantee — no
# extra QR needed.

function _eigen_dense(H_sparse::SparseMatrixCSC; n::Int, which::Symbol)
    # Hermiticity was already verified at the public-API entry; we
    # wrap with `Hermitian` only to dispatch to LAPACK's symmetric
    # eigensolver (SYEVR). The wrap uses the upper triangle as truth,
    # which is exact for Hermitian input.
    M_H = Hermitian(Matrix(H_sparse))
    E_full = LinearAlgebra.eigen(M_H)                  # all N eigvals (LAPACK SYEVR)
    sel = _select_indices(E_full.values, n, which)
    return LinearAlgebra.Eigen(Float64.(E_full.values[sel]),
                               Matrix(E_full.vectors[:, sel]))
end

# --- Krylov backend ---------------------------------------------------

function _eigen_krylov(H_sparse::SparseMatrixCSC; n::Int, which::Symbol,
                       tol::Real, degen_tol::Real,
                       maxiter::Int, krylovdim::Int,
                       x0::Union{Nothing, AbstractVector})
    T = eltype(H_sparse)
    x0_actual = if x0 === nothing
        _default_x0(size(H_sparse, 1), T)
    else
        # Always convert user-supplied x0 to Vector{T} so KrylovKit sees a
        # vector that matches the matrix element type (handles Float32 ↔
        # Float64, BigFloat → Float64, real → complex promotion). The
        # complex-to-real direction is already rejected by validation.
        convert(Vector{T}, x0)
    end

    vals, vecs, info = KrylovKit.eigsolve(H_sparse, x0_actual, n, which;
                                          tol = tol,
                                          maxiter = maxiter,
                                          krylovdim = krylovdim,
                                          ishermitian = true)

    if info.converged < n
        # KrylovKit's `info.normres` is a Vector{Float64} of per-eigenpair
        # residual norms — that's the scalar diagnostic we want. Do NOT
        # use info.residual: it's a Vector of residual VECTORS, not norms.
        normres = Float64[Float64(r) for r in info.normres]
        throw(ConvergenceError(info.converged, normres, info))
    end

    λ = Float64[Float64(real(v)) for v in vals[1:n]]
    V = reduce(hcat, vecs[1:n])::Matrix{T}
    _orthonormalize_clusters!(V, λ, degen_tol)
    return LinearAlgebra.Eigen(λ, V)
end
