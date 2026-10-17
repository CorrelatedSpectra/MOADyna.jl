function LinearAlgebra.eigen(H::OperatorSum, basis::EagerBasis;
                             backend::Symbol = :sparse, threaded = :auto, kwargs...)
    action = _build_action(H, basis; backend, threaded)
    return LinearAlgebra.eigen(action, basis; threaded, kwargs...)
end

# Shared sparse and matrix-free solver.

"""
    eigen(H, basis; n=1, which=:SR, kwargs...) -> LinearAlgebra.Eigen

Compute `n` eigenpairs of `H` in `basis`. Sparse matrices use dense
`eigen(Hermitian(...))` for small bases and `KrylovKit.eigsolve` otherwise.
Matrix-free operators always use the iterative solver. Multiple requested
eigenvectors use block Lanczos, including repeated eigenvalues.
Returned iterative eigenpairs are checked against `H` with residual norm `tol`.
The Hamiltonian is assumed Hermitian; eigenvalues are returned as
`Vector{Float64}`, eigenvectors keep `eltype(H)`.

# Arguments
- `H`     :: `OperatorSum`, `SparseMatrixCSC`, or `MatrixFreeOperator`.
- `basis` :: `EagerBasis`.

# Keyword arguments
| kwarg          | default                  | meaning                                            |
|----------------|--------------------------|----------------------------------------------------|
| `n`            | `1`                      | number of eigenpairs                               |
| `backend`      | `:sparse`                | representation for symbolic `H`: `:sparse` or `:matrixfree` |
| `which`        | `:SR`                    | selection: `:SR`, `:LR`, `:LM` (Hermitian-only)    |
| `tol`          | `1e-10`                  | Krylov convergence tolerance                       |
| `degen_tol`    | `1e-8`                   | tolerance for grouping degenerate eigvals (QR)     |
| `maxiter`      | `300`                    | Krylov max iterations                              |
| `krylovdim`    | `max(20, 2n + 10)`       | Krylov subspace dimension                          |
| `dense_below`  | `1024`                   | sparse dense-path threshold; ignored for matrix-free input |
| `x0`           | `nothing`                | initial vector or block with up to `n` columns (seeded random if `nothing`) |
| `threaded`     | `:auto`                  | Hamiltonian action: `:auto`, `true`, or `false` |

`threaded=:auto` selects a parallel sparse action when the matrix workload and
available Julia threads justify it. `true` forces that kernel using the available
Julia thread pool (which may contain only one thread); `false` uses serial sparse
products. For a matrix-free operator, `:auto` preserves its construction setting;
`true` and `false` override it for this solve. Start Julia with `--threads=N` to
make multiple threads available. This control affects iterative operator products;
dense diagonalization, basis/assembly scheduling, and BLAS thread settings are
independent.

# Returns
A `LinearAlgebra.Eigen{Float64, T, Matrix{T}}` with:
- `E.values  :: Vector{Float64}`   — length `n`, sorted by `which`
- `E.vectors :: Matrix{T}`         — `length(basis) × n`, orthonormal columns

# Throws
- `ArgumentError`     — bad `which`, `n` out of range, `x0` shape/eltype mismatch
- `DimensionMismatch` — H/basis size disagree, x0 wrong length
- `ConvergenceError`  — Krylov path failed to converge `n` eigenpairs
"""
function LinearAlgebra.eigen(H::Union{SparseMatrixCSC,MatrixFreeOperator}, basis::EagerBasis;
                             n::Int = 1,
                             which::Symbol = :SR,
                             tol::Real = 1e-10,
                             degen_tol::Real = 1e-8,
                             maxiter::Int = 300,
                             krylovdim::Int = max(20, 2n + 10),
                             dense_below::Int = 1024,
                             x0::Union{Nothing, AbstractVecOrMat} = nothing,
                             threaded = :auto)
    _validate_threading(threaded)
    _validate_eigen_inputs(H, basis;
                           n, which, krylovdim, x0)
    _validate_action_basis(H, basis)

    # A Hermitian wrapper alone would silently replace one input triangle.
    ishermitian(H) || throw(ArgumentError(
        "H is not Hermitian. MOADyna's eigen contract requires Hermitian " *
        "input. For a non-Hermitian operator, use a different solver."))
    if H isa SparseMatrixCSC && length(basis) ≤ dense_below
        return _eigen_dense(H; n, which)
    else
        return _eigen_krylov(H; n, which, tol, degen_tol,
                             maxiter, krylovdim, x0, threaded)
    end
end

# --- Dense backend ----------------------------------------------------
#
# `eigen(Hermitian(M))` calls LAPACK's symmetric/Hermitian eigensolver
# (SYEVR / HEEVR). Eigenvectors are orthonormal by LAPACK guarantee — no
# extra QR needed.

function _eigen_dense(H::SparseMatrixCSC; n::Int, which::Symbol)
    # Hermiticity was already verified at the public-API entry; we
    # wrap with `Hermitian` only to dispatch to LAPACK's symmetric
    # eigensolver (SYEVR). The wrap uses the upper triangle as truth,
    # which is exact for Hermitian input.
    M_H = Hermitian(Matrix(H))
    N = size(H, 1)
    if n < N && which in (:SR, :LR) &&
       eltype(H) <: Union{Float32, Float64, ComplexF32, ComplexF64}
        # LAPACK can return only the requested edge of the sorted spectrum.
        indices = which === :SR ? (1:n) : (N-n+1:N)
        E = LinearAlgebra.eigen(M_H, indices)
        sel = which === :SR ? (1:n) : (n:-1:1)
    else
        E = LinearAlgebra.eigen(M_H)
        sel = _select_indices(E.values, n, which)
    end
    return LinearAlgebra.Eigen(Float64.(E.values[sel]),
                               Matrix(E.vectors[:, sel]))
end

# --- Krylov backend ---------------------------------------------------

function _eigen_krylov(H::AbstractMatrix; n::Int, which::Symbol,
                       tol::Real, degen_tol::Real,
                       maxiter::Int, krylovdim::Int,
                       x0::Union{Nothing, AbstractVecOrMat}, threaded = :auto)
    T = eltype(H)
    x0_actual = if x0 === nothing
        _default_x0(size(H, 1), T)
    else
        seed = Array{T}(x0)
        for column in eachcol(reshape(seed, size(seed, 1), :))
            seed_norm = norm(column)
            isfinite(seed_norm) && seed_norm > 0 || throw(ArgumentError(
                "each x0 column must have a finite, nonzero norm"))
            column ./= seed_norm
        end
        seed
    end

    action = _prepare_action(H; nrhs=n, threaded)
    vals, vecs, info = _krylov_eigenpairs(action, x0_actual, n, which;
                                         tol, maxiter, krylovdim)

    if info.converged < n
        normres = Float64[Float64(r) for r in info.normres]
        throw(ConvergenceError(info.converged, normres, info))
    end

    λ = Float64[Float64(real(v)) for v in vals[1:n]]
    V = reduce(hcat, vecs[1:n])::Matrix{T}
    for column in eachcol(V)
        vector_norm = norm(column)
        isfinite(vector_norm) && vector_norm > 0 ||
            throw(ConvergenceError(0, fill(Inf, n), info))
        column ./= vector_norm
    end
    _orthonormalize_clusters!(V, λ, degen_tol)
    residual = action * V - V * Diagonal(λ)
    normres = Float64[norm(column) for column in eachcol(residual)]
    converged = count(r -> isfinite(r) && r <= tol, normres)
    converged == n || throw(ConvergenceError(converged, normres, info))
    return LinearAlgebra.Eigen(λ, V)
end
