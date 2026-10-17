# Batch block products so matrix-free actions generate each transition once.
struct _KrylovBlockAction{A}
    operator::A
end

(A::_KrylovBlockAction)(v::AbstractVector) = A.operator * v

function KrylovKit.apply(A::_KrylovBlockAction, block::KrylovKit.Block)
    X = reduce(hcat, block.vec)
    Y = A.operator * X
    return KrylovKit.Block([Y[:, j] for j in axes(Y, 2)])
end

function _krylov_eigenpairs(action, x0, n, which; tol, maxiter, krylovdim)
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    if n == 1
        return KrylovKit.eigsolve(action, x0, n, which;
                                  tol, maxiter, krylovdim, ishermitian=true)
    end
    rng = copy(_DEFAULT_X0_RNG)
    X = randn(rng, eltype(x0), length(x0), n)
    X[:, 1] = x0
    # Complete the requested starting space even if a supplied seed coincides
    # with one of the random columns. Householder QR supplies orthogonal columns.
    Q = Matrix(qr(X).Q)
    initial = KrylovKit.Block([Q[:, j] for j in 1:n])
    # Discarding a residual block must fit within the requested residual norm.
    qr_tol = min(tol / sqrt(n), eps(real(eltype(x0))))
    qr_tol > 0 || throw(ArgumentError("tol is too small for block rank detection"))
    algorithm = KrylovKit.BlockLanczos(; tol, qr_tol, maxiter, krylovdim)
    return KrylovKit.eigsolve(_KrylovBlockAction(action), initial, n, which, algorithm)
end
