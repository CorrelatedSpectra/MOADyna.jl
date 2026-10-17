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
        return KrylovKit.eigsolve(action, vec(x0), n, which;
                                  tol, maxiter, krylovdim, ishermitian=true)
    end
    rng = copy(_DEFAULT_X0_RNG)
    X = randn(rng, eltype(x0), size(x0, 1), n)
    X[:, 1:size(x0, 2)] = reshape(x0, size(x0, 1), :)
    # Complete the requested starting space even if a supplied seed coincides
    # with one of the random columns. Householder QR supplies orthogonal columns.
    Q = Matrix(qr(X).Q)
    initial = KrylovKit.Block([Q[:, j] for j in 1:n])
    # Share the residual budget across columns without resolving roundoff noise.
    # The unit bound also keeps the normalized initial block at loose tolerances.
    qr_tol = min(tol, one(tol)) / sqrt(n)
    qr_tol > 0 || throw(ArgumentError("tol is too small for block rank detection"))
    algorithm = KrylovKit.BlockLanczos(; tol, qr_tol, maxiter, krylovdim)
    return KrylovKit.eigsolve(_KrylovBlockAction(action), initial, n, which, algorithm)
end
