# Prepared CPU sparse action for repeated Krylov products. Each task owns
# complete output rows, so no atomics or per-thread output reductions are needed.
# This is solve-local: the source matrix must stay fixed while the action is used.

struct _SparseRowAction{T,Ti<:Integer,Conjugate} <: AbstractMatrix{T}
    parent::SparseMatrixCSC{T,Ti}
    row_storage::SparseMatrixCSC{T,Ti}
    chunks::Vector{UnitRange{Int}}
end

Base.size(A::_SparseRowAction) = size(A.parent)
Base.getindex(A::_SparseRowAction, i::Int, j::Int) = A.parent[i, j]

function _sparse_row_chunks(storage::SparseMatrixCSC, nchunks::Int)
    nchunks > 0 || throw(ArgumentError("sparse action: nchunks must be positive"))
    nrows = size(storage, 2)
    chunks = UnitRange{Int}[]
    nrows == 0 && return chunks
    # Count one unit per row as well as its nonzeros, so empty rows also
    # receive an owner. Boundaries keep each row's sum on a single task.
    prefix = Int.(storage.colptr) .+ (0:nrows) .- 1
    count = min(nchunks, nrows)
    firstrow = 1
    for k in 1:(count-1)
        nextrow = searchsortedfirst(prefix, cld(prefix[end] * k, count))
        if nextrow > firstrow
            push!(chunks, firstrow:(nextrow-1))
            firstrow = nextrow
        end
    end
    firstrow <= nrows && push!(chunks, firstrow:nrows)
    return chunks
end

function _SparseRowAction(A::SparseMatrixCSC{T,Ti};
                          nchunks::Int = 4 * Threads.nthreads()) where {T,Ti}
    # Only exact Hermiticity licenses conjugate-column row access. A
    # general complex operator instead needs transpose storage, not adjoint.
    conjugate = ishermitian(A)
    storage = conjugate ? A : copy(transpose(A))
    chunks = _sparse_row_chunks(storage, nchunks)
    return _SparseRowAction{T,Ti,conjugate}(A, storage, chunks)
end

function _validate_sparse_threading(threaded)
    (threaded === :auto || threaded === true || threaded === false) ||
        throw(ArgumentError("threaded must be :auto, true, or false; got $(repr(threaded))"))
    return threaded
end

function _prepare_sparse_action(H; nrhs::Int = 1, threaded = :auto)
    _validate_sparse_threading(threaded)
    return H
end

function _prepare_sparse_action(H::_SparseRowAction; nrhs::Int = 1, threaded = :auto)
    _validate_sparse_threading(threaded)
    return threaded === false ? H.parent : H
end

# Conservative single-RHS floors from physical-core sweeps. An interactive
# caller pays an extra handoff to the worker pool at every product. Ignore the
# initial nrhs for selection: block deflation can later leave just one column.
const _SPARSE_ACTION_MIN_NNZ = 100_000
const _SPARSE_ACTION_INTERACTIVE_MIN_NNZ = 1_500_000
function _prepare_sparse_action(H::SparseMatrixCSC{T}; nrhs::Int = 1,
                                threaded = :auto) where
        {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    _validate_sparse_threading(threaded)
    threaded === false && return H
    threaded === true && return _SparseRowAction(H)
    minimum_nnz = Threads.threadpool() === :interactive ?
        _SPARSE_ACTION_INTERACTIVE_MIN_NNZ : _SPARSE_ACTION_MIN_NNZ
    if Threads.nthreads() == 1 || size(H, 1) < 4 * Threads.nthreads() ||
       nnz(H) < minimum_nnz
        return H
    end
    return _SparseRowAction(H)
end

function _mul_sparse_rows!(Y, A::_SparseRowAction{T,Ti,C}, X,
                           alpha, beta, rows) where {T,Ti,C}
    storage = A.row_storage
    values, indices = nonzeros(storage), rowvals(storage)
    S = promote_type(T, eltype(X), eltype(Y))
    @inbounds for col in axes(X, 2), row in rows
        value = zero(S)
        for p in nzrange(storage, row)
            a = C ? conj(values[p]) : values[p]
            value += a * X[indices[p], col]
        end
        # beta=0 must not read uninitialized (or NaN) output storage.
        Y[row, col] = iszero(beta) ? alpha * value :
                                   alpha * value + beta * Y[row, col]
    end
    return nothing
end

function _sparse_mul!(Y, A::_SparseRowAction, X, alpha, beta)
    Base.require_one_based_indexing(Y, X)
    size(X, 1) == size(A, 2) && size(Y, 1) == size(A, 1) &&
        size(Y, 2) == size(X, 2) ||
        throw(DimensionMismatch("sparse action: incompatible input/output dimensions"))
    Base.mightalias(Y, X) &&
        throw(ArgumentError("sparse action: output must not alias the input"))
    (Base.mightalias(Y, A.parent) || Base.mightalias(Y, A.row_storage)) &&
        throw(ArgumentError("sparse action: output must not alias matrix storage"))
    if iszero(alpha)
        if iszero(beta)
            fill!(Y, zero(eltype(Y)))
        else
            Y .*= beta
        end
        return Y
    end
    @tasks for rows in A.chunks
        @set scheduler = :dynamic
        _mul_sparse_rows!(Y, A, X, alpha, beta, rows)
    end
    return Y
end

# Specialize the dense Krylov RHS types only. Broader AbstractMatrix RHS
# methods intersect structured-matrix multiplication on supported Julia 1.10.
LinearAlgebra.mul!(Y::StridedVector, A::_SparseRowAction, X::StridedVector,
                   alpha::Number, beta::Number) = _sparse_mul!(Y, A, X, alpha, beta)
LinearAlgebra.mul!(Y::StridedMatrix, A::_SparseRowAction, X::StridedMatrix,
                   alpha::Number, beta::Number) = _sparse_mul!(Y, A, X, alpha, beta)
LinearAlgebra.mul!(Y::StridedVector, A::_SparseRowAction, X::StridedVector) =
    _sparse_mul!(Y, A, X, true, false)
LinearAlgebra.mul!(Y::StridedMatrix, A::_SparseRowAction, X::StridedMatrix) =
    _sparse_mul!(Y, A, X, true, false)

function Base.:*(A::_SparseRowAction{T}, x::StridedVector{S}) where {T,S}
    y = Vector{promote_type(T,S)}(undef, size(A, 1))
    return mul!(y, A, x)
end

function Base.:*(A::_SparseRowAction{T}, X::StridedMatrix{S}) where {T,S}
    Y = Matrix{promote_type(T,S)}(undef, size(A, 1), size(X, 2))
    return mul!(Y, A, X)
end
