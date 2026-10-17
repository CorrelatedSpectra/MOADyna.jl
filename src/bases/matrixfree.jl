"""
    MatrixFreeOperator <: AbstractMatrix

Exact action of a compiled operator on fixed source and destination bases,
constructed with [`matrixfree`](@ref). Stores operator terms and determinant
lookup tables, without storing matrix elements. Basis data must remain unchanged
while the operator is in use.
"""
struct MatrixFreeOperator{T,BI<:_StateIndex,KI<:_StateIndex} <: AbstractMatrix{T}
    forward::CompiledHamiltonian{T}
    reverse::CompiledHamiltonian{T}
    bra::EagerBasis
    ket::EagerBasis
    bra_index::BI
    ket_index::KI
    threaded::Bool
end

"""
    matrixfree(A, basis; threaded=:auto)
    matrixfree(A, bra, ket; threaded=:auto)

Represent `P_bra * A * P_ket` without assembling its matrix. `A` may be an
`OperatorSum` or a `CompiledHamiltonian` compiled against `ket`. The bases must
use identical site, mode and bit encodings; their restrictions may differ.
The one-basis form creates a square operator.

Supports vector and dense-block multiplication, `mul!`, adjoints and explicit
conversion with `sparse` or `Matrix`. No coefficients or determinants are
discarded during multiplication. `threaded=:auto` uses the available Julia
threads; `true` and `false` explicitly select parallel or serial products.
Scratch storage belongs to each call, so independent products may run concurrently.
Scalar multiplication and addition of actions on identical bases combine their
compiled terms. The result inherits the first action's threading setting.
"""
matrixfree(A, basis::EagerBasis; kwargs...) = matrixfree(A, basis, basis; kwargs...)

function matrixfree(A::OperatorSum, bra::EagerBasis, ket::EagerBasis; kwargs...)
    _same_encoding(bra.encoding, ket.encoding) || throw(ArgumentError(
        "source and destination bases must have identical site and mode encodings"))
    return matrixfree(compile(A, ket), bra, ket; kwargs...)
end

function matrixfree(A::CompiledHamiltonian, bra::EagerBasis, ket::EagerBasis;
                    threaded=:auto)
    _validate_threading(threaded)
    A.basis_id == basis_id(ket) || throw(ArgumentError(
        "compiled operator does not belong to the source basis"))
    _same_encoding(bra.encoding, ket.encoding) || throw(ArgumentError(
        "source and destination bases must have identical site and mode encodings"))
    ket_index = _StateIndex(ket)
    bra_index = bra === ket ? ket_index : _StateIndex(bra)
    parallel = threaded === :auto ? Threads.nthreads() > 1 : threaded
    return MatrixFreeOperator(A, _adjoint_plan(A, basis_id(bra)), bra, ket,
                               bra_index, ket_index, parallel)
end

Base.size(A::MatrixFreeOperator) = (length(A.bra), length(A.ket))
Base.size(A::MatrixFreeOperator, d::Integer) = d == 1 ? length(A.bra) :
    d == 2 ? length(A.ket) : d > 2 ? 1 : throw(ArgumentError("dimension must be positive"))
Base.copy(A::MatrixFreeOperator) = A
Base.adjoint(A::MatrixFreeOperator) = MatrixFreeOperator(
    A.reverse, A.forward, A.ket, A.bra, A.ket_index, A.bra_index, A.threaded)

function _prepare_action(A::MatrixFreeOperator; nrhs::Int=1, threaded=:auto)
    _validate_threading(threaded)
    (threaded === :auto || threaded === A.threaded) && return A
    return MatrixFreeOperator(A.forward, A.reverse, A.bra, A.ket,
                               A.bra_index, A.ket_index, threaded)
end

_action_backend(::MatrixFreeOperator) = :matrixfree

_transition_action(A::MatrixFreeOperator, basis, H; threaded=:auto) =
    _prepare_action(A; threaded)

function _validate_action_coordinates(A::MatrixFreeOperator, basis)
    basis_id(A.bra) == basis_id(A.ket) == basis_id(basis) ||
        throw(ArgumentError("matrix-free Hamiltonian must use the supplied basis on both sides"))
    return nothing
end

function Base.show(io::IO, A::MatrixFreeOperator{T}) where {T}
    print(io, "MatrixFreeOperator{", T, "}(", size(A, 1), " × ", size(A, 2),
          ", ", length(A.forward), " terms)")
end

@inline function _foreach_transition(f, plan, basis, i, scratch)
    offset = state_offset(basis, i)
    for term in plan.terms
        _term_precheck_fails(plan, term, basis.states, offset) && continue
        copy_state!(scratch, basis.states, offset, basis.nwords)
        phase = _apply_chain_rtl!(scratch, plan, term)
        iszero(phase) && continue
        f(scratch, term.coef * phase)
    end
    return nothing
end

function _projected_column!(column, plan, basis, i, index, scratch)
    empty!(column)
    _foreach_transition(plan, basis, i, scratch) do state, value
        j = _state_index(index, state)
        j == 0 && return
        column[j] = get(column, j, zero(value)) + value
    end
    filter!(p -> !iszero(last(p)), column)
    return column
end

function Base.getindex(A::MatrixFreeOperator{T}, i::Int, j::Int) where {T}
    checkbounds(A, i, j)
    column = _projected_column!(Dict{Int,T}(), A.forward, A.ket, j, A.bra_index,
                                Vector{UInt64}(undef, A.ket.nwords))
    return get(column, i, zero(T))
end

function SparseArrays.sparse(A::MatrixFreeOperator{T}) where {T}
    rows, cols, values = Int[], Int[], T[]
    column = Dict{Int,T}()
    scratch = Vector{UInt64}(undef, A.ket.nwords)
    for j in 1:size(A, 2)
        _projected_column!(column, A.forward, A.ket, j, A.bra_index, scratch)
        for (i, value) in column
            push!(rows, i)
            push!(cols, j)
            push!(values, value)
        end
    end
    return sparse(rows, cols, values, size(A)...)
end

Base.Matrix(A::MatrixFreeOperator) = Matrix(sparse(A))

function LinearAlgebra.ishermitian(A::MatrixFreeOperator{T}) where {T}
    size(A, 1) == size(A, 2) || return false
    column, row = Dict{Int,T}(), Dict{Int,T}()
    scratch = Vector{UInt64}(undef, A.ket.nwords)
    for i in 1:size(A, 1)
        _projected_column!(column, A.forward, A.ket, i, A.bra_index, scratch)
        _projected_column!(row, A.reverse, A.bra, i, A.ket_index, scratch)
        column == row || return false
    end
    return true
end

function _mul_matrixfree_rows!(Y, A::MatrixFreeOperator{T}, X, alpha, beta, rows) where {T}
    scratch = Vector{UInt64}(undef, A.bra.nwords)
    coefficients = Dict{Int,T}()
    sums = zeros(promote_type(T, eltype(X), eltype(Y)), size(X, 2))
    for i in rows
        fill!(sums, zero(eltype(sums)))
        # Combine contributions before multiplying: cancellation can prevent
        # overflow even when the individual term-times-vector products cannot.
        _projected_column!(coefficients, A.reverse, A.bra, i, A.ket_index, scratch)
        for (j, value) in coefficients
            a = conj(value)
            @inbounds for q in axes(X, 2)
                sums[q] += a * X[j, q]
            end
        end
        @inbounds for q in axes(X, 2)
            Y[i, q] = iszero(beta) ? alpha * sums[q] : alpha * sums[q] + beta * Y[i, q]
        end
    end
    return nothing
end

function _matrixfree_mul!(Y, A::MatrixFreeOperator, X, alpha, beta)
    Base.require_one_based_indexing(Y, X)
    size(X, 1) == size(A, 2) && size(Y, 1) == size(A, 1) && size(X, 2) == size(Y, 2) ||
        throw(DimensionMismatch("matrix-free action: incompatible input/output dimensions"))
    Base.mightalias(Y, X) && throw(ArgumentError("output must not alias the input"))
    (Base.mightalias(Y, A.bra.states) || Base.mightalias(Y, A.ket.states)) &&
        throw(ArgumentError("output must not alias basis storage"))
    if iszero(alpha)
        iszero(beta) ? fill!(Y, zero(eltype(Y))) : (Y .*= beta)
        return Y
    end
    if A.threaded && size(A, 1) > 1
        width = max(1, cld(size(A, 1), 4 * Threads.nthreads()))
        @tasks for rows in collect(Iterators.partition(1:size(A, 1), width))
            @set scheduler = :dynamic
            _mul_matrixfree_rows!(Y, A, X, alpha, beta, rows)
        end
    else
        _mul_matrixfree_rows!(Y, A, X, alpha, beta, 1:size(A, 1))
    end
    return Y
end

LinearAlgebra.mul!(Y::StridedVector, A::MatrixFreeOperator, X::StridedVector,
                   alpha::Number, beta::Number) = _matrixfree_mul!(Y, A, X, alpha, beta)
LinearAlgebra.mul!(Y::StridedMatrix, A::MatrixFreeOperator, X::StridedMatrix,
                   alpha::Number, beta::Number) = _matrixfree_mul!(Y, A, X, alpha, beta)
LinearAlgebra.mul!(Y::StridedVector, A::MatrixFreeOperator, X::StridedVector) =
    _matrixfree_mul!(Y, A, X, true, false)
LinearAlgebra.mul!(Y::StridedMatrix, A::MatrixFreeOperator, X::StridedMatrix) =
    _matrixfree_mul!(Y, A, X, true, false)

function Base.:*(A::MatrixFreeOperator{T}, x::StridedVector{S}) where {T,S}
    return mul!(Vector{promote_type(T,S)}(undef, size(A, 1)), A, x)
end
function Base.:*(A::MatrixFreeOperator{T}, X::StridedMatrix{S}) where {T,S}
    return mul!(Matrix{promote_type(T,S)}(undef, size(A, 1), size(X, 2)), A, X)
end
