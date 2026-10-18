# Single-column specialization of the block recurrence. Only operators with
# a known efficient vector action opt in; arbitrary matrix-only operators keep
# the existing block path. An applicable three-argument mul! is insufficient:
# LinearAlgebra provides a generic fallback even for unsupported operators.
_scalar_lanczos_type(::Type{T}) where {T} =
    T <: Union{Float32,Float64,ComplexF32,ComplexF64}

_supports_scalar_lanczos_action(::Any) = false
_supports_scalar_lanczos_action(
    ::Union{SparseMatrixCSC{T},StridedMatrix{T},Diagonal{T},Bases._SparseRowAction{T},MatrixFreeOperator{T}}
) where {T} = _scalar_lanczos_type(T)

function _scalar_lanczos(action, X::Matrix{T}, source_norm;
                         krylovdim::Int, reorth::Symbol, tol::Real,
                         min_iter::Int, max_iter::Int, deflate_tol::Real,
                         restrictions, basis, retain_basis::Bool) where {T}
    # X is already projected. A positive-norm scalar gauge replaces the QR
    # phase while preserving V1*R and all physical/projected responses.
    q = copy(vec(X))
    q ./= source_norm
    q_prev = similar(q)
    work = similar(q)
    R_init = fill(T(source_norm), 1, 1)

    α_blocks = Matrix{T}[]
    β_blocks = Matrix{T}[]
    keep_history = retain_basis || reorth === :full
    V_blocks = keep_history ? Matrix{T}[reshape(copy(q), :, 1)] : Matrix{T}[]
    β_prev = zero(T)
    β1_norm = oftype(source_norm, NaN)
    initial_threshold = deflate_tol * source_norm
    consecutive = 0
    converged = false
    n_iter = 0

    for k in 1:min(krylovdim, max_iter)
        mul!(work, action, q)
        if restrictions !== nothing
            apply_restriction!(work, restrictions, basis)
        end

        # Preserve the block kernel's order: compute alpha before subtracting
        # either recurrence term, including with imperfect orthogonality.
        α_k = dot(q, work)
        push!(α_blocks, fill(T(α_k), 1, 1))
        work .-= q .* α_k
        if k > 1
            work .-= q_prev .* conj(β_prev)
        end

        if reorth === :full
            for _ in 1:2
                for V_old in V_blocks
                    q_old = vec(V_old)
                    coefficient = dot(q_old, work)
                    work .-= q_old .* coefficient
                end
            end
        end
        n_iter = k

        # Exhaustion and relative deflation both stop before appending beta.
        # Dispatch guarantees a positive initial threshold, so an exactly
        # zero work vector never reaches normalization.
        β_norm = norm(work)
        if β_norm < initial_threshold
            converged = true
            break
        end
        β_threshold = deflate_tol * (isnan(β1_norm) ? β_norm : β1_norm)
        if β_norm < β_threshold
            converged = true
            break
        end

        β_k = T(β_norm)
        push!(β_blocks, fill(β_k, 1, 1))
        if k == 1
            β1_norm = β_norm
        end
        if β_norm < tol * β1_norm
            consecutive += 1
            if consecutive >= min_iter
                converged = true
                # Soft-stop retains beta_k but not the candidate q_(k+1).
                break
            end
        else
            consecutive = 0
        end

        work ./= β_norm
        q_prev, q, work = q, work, q_prev
        β_prev = β_k
        if keep_history
            # Owning copies survive buffer rotation. On an iteration cap this
            # also preserves the trailing K+1 vector expected by RIXS/FY.
            push!(V_blocks, reshape(copy(q), :, 1))
        end
    end

    return (α = α_blocks, β = β_blocks, R = R_init,
            V_basis = retain_basis ? V_blocks : nothing,
            n_iter = n_iter, converged = converged)
end
