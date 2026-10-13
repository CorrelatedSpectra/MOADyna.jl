# Amortize the two task barriers for each historical block. This is a
# conservative work heuristic, not a hardware-specific crossover.
const _BLOCK_REORTH_MIN_CHUNK_WORK = 16_384

_block_reorth_work(n, b, r, workers) =
    fld(n, workers) * b * r >= _BLOCK_REORTH_MIN_CHUNK_WORK

function _prepare_block_reorth(n, coeff_work, threaded)
    (threaded === false || coeff_work === nothing) && return nothing
    T = eltype(coeff_work)
    _scalar_lanczos_type(T) || return nothing
    workers = min(n, Threads.nthreads(:default))
    b = size(coeff_work, 1)
    if threaded === :auto
        (workers <= 1 ||
         !_block_reorth_work(n, b, b, workers)) && return nothing
    end
    rows = [fld((i-1)*n, workers)+1:fld(i*n, workers) for i in 1:workers]
    return (; rows, partial=Array{T}(undef, b, b, workers), force=threaded === true)
end

function _reorthogonalize_block!(W, blocks, coeff_work, row_work)
    T = coeff_work === nothing ? eltype(W) : eltype(coeff_work)
    use_rows = row_work !== nothing && coeff_work !== nothing &&
        W isa StridedMatrix{T} &&
        !Base.mightalias(W, coeff_work) &&
        all(V -> V isa StridedMatrix{T} && !Base.mightalias(W, V) &&
                 !Base.mightalias(coeff_work, V), blocks)
    # Keep both passes and every historical-block visit in the original order.
    for _ in 1:2, V_old in blocks
        r, b = size(V_old, 2), size(W, 2)
        if use_rows && (row_work.force ||
                       _block_reorth_work(size(W, 1), b, r, length(row_work.rows)))
            @tasks for index in eachindex(row_work.rows)
                @set scheduler=:dynamic
                rows = row_work.rows[index]
                mul!(view(row_work.partial, 1:r, 1:b, index),
                     adjoint(view(V_old, rows, :)), view(W, rows, :))
            end
            C = view(coeff_work, 1:r, 1:b)
            fill!(C, zero(T))
            # Join coefficient tasks before this deterministic chunk reduction.
            for index in eachindex(row_work.rows), j in 1:b, i in 1:r
                C[i, j] += row_work.partial[i, j, index]
            end
            @tasks for rows in row_work.rows
                @set scheduler=:dynamic
                mul!(view(W, rows, :), view(V_old, rows, :), C, -one(T), one(T))
            end
        else
            if coeff_work === nothing || !(W isa StridedMatrix{T})
                coeff = adjoint(V_old) * W
            else
                coeff = view(coeff_work, 1:size(V_old, 2), 1:size(W, 2))
                mul!(coeff, adjoint(V_old), W)
            end
            _subtract_block_product!(W, V_old, coeff)
        end
    end
    return W
end
