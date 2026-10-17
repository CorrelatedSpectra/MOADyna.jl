struct _CandidateLimit <: Exception
    limit::Int
end
Base.showerror(io::IO, e::_CandidateLimit) =
    print(io, "generated-state count exceeds max_candidates=", e.limit)

# Flat, growing table: one packed state and one coefficient row per key.
mutable struct _StateAccumulator{T}
    states::Vector{UInt64}
    values::Vector{T}
    slots::Vector{Int}
    nwords::Int
    width::Int
    limit::Int
end
_StateAccumulator(::Type{T}, nw, width, limit) where {T} =
    _StateAccumulator(UInt64[], T[], zeros(Int, 16), nw, width, limit)
Base.length(a::_StateAccumulator) = length(a.states) ÷ a.nwords

function Base.empty!(a::_StateAccumulator)
    empty!(a.states)
    empty!(a.values)
    fill!(a.slots, 0)
    return a
end

function _rehash!(a::_StateAccumulator)
    slots = zeros(Int, 2length(a.slots))
    mask = length(slots) - 1
    for i in 1:length(a)
        slot = Int(_state_hash(a.states, (i - 1) * a.nwords + 1, a.nwords) & UInt(mask)) + 1
        while slots[slot] != 0
            slot = (slot & mask) + 1
        end
        slots[slot] = i
    end
    a.slots = slots
end

function _accumulator_row!(a::_StateAccumulator{T}, state) where {T}
    mask = length(a.slots) - 1
    slot = Int(_state_hash(state, 1, a.nwords) & UInt(mask)) + 1
    @inbounds while a.slots[slot] != 0
        i = a.slots[slot]
        offset = (i - 1) * a.nwords
        all(w -> a.states[offset + w] == state[w], 1:a.nwords) && return i
        slot = (slot & mask) + 1
    end
    length(a) < a.limit || throw(_CandidateLimit(a.limit))
    if 2(length(a) + 1) > length(a.slots)
        _rehash!(a)
        return _accumulator_row!(a, state)
    end
    append!(a.states, state)
    append!(a.values, Iterators.repeated(zero(T), a.width))
    a.slots[slot] = length(a)
    return length(a)
end

function _external_chunk(A, X, restrictions, rows, limit)
    T = promote_type(eltype(A), eltype(X))
    width, nw = size(X, 2), A.ket.nwords
    result = _StateAccumulator(T, nw, width, limit)
    column = _StateAccumulator(eltype(A), nw, 1, limit)
    scratch = Vector{UInt64}(undef, nw)
    for j in rows
        all(iszero, @view(X[j, :])) && continue
        empty!(column)
        _foreach_transition(A.forward, A.ket, j, scratch) do state, value
            _state_index(A.bra_index, state) == 0 || return
            all(r -> _check(r, state, 1), restrictions) || return
            row = _accumulator_row!(column, state)
            column.values[row] += value
        end
        for k in 1:length(column)
            value = column.values[k]
            iszero(value) && continue
            state = @view column.states[(k - 1) * nw + 1:k * nw]
            row = _accumulator_row!(result, state)
            @inbounds for q in 1:width
                result.values[(row - 1) * width + q] += value * X[j, q]
            end
        end
    end
    return result
end

function _external_action(A::MatrixFreeOperator, X::AbstractMatrix;
                          restrictions, max_candidates::Int, threaded=:auto)
    _validate_threading(threaded)
    _validate_action_basis(A, A.ket)
    size(X, 1) == size(A, 2) || throw(DimensionMismatch("input block and basis disagree"))
    max_candidates > 0 || throw(ArgumentError("max_candidates must be positive"))
    compiled = [compile_restriction(r, A.ket.encoding) for r in _restriction_list(restrictions)]
    parallel = threaded === :auto ? A.threaded : threaded
    chunks = parallel ? min(Threads.nthreads(), max(1, length(A.ket))) : 1
    ranges = collect(Iterators.partition(1:length(A.ket), max(1, cld(length(A.ket), chunks))))
    T = promote_type(eltype(A), eltype(X))
    parts = Vector{Union{_StateAccumulator{T},_CandidateLimit}}(undef, length(ranges))
    run(k) = try
        _external_chunk(A, X, compiled, ranges[k], max_candidates)
    catch e
        e isa _CandidateLimit || rethrow()
        e
    end
    if parallel && length(ranges) > 1
        @tasks for k in eachindex(ranges)
            @set scheduler = :dynamic
            parts[k] = run(k)
        end
    else
        for k in eachindex(ranges)
            parts[k] = run(k)
        end
    end
    nw, width = A.ket.nwords, size(X, 2)
    result = _StateAccumulator(T, nw, width, max_candidates)
    for part in parts
        part isa _CandidateLimit && throw(part)
        for i in 1:length(part)
            state = @view part.states[(i - 1) * nw + 1:i * nw]
            row = _accumulator_row!(result, state)
            @inbounds for q in 1:width
                result.values[(row - 1) * width + q] += part.values[(i - 1) * width + q]
            end
        end
    end
    rows = [i for i in 1:length(result) if
        any(!iszero, @view(result.values[(i - 1) * width + 1:i * width]))]
    sort!(rows; lt=(i, j) -> _state_less_flat(result.states, nw, i, j))
    states = Vector{UInt64}(undef, length(rows) * nw)
    values = Matrix{T}(undef, length(rows), width)
    for (i, row) in enumerate(rows)
        copyto!(states, (i - 1) * nw + 1, result.states, (row - 1) * nw + 1, nw)
        @inbounds for q in 1:width
            values[i, q] = result.values[(row - 1) * width + q]
        end
    end
    return (; states, values)
end
