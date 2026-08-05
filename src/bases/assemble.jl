# =====================================================================
# assemble — parallel sparse matrix construction
# =====================================================================
#
# Per §6 of docs/architecture/02_hilbert.md. Two-pass build, but we
# skip the COO scratch and write CSC directly:
#   Pass 1: each chunk produces a *mini-CSC* (colptr, rowval, nzval)
#           for its column slab. Per-column Dict dedup; sort by row.
#   Pass 2: stitch the mini-CSCs into the final SparseMatrixCSC by
#           concatenating rowval/nzval and offsetting each chunk's
#           colptr. No global COO triplets, no sparse(I,J,V) scratch.
#
# Index type Idx is auto-selected: Int32 when length(basis) <= typemax(Int32)-1
# AND nnz fits Int32 (almost always for ED), else Int. Halves the memory
# for colptr / rowval / nzval (the 8-byte "Int" → 4-byte "Int32" slim).

"""
    assemble(H::CompiledHamiltonian{T}, basis::EagerBasis) -> SparseMatrixCSC{T,<:Integer}

Build the sparse-matrix form Hamiltonian. Convention:

    H ψ  →  result[i] = Σⱼ H[i,j] ψ[j]

so column `j` is "starting state index", row `i` is "outgoing state index".

The integer index type is `Int32` when the basis size and the resulting
nnz both fit in `Int32` (saves ~50% on `colptr`/`rowval` memory),
otherwise `Int`.
"""
function assemble(H::CompiledHamiltonian{T}, basis::EagerBasis) where {T}
    H.basis_id == basis.basis_id ||
        throw(ArgumentError("CompiledHamiltonian was compiled against a different " *
                            "basis (got id $(H.basis_id), basis id $(basis.basis_id))."))

    N = length(basis)
    if N <= typemax(Int32) - 1
        return _assemble_with_idx(H, basis, Int32)
    else
        return _assemble_with_idx(H, basis, Int)
    end
end

function _assemble_with_idx(H::CompiledHamiltonian{T}, basis::EagerBasis,
                            ::Type{Idx}) where {T, Idx<:Integer}
    N = length(basis)
    nwords = basis.nwords
    chunks = collect(Iterators.partition(1:N, _assemble_chunk_size(N)))

    # ---- Pass 1: per-chunk mini-CSC with per-column Dict dedup -----------
    #
    # Each chunk owns a contiguous range of columns (chunks are produced
    # by `Iterators.partition(1:N, …)`), so its mini-CSC slots in
    # cleanly into the final colptr/rowval/nzval without needing any
    # cross-chunk merge. An eager assemble can hit the same (i, j) slot
    # up to R times per column (R = compiled-term count); the Dict
    # accumulator dedups per column at almost no memory cost (Dict stays
    # tiny — ~hundreds of entries — and is cleared+reused).
    locals = tmap(chunks; scheduler = :dynamic) do chunk
        _assemble_chunk(chunk, H, basis, nwords, T, Idx)
    end

    # If Idx is Int32 but nnz overflows it, retry with Int. We can't
    # know nnz before pass-1, so this is the natural place to check.
    # In practice ED problems never come close — the 2.6M-state benchmark has nnz≈3.6e7,
    # well below the 2.15e9 Int32 ceiling.
    total_nnz_int = 0
    @inbounds for k in eachindex(locals)
        total_nnz_int += length(locals[k][2])
    end
    if Idx === Int32 && total_nnz_int > typemax(Int32) - 1
        # Repack and retry. Rare path; not worth optimizing.
        locals = nothing
        GC.gc()
        return _assemble_with_idx(H, basis, Int)
    end

    return _pass2_stitch(locals, chunks, N, total_nnz_int, T, Idx)
end

# Helper kept out-of-line so the inner `@tasks` does not capture local
# variables of `_assemble_with_idx` (OhMyThreads' boxing detector
# otherwise refuses to compile the loop).
function _pass2_stitch(locals, chunks, N::Int, total_nnz::Int,
                       ::Type{T}, ::Type{Idx}) where {T, Idx<:Integer}
    n_chunks = length(locals)

    # Per-chunk row offset into the global rowval/nzval arrays.
    row_offsets = Vector{Int}(undef, n_chunks)
    acc = 0
    @inbounds for k in 1:n_chunks
        row_offsets[k] = acc
        acc += length(locals[k][2])
    end
    @assert acc == total_nnz

    colptr = Vector{Idx}(undef, N + 1)
    rowval = Vector{Idx}(undef, total_nnz)
    nzval  = Vector{T}(undef, total_nnz)
    colptr[1] = Idx(1)

    @tasks for k in 1:n_chunks
        @set scheduler = :dynamic
        chunk = chunks[k]
        local_colptr, local_rowval, local_nzval = locals[k]
        row_off = row_offsets[k]
        n_cols = length(chunk)

        # Stitch colptr: column j of the global matrix ends at
        # local_colptr[j_local+1] + row_off. (local_colptr is Int and
        # 1-based within the chunk; row_off shifts it into global row
        # position. The conversion to `Idx` is safe here because the
        # caller has verified `total_nnz ≤ typemax(Idx) - 1`.)
        @inbounds for j_local in 1:n_cols
            j = chunk[j_local]
            colptr[j + 1] = Idx(local_colptr[j_local + 1] + row_off)
        end

        # Copy rowval / nzval into global slot.
        n_chunk = length(local_rowval)
        @inbounds for t in 1:n_chunk
            rowval[row_off + t] = local_rowval[t]
            nzval[row_off + t]  = local_nzval[t]
        end
    end

    return SparseMatrixCSC{T, Idx}(N, N, colptr, rowval, nzval)
end

@inline _assemble_chunk_size(N::Int) = max(64, N ÷ (8 * Threads.nthreads()))

# Per-chunk worker. Lifted out of `_assemble_with_idx` so the generic
# compiler specializes on (CompiledHamiltonian{T}, EagerBasis, T, Idx).
#
# Returns a chunk-local mini-CSC:
#   colptr_chunk : Vector{Int}  of length n_cols+1, 1-based, local positions
#   rowval_chunk : Vector{Idx}  of length nnz_chunk, sorted within each column
#   nzval_chunk  : Vector{T}    of length nnz_chunk
#
# Note `colptr_chunk` uses `Int` (not `Idx`) so a chunk whose local nnz
# would temporarily exceed `typemax(Idx)` does not throw before
# `_assemble_with_idx` has had a chance to verify the *global* nnz against
# `typemax(Idx)` and (if needed) retry with `Idx = Int`. Row indices are
# bounded by `length(basis) ≤ typemax(Idx)-1` (checked at the call site)
# so converting them to `Idx` here is always safe.
function _assemble_chunk(chunk, H::CompiledHamiltonian{T}, basis::EagerBasis,
                         nwords::Int, ::Type{T}, ::Type{Idx}) where {T, Idx<:Integer}
    n_cols = length(chunk)
    accum     = Dict{Int, T}()       # i → accumulated coefficient at column j
    keys_buf  = Int[]                # scratch for per-column key sort

    colptr_chunk = Vector{Int}(undef, n_cols + 1)
    rowval_chunk = Idx[]
    nzval_chunk  = T[]
    new_state    = Vector{UInt64}(undef, nwords)

    colptr_chunk[1] = 1

    for (j_local, j) in enumerate(chunk)
        off_j = state_offset(basis, j)
        empty!(accum)
        for term in H.terms
            _term_precheck_fails(H, term, basis.states, off_j) && continue
            copy_state!(new_state, basis.states, off_j, nwords)
            phase = _apply_chain_rtl!(new_state, H, term)
            phase == zero(T) && continue
            i = get_index(basis, new_state)
            i == 0 && continue
            v = term.coef * phase
            new_v = get(accum, i, zero(T)) + v
            # Drop exact cancellations rather than storing explicit zeros.
            # Use `iszero`, never tolerance — small but legitimate matrix
            # elements must survive.
            if iszero(new_v)
                haskey(accum, i) && delete!(accum, i)
            else
                accum[i] = new_v
            end
        end

        # Sort the column's row indices and write them out. CSC requires
        # row indices be sorted within each column for downstream BLAS /
        # KrylovKit to be happy.
        empty!(keys_buf)
        for k in keys(accum)
            push!(keys_buf, k)
        end
        sort!(keys_buf)
        for i in keys_buf
            push!(rowval_chunk, Idx(i))
            push!(nzval_chunk, accum[i])
        end

        colptr_chunk[j_local + 1] = length(rowval_chunk) + 1
    end

    # Drop push!-grown capacity slack (up to ~2× length) so it does not
    # coexist with the final CSC arrays during pass-2 stitch. `copy(v)`
    # allocates a fresh, tight-capacity buffer; the original goes to GC.
    # Equivalent to `sizehint!(v, length(v); shrink=true)` but works on
    # Julia 1.10 (the `shrink` keyword was added in 1.11).
    rowval_chunk = copy(rowval_chunk)
    nzval_chunk  = copy(nzval_chunk)

    return (colptr_chunk, rowval_chunk, nzval_chunk)
end

# =====================================================================
# _apply_chain_rtl! — walk a compiled term right-to-left
# =====================================================================
#
# Returns the accumulated phase as a value of the same eltype as
# H.terms[i].coef. Phase 0 means "term vanishes on this state".
#
# Per §6.2 of the layer-2 chapter: rightmost entry hits |ψ⟩ first.
# Mutates `state` in place to the resulting |φ⟩.

function _apply_chain_rtl!(state::AbstractVector{UInt64},
                           H::CompiledHamiltonian{T},
                           term::CompiledTerm{T})::T where {T}
    if term.kind === EMPTY_CHAIN
        return one(T)
    end
    phase = one(T)
    # Walk entries from chain_length down to 1 (right-to-left).
    @inbounds for k in term.chain_length:-1:1
        ce = H.entries[term.chain_offset + k]
        kind = ce.kind
        if kind === FERM_C
            # Annihilate fermion at (word_idx, bit_offset). Must be set;
            # phase ×= JW sign = (-1)^(parity of bits below this position).
            v = get_bit(state, 1, ce.word_idx, ce.bit_offset)
            v == 0 && return zero(T)
            sgn = _jw_parity(state, ce.word_idx, ce.bit_offset, H.nwords)
            clear_bit!(state, 1, ce.word_idx, ce.bit_offset)
            phase = sgn == 0 ? phase : -phase
        elseif kind === FERM_CDAG
            v = get_bit(state, 1, ce.word_idx, ce.bit_offset)
            v == 1 && return zero(T)
            sgn = _jw_parity(state, ce.word_idx, ce.bit_offset, H.nwords)
            set_bit!(state, 1, ce.word_idx, ce.bit_offset)
            phase = sgn == 0 ? phase : -phase
        elseif kind === BOS_B
            n = get_span(state, 1, ce.word_idx, ce.bit_offset, ce.nbits)
            n == 0 && return zero(T)
            phase *= T(sqrt(n))
            set_span!(state, 1, ce.word_idx, ce.bit_offset, ce.nbits, n - 1)
        elseif kind === BOS_BDAG
            n = get_span(state, 1, ce.word_idx, ce.bit_offset, ce.nbits)
            n >= ce.max_val && return zero(T)
            phase *= T(sqrt(n + 1))
            set_span!(state, 1, ce.word_idx, ce.bit_offset, ce.nbits, n + 1)
        elseif kind === SPIN_SZ
            mz_enc = get_span(state, 1, ce.word_idx, ce.bit_offset, ce.nbits)
            mz_enc == ce.target_value || return zero(T)
            # max_val = 2S; mz = encoded - S → 2·mz = 2·encoded - max_val.
            mz_doubled = 2 * mz_enc - ce.max_val
            phase *= T(mz_doubled // 2)
        elseif kind === SPIN_SPLUS
            mz_enc = get_span(state, 1, ce.word_idx, ce.bit_offset, ce.nbits)
            mz_enc == ce.target_value || return zero(T)
            mz_enc >= ce.max_val && return zero(T)
            # S+|S, m⟩ = √((S-m)(S+m+1)) |S, m+1⟩
            # m = mz_enc - S, S = max_val/2.
            S_f = ce.max_val / 2
            m_f = mz_enc - S_f
            coef = sqrt((S_f - m_f) * (S_f + m_f + 1))
            phase *= T(coef)
            set_span!(state, 1, ce.word_idx, ce.bit_offset, ce.nbits, mz_enc + 1)
        elseif kind === SPIN_SMINUS
            mz_enc = get_span(state, 1, ce.word_idx, ce.bit_offset, ce.nbits)
            mz_enc == ce.target_value || return zero(T)
            mz_enc == 0 && return zero(T)
            # S-|S, m⟩ = √((S+m)(S-m+1)) |S, m-1⟩
            S_f = ce.max_val / 2
            m_f = mz_enc - S_f
            coef = sqrt((S_f + m_f) * (S_f - m_f + 1))
            phase *= T(coef)
            set_span!(state, 1, ce.word_idx, ce.bit_offset, ce.nbits, mz_enc - 1)
        end
    end
    return phase
end

# Jordan-Wigner parity: count set bits *strictly below* (word_idx, bit_offset)
# in `state`. Returns 0 for even parity (no sign flip), 1 for odd.
@inline function _jw_parity(state::AbstractVector{UInt64}, word_idx::Int,
                            bit_offset::Int, nwords::Int)
    n = 0
    @inbounds for w in 1:word_idx-1
        n += count_ones(state[w])
    end
    @inbounds if bit_offset > 0
        mask_below = (UInt64(1) << bit_offset) - UInt64(1)
        n += count_ones(state[word_idx] & mask_below)
    end
    return n & 1
end

# =====================================================================
# Per-term prechecks
# =====================================================================

@inline function _term_precheck_fails(H::CompiledHamiltonian, term::CompiledTerm,
                                      buf::AbstractVector{UInt64}, off::Int)
    # Fermionic prechecks.
    if term.ann_mask_offset >= 0
        @inbounds for w in 1:H.nwords
            ann = H.masks_flat[term.ann_mask_offset + w]
            if (buf[off + w - 1] & ann) != ann
                return true
            end
        end
    end
    if term.cre_forbid_mask_offset >= 0
        @inbounds for w in 1:H.nwords
            cre = H.masks_flat[term.cre_forbid_mask_offset + w]
            if (buf[off + w - 1] & cre) != UInt64(0)
                return true
            end
        end
    end
    # Boson prechecks.
    if term.boson_req_count > 0
        @inbounds for k in 0:term.boson_req_count-1
            (entry_idx, min_occ, _max) = H.boson_reqs[term.boson_req_start + k]
            ce = H.entries[entry_idx]
            occ = get_span(buf, off, ce.word_idx, ce.bit_offset, ce.nbits)
            occ < min_occ && return true
        end
    end
    return false
end
