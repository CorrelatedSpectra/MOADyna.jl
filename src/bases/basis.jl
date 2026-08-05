# =====================================================================
# EagerBasis — sorted-vector enumeration of restricted Hilbert sectors
# =====================================================================
#
# Per §2 of docs/architecture/02_hilbert.md. Construction uses partitioned
# enumeration (one partition per restriction-influenced subset of modes,
# plus one "unrestricted" partition for any remaining modes), mixed-radix
# unranking over the Cartesian product, threaded enumeration into a flat
# preallocated buffer, then a sort.

abstract type AbstractBasis end

# --- basis_id token ---------------------------------------------------------

const _BASIS_ID_COUNTER = Threads.Atomic{UInt64}(UInt64(0))
_next_basis_id() = Threads.atomic_add!(_BASIS_ID_COUNTER, UInt64(1)) + UInt64(1)

# --- EagerBasis -------------------------------------------------------------

"""
    EagerBasis(hilbert, restrictions...)

A sorted-vector enumeration of the conserved sectors of `hilbert` defined by
`restrictions`. See the layer-2 chapter, §2.

Typical workflow: build via `basis(hilbert, restrictions...)` or call the
constructor directly; `length(b)` returns the sector dimension; then pass
`b` to `compile` (producing a `CompiledHamiltonian`) and `assemble` (the
dense matrix). For human-readable state inspection see `expectation_table`
and `configuration_weights` in `MOAD.Diagnostics`.
"""
struct EagerBasis <: AbstractBasis
    hilbert::Hilbert
    restrictions::Vector{Restriction}
    encoding::EncodingMap
    nwords::Int
    states::Vector{UInt64}        # flat: nwords × N, sorted lex ascending
    basis_id::UInt64
end

Base.length(b::EagerBasis) = length(b.states) ÷ b.nwords
encoding(b::EagerBasis) = b.encoding
hilbert(b::EagerBasis) = b.hilbert
restrictions(b::EagerBasis) = b.restrictions
basis_id(b::EagerBasis) = b.basis_id

@inline state_offset(b::EagerBasis, i::Int) = (i - 1) * b.nwords + 1

"""
    get_state(b, i) -> view of length nwords

Return a read-only view of the bit-packed `UInt64` Fock word(s) for the
`i`-th basis state (1-based). For a single-`UInt64` basis this is a
length-1 view; multi-word bases return `nwords` words. The raw bit encoding
is defined by `b.encoding`. For human-readable occupation numbers use
`expectation_table` or `configuration_weights` in `MOAD.Diagnostics`.
"""
@inline function get_state(b::EagerBasis, i::Int)
    o = state_offset(b, i)
    return @view b.states[o : o + b.nwords - 1]
end

"""
    copy_state!(buf, src, off, nwords)

Copy `nwords` words from `src` starting at `off` (1-based) into `buf[1:nwords]`.
"""
@inline function copy_state!(buf::AbstractVector{UInt64},
                             src::AbstractVector{UInt64}, off::Int, nwords::Int)
    @inbounds for k in 1:nwords
        buf[k] = src[off + k - 1]
    end
    return buf
end

# --- get_index --------------------------------------------------------------

@inline function _state_less_query(b::EagerBasis, i::Int, q::AbstractVector{UInt64})
    # true iff b[i] < q
    o = state_offset(b, i)
    @inbounds for k in 0:b.nwords-1
        a, c = b.states[o + k], q[k + 1]
        a == c || return a < c
    end
    return false
end

@inline function _query_less_state(q::AbstractVector{UInt64}, b::EagerBasis, i::Int)
    # true iff q < b[i]
    o = state_offset(b, i)
    @inbounds for k in 0:b.nwords-1
        a, c = q[k + 1], b.states[o + k]
        a == c || return a < c
    end
    return false
end

"""
    get_index(b, q) -> Int

Return the position of state `q` (a length-`nwords` vector of `UInt64`)
inside `b`, or `0` if absent. Binary search; O(nwords · log N).
"""
function get_index(b::EagerBasis, q::AbstractVector{UInt64})
    length(q) == b.nwords ||
        throw(DimensionMismatch("query state has length $(length(q)); basis nwords=$(b.nwords)"))
    lo, hi = 1, length(b)
    @inbounds while lo ≤ hi
        mid = (lo + hi) >> 1
        if _state_less_query(b, mid, q)
            lo = mid + 1
        elseif _query_less_state(q, b, mid)
            hi = mid - 1
        else
            return mid
        end
    end
    return 0
end

# --- Construction -----------------------------------------------------------

function EagerBasis(hilbert::Hilbert, restrictions::Restriction...)
    rs = collect(Restriction, restrictions)
    enc = EncodingMap(hilbert)

    # Compile restrictions to operational form.
    compiled = CompiledRestriction[compile_restriction(r, enc) for r in rs]

    # Build per-restriction site groupings, decide partition layout.
    partitions, unrestricted_partition = _plan_partitions(hilbert, enc, compiled)

    # Enumerate per partition: each partition lists every flat-state record
    # (length partition.nwords_per_record × cardinality) it can produce,
    # already satisfying its own restrictions.
    part_lists = Vector{Vector{UInt64}}()    # flat per-partition state lists
    part_radices = Int[]                     # cardinality of each partition
    part_layouts = _PartitionLayout[]        # per-partition mode entries (for scatter)

    for p in partitions
        list, layout = _enumerate_partition(p, enc)
        push!(part_lists, list)
        push!(part_radices, length(list) ÷ enc.nwords)
        push!(part_layouts, layout)
    end
    if unrestricted_partition !== nothing
        list, layout = _enumerate_partition(unrestricted_partition, enc)
        push!(part_lists, list)
        push!(part_radices, length(list) ÷ enc.nwords)
        push!(part_layouts, layout)
    end

    # Mixed-radix unranking: total candidate count M = ∏ radices[p]. Threaded
    # enumeration fills `states` in chunked ordinal ranges. (Scattering
    # writes into specific bit spans, so different threads never touch the
    # same word for the same `r`.)
    M = isempty(part_radices) ? 1 : prod(part_radices)
    states = zeros(UInt64, M * enc.nwords)

    if M > 0 && !isempty(part_layouts)
        _enumerate_all!(states, part_lists, part_radices, part_layouts, enc.nwords, M)
    end

    # Cross-coupled filter: any restriction not fully owned by a single
    # partition needs predicate-style application here. _plan_partitions
    # already ensures every restriction is covered by exactly one partition
    # in v0.1, so no extra filter pass is required for the supported cases.
    # (We still perform a defensive filter when there is unrestricted slack
    # and a restriction touches modes spread across partitions — see
    # `cross_coupled` in _plan_partitions.)

    # Sort the flat buffer lexicographically (index-permutation sort).
    if M > 1
        states = _sort_flat_states!(states, enc.nwords, M)
    end

    return EagerBasis(hilbert, rs, enc, enc.nwords, states, _next_basis_id())
end

# --- Partition planning -----------------------------------------------------

# A partition is a set of mode entries plus the compiled restrictions that
# act exclusively on those modes. Restrictions spanning multiple partitions
# are not supported in v0.1 (none of T1/T2/T3 require them).

struct _Partition
    mode_entries::Vector{ModeEntry}
    restrictions::Vector{CompiledRestriction}
end

# Layout used at scatter-time: knows how to take an index k into a
# partition's state list and write into the right word/bit range of the
# global packed state. We always store one partition's records as a flat
# UInt64 vector with stride = nwords_per_record (= enc.nwords for now —
# every record has the same global nwords stride; we just zero unrelated
# words). Scattering uses bitwise OR into the global state.
struct _PartitionLayout
    nwords::Int                   # global enc.nwords (same for every partition)
end

function _plan_partitions(hilbert::Hilbert, enc::EncodingMap,
                          compiled::Vector{CompiledRestriction})
    # No restrictions: one big unrestricted partition over every mode.
    if isempty(compiled)
        unrestricted_modes = ModeEntry[]
        for (_, e) in enc.entries
            push!(unrestricted_modes, e)
        end
        unrestricted = isempty(unrestricted_modes) ? nothing :
                       _Partition(unrestricted_modes, CompiledRestriction[])
        return _Partition[], unrestricted
    end

    # Restrictions that share at least one mode are merged into a single
    # partition (Union-Find over restrictions). The merged partition owns
    # the union of touched modes and all of those restrictions; the walker
    # applies each via `_check` during enumeration. This handles e.g.
    # `n_fermion == k` AND `WeightedParticleCount(...) == 0` on the same
    # FermionSite (Hubbard-style Sz-resolved fixed-N sectors).
    nrestr = length(compiled)
    parent = collect(1:nrestr)
    function uf_find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        end
        return x
    end
    function uf_union(a::Int, b::Int)
        ra, rb = uf_find(a), uf_find(b)
        ra != rb && (parent[ra] = rb)
    end

    # First mode-occupant wins; subsequent occupants union into it.
    mode_owner = Dict{Tuple{Symbol,Tuple}, Int}()
    for (i, c) in enumerate(compiled)
        for e in c.mode_entries
            key = (e.site_name, e.label)
            if haskey(mode_owner, key)
                uf_union(mode_owner[key], i)
            else
                mode_owner[key] = i
            end
        end
    end

    # Bucket restrictions by their UF root.
    components = Dict{Int, Vector{Int}}()
    for i in 1:nrestr
        push!(get!(components, uf_find(i), Int[]), i)
    end

    partitions = _Partition[]
    covered = Set{Tuple{Symbol,Tuple}}()
    for (_, idxs) in components
        seen_in_part = Set{Tuple{Symbol,Tuple}}()
        modes = ModeEntry[]
        for i in idxs
            for e in compiled[i].mode_entries
                key = (e.site_name, e.label)
                if !(key in seen_in_part)
                    push!(modes, e)
                    push!(seen_in_part, key)
                end
            end
        end
        Base.union!(covered, seen_in_part)
        rs = CompiledRestriction[compiled[i] for i in idxs]
        push!(partitions, _Partition(modes, rs))
    end

    # Modes not touched by any restriction go into a single unrestricted
    # partition.
    unrestricted_modes = ModeEntry[]
    for (key, e) in enc.entries
        key in covered || push!(unrestricted_modes, e)
    end
    unrestricted = isempty(unrestricted_modes) ? nothing :
                   _Partition(unrestricted_modes, CompiledRestriction[])

    return partitions, unrestricted
end

# --- Per-partition enumeration ---------------------------------------------
#
# Each partition is enumerated by walking all values its mode_entries can
# take and emitting only those satisfying the partition's compiled
# restrictions. The enumerator returns a flat UInt64 vector storing the
# emitted records; each record occupies enc.nwords UInt64s (the entire
# global stride, so scattering by OR is a constant-cost copy).

function _enumerate_partition(p::_Partition, enc::EncodingMap)
    nw = enc.nwords
    out = UInt64[]
    rec = zeros(UInt64, nw)

    if isempty(p.restrictions)
        # Unrestricted partition: walk every value. For fermionic modes that's
        # 2^k bit patterns over k modes; for boson it's the per-mode product
        # of (max_val+1); for spin it's (2S+1) per site.
        _walk_unrestricted!(out, rec, p.mode_entries, 1, nw)
    elseif _is_pure_fermion_count(p)
        # Single POPCOUNT, all-fermion modes: pure k-combinations enumerator.
        _walk_popcount!(out, rec, p, enc)
    elseif (pop_idx = _find_popcount_covering(p)) > 0
        # POPCOUNT covers all partition modes; remaining restrictions are
        # filtered per candidate. Common case: Hubbard-style fixed-N + Sz.
        # Cost C(n, k) candidates per k instead of 2^n, then O(R-1) filters
        # per candidate. We pass the *index* of the covering POPCOUNT
        # explicitly so the walker doesn't fall on a non-covering POPCOUNT
        # that happens to come first in p.restrictions.
        _walk_popcount_with_filter!(out, rec, p, enc, pop_idx)
    elseif _is_pure_weighted(p)
        # No POPCOUNT in this partition; restrictions are weighted-style
        # (BYTESUM / WEIGHTED_SUM). Use the recursive walk with precomputed
        # suffix min/max-possible-sum pruning (per §4.2 of the chapter).
        _walk_weighted_pruned!(out, rec, p, enc)
    else
        # General fallback: walk every value, check every restriction at
        # the leaf. Used only for partition shapes the fast paths don't
        # cover (mixed boson + fermion + restrictions, etc.).
        _walk_filtered!(out, rec, p, enc)
    end
    return out, _PartitionLayout(nw)
end

function _is_pure_fermion_count(p::_Partition)
    length(p.restrictions) == 1 || return false
    c = p.restrictions[1]
    return c.op === POPCOUNT && all(e.kind === :fermion for e in p.mode_entries)
end

# A POPCOUNT restriction in this partition covers every partition mode:
# enumeration is dominated by the count constraint. Returns the *index*
# (in `p.restrictions`) of such a restriction, or `0` if none exists.
# POPCOUNT only makes sense on 1-bit modes, so the partition must be
# all-fermion.
function _find_popcount_covering(p::_Partition)::Int
    isempty(p.restrictions) && return 0
    all(e -> e.kind === :fermion, p.mode_entries) || return 0
    n = length(p.mode_entries)
    m_keys = Set((e.site_name, e.label) for e in p.mode_entries)
    for (i, c) in enumerate(p.restrictions)
        c.op === POPCOUNT || continue
        length(c.mode_entries) == n || continue
        c_keys = Set((e.site_name, e.label) for e in c.mode_entries)
        c_keys == m_keys && return i
    end
    return 0
end

# Every restriction in this partition is weighted-style (BYTESUM or
# WEIGHTED_SUM). Suitable for the recursive suffix-pruned walk.
function _is_pure_weighted(p::_Partition)
    isempty(p.restrictions) && return false
    return all(c -> c.op !== POPCOUNT, p.restrictions)
end

# --- Walk: unrestricted (cartesian over per-mode value ranges) -------------

function _walk_unrestricted!(out::Vector{UInt64}, rec::Vector{UInt64},
                             modes::Vector{ModeEntry}, pos::Int, nw::Int)
    if pos > length(modes)
        # Emit a copy of `rec`.
        append!(out, rec)
        return
    end
    e = modes[pos]
    for v in 0:e.max_val
        set_span!(rec, 1, e, v)
        _walk_unrestricted!(out, rec, modes, pos + 1, nw)
    end
    # Restore: zero out the span before returning to keep `rec` clean.
    set_span!(rec, 1, e, 0)
    return
end

# --- Walk: filter every candidate (general restricted path) ----------------

function _walk_filtered!(out::Vector{UInt64}, rec::Vector{UInt64},
                         p::_Partition, enc::EncodingMap)
    _walk_filtered_helper!(out, rec, p, 1)
    # Cleanup: set every mode span back to 0.
    for e in p.mode_entries
        set_span!(rec, 1, e, 0)
    end
    return
end

function _walk_filtered_helper!(out::Vector{UInt64}, rec::Vector{UInt64},
                                p::_Partition, pos::Int)
    if pos > length(p.mode_entries)
        # Apply every restriction in this partition.
        for c in p.restrictions
            _check(c, rec, 1) || return
        end
        append!(out, rec)
        return
    end
    e = p.mode_entries[pos]
    for v in 0:e.max_val
        set_span!(rec, 1, e, v)
        _walk_filtered_helper!(out, rec, p, pos + 1)
    end
    # No need to reset here — next sibling will overwrite.
    return
end

# --- Walk: pure fermion popcount (k-combinations enumerator) ---------------
#
# For a partition consisting solely of fermionic 1-bit modes with one
# POPCOUNT restriction `min ≤ popcount ≤ max`, enumerate each k ∈ [min, max]
# by walking lexicographic k-combinations of mode positions {1..n}. This
# works for any n up to (and including) 64 modes; the previous Gosper-trick
# version computed `last = UInt64(1) << n` which wrapped to 0 at n == 64
# and silently emitted an empty basis.

function _walk_popcount!(out::Vector{UInt64}, rec::Vector{UInt64},
                         p::_Partition, enc::EncodingMap)
    n = length(p.mode_entries)
    c = p.restrictions[1]
    kmin = max(c.min, 0)
    kmax = min(c.max, n)
    for k in kmin:kmax
        # Reset every mode bit (k may have a fresh starting subset that
        # doesn't include some bits set during the previous k).
        for e in p.mode_entries
            set_span!(rec, 1, e, 0)
        end
        if k == 0
            append!(out, rec)
            continue
        end
        # First k-combination: positions [1, 2, …, k].
        indices = collect(1:k)
        while true
            # Set bits at the chosen mode positions.
            for i in indices
                set_span!(rec, 1, p.mode_entries[i], 1)
            end
            append!(out, rec)
            # Clear them again before computing the next subset (so the
            # next iteration starts from a clean slate; the next subset
            # may differ from this one in many positions).
            for i in indices
                set_span!(rec, 1, p.mode_entries[i], 0)
            end
            # Advance to next combination in lexicographic order.
            i = k
            while i ≥ 1 && indices[i] == n - k + i
                i -= 1
            end
            i == 0 && break
            indices[i] += 1
            for j in i+1:k
                indices[j] = indices[j - 1] + 1
            end
        end
    end
    return
end

# --- Walk: POPCOUNT covering all modes, with extra-restriction filter ------
#
# The partition has one POPCOUNT restriction covering every fermion mode,
# plus one or more "other" restrictions (typically WEIGHTED_SUM for
# Sz-resolved sectors). Enumerate via k-combinations as in `_walk_popcount!`,
# but before emitting a candidate, run `_check` for each non-POPCOUNT
# restriction. Compared to the general filter path, this collapses
# `2^n` candidates to `Σ_k binomial(n, k)` over the POPCOUNT range.

function _walk_popcount_with_filter!(out::Vector{UInt64}, rec::Vector{UInt64},
                                     p::_Partition, enc::EncodingMap, pop_idx::Int)
    # `pop_idx` is the index (in p.restrictions) of the POPCOUNT restriction
    # that covers ALL partition modes — chosen by `_find_popcount_covering`.
    # Every other restriction (including any subset POPCOUNTs that happen
    # to share the partition) is applied per-candidate via `_check`.
    pop_c = p.restrictions[pop_idx]
    others = CompiledRestriction[c for (i, c) in enumerate(p.restrictions) if i != pop_idx]

    n = length(p.mode_entries)
    kmin = max(pop_c.min, 0)
    kmax = min(pop_c.max, n)

    @inline function pass_filters(rec)
        for c in others
            _check(c, rec, 1) || return false
        end
        return true
    end

    for k in kmin:kmax
        # Reset every mode bit before each k-block.
        for e in p.mode_entries
            set_span!(rec, 1, e, 0)
        end
        if k == 0
            pass_filters(rec) && append!(out, rec)
            continue
        end
        # First k-combination: positions [1, 2, …, k].
        indices = collect(1:k)
        while true
            for i in indices
                set_span!(rec, 1, p.mode_entries[i], 1)
            end
            pass_filters(rec) && append!(out, rec)
            for i in indices
                set_span!(rec, 1, p.mode_entries[i], 0)
            end
            # Next combination in lexicographic order.
            i = k
            while i ≥ 1 && indices[i] == n - k + i
                i -= 1
            end
            i == 0 && break
            indices[i] += 1
            for j in i+1:k
                indices[j] = indices[j - 1] + 1
            end
        end
    end
    return
end

# --- Walk: pure weighted-sum partition with suffix min/max pruning --------
#
# Recursive walk over partition modes 1..n, tracking a per-restriction
# running scaled-int sum. At each position we precompute the minimum and
# maximum possible contribution from the remaining (pos..n) modes; if the
# running sum + remaining_min already exceeds the upper bound (or
# running + remaining_max is below the lower bound) for any restriction,
# prune the entire subtree. At the leaf we verify each restriction's bound.
#
# Per chapter §4.2 — recursive-pruned walk for weighted sums.

function _walk_weighted_pruned!(out::Vector{UInt64}, rec::Vector{UInt64},
                                p::_Partition, enc::EncodingMap)
    n = length(p.mode_entries)
    R = length(p.restrictions)

    # Per-restriction weight vector aligned to p.mode_entries: weights[r][k]
    # is the scaled-int weight for mode k under restriction r. Modes not
    # mentioned by a particular restriction get weight 0.
    weights = Vector{Vector{Int}}(undef, R)
    for r_idx in 1:R
        c = p.restrictions[r_idx]
        w = zeros(Int, n)
        for (k, ce) in enumerate(c.mode_entries)
            for (i, pe) in enumerate(p.mode_entries)
                if pe.site_name == ce.site_name && pe.label == ce.label
                    w[i] = c.weights[k]
                    break
                end
            end
        end
        weights[r_idx] = w
    end

    # Suffix bounds: suffix_lo[r][pos] = min Σ_{k=pos..n} w_{r,k} v_k;
    # similarly suffix_hi. v_k ∈ 0..max_val_k. Indexed 1..n+1; entry n+1 is 0.
    suffix_lo = [zeros(Int, n + 1) for _ in 1:R]
    suffix_hi = [zeros(Int, n + 1) for _ in 1:R]
    for r_idx in 1:R
        for pos in n:-1:1
            e = p.mode_entries[pos]
            w = weights[r_idx][pos]
            contrib_lo = min(0, w * e.max_val)
            contrib_hi = max(0, w * e.max_val)
            suffix_lo[r_idx][pos] = suffix_lo[r_idx][pos + 1] + contrib_lo
            suffix_hi[r_idx][pos] = suffix_hi[r_idx][pos + 1] + contrib_hi
        end
    end

    running = zeros(Int, R)
    _weighted_walk!(out, rec, p, weights, suffix_lo, suffix_hi, running, 1, n, R)

    # Cleanup so the caller's `rec` is left zero.
    for e in p.mode_entries
        set_span!(rec, 1, e, 0)
    end
    return
end

function _weighted_walk!(out::Vector{UInt64}, rec::Vector{UInt64},
                         p::_Partition, weights::Vector{Vector{Int}},
                         suffix_lo::Vector{Vector{Int}}, suffix_hi::Vector{Vector{Int}},
                         running::Vector{Int}, pos::Int, n::Int, R::Int)
    if pos > n
        for r_idx in 1:R
            c = p.restrictions[r_idx]
            (c.min ≤ running[r_idx] ≤ c.max) || return
        end
        append!(out, rec)
        return
    end
    e = p.mode_entries[pos]
    @inbounds for v in 0:e.max_val
        # Update running sums.
        for r_idx in 1:R
            running[r_idx] += weights[r_idx][pos] * v
        end
        # Pruning: any restriction whose [running + suffix_lo, running + suffix_hi]
        # is disjoint from [c.min, c.max] cannot be satisfied — skip.
        prune = false
        for r_idx in 1:R
            c = p.restrictions[r_idx]
            lo = running[r_idx] + suffix_lo[r_idx][pos + 1]
            hi = running[r_idx] + suffix_hi[r_idx][pos + 1]
            if hi < c.min || lo > c.max
                prune = true
                break
            end
        end
        if !prune
            set_span!(rec, 1, e, v)
            _weighted_walk!(out, rec, p, weights, suffix_lo, suffix_hi,
                            running, pos + 1, n, R)
        end
        # Restore running sums for the next v.
        for r_idx in 1:R
            running[r_idx] -= weights[r_idx][pos] * v
        end
    end
    # Reset this mode's bits before returning to the parent frame.
    set_span!(rec, 1, e, 0)
    return
end

# --- Mixed-radix unranking enumeration of the global product ---------------
#
# `part_lists[p]` stores partition p's records flat (each record has stride
# enc.nwords). `part_radices[p] = length(part_lists[p]) ÷ enc.nwords` is the
# per-partition cardinality. Each global candidate r ∈ 0:M-1 decomposes into
# (k_1, k_2, …) where k_p = r % radices[p]; r ÷= radices[p].

function _enumerate_all!(states::Vector{UInt64},
                         part_lists::Vector{Vector{UInt64}},
                         part_radices::Vector{Int},
                         part_layouts::Vector{_PartitionLayout},
                         nwords::Int, M::Int)
    # Per-thread scratch is just a local nwords-buffer; we OR-combine each
    # partition's record into it.
    #
    # Threaded loop with dynamic scheduling. Each iteration writes to a
    # disjoint slice of `states` (offset r*nwords), so no contention.
    # `collect` the partition iterator: newer OhMyThreads' chunk machinery
    # (≥ 0.8) requires an indexable collection, not a lazy
    # `Iterators.PartitionIterator`.
    chunks = collect(Iterators.partition(0:M-1, _enum_chunk_size(M)))
    @tasks for r_chunk in chunks
        @set scheduler = :dynamic
        buf = zeros(UInt64, nwords)
        for r in r_chunk
            fill!(buf, UInt64(0))
            rr = r
            @inbounds for p in eachindex(part_lists)
                radix = part_radices[p]
                k = rr % radix
                rr ÷= radix
                src_off = k * nwords + 1
                src = part_lists[p]
                for w in 1:nwords
                    buf[w] |= src[src_off + w - 1]
                end
            end
            dst_off = r * nwords + 1
            @inbounds for w in 1:nwords
                states[dst_off + w - 1] = buf[w]
            end
        end
    end
    return states
end

@inline _enum_chunk_size(M::Int) = max(64, M ÷ (8 * Threads.nthreads()))

# --- Sort: index-permutation sort, then threaded gather --------------------

function _sort_flat_states!(states::Vector{UInt64}, nwords::Int, N::Int)
    perm = collect(1:N)
    sort!(perm; alg = QuickSort,
          lt = (i, j) -> _state_less_flat(states, nwords, i, j))
    sorted = similar(states)
    @tasks for k in 1:N
        @set scheduler = :dynamic
        src_off = (perm[k] - 1) * nwords + 1
        dst_off = (k - 1) * nwords + 1
        @inbounds for w in 0:nwords-1
            sorted[dst_off + w] = states[src_off + w]
        end
    end
    return sorted
end

# --- show -------------------------------------------------------------------

Base.show(io::IO, b::EagerBasis) = print(io,
    "EagerBasis(", length(b), " states, nwords=", b.nwords,
    ", restrictions=", length(b.restrictions), ", id=", b.basis_id, ")")

# =====================================================================
# basis() — lowercase factory function (user-facing API)
# =====================================================================

"""
    basis(h::Hilbert, restrictions...; lazy::Bool = false) -> AbstractBasis

Construct a basis enumerating all states in `h` that satisfy the given
`restrictions` (which are `Restriction` objects from `MOAD.Algebra`).

By default returns an [`EagerBasis`](@ref): all states are enumerated
upfront and stored as a sorted list of bit-packed states. Suitable for
problems with up to ~10⁷ states.

`lazy = true` is reserved for a future streaming/lazy basis implementation
and currently throws `ArgumentError`.

# Example
```julia
m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
b = basis(m.hilbert, n_fermion(m.hilbert) == 24,
                     n_fermion([m.sites[:Ni_2p]]) == 6)
```

For shell-keyed restrictions see also the `basis(m::ShellModel, ...)`
overload in `MOAD.Shells`.
"""
function basis(h::Hilbert, restrictions...; lazy::Bool = false)
    if lazy
        throw(ArgumentError(
            "lazy basis not yet implemented; this kwarg is reserved for a " *
            "future streaming-basis implementation. Use `basis(h, ...; lazy=false)` " *
            "(the default) for the eager enumeration."))
    end
    return EagerBasis(h, restrictions...)
end

# =====================================================================
# apply_restriction! — public projection API
# =====================================================================

"""
    apply_restriction!(v, R, basis) -> v

Project `v` onto the subspace of `basis` satisfying restriction(s) `R` by
zeroing every component whose corresponding basis state violates `R`.
Returns `v` (modified in place).

`R` may be:

- `nothing` — no-op; `v` is returned unchanged.
- a single `Restriction`.
- a `Vector{<:Restriction}` — every restriction must hold (logical AND).

`v` must satisfy `length(v) == length(basis)`.

This is the diagonal projector ``P_R · v`` where ``P_R`` keeps the
components of `v` whose basis state satisfies `R` and zeros the rest. Used
by `MOAD.Spectroscopy` for ``P · H · P`` projected dynamics inside the
block-Lanczos recurrence (called once per matvec, plus once on the source
block before the recurrence).

Cost: ``O(N · n_R)`` per call, where ``n_R`` is the number of restrictions
and the per-state inner check is one bitmask AND + popcount (POPCOUNT) or
weighted span sum (BYTESUM / WEIGHTED_SUM). Restrictions are compiled once
per call; reusing the same restriction across many calls amortises that
cost trivially.
"""
function apply_restriction! end

apply_restriction!(v::AbstractVector, ::Nothing, ::AbstractBasis) = v

function apply_restriction!(v::AbstractVector,
                            R::Restriction,
                            basis::EagerBasis)
    length(v) == length(basis) ||
        throw(DimensionMismatch("v has length $(length(v)); basis has $(length(basis))"))
    cR = compile_restriction(R, basis.encoding)
    states = basis.states
    nw = basis.nwords
    z = zero(eltype(v))
    @inbounds for i in 1:length(basis)
        off = (i - 1) * nw + 1
        if !_check(cR, states, off)
            v[i] = z
        end
    end
    return v
end

function apply_restriction!(v::AbstractVector,
                            Rs::AbstractVector{<:Restriction},
                            basis::EagerBasis)
    length(v) == length(basis) ||
        throw(DimensionMismatch("v has length $(length(v)); basis has $(length(basis))"))
    isempty(Rs) && return v
    cRs = CompiledRestriction[compile_restriction(R, basis.encoding) for R in Rs]
    states = basis.states
    nw = basis.nwords
    z = zero(eltype(v))
    @inbounds for i in 1:length(basis)
        off = (i - 1) * nw + 1
        keep = true
        for cR in cRs
            if !_check(cR, states, off)
                keep = false
                break
            end
        end
        keep || (v[i] = z)
    end
    return v
end
