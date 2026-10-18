# Count restricted suffixes so a small seed can be sampled without enumerating
# the allowed space. Bounds use encoded occupations, including spin offsets.
mutable struct _BasisCounter
    entries::Vector{ModeEntry}
    weights::Matrix{BigInt}
    lower::Matrix{BigInt}
    upper::Matrix{BigInt}
    dimensions::Vector{BigInt}
    cache::Vector{Dict{Tuple{Vararg{Tuple{BigInt,BigInt}}},BigInt}}
    work::Int
    max_work::Int
end

function _BasisCounter(enc::EncodingMap, restrictions, max_work::Int)
    entries = sort!(collect(values(enc.entries)); by=e -> (e.word_idx, e.bit_offset))
    compiled = [compile_restriction(r, enc) for r in restrictions]
    nr, nm = length(compiled), length(entries)
    indices = Dict((e.word_idx, e.bit_offset) => i for (i, e) in enumerate(entries))
    weights = zeros(BigInt, nr, nm)
    for (i, r) in enumerate(compiled)
        if r.op === POPCOUNT
            for (j, e) in enumerate(entries)
                weights[i, j] = (r.mask[e.word_idx] >> e.bit_offset) & 1
            end
        else
            for (e, w) in zip(r.mode_entries, r.weights)
                weights[i, indices[(e.word_idx, e.bit_offset)]] += w
            end
        end
    end
    lower, upper = zeros(BigInt, nr, nm + 1), zeros(BigInt, nr, nm + 1)
    dimensions = ones(BigInt, nm + 1)
    for j in nm:-1:1
        dimensions[j] = dimensions[j + 1] * (big(entries[j].max_val) + 1)
        for i in 1:nr
            extent = weights[i, j] * entries[j].max_val
            lower[i, j] = lower[i, j + 1] + min(0, extent)
            upper[i, j] = upper[i, j + 1] + max(0, extent)
        end
    end
    cache = [Dict{Tuple{Vararg{Tuple{BigInt,BigInt}}},BigInt}() for _ in 1:nm]
    bounds = Tuple((big(r.min), big(r.max)) for r in compiled)
    return _BasisCounter(entries, weights, lower, upper, dimensions, cache, 0, max_work), bounds
end

function _remaining_bounds(counter::_BasisCounter, bounds, pos, occupation)
    return Tuple((lo - counter.weights[i, pos] * occupation,
                  hi - counter.weights[i, pos] * occupation)
                 for (i, (lo, hi)) in enumerate(bounds))
end

function _count_completions(counter::_BasisCounter, pos::Int, bounds)::BigInt
    counter.work < counter.max_work || throw(ArgumentError(
        "automatic seed construction exceeded seed_max_work=$(counter.max_work); " *
        "supply an explicit seed or increase seed_max_work"))
    counter.work += 1
    # Clipping merges equivalent subproblems, including completed restrictions.
    clipped = Tuple((max(lo, counter.lower[i, pos]), min(hi, counter.upper[i, pos]))
                    for (i, (lo, hi)) in enumerate(bounds))
    any(b -> b[1] > b[2], clipped) && return big(0)
    all(i -> clipped[i] == (counter.lower[i, pos], counter.upper[i, pos]),
        eachindex(clipped)) && return counter.dimensions[pos]
    return get!(counter.cache[pos], clipped) do
        sum((_count_completions(counter, pos + 1,
                                _remaining_bounds(counter, clipped, pos, v))
             for v in 0:counter.entries[pos].max_val); init=big(0))
    end
end

function _sample_basis(h::Hilbert, restrictions, count::Int; rng, max_work::Int=100_000)
    count > 0 || throw(ArgumentError("seed_size must be positive"))
    max_work > 0 || throw(ArgumentError("seed_max_work must be positive"))
    rs = _restriction_list(restrictions)
    enc = EncodingMap(h)
    counter, bounds = _BasisCounter(enc, rs, max_work)
    dimension = _count_completions(counter, 1, bounds)
    count = Int(min(count, dimension))
    # Floyd's algorithm samples distinct ranks without storing the population.
    ranks = Set{BigInt}()
    for j in dimension-count:dimension-1
        rank = rand(rng, big(0):j)
        push!(ranks, rank in ranks ? j : rank)
    end
    states = zeros(UInt64, count * enc.nwords)
    for (i, rank) in enumerate(sort!(collect(ranks)))
        remaining = bounds
        for (pos, entry) in enumerate(counter.entries)
            for v in 0:entry.max_val
                next = _remaining_bounds(counter, remaining, pos, v)
                branch = _count_completions(counter, pos + 1, next)
                if rank < branch
                    set_span!(states, (i - 1) * enc.nwords + 1, entry, v)
                    remaining = next
                    break
                end
                rank -= branch
            end
        end
    end
    template = EagerBasis(h, rs, enc, enc.nwords, UInt64[], _next_basis_id())
    return _selected_basis(template, states; restrictions=rs), dimension
end
