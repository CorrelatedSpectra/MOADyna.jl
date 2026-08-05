"""
    configuration_weights(psi, basis::AbstractBasis, m::ShellModel;
                          group_by::Tuple{Vararg{Symbol}})
        -> Vector{Pair{NamedTuple, Float64}}

Bucket the basis states of `basis` by their per-shell fermion occupation and
sum `|ψ_i|²` per bucket.

Each basis state has a shell-resolved occupation tuple (one integer per
shell in `group_by`). The function walks the basis, computes the occupation
on every requested shell (by counting set bits in that shell's mode range
inside the bit-packed state), and accumulates the weight `abs2(psi[i])`
into a `Dict` keyed by the resulting `NamedTuple`. Buckets are returned
sorted by total weight in descending order.

# Arguments
- `psi`      — coefficient vector with `length(psi) == length(basis)`.
- `basis`    — the `AbstractBasis` `psi` is expanded over.
- `m`        — `ShellModel` providing the shell→mode-range map.

# Keyword arguments
- `group_by` (**required**) — non-empty tuple of shell tags (subset of
  `m.shells`) to bucket by.  There is no default; omitting this keyword
  is a method-dispatch error.  Passing an empty tuple raises
  `ArgumentError`.

# Returns
A `Vector{Pair{NamedTuple, Float64}}` sorted by weight descending. Each
key is a `NamedTuple` whose field names are the shell tags in `group_by`
and whose values are integer occupations.

# Example
```julia
m   = ShellModel([:Ni_3d, :L_3d])
bas = basis(m, n_fermion(m.hilbert) == 18)
sys = eigen(H, bas; n = 1)
ψ   = sys.vectors[:, 1]
configuration_weights(ψ, bas, m; group_by = (:Ni_3d, :L_3d))
# e.g. [(Ni_3d=8, L_3d=10) => 0.93, (Ni_3d=9, L_3d=9) => 0.06, ...]
```
"""
function configuration_weights(psi::AbstractVector,
                               basis::AbstractBasis,
                               m::ShellModel;
                               group_by::Tuple{Vararg{Symbol}})
    isempty(group_by) &&
        throw(ArgumentError("configuration_weights: group_by must not be empty"))
    length(psi) == length(basis) ||
        throw(DimensionMismatch(
            "configuration_weights: length(psi) = $(length(psi)) != length(basis) = $(length(basis))"))
    for s in group_by
        haskey(m.ranges, s) || throw(ArgumentError(
            "configuration_weights: shell $(repr(s)) not in ShellModel " *
            "(known shells: $(m.shells))"))
    end

    enc = encoding(basis)

    # Pre-resolve the ModeEntry list for each requested shell. Each shell
    # has length(range_of(m, s)) fermionic modes, labelled (1,)..(N,).
    shell_entries = Vector{Vector{ModeEntry}}(undef, length(group_by))
    for (k, s) in enumerate(group_by)
        n_modes = length(m.ranges[s])
        entries = Vector{ModeEntry}(undef, n_modes)
        for i in 1:n_modes
            entries[i] = enc.entries[(s, (i,))]
        end
        shell_entries[k] = entries
    end

    # Bucket dict: NamedTuple → Float64.
    NT = NamedTuple{group_by, NTuple{length(group_by), Int}}
    buckets = Dict{NT, Float64}()
    n_states = length(basis)

    occ_buf = Vector{Int}(undef, length(group_by))
    @inbounds for i in 1:n_states
        ψ_i = psi[i]
        weight = abs2(ψ_i)
        weight == 0.0 && continue
        off = (i - 1) * basis.nwords + 1
        for k in eachindex(shell_entries)
            occ_buf[k] = _count_shell_occupation(basis.states, off, shell_entries[k])
        end
        key = NT(Tuple(occ_buf))
        buckets[key] = get(buckets, key, 0.0) + weight
    end

    pairs = collect(buckets)
    sort!(pairs; by = p -> p.second, rev = true)
    return pairs
end

# Sum the bit values across every mode of one shell.
@inline function _count_shell_occupation(states::AbstractVector{UInt64},
                                         off::Int,
                                         entries::Vector{ModeEntry})
    c = 0
    @inbounds for e in entries
        c += get_bit(states, off, e.word_idx, e.bit_offset)
    end
    return c
end
