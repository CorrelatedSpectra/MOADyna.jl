# Flat lookup keeps determinant storage in the basis and stores only indices.
# The table is sized automatically at construction and never resized by mul!.
struct _StateIndex{I<:Integer}
    basis::EagerBasis
    slots::Vector{I}
end

@inline function _state_hash(states, offset, nwords)
    h = zero(UInt)
    @inbounds for k in 0:nwords-1
        h = hash(states[offset + k], h)
    end
    return h
end

function _StateIndex(b::EagerBasis)
    I = length(b) <= typemax(Int32) ? Int32 : Int
    capacity = nextpow(2, max(1, Base.checked_mul(2, length(b))))
    slots = zeros(I, capacity)
    for i in 1:length(b)
        slot = Int(_state_hash(b.states, state_offset(b, i), b.nwords) & UInt(capacity-1)) + 1
        while slots[slot] != 0
            slot = (slot & (capacity-1)) + 1
        end
        slots[slot] = i
    end
    return _StateIndex(b, slots)
end

@inline function _state_index(index::_StateIndex, state)
    b, slots = index.basis, index.slots
    mask = length(slots) - 1
    slot = Int(_state_hash(state, 1, b.nwords) & UInt(mask)) + 1
    @inbounds while slots[slot] != 0
        i = Int(slots[slot])
        offset = state_offset(b, i)
        equal = true
        for k in 1:b.nwords
            if b.states[offset + k - 1] != state[k]
                equal = false
                break
            end
        end
        equal && return i
        slot = (slot & mask) + 1
    end
    return 0
end

function _same_encoding(a::EncodingMap, b::EncodingMap)
    a === b && return true
    a.nwords == b.nwords && a.total_bits == b.total_bits &&
        a.site_kind == b.site_kind && a.site_nbits == b.site_nbits &&
        a.site_word_idx == b.site_word_idx && a.site_bit_offset == b.site_bit_offset &&
        a.site_max_vals == b.site_max_vals && keys(a.entries) == keys(b.entries) || return false
    return all(keys(a.entries)) do key
        x, y = a.entries[key], b.entries[key]
        all(name -> getfield(x, name) == getfield(y, name), fieldnames(ModeEntry))
    end
end
