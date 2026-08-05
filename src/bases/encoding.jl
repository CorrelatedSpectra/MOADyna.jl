# =====================================================================
# EncodingMap — site/label/mode → bit position in packed Vector{UInt64}
# =====================================================================
#
# For a Hilbert with sites s_1, …, s_K, the encoding lays out per-mode bit
# spans contiguously in a flat bit-string and packs them into a UInt64 word
# vector. Per §3 of docs/architecture/02_hilbert.md:
#
#   - FermionSite{N}:  N bits, 1 per mode (occupation 0/1)
#   - BosonSite{Nmax}: ⌈log₂(Nmax+1)⌉ bits per site (occupation 0..Nmax)
#   - SpinSite{S}:     ⌈log₂(2S+1)⌉ bits per site, encoding mz - (-S)
#                      (so 0 ↔ mz=-S, monotone increasing).
#
# nwords = ⌈total_bits / 64⌉.

# --- Per-mode entry inside the EncodingMap ---

"""
    ModeEntry(site_name, label, word_idx, bit_offset, nbits, max_val, kind)

Locates one mode of one site inside a packed `Vector{UInt64}` state. Used by
`compile` (to lower `LadderEntry`s) and by `CompiledRestriction` (to build
masks).

* `word_idx`     — 1-based UInt64 word index inside the packed state
* `bit_offset`   — bit offset (0..63) inside that word
* `nbits`        — width of the bit span (1 for fermion; ≥1 for boson/spin)
* `max_val`      — maximum value the span can hold (1 for fermion; Nmax for
                   boson; 2S for spin's encoded mz)
* `kind`         — :fermion, :boson, or :spin
"""
struct ModeEntry
    site_name::Symbol
    label::Tuple                 # canonical label tuple, matches LadderEntry.label
    word_idx::Int
    bit_offset::Int
    nbits::Int
    max_val::Int
    kind::Symbol                 # :fermion | :boson | :spin
end

"""
    EncodingMap

A precomputed mapping from `(site_name, canonical_label) → ModeEntry`, plus
the global `nwords` count and the total bit width. Built once at basis
construction and consumed by `compile` and the basis enumerator.

Spans never straddle a UInt64 word boundary — the layout pads to the next
word when a span would not fit. Most realistic problems (≤ 64 fermionic
modes, or ≤ 21 boson sites with 3-bit spans) fit in `nwords == 1`.
"""
struct EncodingMap
    entries::Dict{Tuple{Symbol, Tuple}, ModeEntry}   # (site_name, label) → entry
    site_kind::Dict{Symbol, Symbol}                  # site_name → :fermion/:boson/:spin
    site_nbits::Dict{Symbol, Int}                    # per-site total bit count (sum of mode nbits)
    site_word_idx::Dict{Symbol, Int}                 # first word containing the site's span
    site_bit_offset::Dict{Symbol, Int}               # first bit of the site's span (within word)
    site_max_vals::Dict{Symbol, Vector{Int}}         # per-mode max_val, in mode order
    nwords::Int
    total_bits::Int
end

function EncodingMap(hilbert::Hilbert)
    entries = Dict{Tuple{Symbol, Tuple}, ModeEntry}()
    site_kind = Dict{Symbol, Symbol}()
    site_nbits = Dict{Symbol, Int}()
    site_word_idx = Dict{Symbol, Int}()
    site_bit_offset = Dict{Symbol, Int}()
    site_max_vals = Dict{Symbol, Vector{Int}}()

    bit_cursor = 0     # global bit position into the packed state

    for (site_name, site) in pairs(hilbert)
        kind = _site_kind(site)
        site_kind[site_name] = kind
        site_total = encoding_bits(site)

        # Reserve the site's span; pad to the next word boundary if it would
        # straddle. v0.1 acceptance tests fit in nwords == 1 so straddling
        # is rare; padding keeps every mode addressable inside a single word.
        word_idx_0based  = bit_cursor ÷ 64
        bit_in_word      = bit_cursor % 64
        if bit_in_word + site_total > 64
            bit_cursor   = (word_idx_0based + 1) * 64
            word_idx_0based  = bit_cursor ÷ 64
            bit_in_word      = 0
        end
        site_word_idx[site_name]    = word_idx_0based + 1   # 1-based
        site_bit_offset[site_name]  = bit_in_word
        site_nbits[site_name]       = site_total

        # Per-mode entries.
        max_vals = Int[]
        for (canon_label, span) in _mode_spans(site)
            mode_bit_off = bit_in_word + span.offset_in_site
            entry = ModeEntry(site_name, canon_label,
                              word_idx_0based + 1, mode_bit_off,
                              span.nbits, span.max_val, kind)
            entries[(site_name, canon_label)] = entry
            push!(max_vals, span.max_val)
        end
        site_max_vals[site_name] = max_vals

        bit_cursor += site_total
    end

    nwords = bit_cursor == 0 ? 1 : ((bit_cursor - 1) ÷ 64) + 1
    return EncodingMap(entries, site_kind, site_nbits, site_word_idx,
                       site_bit_offset, site_max_vals, nwords, bit_cursor)
end

# --- Per-site mode-span helpers --------------------------------------------

# Internal: a (canonical_label, span) iterator describing each mode within a
# site, where span tells us its offset inside the site's bit run, its bit
# width, and the max value it can hold.
struct _ModeSpan
    offset_in_site::Int
    nbits::Int
    max_val::Int
end

_site_kind(::FermionSite) = :fermion
_site_kind(::BosonSite)   = :boson
_site_kind(::SpinSite)    = :spin

function _mode_spans(s::FermionSite{N}) where {N}
    # One mode per bit. Mode 1 occupies bit 0 of the site span, mode 2 bit 1,
    # etc. `label` is the canonical 1-tuple `(i,)`.
    return [((Int(i),), _ModeSpan(i - 1, 1, 1)) for i in 1:N]
end

function _mode_spans(s::BosonSite{Nmax}) where {Nmax}
    nbits = encoding_bits(s)
    return [((), _ModeSpan(0, nbits, Nmax))]
end

function _mode_spans(s::SpinSite{Sval}) where {Sval}
    # Single mz axis, encoded as mz - (-S) ∈ 0:2S. The canonical label is the
    # mz value itself (Rational{Int}); but spin LadderEntries also act on a
    # specific mz, which selects |mz⟩ within the local Hilbert. For encoding
    # purposes there is one mode per site; we return a single entry under
    # the empty-tuple label so the encoding-lookup ignores the LadderEntry's
    # mz value (the operator dispatches on it during apply, not on encoding).
    nbits = encoding_bits(s)
    max_val = Int(2 * Sval)
    return [((), _ModeSpan(0, nbits, max_val))]
end

# --- Mode lookup ------------------------------------------------------------

"""
    mode_entry(enc, site_name, label) -> ModeEntry

Resolve an entry. For sites whose canonical label is `()` (boson, spin), the
caller may pass either `()` or any other label; only the site is used for
those kinds.
"""
function mode_entry(enc::EncodingMap, site_name::Symbol, label::Tuple)
    kind = enc.site_kind[site_name]
    if kind === :fermion
        return enc.entries[(site_name, label)]
    else
        return enc.entries[(site_name, ())]
    end
end

# --- Bit get/set on a flat buffer -------------------------------------------
#
# All hot-path accesses use a flat Vector{UInt64} `buf` plus a 1-based offset
# `off` (the start of the encoded state's `nwords` slice). This keeps the
# interface uniform across direct buffer access and basis-state access.

@inline function get_span(buf::AbstractVector{UInt64}, off::Int,
                          word_idx::Int, bit_offset::Int, nbits::Int)
    @inbounds w = buf[off + word_idx - 1]
    mask = (UInt64(1) << nbits) - UInt64(1)
    return Int((w >> bit_offset) & mask)
end

@inline function set_span!(buf::AbstractVector{UInt64}, off::Int,
                           word_idx::Int, bit_offset::Int, nbits::Int, value::Int)
    mask = (UInt64(1) << nbits) - UInt64(1)
    @inbounds w = buf[off + word_idx - 1]
    w &= ~(mask << bit_offset)
    w |= (UInt64(value) & mask) << bit_offset
    @inbounds buf[off + word_idx - 1] = w
    return nothing
end

@inline get_span(buf::AbstractVector{UInt64}, off::Int, e::ModeEntry) =
    get_span(buf, off, e.word_idx, e.bit_offset, e.nbits)

@inline set_span!(buf::AbstractVector{UInt64}, off::Int, e::ModeEntry, value::Int) =
    set_span!(buf, off, e.word_idx, e.bit_offset, e.nbits, value)

# --- Single-fermion-bit helpers (the hot-path case) -------------------------

@inline function get_bit(buf::AbstractVector{UInt64}, off::Int,
                         word_idx::Int, bit_offset::Int)
    @inbounds w = buf[off + word_idx - 1]
    return Int((w >> bit_offset) & UInt64(1))
end

@inline function set_bit!(buf::AbstractVector{UInt64}, off::Int,
                          word_idx::Int, bit_offset::Int)
    @inbounds buf[off + word_idx - 1] |= UInt64(1) << bit_offset
    return nothing
end

@inline function clear_bit!(buf::AbstractVector{UInt64}, off::Int,
                            word_idx::Int, bit_offset::Int)
    @inbounds buf[off + word_idx - 1] &= ~(UInt64(1) << bit_offset)
    return nothing
end

# --- Lexicographic comparator over flat storage (§2.1) ----------------------
#
# Used both at sort time (before any EagerBasis exists) and post-construction
# via the basis-aware wrapper.

@inline function _state_less_flat(buf::AbstractVector{UInt64}, nwords::Int, i::Int, j::Int)
    oi = (i - 1) * nwords + 1
    oj = (j - 1) * nwords + 1
    @inbounds for k in 0:nwords-1
        a, c = buf[oi + k], buf[oj + k]
        a == c || return a < c
    end
    return false
end

@inline function _state_equal_flat(buf::AbstractVector{UInt64}, nwords::Int, i::Int, j::Int)
    oi = (i - 1) * nwords + 1
    oj = (j - 1) * nwords + 1
    @inbounds for k in 0:nwords-1
        buf[oi + k] == buf[oj + k] || return false
    end
    return true
end
