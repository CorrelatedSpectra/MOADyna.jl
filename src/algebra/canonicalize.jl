# =====================================================================
# Canonicalize — Tier-2 normal ordering for operator chains
# =====================================================================
#
# Within each statistics group (fermionic, bosonic, spin), operators are
# normal-ordered:
#   - Fermionic: cdag's left, c's right; sorted by (site_name, label) within
#                each side. Swaps generate -1 sign; same-mode c/cdag swaps
#                generate an additional δ (constant) term.
#   - Bosonic:   bdag's left, b's right; sorted similarly. Same-mode
#                swaps generate a +δ constant term, no sign flip.
#   - Spin:      different sites commute and are sorted; same-site spin
#                operators are LEFT IN USER ORDER (Tier-2 does not apply
#                [S+,S-]=2Sz; that's Tier 3 / `normal_order` in layer 2).
#
# Across statistics groups, operators commute and are sorted in canonical
# group order: fermionic < bosonic < spin.
#
# Pauli zeros (`c*c` or `c'*c'` on the same fermionic mode) vanish as a
# consequence of normal ordering.

"""
    canonicalize(chain) -> Vector{Tuple{Float64, Chain}}

Apply Tier-2 canonicalization to a chain. Returns a list of
`(coefficient_factor, canonical_chain)` pairs whose sum is algebraically
equal to the input chain (treated as the product of its entries).

A list is returned because (anti)commutators can spawn additional shorter
chains. An empty list means the chain reduces to zero.
"""
function canonicalize(chain::Chain)
    return _canonicalize(chain, 1.0)
end

# Recursive worker; carries a running coefficient factor.
function _canonicalize(chain::Chain, coef::Real)
    out = Tuple{Float64, Chain}[]
    _canonicalize!(out, chain, Float64(coef))
    return out
end

function _canonicalize!(out::Vector{Tuple{Float64, Chain}}, chain::Chain, coef::Float64)
    n = length(chain)
    if n <= 1
        push!(out, (coef, chain))
        return out
    end

    # Find the leftmost adjacent pair that's out of canonical order
    for i in 1:(n - 1)
        a = chain[i]; b = chain[i + 1]
        if _should_swap(a, b)
            # Optional constant term from (anti)commutator on same-mode complementary pair
            cterm = _swap_constant(a, b)
            if cterm != 0
                shorter = _drop_two(chain, i)
                _canonicalize!(out, shorter, coef * cterm)
            end
            # Swapped chain (always emitted)
            sgn = _swap_sign(a, b)
            swapped = _swap_two(chain, i)
            _canonicalize!(out, swapped, coef * sgn)
            return out
        end
    end

    # No inversion found — chain is in canonical order. Final check: Pauli zeros.
    for i in 1:(n - 1)
        if _is_pauli_zero(chain[i], chain[i + 1])
            return out  # drop entire term
        end
    end
    push!(out, (coef, chain))
    return out
end

# --- Helpers: chain manipulation ---

@inline _drop_two(chain::Chain, i::Int) = (chain[1:(i - 1)]..., chain[(i + 2):end]...)
@inline _swap_two(chain::Chain, i::Int) = (chain[1:(i - 1)]..., chain[i + 1], chain[i], chain[(i + 2):end]...)

# --- Statistics group classification ---

@inline _is_fermionic(e::LadderEntry) = e.kind === :c || e.kind === :cdag
@inline _is_bosonic(e::LadderEntry)   = e.kind === :b || e.kind === :bdag
@inline _is_spin(e::LadderEntry)      = e.kind === :Sx || e.kind === :Sy || e.kind === :Sz ||
                                       e.kind === :Splus || e.kind === :Sminus

@inline function _statistics_priority(e::LadderEntry)
    _is_fermionic(e) && return 0
    _is_bosonic(e)   && return 1
    return 2  # spin
end

@inline function _kind_priority(e::LadderEntry)
    k = e.kind
    (k === :cdag || k === :bdag) && return 0   # creations first within group
    (k === :c    || k === :b)    && return 1   # then annihilations
    # Spin priority order: Sx < Sy < Sz < S+ < S-  (arbitrary but consistent)
    k === :Sx     && return 0
    k === :Sy     && return 1
    k === :Sz     && return 2
    k === :Splus  && return 3
    k === :Sminus && return 4
    return 99
end

# Total ordering key for canonical position
@inline function _order_key(e::LadderEntry)
    return (_statistics_priority(e), _kind_priority(e), String(name(e.site)), e.label)
end

@inline _same_mode(a::LadderEntry, b::LadderEntry) = a.site == b.site && a.label == b.label

# --- Swap rules ---

"""
    _should_swap(a, b) -> Bool

True if `a` is currently before `b` in the chain but should come after in
canonical order. Same-site spin operators are NEVER swapped (Tier-2 leaves
the user's order intact within a site).
"""
function _should_swap(a::LadderEntry, b::LadderEntry)
    # Same-site spin pairs: do not swap (Tier 2 punt)
    if _is_spin(a) && _is_spin(b) && name(a.site) == name(b.site)
        return false
    end
    return _order_key(a) > _order_key(b)
end

"""
    _swap_constant(a, b) -> Float64

The constant (δ) generated when swapping `a*b` to `b*a`, for same-mode
complementary pairs. Zero otherwise.
"""
function _swap_constant(a::LadderEntry, b::LadderEntry)
    _same_mode(a, b) || return 0.0
    if _is_fermionic(a) && _is_fermionic(b)
        # c*cdag = δ - cdag*c — when we swap c↔cdag (so a=c, b=cdag)
        if a.kind === :c && b.kind === :cdag
            return 1.0
        end
        # cdag*c is already canonical, never reached via _should_swap
    elseif _is_bosonic(a) && _is_bosonic(b)
        # b*bdag = δ + bdag*b — swap b↔bdag
        if a.kind === :b && b.kind === :bdag
            return 1.0
        end
    end
    return 0.0
end

"""
    _swap_sign(a, b) -> Float64

Sign factor applied to the swapped term `b*a`. -1 for fermionic
anticommutation, +1 for everything else.
"""
function _swap_sign(a::LadderEntry, b::LadderEntry)
    if _is_fermionic(a) && _is_fermionic(b)
        return -1.0
    end
    return 1.0
end

"""
    _is_pauli_zero(a, b) -> Bool

True if `a*b` is identically zero by Pauli exclusion. Detected as adjacent
same-kind same-mode fermionic pairs.
"""
function _is_pauli_zero(a::LadderEntry, b::LadderEntry)
    _is_fermionic(a) && _is_fermionic(b) && a.kind === b.kind && _same_mode(a, b)
end
