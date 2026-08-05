# src/shells/mode_layout.jl
#
# Mode-layout helpers encoding the m-major + dn-then-up convention for
# fermionic shells. Local mode indices within a shell of orbital angular
# momentum ℓ are 1-based and run as
#   (m=-ℓ, dn), (m=-ℓ, up), (m=-ℓ+1, dn), (m=-ℓ+1, up), …, (m=ℓ, dn), (m=ℓ, up)
# giving 2(2ℓ+1) modes total. This matches Quanty's IndexDn/IndexUp layout
# (even = dn, odd = up within each shell) and saves a permutation in the
# QuantyIO regression path.

_check_ell(ell::Integer) =
    ell >= 0 || throw(ArgumentError("ell must be ≥ 0, got $ell"))

"""
    _m_values(ell) -> UnitRange{Int}

The orbital magnetic quantum numbers for orbital angular momentum ℓ:
`-ℓ, -ℓ+1, …, ℓ`. Returns a `UnitRange{Int}` of length `2ℓ+1`.

Throws `ArgumentError` if `ell < 0`.
"""
function _m_values(ell::Integer)
    _check_ell(ell)
    return -ell:ell
end

"""
    _orbital_mode_pair(ell, m) -> (dn_idx, up_idx)

Within a shell of orbital angular momentum ℓ, return the 1-based local
mode indices for the (m, ↓) and (m, ↑) fermionic modes under the
m-major + dn-then-up convention.

For ℓ=2, m=0 → (5, 6); ℓ=1, m=-1 → (1, 2).

Throws `ArgumentError` if `ell < 0` or if `m` is outside the valid range `-ell:ell`.
"""
function _orbital_mode_pair(ell::Integer, m::Integer)
    _check_ell(ell)
    -ell <= m <= ell || throw(ArgumentError(
        "m=$m out of range for ℓ=$ell (valid range: $(-ell):$ell)"))
    base = 2 * (m + ell)   # 0-indexed offset of the (m, dn) mode
    return (base + 1, base + 2)
end

"""
    _shell_mode_labels(ell) -> Vector{NamedTuple}

The full mode-label list for a shell of orbital angular momentum ℓ,
in m-major + dn-then-up order. Each entry is `(m = ..., sigma = :dn or :up)`.

For ℓ=2 returns a 10-element vector:
`[(m=-2,sigma=:dn), (m=-2,sigma=:up), (m=-1,sigma=:dn), …, (m=2,sigma=:up)]`.

Throws `ArgumentError` if `ell < 0`.
"""
function _shell_mode_labels(ell::Integer)
    _check_ell(ell)
    return [(m = m, sigma = σ) for m in -ell:ell for σ in (:dn, :up)]
end
