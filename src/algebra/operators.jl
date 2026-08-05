# =====================================================================
# Operators — ladder/spin operator construction
# =====================================================================
#
# An operator term is a sequence of "ladder entries", each identifying a
# specific named action on a specific (site, label). Operators are built by
# constructor functions (c, c', b, b', S±, etc.) which return single-term
# OperatorSums; the user composes via +/*/scalar to build full Hamiltonians.

"""
    LadderKind

A `Symbol` alias used as the `.kind` field of a `LadderEntry`; rarely
needed directly — inspect it via `entry.kind` when iterating a term's chain.

Symbol identifying the kind of single-mode ladder/spin operator. Drives apply!
dispatch and adjoint mapping.

| Symbol     | Statistics  | Meaning                              |
|------------|-------------|--------------------------------------|
| `:c`       | Fermionic   | annihilation on a fermionic mode     |
| `:cdag`    | Fermionic   | creation on a fermionic mode         |
| `:b`       | Bosonic     | annihilation on a bosonic mode       |
| `:bdag`    | Bosonic     | creation on a bosonic mode           |
| `:Sx`      | Spin        | Sx component on a spin mode          |
| `:Sy`      | Spin        | Sy component                         |
| `:Sz`      | Spin        | Sz component                         |
| `:Splus`   | Spin        | S+ raising operator                  |
| `:Sminus`  | Spin        | S- lowering operator                 |
"""
const LadderKind = Symbol

"""
    LadderEntry{S<:AbstractSite, L}

A single-mode action: kind (`:c`, `:cdag`, …), the site it acts on, and the
canonical-form label within that site.
"""
struct LadderEntry{S<:AbstractSite, L}
    kind::LadderKind
    site::S
    label::L
end

# Equality and hashing — needed for OrderedDict keys
function Base.:(==)(a::LadderEntry, b::LadderEntry)
    a.kind === b.kind && a.site == b.site && a.label == b.label
end
Base.hash(e::LadderEntry, h::UInt) = hash(e.kind, hash(e.site, hash(e.label, h)))

Base.show(io::IO, e::LadderEntry) = print(io, e.kind, "(", e.site, ", ", e.label, ")")

# --- Adjoint mapping for kinds ---

const _ADJOINT_KIND = Dict(
    :c => :cdag,
    :cdag => :c,
    :b => :bdag,
    :bdag => :b,
    :Sx => :Sx,
    :Sy => :Sy,
    :Sz => :Sz,
    :Splus => :Sminus,
    :Sminus => :Splus,
)

adjoint_kind(k::LadderKind) = _ADJOINT_KIND[k]

# =====================================================================
# Single-entry constructors
# =====================================================================
#
# Each returns a single-term OperatorSum (defined in operator_sum.jl).
# We forward-declare _make_oneterm here; concrete definition lives there.

function _make_oneterm end   # filled in operator_sum.jl

# --- Fermionic ---

"""
    c(site::FermionSite, label...)

Annihilation operator on a fermionic mode of `site`, identified by `label`.
"""
function c(site::FermionSite, label...)
    canon = normalize_label(site, label...)
    return _make_oneterm(LadderEntry(:c, site, canon))
end

"""
    cdag(site::FermionSite, label...)

Creation operator on a fermionic mode (alias for `c'(site, label...)`).
"""
function cdag(site::FermionSite, label...)
    canon = normalize_label(site, label...)
    return _make_oneterm(LadderEntry(:cdag, site, canon))
end

# --- Bosonic ---

"""
    b(site::BosonSite)

Annihilation operator on the (single) bosonic mode of `site`.
"""
b(site::BosonSite) = _make_oneterm(LadderEntry(:b, site, ()))

"""
    bdag(site::BosonSite)

Creation operator on the (single) bosonic mode of `site`.
"""
bdag(site::BosonSite) = _make_oneterm(LadderEntry(:bdag, site, ()))

# --- Constructor adjoints: make c'(...) and b'(...) work ---
#
# Julia parses `c'(s, 1)` as `adjoint(c)(s, 1)`. Without these methods,
# adjoint of the function `c` is undefined and the call errors out. With
# them, c' is the function cdag and c'(...) returns the creation operator.
Base.adjoint(::typeof(c))    = cdag
Base.adjoint(::typeof(cdag)) = c
Base.adjoint(::typeof(b))    = bdag
Base.adjoint(::typeof(bdag)) = b

# --- Spin ---

"""
    Sx(site::SpinSite, mz)
    Sy(site::SpinSite, mz)
    Sz(site::SpinSite, mz)
    Splus(site::SpinSite, mz)
    Sminus(site::SpinSite, mz)

    Sx(site::FermionSite, orbital_pairs)
    Sy(site::FermionSite, orbital_pairs)
    Sz(site::FermionSite, orbital_pairs)
    Splus(site::FermionSite, orbital_pairs)
    Sminus(site::FermionSite, orbital_pairs)

Spin operators with two dispatch forms:

**`SpinSite` form** — single-`mz` spin operators on a `SpinSite`. The `mz`
argument selects which local Hilbert sub-space the operator acts on
(`-S ≤ mz ≤ S`).

**`FermionSite` form** — spin component summed over a list of
`(dn_idx, up_idx)` orbital-mode pairs on a `FermionSite`, where each tuple
gives the 1-based mode indices for the spin-down and spin-up electrons of one
orbital:

    Sz     = (1/2) Σ_α (n(α,↑) − n(α,↓))
    S₊     = Σ_α c†(α,↑) c(α,↓)
    S₋     = (S₊)†
    Sx     = (1/2)(S₊ + S₋)
    Sy     = (−i/2)(S₊ − S₋)

The `FermionSite` methods are defined in `interactions.jl`.
"""
function Sx(site::SpinSite, mz)
    canon = normalize_label(site, mz)
    return _make_oneterm(LadderEntry(:Sx, site, canon))
end

function Sy(site::SpinSite, mz)
    canon = normalize_label(site, mz)
    return _make_oneterm(LadderEntry(:Sy, site, canon))
end

function Sz(site::SpinSite, mz)
    canon = normalize_label(site, mz)
    return _make_oneterm(LadderEntry(:Sz, site, canon))
end

function Splus(site::SpinSite, mz)
    canon = normalize_label(site, mz)
    return _make_oneterm(LadderEntry(:Splus, site, canon))
end

function Sminus(site::SpinSite, mz)
    canon = normalize_label(site, mz)
    return _make_oneterm(LadderEntry(:Sminus, site, canon))
end

# --- Convenience: number operators ---

"""
    n(site::FermionSite, label...)

Number operator on a single fermionic mode: `c'(site, label) * c(site, label)`.
"""
n(site::FermionSite, label...) = cdag(site, label...) * c(site, label...)

"""
    n(site::FermionSite)

Total number operator on a fermionic site: sum of `n` over all modes.
"""
n(site::FermionSite{N}) where {N} = sum(n(site, i) for i in 1:N)

"""
    n_b(site::BosonSite)

Number operator on a bosonic site: `b'(site) * b(site)`.
"""
n_b(site::BosonSite) = bdag(site) * b(site)

# --- S(site) bundle for Heisenberg dot products ---

"""
    S(site::SpinSite)

Returns the bundle `(Sx_total, Sy_total, Sz_total)` summed over all mz on the
site. Use with `⋅` from `LinearAlgebra` to write Heisenberg products:

    H = J * sum(S(s[i]) ⋅ S(s[i+1]) for i = 1:L-1)
"""
function S(site::SpinSite{Sval}) where {Sval}
    Sx_t = sum(Sx(site, mz) for mz in (-Sval):1:Sval)
    Sy_t = sum(Sy(site, mz) for mz in (-Sval):1:Sval)
    Sz_t = sum(Sz(site, mz) for mz in (-Sval):1:Sval)
    return (Sx_t, Sy_t, Sz_t)
end
