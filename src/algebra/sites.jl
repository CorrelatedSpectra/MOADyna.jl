# =====================================================================
# Sites — local Hilbert spaces
# =====================================================================
#
# Generic site types with no domain-specific physics. Multiplet semantics
# (atomic shells, orbital labels, point-group machinery) live in higher
# modules (MOAD.Shells) and produce these primitive sites with auxiliary
# label maps.

# --- Statistics tags ---

abstract type Statistics end
struct Fermionic <: Statistics end
struct Bosonic   <: Statistics end

# --- Abstract site ---

"""
    AbstractSite

Supertype for the local Hilbert spaces ("sites") a `Hilbert` is built from.
Concrete sites — `FermionSite`, `BosonSite`, `SpinSite` — carry a `name` and
define their local dimension, particle statistics, and mode labelling. Two
sites are equal iff they have the same type and the same name.
"""
abstract type AbstractSite end

"""
    name(site::AbstractSite) -> Symbol

Human-readable identifier for the site. Two sites are equal iff they have
the same type and the same name (see §1.4 / §8 of 01_algebra.md).
"""
function name end

Base.:(==)(a::AbstractSite, b::AbstractSite) = typeof(a) == typeof(b) && name(a) == name(b)
Base.hash(s::AbstractSite, h::UInt) = hash(name(s), hash(typeof(s), h))

# --- FermionSite{N} ---
# N indistinguishable fermionic modes, labeled 1..N.

"""
    FermionSite{N}(name::Symbol)

A site of `N` indistinguishable fermionic modes (e.g. the spin-orbitals of a
shell), labelled `1:N`. Local dimension `2^N`, fermionic statistics, encoded in
`N` bits. Build ladder operators on it with `c` / `cdag`.

```julia
s = FermionSite{2}(:imp)   # 2-mode (↑,↓) fermion site; local_dim(s) == 4
```
"""
struct FermionSite{N} <: AbstractSite
    name::Symbol
end

name(s::FermionSite) = s.name
local_dim(::FermionSite{N}) where {N} = 2^N
statistics(::FermionSite) = Fermionic()
encoding_bits(::FermionSite{N}) where {N} = N

mode_labels(::FermionSite{N}) where {N} = 1:N
mode_labels_canonical(::FermionSite{N}) where {N} = ((i,) for i in 1:N)

function normalize_label(::FermionSite{N}, i::Integer) where {N}
    1 <= i <= N || throw(ArgumentError(
        "mode $i out of range for FermionSite{$N}: valid labels are 1:$N"))
    return (Int(i),)
end
normalize_label(s::FermionSite, i::Tuple{Integer}) = normalize_label(s, i[1])

# --- BosonSite{Nmax} ---
# One bosonic mode per site, occupation 0..Nmax.

"""
    BosonSite{Nmax}(name::Symbol)

A single bosonic mode with occupation truncated to `0:Nmax`. Local dimension
`Nmax + 1`, bosonic statistics. Build ladder/number operators on it with `b` /
`bdag` / `n_b`.

```julia
s = BosonSite{5}(:ph)   # phonon mode with up to 5 quanta; local_dim(s) == 6
```
"""
struct BosonSite{Nmax} <: AbstractSite
    name::Symbol
end

name(s::BosonSite) = s.name
local_dim(::BosonSite{Nmax}) where {Nmax} = Nmax + 1
statistics(::BosonSite) = Bosonic()
encoding_bits(::BosonSite{Nmax}) where {Nmax} = ceil(Int, log2(Nmax + 1))

mode_labels(::BosonSite) = ((),)
mode_labels_canonical(::BosonSite) = ((),)

normalize_label(::BosonSite) = ()
normalize_label(::BosonSite, ::Tuple{}) = ()

# --- SpinSite{S} ---
# Spin-S, local dim 2S+1, labeled by mz ∈ -S:1:S.

"""
    SpinSite{S}(name::Symbol)

A spin-`S` site (`S` integer or half-integer), local dimension `2S + 1`, basis
states labelled by `mz ∈ -S:1:S`. Build spin operators on it with `Sz` /
`Splus` / `Sminus` / `Sx` / `Sy`.

```julia
s = SpinSite{1//2}(:q)   # spin-½ site; local_dim(s) == 2
```
"""
struct SpinSite{S} <: AbstractSite
    name::Symbol
end

name(s::SpinSite) = s.name
local_dim(::SpinSite{S}) where {S} = Int(2S + 1)
statistics(::SpinSite) = Bosonic()
encoding_bits(::SpinSite{S}) where {S} = ceil(Int, log2(2S + 1))

# mode_labels iterates mz from -S to +S in steps of 1
mode_labels(::SpinSite{S}) where {S} = (-S):1:S
mode_labels_canonical(::SpinSite{S}) where {S} = ((mz,) for mz in (-S):1:S)

normalize_label(s::SpinSite{S}, mz) where {S} = (_canonical_mz(S, mz),)
normalize_label(s::SpinSite{S}, mz::Tuple) where {S} = (_canonical_mz(S, mz[1]),)

# Canonical mz: a Rational that fits the spin S allowed values
function _canonical_mz(S, mz)
    canonical = Rational{Int}(mz)
    -S <= canonical <= S || throw(ArgumentError("mz=$mz outside allowed range -$S:$S"))
    # Half-integer S allows half-integer mz; integer S allows integer mz.
    isinteger(S - canonical) || throw(
        ArgumentError("mz=$mz incompatible with spin S=$S (step must be integer)")
    )
    return canonical
end

# --- Display ---

Base.show(io::IO, s::FermionSite{N}) where {N} = print(io, "FermionSite{$N}(:", s.name, ")")
Base.show(io::IO, s::BosonSite{Nmax}) where {Nmax} = print(io, "BosonSite{$Nmax}(:", s.name, ")")
Base.show(io::IO, s::SpinSite{S}) where {S} = print(io, "SpinSite{$S}(:", s.name, ")")
