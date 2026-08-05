# =====================================================================
# Observables algebra — typed conserved-quantity expressions
# =====================================================================
#
# Provides a small typed-expression vocabulary for conserved quantities
# (particle counts, total Sz) and the comparisons that turn them into
# Restriction objects. `MOAD.Bases` introspects Restrictions to drive
# symmetry-sector enumeration when the conserved quantity is recognised;
# otherwise it falls back to a predicate-filter pass over candidate states.

"""
    QuantumNumber

Supertype for the symbolic **abelian quantum numbers**: additive labels that are
diagonal in the occupation-number (Fock) basis. Comparing one to a value selects a
*coordinate* subspace of Fock states; that subspace is a genuinely *conserved* sector
precisely when the Hamiltonian commutes with the label — a property of the model, not of
the label itself. The concrete cases are [`ParticleCount`](@ref) (a U(1) particle
number), [`TotalSz`](@ref) (total ``S_z``), and [`WeightedParticleCount`](@ref)
(a general ``\\sum_i w_i n_i``), with the convenience builders [`n_fermion`](@ref),
[`n_boson`](@ref), [`Sz_total`](@ref).

The comparison produces a [`Restriction`](@ref), which `MOAD.Bases` recognises to
enumerate exactly the Fock states of the selected subspace. Non-abelian conserved
quantities (total ``S^2``,
``L^2``, ``J^2``, point-group irreps) are *not* `QuantumNumber`s — they are not
Fock-diagonal, so they cannot select a coordinate subspace; instead they are
applied to the diagonalized states as operators / projectors (e.g. `Ssqr`,
`project`, `classify_subspace`).

`QN` is a short alias for `QuantumNumber`.
"""
abstract type QuantumNumber end

"Short alias for [`QuantumNumber`](@ref)."
const QN = QuantumNumber

# --- ParticleCount ---

"""
    ParticleCount(sites; statistics)

Counts particles (fermions or bosons) on a list of sites. The type
parameter is `Fermionic` or `Bosonic`; prefer the convenience constructors
`n_fermion` / `n_boson` over calling this directly. Combine with `==` or
`∈` to form a `Restriction` for use in `basis(...)`:

```julia
n_fermion(hilbert) == 8          # exact sector
n_boson([boson_site]) ∈ 0:4      # bounded sector
```
"""
struct ParticleCount{Stat<:Statistics} <: QuantumNumber
    sites::Vector{AbstractSite}
    # Inner constructor accepts any iterable of sites; we collect into the
    # canonical Vector{AbstractSite} once, here.
    function ParticleCount{Stat}(sites) where {Stat<:Statistics}
        return new{Stat}(collect(AbstractSite, sites))
    end
end

# --- TotalSz ---

"""
    TotalSz(sites)

Total z-component of spin summed over the given sites. Restricted to
`SpinSite`s; bare `FermionSite`s carry no spin convention (the up/down
labelling is provided by `MOAD.Shells`, not by layer 1), so a fermionic
total-Sz is expressed via `WeightedParticleCount` instead — see the
**Hilbert spaces & bases** chapter of the manual.
"""
struct TotalSz <: QuantumNumber
    sites::Vector{AbstractSite}
    function TotalSz(sites)
        collected = collect(AbstractSite, sites)
        for s in collected
            s isa FermionSite && throw(ArgumentError(
                "TotalSz on a FermionSite (got :$(name(s))) is not defined: " *
                "bare FermionSite{N} carries no spin convention. Use " *
                "`WeightedParticleCount(sites, weights)` with explicit " *
                "+1//2 / -1//2 weights on the up/down modes."))
        end
        return new(collected)
    end
end

# --- WeightedParticleCount ---

"""
    WeightedParticleCount(sites, weights)

A linear combination ``Σᵢ wᵢ nᵢ`` over fermionic modes (or a mix thereof),
with one weight per mode in mode-concatenation order across `sites`.

`weights` must be `Vector{Int}` or `Vector{Rational{Int}}` — never `Float64`.
Sector-defining quantities must be exact at boundaries; layer 2 lowers
rationals to scaled integers via the LCM of denominators.

# Examples

    # Total Sz on two FermionSite{2}s (modes 1=up, 2=down each):
    WeightedParticleCount([s1, s2], [1//2, -1//2, 1//2, -1//2])

    # Sublattice imbalance (modes 1,2 on s1 are A; 1,2 on s2 are B):
    WeightedParticleCount([s1, s2], [1, 1, -1, -1])
"""
struct WeightedParticleCount{W<:Union{Int, Rational{Int}}} <: QuantumNumber
    sites::Vector{AbstractSite}
    weights::Vector{W}

    function WeightedParticleCount{W}(sites, weights::AbstractVector{W}) where {W<:Union{Int, Rational{Int}}}
        collected_sites = collect(AbstractSite, sites)
        # Verify every site is fermionic — only fermionic occupations are
        # bit-encoded as 0/1 per mode, which is what the weighted sum sums over.
        # Bosonic and spin sites have richer occupation domains and are
        # represented by ParticleCount{Bosonic} / TotalSz respectively.
        for s in collected_sites
            s isa FermionSite || throw(ArgumentError(
                "WeightedParticleCount currently supports only FermionSite; " *
                "got $(typeof(s)) for site :$(name(s)). For boson/spin " *
                "totals use ParticleCount / TotalSz."))
        end
        # Total number of modes across all sites must match weights length.
        nmodes = sum(_n_modes(s) for s in collected_sites; init = 0)
        length(weights) == nmodes || throw(ArgumentError(
            "WeightedParticleCount: weights length $(length(weights)) does not " *
            "match total mode count $nmodes across $(length(collected_sites)) site(s)."))
        return new{W}(collected_sites, collect(W, weights))
    end
end

# Type-inferring outer constructor: deduce W from the eltype of `weights`.
function WeightedParticleCount(sites, weights::AbstractVector{<:Union{Int, Rational{Int}}})
    W = eltype(weights)
    return WeightedParticleCount{W}(sites, weights)
end

# Reject Float weights explicitly with a clear message.
function WeightedParticleCount(::Any, ::AbstractVector{<:AbstractFloat})
    throw(ArgumentError(
        "WeightedParticleCount: Float weights are not accepted (sector-defining " *
        "quantities must be exact). Use `Int` or `Rational{Int}` weights " *
        "(e.g. `1//2` for ±½ Sz)."))
end

_n_modes(s::FermionSite{N}) where {N} = N
_n_modes(::AbstractSite) = 1   # not currently reached but keeps the helper total

# --- Convenience constructors over a Hilbert ---

"""
    n_fermion(hilbert) -> ParticleCount{Fermionic}
    n_fermion(sites)   -> ParticleCount{Fermionic}

Total fermion count over all FermionSites in the Hilbert (or over a
user-specified vector of sites).
"""
n_fermion(h::Hilbert) = ParticleCount{Fermionic}(_filter_sites(h, Fermionic))
n_fermion(sites::AbstractVector{<:AbstractSite}) = ParticleCount{Fermionic}(sites)

"""
    n_boson(hilbert) -> ParticleCount{Bosonic}

Total boson count over all BosonSites in the Hilbert. (Spin sites have
`statistics(...) == Bosonic()` but no particle-count meaning, so they are
excluded.)
"""
n_boson(h::Hilbert) = ParticleCount{Bosonic}(_filter_sites_kind(h, BosonSite))
n_boson(sites::AbstractVector{<:AbstractSite}) = ParticleCount{Bosonic}(sites)

"""
    Sz_total(hilbert) -> TotalSz
    Sz_total(sites)   -> TotalSz

Sum of Sz over `SpinSite`s. `FermionSite`s are NOT auto-summed: a bare
`FermionSite{N}` carries no spin convention (no built-in up/down
labelling), so `Sz_total` over fermions is ill-defined at layer 1. Use
`WeightedParticleCount(sites, weights)` with explicit `+1//2`/`-1//2`
weights instead — see §4.3 of `docs/architecture/02_hilbert.md`.

`BosonSite`s are also excluded (no spin meaning).
"""
Sz_total(h::Hilbert) = TotalSz(_spin_carrying_sites(h))
Sz_total(sites::AbstractVector{<:AbstractSite}) = TotalSz(sites)

# Direct misuse: Sz_total on a single FermionSite. Throw with a hint to
# WeightedParticleCount; do NOT silently lift to TotalSz([s]) (which would
# itself throw inside the inner constructor, but with a less direct hint).
function Sz_total(s::FermionSite, args...)
    throw(ArgumentError(
        "Sz_total(::FermionSite) is not defined: bare FermionSite{N} carries " *
        "no spin convention. Use `WeightedParticleCount([s], weights)` with " *
        "explicit ±1//2 weights, e.g. " *
        "`WeightedParticleCount([s], [1//2, -1//2, 1//2, -1//2])` for a " *
        "two-orbital up/down/up/down ordering. See §4.3 of " *
        "docs/architecture/02_hilbert.md."))
end

function _spin_carrying_sites(h::Hilbert)
    out = AbstractSite[]
    for site in values(h.sites)
        # Only SpinSites carry an unambiguous spin label. FermionSites are
        # excluded; users requesting Sz over fermions must say so via
        # WeightedParticleCount.
        site isa SpinSite && push!(out, site)
    end
    return out
end

# --- Internal helpers ---

function _filter_sites(h::Hilbert, ::Type{S}) where {S<:Statistics}
    out = AbstractSite[]
    for site in values(h.sites)
        statistics(site) isa S && push!(out, site)
    end
    return out
end

function _filter_sites_kind(h::Hilbert, ::Type{T}) where {T<:AbstractSite}
    out = AbstractSite[]
    for site in values(h.sites)
        site isa T && push!(out, site)
    end
    return out
end

# --- Composition: ParticleCount + ParticleCount ---

"""
    +(q₁::ParticleCount, q₂::ParticleCount)

Compose two same-statistics particle counts on (possibly disjoint) site
lists into one ParticleCount over the union.
"""
function Base.:+(q₁::ParticleCount{Stat}, q₂::ParticleCount{Stat}) where {Stat}
    return ParticleCount{Stat}(unique(vcat(q₁.sites, q₂.sites)))
end

# Composition for TotalSz
Base.:+(q₁::TotalSz, q₂::TotalSz) = TotalSz(unique(vcat(q₁.sites, q₂.sites)))

# Composition for WeightedParticleCount: concatenate site lists and weights.
# Distinct sites only — overlapping site lists would require resolving which
# weights apply, which is not unambiguously defined. Promotes weight types
# (e.g. Int + Rational{Int} → Rational{Int}).
function Base.:+(q₁::WeightedParticleCount, q₂::WeightedParticleCount)
    overlap = intersect(Set(name(s) for s in q₁.sites), Set(name(s) for s in q₂.sites))
    isempty(overlap) || throw(ArgumentError(
        "WeightedParticleCount + WeightedParticleCount: site name overlap $overlap. " *
        "Compose disjoint site lists, or build a single WeightedParticleCount " *
        "with the full weight vector."))
    W = promote_type(eltype(q₁.weights), eltype(q₂.weights))
    return WeightedParticleCount{W}(vcat(q₁.sites, q₂.sites),
                                    vcat(W.(q₁.weights), W.(q₂.weights)))
end

# =====================================================================
# Restriction
# =====================================================================

"""
    Restriction(quantity, bounds)

A constraint that the value of `quantity` (a QuantumNumber) lies in
`bounds` (a UnitRange of integers, or for TotalSz a UnitRange of half-integers).

Construct via comparison operators on a QuantumNumber:

    n_fermion(hilbert) == 8
    n_fermion(d_sites) ∈ 6:8
    Sz_total(hilbert) == 0
"""
struct Restriction{Q<:QuantumNumber, B}
    quantity::Q
    bounds::B
end

# `==` makes a singleton-bound restriction. One method accepting Real
# covers Int, Rational, Float — bounds become a UnitRange or StepRangeLen
# depending on the value type, both of which layer 2 can consume.
Base.:(==)(q::QuantumNumber, n::Real) = Restriction(q, n:n)
Base.:(==)(n::Real, q::QuantumNumber) = q == n

# `∈` for explicit ranges
Base.:in(q::QuantumNumber, r::AbstractRange) = Restriction(q, r)

# `≥` / `≤` for one-sided bounds (integer-only — these counts are always integral)
Base.:(>=)(q::QuantumNumber, n::Integer) = Restriction(q, n:typemax(Int))
Base.:(<=)(q::QuantumNumber, n::Integer) = Restriction(q, 0:n)

# Reject Float bounds for WeightedParticleCount/TotalSz with a clear message:
# the same exactness argument as for the weights themselves applies.
Base.:(==)(q::WeightedParticleCount, ::AbstractFloat) =
    throw(ArgumentError("WeightedParticleCount == Float is not accepted; pass " *
                        "Int or Rational{Int} for exact sector boundaries."))
Base.:(==)(::AbstractFloat, q::WeightedParticleCount) =
    throw(ArgumentError("WeightedParticleCount == Float is not accepted; pass " *
                        "Int or Rational{Int} for exact sector boundaries."))

# --- Display ---

Base.show(io::IO, q::ParticleCount{Fermionic}) =
    print(io, "n_fermion[", join((name(s) for s in q.sites), ", "), "]")
Base.show(io::IO, q::ParticleCount{Bosonic}) =
    print(io, "n_boson[", join((name(s) for s in q.sites), ", "), "]")
Base.show(io::IO, q::TotalSz) =
    print(io, "Sz_total[", join((name(s) for s in q.sites), ", "), "]")
Base.show(io::IO, q::WeightedParticleCount) =
    print(io, "WeightedParticleCount[", join((name(s) for s in q.sites), ", "),
              "; weights=", q.weights, "]")
Base.show(io::IO, r::Restriction) =
    print(io, r.quantity, " ∈ ", r.bounds)
