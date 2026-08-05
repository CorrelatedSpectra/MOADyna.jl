# src/shells/shell_model.jl
#
# ShellModel — registry mapping shell tags to fermionic mode ranges.
# Each tag like :Ni_3d allocates 2(2ℓ+1) fermionic modes per shell in
# m-major + dn-then-up order. The atom label and principal quantum
# number in the tag are user labels with no physical meaning; only
# the orbital character drives mode allocation.

"""
    ShellModel(shell_tags::Vector{Symbol})

Registry mapping shell tags (e.g. `:Ni_3d`) to fermionic mode ranges.

Each tag is parsed for orbital character (`s, p, d, f, g, h`) and
allocates `2 × (2ℓ + 1)` fermionic modes per shell, in m-major +
dn-then-up order. The atom label and principal quantum number in the
tag are user labels with no physical meaning.

# Example
```julia
m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
# 6 + 10 + 10 = 26 fermionic modes
```
"""
struct ShellModel
    hilbert::Hilbert
    shells::Vector{Symbol}
    ranges::Dict{Symbol, UnitRange{Int}}
    ell::Dict{Symbol, Int}
    sites::Dict{Symbol, FermionSite}
end

function ShellModel(tags::AbstractVector{Symbol})
    isempty(tags) && throw(ArgumentError("ShellModel requires at least one shell tag"))
    length(unique(tags)) == length(tags) || throw(ArgumentError(
        "ShellModel: duplicate shell tags in $tags"))

    sites = Dict{Symbol, FermionSite}()
    ells = Dict{Symbol, Int}()
    ranges = Dict{Symbol, UnitRange{Int}}()
    site_pairs = Pair{Symbol, AbstractSite}[]

    cumulative = 0
    for tag in tags
        parsed = _parse_shell_tag(tag)
        nmodes = 2 * (2 * parsed.ell + 1)
        site = _make_fermion_site(tag, nmodes)
        sites[tag] = site
        ells[tag] = parsed.ell
        ranges[tag] = (cumulative + 1):(cumulative + nmodes)
        push!(site_pairs, tag => site)
        cumulative += nmodes
    end

    h = Hilbert(site_pairs...)
    return ShellModel(h, collect(tags), ranges, ells, sites)
end

# Convenience: accept a tuple of symbols
ShellModel(tags::Tuple{Vararg{Symbol}}) = ShellModel(collect(tags))

# Convenience: accept varargs
ShellModel(tags::Symbol...) = ShellModel(collect(tags))

# Type-stable inner helper: produce a FermionSite{N} with the right N.
# Pre-instantiate for the common ℓ values (s/p/d/f/g/h).
_make_fermion_site(tag::Symbol, nmodes::Int) =
    _fermion_site_dispatch(Val(nmodes), tag)

_fermion_site_dispatch(::Val{2},  tag::Symbol) = FermionSite{2}(tag)   # s, ℓ=0
_fermion_site_dispatch(::Val{6},  tag::Symbol) = FermionSite{6}(tag)   # p, ℓ=1
_fermion_site_dispatch(::Val{10}, tag::Symbol) = FermionSite{10}(tag)  # d, ℓ=2
_fermion_site_dispatch(::Val{14}, tag::Symbol) = FermionSite{14}(tag)  # f, ℓ=3
_fermion_site_dispatch(::Val{18}, tag::Symbol) = FermionSite{18}(tag)  # g, ℓ=4
_fermion_site_dispatch(::Val{22}, tag::Symbol) = FermionSite{22}(tag)  # h, ℓ=5

# Catch-all fallback. The parser only accepts s/p/d/f/g/h orbitals so this
# branch is unreachable from normal user input — but if a future change
# extends `_parse_shell_tag` without also extending the dispatch table,
# we want a clear error rather than a confusing MethodError.
_fermion_site_dispatch(::Val{N}, tag::Symbol) where {N} =
    error("ShellModel: no FermionSite defined for $N modes (tag :$tag). " *
          "Supported shells are s/p/d/f/g/h (ℓ = 0..5).")

# --- Accessors ---

"""
    ell_of(m::ShellModel, shell::Symbol) -> Int

Orbital angular momentum ℓ of `shell` within model `m`.
Throws `KeyError` if `shell` is not registered in `m`.

See also: `range_of`, `site_of`.
"""
ell_of(m::ShellModel, shell::Symbol) = m.ell[shell]

"""
    range_of(m::ShellModel, shell::Symbol) -> UnitRange{Int}

The 1-based mode-index range occupied by `shell` in the underlying Hilbert.
Throws `KeyError` if `shell` is not registered in `m`.

See also: `ell_of`, `site_of`.
"""
range_of(m::ShellModel, shell::Symbol) = m.ranges[shell]

"""
    site_of(m::ShellModel, shell::Symbol) -> FermionSite

The `FermionSite{N}` underlying `shell`.

Note: the static return type is the abstract `FermionSite`, since different
shells in the same `ShellModel` may have different `N`. When concrete-`N`
dispatch matters at a call site where `N` is known statically, use a type
assertion: `site_of(m, :Ni_3d)::FermionSite{10}`.
"""
site_of(m::ShellModel, shell::Symbol) = m.sites[shell]

# --- Display ---

function Base.show(io::IO, m::ShellModel)
    tags_str = join((repr(s) for s in m.shells), ", ")
    print(io, "ShellModel([", tags_str, "])")
end

function Base.show(io::IO, ::MIME"text/plain", m::ShellModel)
    show(io, m)   # the compact form first
    n_total = sum(length(r) for r in values(m.ranges); init = 0)
    print(io, "  # ", n_total, " modes")
    for s in m.shells
        print(io, "\n  ", s, " : ℓ=", m.ell[s], ", modes ", m.ranges[s])
    end
end
