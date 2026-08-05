# =====================================================================
# Hilbert — collection of named sites
# =====================================================================

"""
    Hilbert(pairs...)

A Hilbert space description: an ordered collection of named sites. Each entry
maps a `Symbol` key to an `AbstractSite`.

# Examples

    Hilbert(:s1 => FermionSite{2}(:s1), :s2 => FermionSite{2}(:s2))

    sites = [FermionSite{2}(Symbol("s\$i")) for i = 1:8]
    Hilbert(s.name => s for s in sites)
"""
struct Hilbert
    sites::OrderedDict{Symbol, AbstractSite}

    function Hilbert(sites::OrderedDict{Symbol, <:AbstractSite})
        for (k, v) in sites
            k === name(v) || throw(ArgumentError(
                "Hilbert: key :$k does not match site name :$(name(v)). " *
                "Site identity uses the name field for chain ids; " *
                "key and site name must match to avoid ambiguity."))
        end
        return new(OrderedDict{Symbol, AbstractSite}(sites))
    end
end

# Construction from pairs / iterables of pairs
Hilbert(pairs::Pair{Symbol, <:AbstractSite}...) = Hilbert(OrderedDict(pairs))
Hilbert(itr) = Hilbert(OrderedDict(itr))

# --- Accessors ---

Base.getindex(h::Hilbert, k::Symbol) = h.sites[k]
Base.haskey(h::Hilbert, k::Symbol) = haskey(h.sites, k)
Base.keys(h::Hilbert) = keys(h.sites)
Base.values(h::Hilbert) = values(h.sites)
Base.pairs(h::Hilbert) = pairs(h.sites)
Base.length(h::Hilbert) = length(h.sites)
Base.iterate(h::Hilbert, args...) = iterate(h.sites, args...)

sites(h::Hilbert) = collect(values(h.sites))

# --- Composition: ⊗ for disjoint union of Hilberts ---

function ⊗(h₁::Hilbert, h₂::Hilbert)
    overlap = intersect(keys(h₁.sites), keys(h₂.sites))
    isempty(overlap) || throw(ArgumentError(
        "Hilbert ⊗: site name(s) overlap between operands: $overlap"
    ))
    merged = OrderedDict{Symbol, AbstractSite}()
    for (k, v) in h₁.sites; merged[k] = v; end
    for (k, v) in h₂.sites; merged[k] = v; end
    return Hilbert(merged)
end

# --- Display ---

function Base.show(io::IO, h::Hilbert)
    print(io, "Hilbert(")
    first = true
    for (k, v) in h.sites
        first || print(io, ", ")
        print(io, ":", k, " => ", v)
        first = false
    end
    print(io, ")")
end
