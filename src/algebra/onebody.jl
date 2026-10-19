"""
    onebody(h, modes)
    onebody(h, hilbert::Hilbert)

Construct `sum(h[i,j] * cdag(modes[i]...) * c(modes[j]...))` from a square
one-electron matrix. `modes` is an ordered collection of `(FermionSite, label)`
pairs; a `Hilbert` supplies its canonical mode order. Rows create and columns
annihilate. Complex and non-Hermitian entries are preserved, and spin is not
duplicated. Zero entries contribute no terms.
"""
function onebody(h::AbstractMatrix{T}, modes) where {T<:Number}
    entries = collect(modes)
    n = length(entries)
    size(h) == (n, n) || throw(DimensionMismatch("matrix size must match the $n supplied modes"))
    all(isfinite, h) || throw(ArgumentError("one-body matrix entries must be finite"))
    annihilators = LadderEntry[]
    sites = Dict{Symbol,FermionSite}()
    for (site, label) in entries
        site isa FermionSite || throw(ArgumentError("onebody requires fermionic modes"))
        previous = get(sites, site.name, site)
        previous == site || throw(ArgumentError("conflicting definitions of site $(site.name)"))
        sites[site.name] = site
        push!(annihilators, LadderEntry(:c, site, normalize_label(site, label)))
    end
    allunique(annihilators) || throw(ArgumentError("modes must be distinct after label normalization"))
    creators = [LadderEntry(:cdag, a.site, a.label) for a in annihilators]
    out = OperatorSum{T}()
    for j in 1:n, i in 1:n
        iszero(h[i, j]) || _push!(out, h[i, j], (creators[i], annihilators[j]))
    end
    return out
end

function onebody(h::AbstractMatrix{<:Number}, hilbert::Hilbert)
    modes = [(site, label) for site in values(hilbert.sites)
             for label in mode_labels_canonical(site)]
    return onebody(h, modes)
end
