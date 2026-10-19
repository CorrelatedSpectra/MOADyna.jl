"""
    cluster_sites(lattice, positions; centres, neighbors=1:size(positions,2),
                  radius, atol=1e-10)

Select sites around one or more centres in a periodic structure. Columns of
`lattice` are direct lattice vectors and columns of `positions` are Cartesian
site positions, both in Å. Positions need not be wrapped into the home cell.

Each entry in `centres` is a site index in the reference cell, or an explicit
`(cell, site)` pair. `neighbors` lists reference-cell site indices whose
periodic images may be added. Include an image when its distance to any centre
is at most `radius + atol`; both distances are in Å. The search includes all
images within that distance, including for nonorthogonal lattices.

Return ordered `(cell, site)` pairs: distinct centres first in their supplied
order, then the remaining images sorted by cell and site. Centres are always
included, even when absent from `neighbors`. Shared neighbors appear once.
Site indices retain their meaning in the supplied `positions`; for atomic
positions, map the selected atoms to orbitals before calling [`finite_cluster`](@ref).

# Example
```julia
sites = cluster_sites(lattice, atoms; centres=[1, 2], neighbors=3:9, radius=2.5)
```
"""
function cluster_sites(lattice::AbstractMatrix, positions::AbstractMatrix;
                       centres, neighbors=1:size(positions, 2), radius::Real,
                       atol::Real=1e-10)
    D, n = size(positions)
    D > 0 || throw(ArgumentError("positions must have at least one coordinate"))
    A = _geometry_matrix(lattice, (D, D), "lattice")
    x = _geometry_matrix(positions, (D, n), "positions")
    rank(A) == D || throw(ArgumentError("lattice vectors must be independent"))
    isfinite(radius) && radius >= 0 && isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("radius and atol must be finite and nonnegative"))
    distance = Float64(radius + atol)
    isfinite(distance) || throw(ArgumentError("radius + atol must be finite"))
    inverse = inv(A)
    bounds = distance .* [norm(row) for row in eachrow(inverse)]

    selected = Tuple{NTuple{D,Int},Int}[]
    for centre in centres
        cell, site = centre isa Integer ? (ntuple(_ -> 0, D), centre) : centre
        site isa Integer && 1 <= site <= n ||
            throw(ArgumentError("centre site index must lie in 1:$n"))
        push!(selected, (_cell_tuple(cell, Val(D)), Int(site)))
    end
    unique!(selected)
    isempty(selected) && throw(ArgumentError("supply at least one centre"))
    candidates = unique(collect(neighbors))
    all(i -> i isa Integer && 1 <= i <= n, candidates) ||
        throw(ArgumentError("neighbor site indices must lie in 1:$n"))
    images = Set{Tuple{NTuple{D,Int},Int}}()
    for (central_cell, central_site) in selected, site in candidates
        displacement = x[:, central_site] + A*collect(central_cell) - x[:, site]
        midpoint = inverse * displacement
        # Cauchy–Schwarz bounds every integer image inside the Cartesian sphere.
        # Round outward so the distance check decides inclusion at boundaries.
        ranges = ntuple(i -> floor(Int, midpoint[i]-bounds[i]):
                             ceil(Int, midpoint[i]+bounds[i]), D)
        for cell in Iterators.product(ranges...)
            norm(A*collect(cell) - displacement) <= distance && push!(images, (cell, site))
        end
    end
    setdiff!(images, selected)
    append!(selected, sort!(collect(images)))
    return selected
end
