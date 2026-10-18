"""
    KPath(points, distance; labels=[], breaks=[])

A sampled reciprocal-space path. Columns of `points` are fractional k vectors;
`distance` gives the plotting coordinates in Å⁻¹. Set `points=nothing`
for reference bands whose files contain only path distances.

`labels` contains `sample_index => label` pairs. Each index in `breaks` starts
a disconnected section. This constructor preserves the supplied distances,
including any gap between sections; [`kpath`](@ref) generates physical arc-length
coordinates with no contribution from such jumps. Arrays are copied.
"""
struct KPath
    points::Union{Nothing,Matrix{Float64}}
    distance::Vector{Float64}
    labels::Vector{Pair{Int,String}}
    breaks::Vector{Int}

    function KPath(points, distance; labels=[], breaks=[])
        x = Float64.(collect(distance))
        isempty(x) && throw(ArgumentError("a path must contain at least one point"))
        all(isfinite, x) && issorted(x) || throw(ArgumentError(
            "path distances must be finite and nondecreasing"))
        p = if points === nothing
            nothing
        else
            points isa AbstractMatrix{<:Real} || throw(ArgumentError(
                "k points must be a real matrix with one point per column"))
            size(points, 2) == length(x) || throw(DimensionMismatch(
                "the number of k points must match the path distances"))
            all(isfinite, points) || throw(ArgumentError("k points must be finite"))
            Matrix{Float64}(points)
        end
        ticks = Pair{Int,String}[Int(i) => string(label) for (i, label) in labels]
        all(tick -> 1 <= first(tick) <= length(x), ticks) ||
            throw(ArgumentError("label indices must lie within the path"))
        starts = sort!(unique(Int.(collect(breaks))))
        all(i -> 2 <= i <= length(x), starts) ||
            throw(ArgumentError("a path break must lie between the second and last point"))
        new(p, x, ticks, starts)
    end
end

"""
    kpath(model, vertices; intervals=100, labels=nothing, breaks=[])
    kpath(lattice, vertices; intervals=100, labels=nothing, breaks=[])

Interpolate straight sections through the columns of fractional `vertices`.
`intervals` is a positive integer for every section, or a vector with one count
per consecutive vertex pair. `labels` supplies one name per vertex.

An index in `breaks` starts a new section at that vertex, skipping the connection
from its predecessor. Distances use the reciprocal metric of `lattice`, whose
columns are direct vectors in Å. A model must have lattice geometry.
"""
function kpath(model::TightBindingModel, vertices; kwargs...)
    model.lattice === nothing && throw(ArgumentError("a physical k path requires a lattice"))
    return kpath(model.lattice, vertices; kwargs...)
end

function kpath(lattice::AbstractMatrix{<:Real}, vertices::AbstractMatrix{<:Real};
               intervals=100, labels=nothing, breaks=[])
    D, nv = size(vertices)
    nv > 0 || throw(ArgumentError("supply at least one path vertex"))
    A = _geometry_matrix(lattice, (D, D), "lattice")
    rank(A) == D || throw(ArgumentError("lattice vectors must be independent"))
    all(isfinite, vertices) || throw(ArgumentError("path vertices must be finite"))
    counts = intervals isa Integer ? fill(Int(intervals), nv-1) : Int.(collect(intervals))
    length(counts) == nv-1 || throw(DimensionMismatch("supply one interval count per vertex pair"))
    starts = Set(Int.(collect(breaks)))
    all(i -> 2 <= i <= nv, starts) || throw(ArgumentError("invalid path-break vertex"))
    all(j -> j+1 in starts || counts[j] > 0, eachindex(counts)) ||
        throw(ArgumentError("connected sections require positive interval counts"))
    names = labels === nothing ? fill("", nv) : string.(collect(labels))
    length(names) == nv || throw(DimensionMismatch("supply one label per vertex"))
    n = 1 + sum(j+1 in starts ? 1 : counts[j] for j in eachindex(counts); init=0)
    points, distance = Matrix{Float64}(undef, D, n), zeros(n)
    points[:, 1] = vertices[:, 1]
    ticks = Pair{Int,String}[]
    isempty(names[1]) || push!(ticks, 1 => names[1])
    path_breaks = Int[]
    reciprocal = 2π * inv(A')
    pos = 1
    for j in 2:nv
        if j in starts
            pos += 1
            points[:, pos] = vertices[:, j]
            distance[pos] = distance[pos-1]
            push!(path_breaks, pos)
        else
            delta = vertices[:, j] - vertices[:, j-1]
            length_step = norm(reciprocal * delta) / counts[j-1]
            for m in 1:counts[j-1]
                pos += 1
                points[:, pos] = vertices[:, j-1] + (m/counts[j-1]) * delta
                distance[pos] = distance[pos-1] + length_step
            end
        end
        isempty(names[j]) || push!(ticks, pos => names[j])
    end
    return KPath(points, distance; labels=ticks, breaks=path_breaks)
end

"""
    BandStructure(path, energies; metadata=Dict())

Band energies in eV, with bands along rows and k points along columns.
The energy reference is supplied by the source; construction does not shift it.
"""
struct BandStructure
    path::KPath
    energies::Matrix{Float64}
    metadata::Dict{Symbol,Any}

    function BandStructure(path::KPath, energies::AbstractMatrix{<:Real}; metadata=Dict())
        size(energies, 2) == length(path.distance) || throw(DimensionMismatch(
            "energy columns must match the path points"))
        all(isfinite, energies) || throw(ArgumentError("band energies must be finite"))
        new(path, Matrix{Float64}(energies), Dict{Symbol,Any}(metadata))
    end
end

"""
    bandstructure(model, path; gauge=:periodic)

Calculate sorted band energies along a `KPath` with known fractional k points.
The model's energy zero is preserved. The returned `BandStructure` can be plotted
when a RecipesBase-compatible plotting package is loaded.
"""
function bandstructure(model::TightBindingModel, path::KPath; gauge=:periodic)
    path.points === nothing && throw(ArgumentError(
        "this path contains distances only; supply fractional k points to evaluate bands"))
    nk = size(path.points, 2)
    energies = Matrix{Float64}(undef, size(model, 1), nk)
    work = Matrix{ComplexF64}(undef, size(model))
    for j in 1:nk
        bloch_hamiltonian!(work, model, view(path.points, :, j); gauge)
        energies[:, j] = eigvals!(Hermitian(work))
    end
    return BandStructure(path, energies; metadata=Dict(:gauge => gauge))
end
