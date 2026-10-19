function _read_band_tables(file)
    bands = Matrix{Float64}[]
    rows = Vector{Float64}[]
    for raw in eachline(file)
        line = strip(raw)
        if isempty(line) || occursin(r"(?i)^(?:#\s*band-index|bandindex:)", line)
            if !isempty(rows)
                push!(bands, permutedims(reduce(hcat, rows)))
                empty!(rows)
            end
        elseif !startswith(line, '#') && !startswith(line, '!')
            values = _w90_float.(split(line))
            isempty(rows) || length(values) == length(first(rows)) ||
                throw(ArgumentError("inconsistent column counts in $file"))
            push!(rows, values)
        end
    end
    isempty(rows) || push!(bands, permutedims(reduce(hcat, rows)))
    isempty(bands) && throw(ArgumentError("no band data found in $file"))
    all(band -> size(band) == size(first(bands)), bands) ||
        throw(ArgumentError("band blocks must have the same size"))
    all(band -> all(isfinite, band), bands) || throw(ArgumentError("band data must be finite"))
    return bands
end

function _bands_from_tables(tables, file; xcolumn=1, ecolumn=2, path=nothing,
                            distance_scale=1.0, energy_scale=1.0, atol=1e-7, rtol=0.0)
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative"))
    isfinite(rtol) && rtol >= 0 || throw(ArgumentError("rtol must be finite and nonnegative"))
    size(first(tables), 2) >= max(xcolumn, ecolumn) || throw(ArgumentError("not enough band-data columns"))
    distance = distance_scale * first(tables)[:, xcolumn]
    for table in tables
        all(i -> isapprox(distance_scale*table[i, xcolumn], distance[i]; atol, rtol), eachindex(distance)) ||
            throw(ArgumentError("bands do not share a common path"))
    end
    if path === nothing
        path = KPath(nothing, distance; breaks=findall(iszero, diff(distance)) .+ 1)
    else
        length(path.distance) == length(distance) || throw(DimensionMismatch("reference bands and supplied path have different lengths"))
        all(i -> isapprox(path.distance[i], distance[i]; atol, rtol), eachindex(distance)) ||
            throw(ArgumentError("reference distances disagree with the supplied path"))
    end
    energies = permutedims(reduce(hcat, [energy_scale * table[:, ecolumn] for table in tables]))
    return BandStructure(path, energies; metadata=Dict(:source => abspath(file)))
end

"""
    read_wannier_bands(seed_or_file; path=nothing, atol=1e-7, rtol=5e-8)

Read the bands in `seed_band.dat`, preserving the source energy zero in eV.
An optional `KPath` supplies coordinates and labels after a pointwise distance
check. Otherwise the result contains the file's distances and section breaks,
with unknown k coordinates. A third column, when present, is retained as
`metadata[:projection]` with the same band/point axes as the energies. The distance
tolerances accommodate the eight-significant-digit output format.
"""
function read_wannier_bands(seed_or_file::AbstractString; path=nothing, atol::Real=1e-7, rtol::Real=5e-8)
    file = isfile(seed_or_file) ? String(seed_or_file) : String(seed_or_file) * "_band.dat"
    tables = _read_band_tables(file)
    result = _bands_from_tables(tables, file; path, atol, rtol)
    if size(first(tables), 2) >= 3
        result.metadata[:projection] = permutedims(reduce(hcat, [table[:, 3] for table in tables]))
    end
    return result
end

function _wannier_path_from_win(win, lattice)
    haskey(win.blocks, "kpoint_path") || throw(ArgumentError("no kpoint_path block or complete band.kpt/labelinfo pair was found"))
    rows = [_w90_fields(line) for line in win.blocks["kpoint_path"]]
    !isempty(rows) && all(row -> length(row) == 8, rows) ||
        throw(ArgumentError("kpoint_path rows must contain two labels and two three-component points"))
    first_count = parse(Int, get(win.parameters, "bands_num_points", "100"))
    first_count > 0 || throw(ArgumentError("bands_num_points must be positive"))
    reciprocal = 2π * inv(lattice')
    starts = [_w90_float.(row[2:4]) for row in rows]
    stops = [_w90_float.(row[6:8]) for row in rows]
    lengths = [norm(reciprocal * (b-a)) for (a, b) in zip(starts, stops)]
    all(x -> isfinite(x) && x > 0, lengths) || throw(ArgumentError("Wannier path sections must have positive finite length"))
    counts = round.(Int, first_count .* lengths ./ first(lengths), RoundNearestTiesAway)
    all(>(0), counts) || throw(ArgumentError("a path section has no sampling intervals; provide band.kpt/labelinfo or increase bands_num_points"))
    points = Vector{Float64}[]
    distance = Float64[]
    labels, breaks = Pair{Int,String}[], Int[]
    x = 0.0
    for j in eachindex(rows)
        disconnected = j == 1 || sum(abs2, starts[j]-stops[j-1]) > 1e-6 || rows[j][1] != rows[j-1][5]
        if disconnected
            isempty(points) || push!(breaks, length(points)+1)
            push!(points, starts[j])
            push!(distance, x)
            push!(labels, length(points) => rows[j][1])
        end
        for m in 1:counts[j]
            push!(points, starts[j] + (m/counts[j]) * (stops[j]-starts[j]))
            x += lengths[j]/counts[j]
            push!(distance, x)
        end
        push!(labels, length(points) => rows[j][5])
    end
    return KPath(reduce(hcat, points), distance; labels, breaks)
end

function _read_wannier_kpoints(file)
    open(file) do io
        n = parse(Int, only(_w90_line(io)))
        n > 0 || throw(ArgumentError("band.kpt must contain at least one point"))
        points = zeros(3, n)
        for j in 1:n
            row = _w90_line(io)
            length(row) in (3, 4) || throw(ArgumentError("band.kpt rows need three coordinates and an optional weight"))
            points[:, j] = _w90_float.(row[1:3])
        end
        all(line -> isempty(_w90_fields(line)), eachline(io)) || throw(ArgumentError("extra points in band.kpt"))
        return points
    end
end

function _path_from_labelinfo(points, file, atol)
    labels = Pair{Int,String}[]
    declared = Float64[]
    for line in eachline(file)
        row = _w90_fields(line)
        isempty(row) && continue
        length(row) == 6 || throw(ArgumentError("labelinfo records need label, index, distance and three coordinates"))
        i = parse(Int, row[2])
        1 <= i <= size(points, 2) || throw(ArgumentError("labelinfo index lies outside band.kpt"))
        maximum(abs, points[:, i] - _w90_float.(row[4:6])) <= atol ||
            throw(ArgumentError("labelinfo and band.kpt coordinates disagree"))
        push!(labels, i => row[1])
        push!(declared, _w90_float(row[3]))
    end
    isempty(labels) && throw(ArgumentError("labelinfo contains no labels"))
    indices = first.(labels)
    issorted(indices) && allunique(indices) && issorted(declared) ||
        throw(ArgumentError("labelinfo records must follow path order"))
    first(indices) == 1 && last(indices) == size(points, 2) ||
        throw(ArgumentError("labelinfo must include both ends of band.kpt"))
    breaks = [first(labels[j]) for j in 2:length(labels)
              if indices[j] == indices[j-1]+1 && declared[j] == declared[j-1]]
    distance = zeros(size(points, 2))
    distance[1] = first(declared)
    for j in 2:length(labels)
        a, b = indices[j-1], indices[j]
        distance[a:b] = range(declared[j-1], declared[j]; length=b-a+1)
    end
    return KPath(points, distance; labels, breaks)
end

"""
    read_wannier_path(seed; lattice=nothing, atol=1e-6)

Read the explicit `seed_band.kpt` and `seed_band.labelinfo.dat` path when both
exist, keeping the printed k coordinates and uniformly interpolating the
more precise labelinfo distances within each section. This route needs no
lattice input. Otherwise reconstruct the path from `seed.win`, including proportional
interval counts and disconnected sections. A k-point file, if present alone,
is checked against this reconstruction.

For reconstruction, lattice vectors are columns in Å; by default they are read
from unit_cell_cart in the win file. `atol` controls componentwise coordinate
checks and accommodates band.kpt's six-decimal precision. The result is a [`KPath`](@ref).
"""
function read_wannier_path(path::AbstractString; lattice=nothing, atol::Real=1e-6)
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative"))
    seed = replace(_wannier_seed(path), r"_band\.(?:kpt|labelinfo\.dat)$" => "")
    pointfile, labelfile = seed * "_band.kpt", seed * "_band.labelinfo.dat"
    points = isfile(pointfile) ? _read_wannier_kpoints(pointfile) : nothing
    if points !== nothing && isfile(labelfile)
        return _path_from_labelinfo(points, labelfile, atol)
    end
    win = _read_wannier_win(seed * ".win")
    A = if lattice !== nothing
        _geometry_matrix(lattice, (3, 3), "lattice")
    elseif haskey(win.blocks, "unit_cell_cart")
        _win_lattice(win.blocks["unit_cell_cart"])
    else
        throw(ArgumentError("supply lattice=... when the win file contains no lattice vectors"))
    end
    rank(A) == 3 || throw(ArgumentError("lattice vectors must be independent"))
    generated = _wannier_path_from_win(win, A)
    if points !== nothing
        size(points) == size(generated.points) || throw(DimensionMismatch("band.kpt and win sampling counts differ"))
        maximum(abs, points - generated.points) <= atol || throw(ArgumentError("band.kpt and win paths disagree"))
        return KPath(points, generated.distance; labels=generated.labels, breaks=generated.breaks)
    end
    return generated
end
