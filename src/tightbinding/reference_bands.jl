function _band_indices(file, pattern, n)
    indices = Int[]
    for line in eachline(file)
        found = match(pattern, strip(line))
        found === nothing || push!(indices, parse(Int, found[1]))
    end
    isempty(indices) && return collect(1:n)
    length(indices) == n && all(>(0), indices) && allunique(indices) ||
        throw(ArgumentError("invalid or incomplete band indices in $file"))
    return indices
end

function _vaspkit_labels(file, distance, atol)
    labels = Pair{Int,String}[]
    for line in eachline(file)
        row = split(strip(line))
        isempty(row) && continue
        startswith(row[1], '*') && break
        lowercase(row[1]) == "k-label" && continue
        length(row) == 2 || throw(ArgumentError("KLABELS rows need a label and distance"))
        x = _w90_float(row[2])
        delta, i = findmin(abs.(distance .- x))
        delta <= atol || throw(ArgumentError("KLABELS point $(row[1]) does not lie on the band path"))
        push!(labels, i => row[1])
    end
    return labels
end

"""
    read_vaspkit_bands(file; labels_file=..., path=nothing, energy_column=2, atol=5e-4)

Read block-form `BAND.dat` or `PBAND_*.dat` output from VASPKIT. Distances are
in Å⁻¹ and energies in eV, retaining the file's energy zero (usually its Fermi
level). `energy_column` selects one energy column for spin-resolved BAND files.

The adjacent `KLABELS` file supplies tick labels when present; set
`labels_file=nothing` to omit it. Its positions are matched within `atol`, which
allows for the usual three-decimal output. An explicit `path` instead supplies
coordinates and labels after checking distances. K coordinates otherwise remain
unknown. Original band indices and additional named columns, including orbital
projections, are retained in `metadata[:band_indices]` and `metadata[:columns]`.
"""
function read_vaspkit_bands(file::AbstractString;
                            labels_file=isfile(joinpath(dirname(file), "KLABELS")) ? joinpath(dirname(file), "KLABELS") : nothing,
                            path=nothing, energy_column::Integer=2, atol::Real=5e-4)
    energy_column >= 2 || throw(ArgumentError("energy_column must follow the path-distance column"))
    tables = _read_band_tables(file)
    # VASPKIT can alternate the traversal direction between bands.
    for j in eachindex(tables)
        issorted(view(tables[j], :, 1); rev=true) && (tables[j] = reverse(tables[j]; dims=1))
    end
    result = _bands_from_tables(tables, file; path, ecolumn=energy_column, atol)
    if path === nothing && labels_file !== nothing
        p = result.path
        labels = _vaspkit_labels(labels_file, p.distance, atol)
        result = BandStructure(KPath(p.points, p.distance; labels, breaks=p.breaks),
                               result.energies; metadata=result.metadata)
    end
    result.metadata[:band_indices] = _band_indices(file, r"(?i)^#\s*band-index\s+(\d+)", length(tables))
    header = open(file) do io
        split(lstrip(strip(readline(io)), '#'))
    end
    ncols = size(first(tables), 2)
    length(header) == ncols || throw(ArgumentError("VASPKIT column header does not match its data"))
    allunique(header) || throw(ArgumentError("duplicate VASPKIT column labels"))
    result.metadata[:columns] = Dict(String(header[j]) =>
        permutedims(reduce(hcat, [table[:, j] for table in tables]))
        for j in 2:ncols if j != energy_column)
    return result
end

function _wien_insp_file(file)
    found = match(r"^(.*)\.spaghetti(?:up|dn)?_ene$", file)
    return found === nothing ? nothing : found[1] * ".insp"
end

function _wien_insp(file)
    (file === nothing || !isfile(file)) && return nothing
    lines = readlines(file)
    sections = findall(line -> startswith(strip(line), "###"), lines)
    length(sections) >= 2 || throw(ArgumentError("unsupported insp layout; supply energy_unit explicitly"))
    rows = filter(!isempty, _w90_fields.(lines[sections[2]+1:end]))
    length(rows) >= 2 && length(rows[1]) >= 3 && length(rows[2]) >= 2 ||
        throw(ArgumentError("incomplete insp data configuration"))
    switch = parse(Int, rows[1][3])
    switch in (1, 2) || throw(ArgumentError("unknown insp energy switch"))
    fermi = _w90_float(rows[2][2])
    isfinite(fermi) || throw(ArgumentError("insp Fermi energy must be finite"))
    switch == 2 && fermi == 999 && throw(ArgumentError("insp eV mode requires a Fermi energy; 999 means unspecified"))
    return (; unit=switch == 1 ? :Ry : :eV, fermi_Ry=fermi == 999 ? nothing : fermi)
end

"""
    read_wien2k_bands(file; energy_unit=nothing, insp_file=..., path=nothing,
                      distance_scale=1/Units.bohr_angstrom, atol=1e-5)

Read `case.spaghetti_ene` band blocks: three source k coordinates, accumulated
distance and energy. Distances are converted from bohr⁻¹ to Å⁻¹ by default;
`distance_scale` can specify a different conversion for externally modified files.

The adjacent `case.insp` determines the energy unit: switch 1 means Ry on the
internal scale, switch 2 means eV relative to the supplied Fermi energy. Without
that file, specify `energy_unit=:Ry` or `:eV`. Energies are converted to eV with
their original zero; no Fermi energy is subtracted again.

The returned path contains distances only unless a compatible `path` is supplied.
The first three columns are preserved in `metadata[:source_kpoints]`, without
assuming they are fractional coordinates of a Wannier cell. Original band
indices are retained in `metadata[:band_indices]`.
"""
function read_wien2k_bands(file::AbstractString; energy_unit=nothing,
                           insp_file=_wien_insp_file(file),
                           path=nothing, distance_scale::Real=1/bohr_angstrom, atol::Real=1e-5)
    settings = energy_unit === nothing ? _wien_insp(insp_file) : nothing
    if energy_unit === nothing
        settings === nothing && throw(ArgumentError("supply energy_unit=:Ry or :eV when no insp file is available"))
        energy_unit = settings.unit
    end
    energy_unit in (:Ry, :eV) || throw(ArgumentError("energy_unit must be :Ry or :eV"))
    isfinite(distance_scale) && distance_scale > 0 || throw(ArgumentError("distance_scale must be positive and finite"))
    tables = _read_band_tables(file)
    result = _bands_from_tables(tables, file; xcolumn=4, ecolumn=5, path,
                                distance_scale, energy_scale=convert_energy(1, energy_unit => :eV), atol)
    all(table -> isapprox(table[:, 1:3], first(tables)[:, 1:3]; atol, rtol=0), tables) ||
        throw(ArgumentError("spaghetti bands do not share source k coordinates"))
    result.metadata[:source_kpoints] = permutedims(first(tables)[:, 1:3])
    result.metadata[:band_indices] = _band_indices(file, r"(?i)^bandindex:\s*(\d+)", length(tables))
    result.metadata[:source_energy_unit] = energy_unit
    result.metadata[:energy_reference] = settings === nothing ? :source :
        (settings.unit === :eV ? :fermi : :internal)
    if settings !== nothing && settings.fermi_Ry !== nothing
        result.metadata[:fermi_energy_eV] = convert_energy(settings.fermi_Ry, :Ry => :eV)
    end
    return result
end
