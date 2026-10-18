function _w90_length_factor(unit)
    name = lowercase(strip(unit))
    name in ("ang", "angstrom") && return 1.0
    name == "bohr" && return bohr_angstrom
    throw(ArgumentError("unknown Wannier length unit: $unit"))
end

function _win_lattice(block)
    lines = copy(block)
    factor = 1.0
    if length(lines) == 4
        factor = _w90_length_factor(popfirst!(lines))
    end
    length(lines) == 3 || throw(ArgumentError("unit_cell_cart must contain three lattice vectors"))
    rows = [_w90_float.(_w90_fields(line)) for line in lines]
    all(row -> length(row) == 3, rows) || throw(ArgumentError("lattice vectors must have three components"))
    return factor * reduce(hcat, rows)
end

function _win_atoms(win, lattice)
    keys = filter(key -> haskey(win.blocks, key), ("atoms_cart", "atoms_frac"))
    isempty(keys) && return nothing
    length(keys) == 1 || throw(ArgumentError("supply either atoms_cart or atoms_frac, not both"))
    fractional = only(keys) == "atoms_frac"
    lines = copy(win.blocks[only(keys)])
    factor = 1.0
    if !fractional && !isempty(lines) && length(_w90_fields(first(lines))) == 1
        factor = _w90_length_factor(popfirst!(lines))
    end
    rows = _w90_fields.(lines)
    !isempty(rows) && all(row -> length(row) == 4, rows) ||
        throw(ArgumentError("atom records must contain a label and three coordinates"))
    positions = reduce(hcat, [_w90_float.(row[2:4]) for row in rows])
    all(isfinite, positions) || throw(ArgumentError("atomic coordinates must be finite"))
    if fractional
        lattice === nothing && throw(ArgumentError("atoms_frac requires lattice vectors"))
        positions = lattice * positions
    else
        positions *= factor
    end
    return (; labels=String[first(row) for row in rows], positions)
end

function _wout_lattice(text, factor)
    rows = Vector{Float64}[]
    for i in 1:3
        row = match(Regex("(?m)^\\s*a_" * string(i) * "\\s+([^\\n]+)"), text)
        row === nothing && return nothing
        values = _w90_float.(split(row[1]))
        length(values) == 3 || throw(ArgumentError("invalid lattice vector in wout"))
        push!(rows, values)
    end
    return factor * reduce(hcat, rows)
end

function _xyz_centres(file, n)
    open(file) do io
        count = parse(Int, strip(readline(io)))
        readline(io)
        positions = Vector{Float64}[]
        for _ in 1:count
            f = _w90_line(io)
            length(f) == 4 || throw(ArgumentError("xyz records must contain a label and three coordinates"))
            f[1] == "X" && push!(positions, _w90_float.(f[2:4]))
        end
        length(positions) == n || throw(ArgumentError("the xyz file must contain $n Wannier centres labelled X"))
        return reduce(hcat, positions)
    end
end

function _wout_centres(text, n, factor)
    sections = split(text, "Final State")
    length(sections) > 1 || return nothing
    positions = zeros(3, n)
    seen = falses(n)
    for row in eachmatch(r"WF centre and spread\s+(\d+)\s+\(([^)]+)\)", last(sections))
        i = parse(Int, row[1])
        1 <= i <= n && !seen[i] || throw(ArgumentError("invalid or duplicate final Wannier centre in wout"))
        values = _w90_float.(_w90_fields(row[2]))
        length(values) == 3 || throw(ArgumentError("Wannier centres must have three components"))
        positions[:, i] = factor * values
        seen[i] = true
    end
    all(seen) || throw(ArgumentError("incomplete final Wannier centres in wout"))
    return positions
end

function _wannier_geometry(seed, win, n; lattice=nothing, centres=nothing,
                           embedded_lattice=nothing, embedded_centres=nothing)
    woutfile = seed * ".wout"
    wout = isfile(woutfile) ? read(woutfile, String) : ""
    release = match(r"(?i)Release\s*:?\s*(\d+\.\d+(?:\.\d+)?)", wout)
    version = release === nothing ? nothing : VersionNumber(release[1])
    output_unit = match(r"(?i)Length Unit\s*:\s*(Ang|Bohr)", wout)
    factor = _w90_length_factor(output_unit === nothing ?
        get(win.parameters, "length_unit", "ang") : output_unit[1])
    A = if lattice !== nothing
        _geometry_matrix(lattice, (3, 3), "lattice")
    elseif embedded_lattice !== nothing
        _geometry_matrix(embedded_lattice, (3, 3), "lattice")
    elseif haskey(win.blocks, "unit_cell_cart")
        _win_lattice(win.blocks["unit_cell_cart"])
    else
        _wout_lattice(wout, factor)
    end
    A === nothing || (all(isfinite, A) && rank(A) == 3) ||
        throw(ArgumentError("Wannier lattice vectors must be finite and independent"))
    translated = occursin("Translated centres", wout) ||
        (something(_w90_option(win.parameters, "bands_plot"), false) &&
         occursin("cut", lowercase(get(win.parameters, "bands_plot_mode", "")))) ||
        (something(_w90_option(win.parameters, "transport"), false) &&
         occursin(r"bulk|lcr", lowercase(get(win.parameters, "transport_mode", "bulk"))))
    wrapping = _w90_option(win.parameters, "translate_home_cell")
    wrapping === nothing && isfile(seed * ".win") && (wrapping = false)
    source = nothing
    tau = if centres !== nothing
        source = :user
        _geometry_matrix(centres, (3, n), "centres")
    elseif A === nothing
        nothing
    elseif translated
        source = :translated_hamiltonian
        nothing
    else
        xyz = seed * "_centres.xyz"
        printed = _wout_centres(wout, n, factor)
        cartesian = if embedded_centres !== nothing
            source = abspath(seed * "_tb.dat")
            embedded_centres
        elseif printed !== nothing
            source = abspath(woutfile)
            printed
        elseif isfile(xyz) && wrapping === false
            source = abspath(xyz)
            _xyz_centres(xyz, n)
        else
            isfile(xyz) && (source = wrapping === true ? :wrapped_xyz : :unknown_xyz_wrapping)
            nothing
        end
        cartesian === nothing ? nothing : A \ cartesian
    end
    tau = _geometry_matrix(tau, (3, n), "centres")
    return (; lattice=A, centres=tau, centres_source=source, version)
end
