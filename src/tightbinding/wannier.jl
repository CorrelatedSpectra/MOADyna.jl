_w90_float(s) = parse(Float64, replace(s, 'D' => 'e', 'd' => 'e'))
_w90_fields(line) = split(replace(strip(first(split(line, r"[!#]"))), ',' => ' '))

function _w90_line(io)
    while !eof(io)
        fields = _w90_fields(readline(io))
        isempty(fields) || return fields
    end
    throw(ArgumentError("unexpected end of Wannier90 file"))
end

function _wannier_seed(path)
    if isdir(path)
        seeds = unique(replace(file, r"_(?:hr|tb)\.dat$" => "") for file in readdir(path)
                       if endswith(file, "_hr.dat") || endswith(file, "_tb.dat"))
        length(seeds) == 1 || throw(ArgumentError("supply a seedname when a directory contains zero or multiple Wannier models"))
        return joinpath(path, only(seeds))
    end
    return replace(String(path), r"_(?:hr|tb)\.dat$|\.win$" => "")
end

function _read_wannier_hoppings(file)
    open(file) do io
        header = readline(io)
        combined = endswith(file, "_tb.dat")
        lattice = combined ? reduce(hcat, [_w90_float.(_w90_line(io)) for _ in 1:3]) : nothing
        lattice === nothing || size(lattice) == (3, 3) || throw(ArgumentError("tb.dat lattice vectors must have three components"))
        n = parse(Int, only(_w90_line(io)))
        nr = parse(Int, only(_w90_line(io)))
        n > 0 && nr > 0 || throw(ArgumentError("Wannier orbital and translation counts must be positive"))
        weights = Int[]
        while length(weights) < nr
            append!(weights, parse.(Int, _w90_line(io)))
        end
        length(weights) == nr && all(>(0), weights) || throw(ArgumentError("invalid Wannier degeneracy list"))
        blocks = Dict{NTuple{3,Int},Matrix{ComplexF64}}()
        degeneracies = Dict{NTuple{3,Int},Int}()
        seen = Dict{NTuple{3,Int},BitMatrix}()
        R = (0, 0, 0)
        block_size = Base.checked_mul(n, n)
        for row in 1:Base.checked_mul(nr, block_size)
            if combined && mod(row-1, block_size) == 0
                f = _w90_line(io)
                length(f) == 3 || throw(ArgumentError("tb.dat translations must have three components"))
                R = Tuple(parse.(Int, f))
            end
            f = _w90_line(io)
            if !combined
                length(f) == 7 || throw(ArgumentError("hr.dat rows must contain R, m, n, real and imaginary parts"))
                R = Tuple(parse.(Int, f[1:3]))
                f = f[4:7]
            end
            length(f) == 4 || throw(ArgumentError("Hamiltonian rows must contain m, n, real and imaginary parts"))
            i, j = parse.(Int, f[1:2])
            1 <= i <= n && 1 <= j <= n || throw(ArgumentError("Wannier matrix index out of range"))
            if !haskey(blocks, R)
                length(blocks) < nr || throw(ArgumentError("too many Wannier translation blocks"))
                degeneracies[R] = weights[length(blocks)+1]
                blocks[R] = zeros(ComplexF64, n, n)
                seen[R] = falses(n, n)
            end
            seen[R][i, j] && throw(ArgumentError("duplicate Wannier matrix element at $R, $i, $j"))
            blocks[R][i, j] = complex(_w90_float(f[3]), _w90_float(f[4]))
            seen[R][i, j] = true
        end
        length(blocks) == nr && all(all, values(seen)) || throw(ArgumentError("incomplete Wannier hopping table"))
        centres = combined ? _read_tb_centres(io, seen, n, nr) : nothing
        all(line -> isempty(_w90_fields(line)), eachline(io)) || throw(ArgumentError("unexpected rows after the Wannier hopping table"))
        return (; blocks, degeneracies, header, n, lattice, centres)
    end
end

function _read_tb_centres(io, seen, n, nr)
    foreach(mask -> fill!(mask, false), values(seen))
    centres = haskey(seen, (0, 0, 0)) ? zeros(3, n) : nothing
    for _ in 1:nr
        line = _w90_line(io)
        length(line) == 3 || throw(ArgumentError("tb.dat position translations must have three components"))
        R = Tuple(parse.(Int, line))
        haskey(seen, R) || throw(ArgumentError("position and Hamiltonian translation lists differ"))
        for _ in 1:n*n
            f = _w90_line(io)
            length(f) == 8 || throw(ArgumentError("position rows must contain m, n and three complex components"))
            i, j = parse.(Int, f[1:2])
            1 <= i <= n && 1 <= j <= n || throw(ArgumentError("position matrix index out of range"))
            seen[R][i, j] && throw(ArgumentError("duplicate position matrix element"))
            value = _w90_float.(f[3:8])
            all(isfinite, value) || throw(ArgumentError("position entries must be finite"))
            R == (0, 0, 0) && i == j && (centres[:, i] = value[1:2:5])
            seen[R][i, j] = true
        end
    end
    all(all, values(seen)) || throw(ArgumentError("incomplete position matrix table"))
    return centres
end

function _read_wannier_win(file)
    parameters = Dict{String,String}()
    blocks = Dict{String,Vector{String}}()
    isfile(file) || return (; parameters, blocks)
    active = nothing
    for raw in eachline(file)
        line = strip(first(split(raw, r"[!#]")))
        isempty(line) && continue
        opening = match(r"(?i)^begin\s+(\w+)\s*$", line)
        closing = match(r"(?i)^end\s+(\w+)\s*$", line)
        if opening !== nothing
            active === nothing || throw(ArgumentError("nested blocks in $file"))
            active = lowercase(opening[1])
            haskey(blocks, active) && throw(ArgumentError("duplicate $active block in $file"))
            blocks[active] = String[]
        elseif closing !== nothing
            active == lowercase(closing[1]) || throw(ArgumentError("unmatched block end in $file"))
            active = nothing
        elseif active !== nothing
            push!(blocks[active], line)
        else
            entry = match(r"^(\w+)(?:\s*[=:]\s*|\s+)(.+)$", line)
            entry === nothing || (parameters[lowercase(entry[1])] = strip(entry[2]))
        end
    end
    active === nothing || throw(ArgumentError("unterminated $active block in $file"))
    return (; parameters, blocks)
end

function _w90_logical(value)
    token = lowercase(strip(value))
    token in ("true", "t", ".true.", ".t.") && return true
    token in ("false", "f", ".false.", ".f.") && return false
    throw(ArgumentError("invalid Wannier logical value: $value"))
end

function _w90_header_option(text, key)
    found = match(Regex("(?i)\\b" * key * "\\s*[=:]\\s*([^\\s,;]*)"), text)
    return found === nothing ? nothing : _w90_logical(found[1])
end

function _w90_option(parameters, key)
    return haskey(parameters, key) ? _w90_logical(parameters[key]) : nothing
end

function _consistent_option(values, name)
    choices = unique(filter(!isnothing, collect(values)))
    length(choices) <= 1 || throw(ArgumentError("conflicting $name declarations in the Wannier files"))
    return isempty(choices) ? nothing : only(choices)
end

"""
    read_wannier90(seed; interpolation=:auto, lattice=nothing, centres=nothing,
                   labels=nothing, atol=1e-10, rtol=1e-10)

Read `seed_hr.dat` or `seed_tb.dat` and the available `.win`, `.wout`, `_centres.xyz` and
`_wsvec.dat` companions into a [`TightBindingModel`](@ref). `seed` may also be
the Hamiltonian filename or a directory containing one model. The hr file is
preferred when both exist unless the tb filename is supplied explicitly. The
combined tb format also supplies lattice vectors and orbital centres; off-site
position-operator entries are not retained. Energies remain in eV
with the source energy zero. Supplied lattice vectors are columns in Å;
supplied centres are columns in fractional coordinates.

`interpolation=:auto` uses the source declarations and version. Pair-dependent
Wigner–Seitz shifts are read from wsvec, or reconstructed from the lattice,
mesh and final centres when that file is missing. Already-expanded files are
recognized and are not shifted again. `:plain` explicitly selects the ordinary
degeneracy-weighted Fourier sum, and `:ws` requests centre-dependent shifts.
Bare hr files need an explicit interpolation choice because they do not record it.

Centres come from tb data, the final wout state, or an untranslated XYZ file, in
that order. An XYZ file needs a companion win file establishing its wrapping
convention. If `translate_home_cell=true`, supply wout data or explicit `centres`.
Hamiltonians written in cut-band or transport modes can use translated orbital
representatives; these modes require explicit `centres` for geometric operations.

All weights are normalized once into the stored hoppings. The selected route
and source metadata are recorded in `model.metadata`; projection text is retained
as input information, without assigning final orbital characters automatically.
"""
function read_wannier90(path::AbstractString; interpolation=:auto, lattice=nothing,
                        centres=nothing, labels=nothing, atol::Real=1e-10, rtol::Real=1e-10)
    interpolation in (:auto, :plain, :ws) || throw(ArgumentError("interpolation must be :auto, :plain or :ws"))
    seed = _wannier_seed(path)
    file = if endswith(path, "_hr.dat") || endswith(path, "_tb.dat")
        String(path)
    elseif isfile(seed * "_hr.dat")
        seed * "_hr.dat"
    else
        seed * "_tb.dat"
    end
    isfile(file) || throw(ArgumentError("Wannier Hamiltonian file not found: $file"))
    raw = _read_wannier_hoppings(file)
    win = _read_wannier_win(seed * ".win")
    haskey(win.parameters, "num_wann") && parse(Int, win.parameters["num_wann"]) != raw.n &&
        throw(ArgumentError("num_wann differs between input and Hamiltonian files"))
    geometry = _wannier_geometry(seed, win, raw.n; lattice, centres,
                                 embedded_lattice=raw.lattice, embedded_centres=raw.centres)
    wsfile = seed * "_wsvec.dat"
    wsheader = isfile(wsfile) ? open(readline, wsfile) : ""
    applied = _consistent_option((_w90_option(win.parameters, "write_ndegen_applied"),
        _w90_header_option(raw.header, "write_ndegen_applied"),
        _w90_header_option(wsheader, "write_ndegen_applied")), "write_ndegen_applied")
    use_ws = _consistent_option((_w90_option(win.parameters, "use_ws_distance"),
        _w90_header_option(wsheader, "use_ws_distance")), "use_ws_distance")
    if use_ws === nothing && geometry.version !== nothing
        use_ws = geometry.version >= v"3.0"
    end
    normalized = Dict(R => h / raw.degeneracies[R] for (R, h) in raw.blocks)
    metadata = Dict{Symbol,Any}(:source => abspath(file), :source_version => geometry.version,
        :projections => get(win.blocks, "projections", String[]),
        :centres_source => geometry.centres_source)
    blocks, route = if applied === true
        all(==(1), values(raw.degeneracies)) || throw(ArgumentError("expanded Wannier files must have unit degeneracy weights"))
        normalized, :expanded
    elseif interpolation === :plain || (interpolation === :auto && use_ws === false)
        normalized, :plain
    elseif interpolation === :ws || use_ws === true
        if isfile(wsfile) && use_ws !== false
            _expand_wsvec(normalized, wsfile, raw.n), :wsvec
        else
            geometry.lattice === nothing && throw(ArgumentError("reconstructing wsvec requires lattice vectors"))
            geometry.centres === nothing && throw(ArgumentError("reconstructing wsvec requires unwrapped final centres from wout, an untranslated XYZ file, or centres=..."))
            haskey(win.parameters, "mp_grid") || throw(ArgumentError("reconstructing wsvec requires mp_grid"))
            mesh = parse.(Int, _w90_fields(win.parameters["mp_grid"]))
            length(mesh) == 3 && all(>(0), mesh) || throw(ArgumentError("mp_grid must contain three positive integers"))
            tolerance = _w90_float(get(win.parameters, "ws_distance_tol", "1e-5"))
            isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("ws_distance_tol must be positive and finite"))
            metadata[:mp_grid] = Tuple(mesh)
            metadata[:ws_distance_tol] = tolerance
            metadata[:ws_search] = :bounded_geometry
            _reconstruct_ws(normalized, geometry.lattice, geometry.centres, mesh, tolerance), :reconstructed
        end
    else
        throw(ArgumentError("interpolation convention is unknown; supply companion files or choose interpolation=:plain or :ws explicitly"))
    end
    metadata[:interpolation] = route
    return TightBindingModel(blocks; lattice=geometry.lattice, centres=geometry.centres,
                             labels, metadata, atol, rtol)
end
