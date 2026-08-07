# =====================================================================
# File I/O — `save_spectra` and `load_spectra`
# =====================================================================
#
# Two formats:
#
# 1. ASCII text (`.txt` / `.dat`) — `#`-prefixed metadata header
#    followed by a numerical table. Human-inspectable; loses Lanczos
#    chunks (round-trips give back chunks = nothing).
#
# 2. HDF5 (`.h5` / `.hdf5`) — full round-trip including chunks.
#    Supports `re_broaden` after load.
#
# `format = :auto` picks by filename extension. RIXS results round-trip
# through HDF5 only; the ASCII writer rejects RIXS with a clear message
# (a directory-of-files 2-D-map variant is on the to-do list).

const _MOADYNA_VERSION       = v"0.1.0-dev"
const _IO_FORMAT_VERSION  = 1

# ---------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------

"""
    save_spectra(result::SpectraTensor, filename::AbstractString;
                 format::Symbol = :auto)

Write `result` to disk. `format`:

- `:auto` — pick by extension: `.txt` / `.dat` → `:txt`,
  `.h5` / `.hdf5` → `:h5`.
- `:txt` — ASCII text with `#`-prefixed header. **Human-readable
  archive only** — preserves the spectrum tensor and ω grid for
  plotting, hand inspection, or third-party post-processing, but
  drops the Lanczos chunks AND most of the dispatch metadata
  (`T_is_vector_input`, `n_T`, `ψ_is_list_input`, …) needed by
  helpers such as `polarise(result, ε)` or `re_broaden(result; …)`.
  Helpers that depend on those keys will reject a txt-loaded
  result. Use `:h5` for full round-trip.
- `:h5` — HDF5 binary; round-trips the spectrum tensor, all chunks,
  and the full metadata dict (with type tags so Symbols / Bools /
  Tuples come back as themselves). `re_broaden`, `polarise`, etc.
  all work on the loaded result.
"""
function save_spectra(result::SpectraTensor, filename::AbstractString;
                      format::Symbol = :auto)
    fmt = format === :auto ? _detect_format(filename) : format
    if fmt === :txt
        return _save_spectra_txt(result, filename)
    elseif fmt === :h5
        return _save_spectra_h5(result, filename)
    else
        throw(ArgumentError("save_spectra: unknown format :$fmt; expected :txt or :h5"))
    end
end

"""
    load_spectra(filename::AbstractString; format::Symbol = :auto)
        -> SpectraTensor

Round-trips `save_spectra` output.

- ASCII (`:txt`): tensor + ω grid + a small set of canonical metadata
  fields (function tag, Eg, Γ, tensor shape, block size, converged
  flag) are restored. The `# Basis id:` line and free-form
  `# meta key: value` lines are written by the saver but not parsed
  on load — they are intended for human inspection only. `chunks`
  come back `nothing`. The loaded result is fine for plotting and
  inspection but rejected by helpers that need full dispatch metadata
  (e.g., `polarise(result, ε)` for tensor-form XAS); see
  `save_spectra` for details.
- HDF5 (`:h5`): full round-trip of tensor, chunks, ω grid, Eg,
  block_size, and the entire metadata dict (type-tagged). All helpers
  work on the loaded result.
"""
function load_spectra(filename::AbstractString; format::Symbol = :auto)
    fmt = format === :auto ? _detect_format(filename) : format
    if fmt === :txt
        return _load_spectra_txt(filename)
    elseif fmt === :h5
        return _load_spectra_h5(filename)
    else
        throw(ArgumentError("load_spectra: unknown format :$fmt; expected :txt or :h5"))
    end
end

function _detect_format(filename::AbstractString)
    ext = lowercase(splitext(filename)[2])
    ext in (".txt", ".dat")          && return :txt
    ext in (".h5",  ".hdf5", ".jld") && return :h5
    throw(ArgumentError("save/load_spectra: cannot infer format from extension '$ext'; pass `format = :txt` or `:h5`"))
end

# ---------------------------------------------------------------------
# ASCII writer
# ---------------------------------------------------------------------

function _save_spectra_txt(result::SpectraTensor, filename::AbstractString)
    result.metadata[:function] === :rixs &&
        throw(ArgumentError("save_spectra: RIXS 2D-map ASCII writer is not implemented; pass `format = :h5` to save a RIXS result."))

    ω_grid = result.ω_grid::AbstractRange
    n_ω    = length(ω_grid)
    tensor = result.tensor

    # Flatten leading axes to a column-major list of (a, b, …) tuples.
    leading = ndims(tensor) == 1 ? (1,) : size(tensor)[1:end-1]
    n_leading_combos = prod(leading)
    is_complex = eltype(tensor) <: Complex
    n_components_per_combo = is_complex ? 2 : 1
    n_data_columns = n_components_per_combo * n_leading_combos

    open(filename, "w") do io
        # ---- Header --------------------------------------------------
        println(io, "# MOADyna.Spectroscopy output")
        println(io, "# Created: ", Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"))
        println(io, "# MOADyna version: ", _MOADYNA_VERSION)
        println(io, "# Format version: ", _IO_FORMAT_VERSION, " (txt)")
        println(io, "# Function: ", result.metadata[:function])
        println(io, "# Tensor shape: ", size(tensor),
                ", eltype = ", eltype(tensor))
        println(io, "# Ground state: E_g = ", result.Eg, " eV")
        if haskey(result.metadata, :Γ)
            println(io, "# Broadening: Γ = ", result.metadata[:Γ],
                    " eV (FWHM Lorentzian)")
        end
        println(io, "# Frequency grid: ", n_ω, " points, ",
                "[", first(ω_grid), ", ", last(ω_grid), "] eV, step ",
                step(ω_grid))
        println(io, "# Block size: ", result.block_size)
        println(io, "# Converged: ", get(result.metadata, :converged, "—"))
        if haskey(result.metadata, :basis_id)
            println(io, "# Basis id: ", result.metadata[:basis_id])
        end
        if haskey(result.metadata, :restrictions) &&
                result.metadata[:restrictions] !== nothing
            println(io, "# Restrictions: ", result.metadata[:restrictions])
        end
        if get(result.metadata, :auto_range, false)
            println(io, "# Auto-range bandwidth: ",
                    get(result.metadata, :auto_range_bandwidth, "—"))
        end
        # Free-form metadata (anything else).
        for (k, v) in result.metadata
            k in (:function, :Γ, :Eg, :converged, :restrictions, :basis_id,
                  :auto_range, :auto_range_bandwidth) && continue
            println(io, "# meta ", k, ": ", v)
        end
        # Column legend.
        println(io, "#")
        println(io, "# Columns:")
        println(io, "#   1: ω (eV, excitation energy above E_g)")
        col = 2
        for combo_idx in 1:n_leading_combos
            ci = _flat_to_cart(combo_idx, leading)
            label = "[" * join(string.(ci), ",") * "]"
            if is_complex
                println(io, "#   ", col, ": Re tensor", label)
                col += 1
                println(io, "#   ", col, ": Im tensor", label)
                col += 1
            else
                println(io, "#   ", col, ": tensor", label)
                col += 1
            end
        end
        println(io, "#")

        # ---- Data ----------------------------------------------------
        # Each row: ω, then for each leading combo, Re/Im (or just real value).
        for i in 1:n_ω
            Printf.@printf(io, "% .8e", ω_grid[i])
            for combo_idx in 1:n_leading_combos
                ci  = _flat_to_cart(combo_idx, leading)
                val = ndims(tensor) == 1 ? tensor[i] : tensor[ci..., i]
                if is_complex
                    Printf.@printf(io, "  % .8e  % .8e", real(val), imag(val))
                else
                    Printf.@printf(io, "  % .8e", val)
                end
            end
            print(io, "\n")
        end
    end
    return filename
end

# Map a column-major flat index `1..prod(shape)` to a Cartesian index in
# the leading axes (the ω axis is excluded; this is a homemade replacement
# for `CartesianIndices(shape)[flat]` so we don't import it just here).
function _flat_to_cart(flat::Int, shape::NTuple{N,Int}) where {N}
    out = Vector{Int}(undef, N)
    rem = flat - 1
    for d in 1:N
        out[d] = rem % shape[d] + 1
        rem    = rem ÷ shape[d]
    end
    return Tuple(out)
end

# ---------------------------------------------------------------------
# ASCII loader
# ---------------------------------------------------------------------

function _load_spectra_txt(filename::AbstractString)
    metadata = Dict{Symbol,Any}()
    Eg::Float64                   = 0.0
    Γ_meta::Union{Float64,Nothing} = nothing
    block_size::Int               = 1
    tensor_shape::Vector{Int}     = Int[]
    is_complex::Bool              = true
    fname_func::Symbol            = :unknown
    headers_done = false
    grid_step::Union{Float64,Nothing} = nothing
    grid_first::Union{Float64,Nothing} = nothing
    grid_last::Union{Float64,Nothing}  = nothing
    n_ω::Int = 0

    rows = Vector{Vector{Float64}}()
    open(filename, "r") do io
        for line in eachline(io)
            if startswith(line, "#")
                # Header line; parse selected keys.
                _parse_header_line!(line, metadata)
                headers_done = false
                continue
            end
            isempty(strip(line)) && continue
            push!(rows, [parse(Float64, s) for s in split(line)])
        end
    end
    isempty(rows) && throw(ArgumentError("load_spectra: no data rows in $filename"))

    # Pull canonical fields out of the parsed metadata.
    haskey(metadata, :function) && (fname_func = Symbol(metadata[:function]))
    haskey(metadata, :Eg)       && (Eg = Float64(metadata[:Eg]))
    haskey(metadata, :Γ)        && (Γ_meta = Float64(metadata[:Γ]))
    haskey(metadata, :block_size) && (block_size = Int(metadata[:block_size]))
    if haskey(metadata, :tensor_shape)
        tensor_shape = collect(metadata[:tensor_shape])
        is_complex = endswith(string(get(metadata, :tensor_eltype, "Complex")), "}") ||
                     occursin("Complex", string(get(metadata, :tensor_eltype, "Complex")))
    end

    n_ω = length(rows)
    ncols = length(rows[1])
    ω_vec = [r[1] for r in rows]
    grid_step = step(LinRange(ω_vec[1], ω_vec[end], n_ω))   # may differ from header
    ω_grid = range(ω_vec[1], ω_vec[end]; length = n_ω)

    # If we have explicit shape from the header, reconstruct; otherwise
    # treat as scalar.
    if isempty(tensor_shape)
        # Scalar case: 1 + 2 (complex) or 1 + 1 (real).
        if ncols == 3
            tensor = ComplexF64[complex(rows[i][2], rows[i][3]) for i in 1:n_ω]
        elseif ncols == 2
            tensor = Float64[rows[i][2] for i in 1:n_ω]
        else
            throw(ArgumentError("load_spectra: ambiguous column layout " *
                                "($(ncols) cols) without `# Tensor shape:` header"))
        end
    else
        n_leading = prod(tensor_shape[1:end-1])
        ncomp = is_complex ? 2 : 1
        expected_cols = 1 + ncomp * n_leading
        ncols == expected_cols ||
            throw(ArgumentError("load_spectra: column count $ncols does not match " *
                                "expected $expected_cols from header shape $(tensor_shape)"))
        tensor = is_complex ?
            Array{ComplexF64}(undef, tensor_shape...) :
            Array{Float64}(undef, tensor_shape...)
        leading = Tuple(tensor_shape[1:end-1])
        @inbounds for i in 1:n_ω
            for combo_idx in 1:n_leading
                ci = _flat_to_cart(combo_idx, leading)
                if is_complex
                    re_col = 1 + 2 * (combo_idx - 1) + 1
                    im_col = re_col + 1
                    tensor[ci..., i] = complex(rows[i][re_col], rows[i][im_col])
                else
                    val_col = 1 + combo_idx
                    tensor[ci..., i] = rows[i][val_col]
                end
            end
        end
    end

    # Strip the bookkeeping keys we already extracted from the metadata
    # we hand back to the user (no surprises).
    delete!(metadata, :tensor_shape)
    delete!(metadata, :tensor_eltype)
    metadata[:function] = fname_func
    metadata[:loaded_from] = filename
    Γ_meta !== nothing && (metadata[:Γ] = Γ_meta)

    F = typeof(ω_grid)
    N = ndims(tensor)
    return SpectraTensor{eltype(tensor), Float64, N, F}(
        tensor, nothing, ω_grid, Eg, block_size, metadata)
end

function _parse_header_line!(line::AbstractString, meta::Dict{Symbol,Any})
    # Strip leading "# " marker; expect "Key: value" or " meta key: value".
    body = strip(replace(line, r"^#\s*" => ""))
    isempty(body) && return
    # Free-form lines we don't parse are ignored — robust.
    parts = split(body, ":"; limit = 2)
    length(parts) == 2 || return
    key, val = strip(parts[1]), strip(parts[2])
    if key == "MOADyna version"
        meta[:moad_version] = val
    elseif key == "Created"
        meta[:created] = val
    elseif key == "Function"
        meta[:function] = val
    elseif key == "Ground state"
        # "E_g = -1.0 eV"
        m = match(r"E_g\s*=\s*([\-+0-9.eE]+)", val)
        m === nothing || (meta[:Eg] = parse(Float64, m.captures[1]))
    elseif key == "Broadening"
        # "Γ = 0.5 eV (FWHM Lorentzian)"
        m = match(r"=\s*([\-+0-9.eE]+)", val)
        m === nothing || (meta[:Γ] = parse(Float64, m.captures[1]))
    elseif key == "Tensor shape"
        # "(3, 3, 401), eltype = ComplexF64"  or  "(181,), eltype = ..."
        sm = match(r"\(([^)]+)\)", val)
        if sm !== nothing
            parts  = strip.(split(sm.captures[1], ","))
            parts  = filter(!isempty, parts)        # drop trailing-comma empties
            shape  = parse.(Int, parts)
            meta[:tensor_shape] = shape
        end
        em = match(r"eltype\s*=\s*(\S+)", val)
        em === nothing || (meta[:tensor_eltype] = em.captures[1])
    elseif key == "Block size"
        meta[:block_size] = parse(Int, val)
    elseif key == "Converged"
        meta[:converged] = lowercase(val) in ("true", "yes")
    end
    return
end

# ---------------------------------------------------------------------
# HDF5 writer / loader
# ---------------------------------------------------------------------

function _save_spectra_h5(result::SpectraTensor, filename::AbstractString)
    HDF5.h5open(filename, "w") do f
        HDF5.attributes(f)["format_version"] = _IO_FORMAT_VERSION
        HDF5.attributes(f)["moad_version"]   = string(_MOADYNA_VERSION)

        # Tensor (raw numerical array).
        f["tensor"] = result.tensor

        # Grids. RIXS uses a tuple (ω_in, ω_out); XAS/FY uses a single range.
        if result.ω_grid isa Tuple
            ω_in, ω_out = result.ω_grid
            grids = HDF5.create_group(f, "grids")
            grids["ω_in_first"]  = first(ω_in)
            grids["ω_in_last"]   = last(ω_in)
            grids["ω_in_npts"]   = length(ω_in)
            grids["ω_out_first"] = first(ω_out)
            grids["ω_out_last"]  = last(ω_out)
            grids["ω_out_npts"]  = length(ω_out)
        else
            ω = result.ω_grid
            grids = HDF5.create_group(f, "grids")
            grids["ω_first"] = first(ω)
            grids["ω_last"]  = last(ω)
            grids["ω_npts"]  = length(ω)
        end

        f["Eg"]         = result.Eg
        f["block_size"] = result.block_size

        # Chunks. Each chunk → group with α/β/R datasets and scalar attrs.
        if result.chunks !== nothing
            cg = HDF5.create_group(f, "chunks")
            for (n, chunk) in enumerate(result.chunks)
                g = HDF5.create_group(cg, string(n))
                # α / β are vectors of small matrices; pack them into 3-D arrays.
                # Each block can be a different size (ragged), so store sizes too.
                g["raw_block_size"] = chunk.raw_block_size
                g["ψ_index"]        = chunk.ψ_index
                g["ω_in_index"]     = chunk.ω_in_index
                g["n_iter"]         = chunk.n_iter
                g["converged"]      = chunk.converged
                g["R_real"]         = real.(chunk.R)
                g["R_imag"]         = imag.(chunk.R)
                _h5_save_block_list(g, "α", chunk.α)
                _h5_save_block_list(g, "β", chunk.β)
            end
        end

        # Metadata: each entry stored as a dataset whose value is the
        # type-coerced payload, with a sibling __type attribute that lets
        # the loader rehydrate Symbols / Bools / Tuples / nothing.
        mg = HDF5.create_group(f, "metadata")
        for (k, v) in result.metadata
            key_str = string(k)
            try
                payload, tag = _h5_meta_pair(v)
                mg[key_str] = payload
                HDF5.attributes(mg[key_str])["__type"] = tag
            catch
                mg[key_str] = string(v)
                HDF5.attributes(mg[key_str])["__type"] = "Repr"
            end
        end
    end
    return filename
end

function _h5_save_block_list(g, name::String,
                              blocks::Vector{<:AbstractMatrix})
    bg = HDF5.create_group(g, name)
    bg["count"] = length(blocks)
    for (i, blk) in enumerate(blocks)
        gi = HDF5.create_group(bg, string(i))
        gi["real"] = real.(blk)
        gi["imag"] = imag.(blk)
    end
    return
end

function _h5_load_block_list(g)
    cnt = read(g["count"])
    out = Matrix{ComplexF64}[]
    for i in 1:cnt
        gi = g[string(i)]
        re = read(gi["real"])
        im = read(gi["imag"])
        push!(out, complex.(re, im))
    end
    return out
end

# Convert a metadata value to (storable_value, type_tag::String). The tag
# is written as an HDF5 attribute alongside the value so `_h5_load_meta`
# can restore the original Julia type. Without tags, every Symbol /
# Bool / Tuple / Nothing degrades to a plain string on load and breaks
# downstream helpers that compare with `=== :rixs`, etc.
_h5_meta_pair(v::Bool)            = (v, "Bool")
_h5_meta_pair(v::Integer)         = (Int(v), "Int")
_h5_meta_pair(v::AbstractFloat)   = (Float64(v), "Float64")
_h5_meta_pair(v::AbstractString)  = (String(v), "String")
_h5_meta_pair(v::Symbol)          = (string(v), "Symbol")
_h5_meta_pair(v::Nothing)         = ("nothing", "Nothing")
_h5_meta_pair(v::Tuple)           = (collect(v), "Tuple")
_h5_meta_pair(v::AbstractVector{<:Number}) = (collect(v), "Vector")
# Fallback: stringify but mark distinctly so we don't rehydrate as Symbol.
_h5_meta_pair(v) = (string(v), "Repr")

# Read a metadata value and rehydrate its original Julia type, when the
# saved file carries a `__type` attribute. Files written by older
# Spectroscopy versions (or by unrelated tooling) lack the tag — we fall
# back to the raw read value in that case.
function _h5_load_meta(d)
    val = read(d)
    HDF5.attributes(d) === nothing && return val
    haskey(HDF5.attributes(d), "__type") || return val
    type_tag = read(HDF5.attributes(d)["__type"])
    if type_tag == "Symbol"
        return Symbol(val)
    elseif type_tag == "Bool"
        return Bool(val)
    elseif type_tag == "Nothing"
        return nothing
    elseif type_tag == "Tuple"
        return Tuple(val)
    else
        return val
    end
end

function _load_spectra_h5(filename::AbstractString)
    HDF5.h5open(filename, "r") do f
        format_version = read(HDF5.attributes(f)["format_version"])
        format_version == _IO_FORMAT_VERSION ||
            throw(ErrorException(
                "load_spectra: unsupported HDF5 format version $format_version " *
                "(this MOADyna supports v$_IO_FORMAT_VERSION)"))

        tensor = read(f["tensor"])

        if HDF5.haskey(f["grids"], "ω_in_first")
            ω_in  = range(read(f["grids"]["ω_in_first"]),  read(f["grids"]["ω_in_last"]);
                          length = read(f["grids"]["ω_in_npts"]))
            ω_out = range(read(f["grids"]["ω_out_first"]), read(f["grids"]["ω_out_last"]);
                          length = read(f["grids"]["ω_out_npts"]))
            ω_grid = (ω_in, ω_out)
        else
            ω_grid = range(read(f["grids"]["ω_first"]), read(f["grids"]["ω_last"]);
                            length = read(f["grids"]["ω_npts"]))
        end

        Eg         = read(f["Eg"])
        block_size = read(f["block_size"])

        chunks = if HDF5.haskey(f, "chunks")
            cs = LanczosChunk{ComplexF64}[]
            cg = f["chunks"]
            keys_sorted = sort(collect(keys(cg)); by = k -> parse(Int, k))
            for k in keys_sorted
                g = cg[k]
                R = complex.(read(g["R_real"]), read(g["R_imag"]))
                α = _h5_load_block_list(g["α"])
                β = _h5_load_block_list(g["β"])
                push!(cs, LanczosChunk{ComplexF64}(
                    α, β, R,
                    read(g["raw_block_size"]),
                    read(g["ψ_index"]),
                    read(g["ω_in_index"]),
                    read(g["n_iter"]),
                    read(g["converged"])))
            end
            cs
        else
            nothing
        end

        meta = Dict{Symbol,Any}()
        if HDF5.haskey(f, "metadata")
            mg = f["metadata"]
            for k in keys(mg)
                meta[Symbol(k)] = _h5_load_meta(mg[k])
            end
        end
        meta[:loaded_from] = filename

        F = typeof(ω_grid)
        N = ndims(tensor)
        return SpectraTensor{eltype(tensor), Float64, N, F}(
            tensor, chunks, ω_grid, Eg, block_size, meta)
    end
end
