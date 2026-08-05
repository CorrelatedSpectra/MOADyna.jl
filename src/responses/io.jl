# =====================================================================
# MOAD.Responses — HDF5 round-trip for AbstractResponse
# =====================================================================
#
# Format versions:
#   "1" — initial layout (no converged field).
#   "2" — adds /converged Int (0/1) on LanczosResponse subgroups.
#
#   /version           attribute, "1" or "2"
#   /type_tag          attribute, concrete type name string
#
# LanczosResponse layout:
#   /alpha/block_<k>   Matrix (one dataset per diagonal block)
#   /beta/block_<k>    Matrix (one dataset per subdiagonal block)
#   /R                 Matrix
#   /Eg                Float64 (scalar dataset)
#   /Gamma             Float64 (scalar dataset)
#   /sign              Int     (scalar dataset)
#   /prefactor_re      Float64 (scalar)
#   /prefactor_im      Float64 (scalar, 0.0 for real T)
#   /converged         Int     (0/1, version "2" only; defaults to 1 when
#                                loading version "1")
#   /eltype            attribute, eltype string
#
# PoleResponse layout:
#   /a0                Matrix
#   /poles             Vector{Float64}
#   /residues/res_<n>  Matrix (one dataset per residue)
#   /Eg                Float64
#   /Gamma             Float64
#   /eltype            attribute
#
# GridResponse layout:
#   /data              Array (all dims preserved)
#   /omega             Vector{Float64}
#   /Eg                Float64
#   /Gamma             Float64
#   /eltype            attribute
#
# GreensFunction layout:
#   /has_addition      Int (0 or 1)
#   /has_removal       Int (0 or 1)
#   /addition/         sub-group with type_tag + fields (if populated)
#   /removal/          sub-group with type_tag + fields (if populated)
#
# Note: this does NOT touch SpectraTensor's on-disk format (which keeps
# its own _IO_FORMAT_VERSION and lives in src/spectroscopy/io.jl).

const _RESPONSES_IO_FORMAT_VERSION = "2"
const _RESPONSES_IO_FORMAT_VERSIONS_SUPPORTED = ("1", "2")

# ---------------------------------------------------------------------------
# save_response dispatch
# ---------------------------------------------------------------------------

"""
    save_response(file::AbstractString, R::AbstractResponse)

Save any `AbstractResponse` to an HDF5 file. The concrete type is
recorded in the `/type_tag` attribute; `load_response` uses it to
dispatch the correct loader.

Round-trips with [`load_response`](@ref).
"""
function save_response(file::AbstractString, R::AbstractResponse)
    HDF5.h5open(file, "w") do f
        HDF5.attributes(f)["version"]  = _RESPONSES_IO_FORMAT_VERSION
        _write_response(f, R)
    end
    return file
end

# Write a response into an open HDF5 group (reused for GreensFunction channels)
function _write_response(grp, R::LanczosResponse{T}) where {T}
    HDF5.attributes(grp)["type_tag"] = "LanczosResponse"
    HDF5.attributes(grp)["eltype"]   = string(T)

    α_grp = HDF5.create_group(grp, "alpha")
    for (k, ak) in enumerate(R.α)
        α_grp["block_$(k)"] = _to_hdf5_array(ak)
    end

    β_grp = HDF5.create_group(grp, "beta")
    for (k, bk) in enumerate(R.β)
        β_grp["block_$(k)"] = _to_hdf5_array(bk)
    end

    grp["R"]     = _to_hdf5_array(R.R)
    grp["Eg"]    = R.Eg
    grp["Gamma"] = R.Γ
    grp["sign"]  = R.sign

    # Prefactor: write real and imag parts separately for generality
    pf = Complex{Float64}(R.prefactor)
    grp["prefactor_re"] = real(pf)
    grp["prefactor_im"] = imag(pf)

    # converged flag (format version "2"). Stored as Int (0/1) since HDF5
    # readers handle scalar Ints uniformly.
    grp["converged"] = R.converged ? 1 : 0
end

function _write_response(grp, P::PoleResponse{T}) where {T}
    HDF5.attributes(grp)["type_tag"] = "PoleResponse"
    HDF5.attributes(grp)["eltype"]   = string(T)

    grp["a0"]    = _to_hdf5_array(P.a0)
    grp["poles"] = Vector{Float64}(P.poles)
    grp["Eg"]    = P.Eg
    grp["Gamma"] = P.Γ

    res_grp = HDF5.create_group(grp, "residues")
    for (n, rn) in enumerate(P.residues)
        res_grp["res_$(n)"] = _to_hdf5_array(rn)
    end
end

function _write_response(grp, G::GridResponse{T, N}) where {T, N}
    HDF5.attributes(grp)["type_tag"] = "GridResponse"
    HDF5.attributes(grp)["eltype"]   = string(T)

    grp["data"]  = _to_hdf5_array(G.data)
    grp["omega"] = Vector{Float64}(G.ω)
    grp["Eg"]    = G.Eg
    grp["Gamma"] = G.Γ
end

function _write_response(grp, GF::GreensFunction)
    HDF5.attributes(grp)["type_tag"] = "GreensFunction"

    grp["has_addition"] = isnothing(GF.addition) ? 0 : 1
    grp["has_removal"]  = isnothing(GF.removal)  ? 0 : 1

    if !isnothing(GF.addition)
        add_grp = HDF5.create_group(grp, "addition")
        _write_response(add_grp, GF.addition)
    end
    if !isnothing(GF.removal)
        rem_grp = HDF5.create_group(grp, "removal")
        _write_response(rem_grp, GF.removal)
    end
end

# ---------------------------------------------------------------------------
# load_response dispatch
# ---------------------------------------------------------------------------

"""
    load_response(file::AbstractString) -> AbstractResponse

Load a response written by [`save_response`](@ref). Dispatches on the
`/type_tag` attribute to the concrete-type loader.

Raises `ArgumentError` if the `/version` attribute is unrecognized.
"""
function load_response(file::AbstractString)
    HDF5.h5open(file, "r") do f
        version = read(HDF5.attributes(f)["version"])
        version ∈ _RESPONSES_IO_FORMAT_VERSIONS_SUPPORTED || throw(ArgumentError(
            "Responses I/O: unrecognized version \"$version\" " *
            "(this MOAD supports $(collect(_RESPONSES_IO_FORMAT_VERSIONS_SUPPORTED)))"))
        return _read_response(f, version)
    end
end

# Read a response from an open HDF5 group (reused for GreensFunction channels)
function _read_response(grp, version::AbstractString = _RESPONSES_IO_FORMAT_VERSION)
    type_tag = read(HDF5.attributes(grp)["type_tag"])
    if type_tag == "LanczosResponse"
        return _load_lanczos(grp, version)
    elseif type_tag == "PoleResponse"
        return _load_pole(grp)
    elseif type_tag == "GridResponse"
        return _load_grid(grp)
    elseif type_tag == "GreensFunction"
        return _load_greens(grp, version)
    else
        throw(ArgumentError(
            "Responses I/O: unrecognized type_tag \"$type_tag\""
        ))
    end
end

function _load_lanczos(grp, version::AbstractString = _RESPONSES_IO_FORMAT_VERSION)
    T_str   = read(HDF5.attributes(grp)["eltype"])
    T       = _parse_eltype(T_str)
    Eg      = read(grp["Eg"])
    Γ       = read(grp["Gamma"])
    sign    = Int(read(grp["sign"]))
    pf_re   = read(grp["prefactor_re"])
    pf_im   = read(grp["prefactor_im"])
    R_mat   = T.(_from_hdf5_array(read(grp["R"])))

    # Load α blocks
    α_grp   = grp["alpha"]
    n_α     = length(α_grp)
    α_vec   = Vector{Matrix{T}}(undef, n_α)
    for k in 1:n_α
        α_vec[k] = T.(_from_hdf5_array(read(α_grp["block_$(k)"])))
    end

    # Load β blocks
    β_grp   = grp["beta"]
    n_β     = length(β_grp)
    β_vec   = Vector{Matrix{T}}(undef, n_β)
    for k in 1:n_β
        β_vec[k] = T.(_from_hdf5_array(read(β_grp["block_$(k)"])))
    end

    # Reconstruct prefactor in type T
    prefactor = if T <: Complex
        T(pf_re + im * pf_im)
    else
        T(pf_re)
    end

    # converged flag added in format version "2"; default to true when
    # loading version "1" (back-compat: the field didn't exist, so we
    # cannot recover the underlying flag — assume the recurrence
    # completed successfully, matching the v0.1 default).
    converged = if version == "1"
        true
    else
        Int(read(grp["converged"])) != 0
    end

    return LanczosResponse{T}(α_vec, β_vec, R_mat, Float64(Eg), Float64(Γ),
                               sign, prefactor, converged)
end

function _load_pole(grp)
    T_str    = read(HDF5.attributes(grp)["eltype"])
    T        = _parse_eltype(T_str)
    Eg       = read(grp["Eg"])
    Γ        = read(grp["Gamma"])
    a0       = T.(_from_hdf5_array(read(grp["a0"])))
    poles    = Float64.(read(grp["poles"]))

    res_grp  = grp["residues"]
    n_res    = length(res_grp)
    residues = Vector{Matrix{T}}(undef, n_res)
    for n in 1:n_res
        residues[n] = T.(_from_hdf5_array(read(res_grp["res_$(n)"])))
    end

    return PoleResponse{T}(a0, poles, residues, Float64(Eg), Float64(Γ))
end

function _load_grid(grp)
    T_str  = read(HDF5.attributes(grp)["eltype"])
    T      = _parse_eltype(T_str)
    Eg     = read(grp["Eg"])
    Γ      = read(grp["Gamma"])
    data   = T.(_from_hdf5_array(read(grp["data"])))
    ω      = Float64.(read(grp["omega"]))
    N      = ndims(data)
    return GridResponse{T, N}(data, ω, Float64(Eg), Float64(Γ))
end

function _load_greens(grp, version::AbstractString = _RESPONSES_IO_FORMAT_VERSION)
    has_addition = Int(read(grp["has_addition"])) != 0
    has_removal  = Int(read(grp["has_removal"]))  != 0

    addition = has_addition ? _read_response(grp["addition"], version) : nothing
    removal  = has_removal  ? _read_response(grp["removal"],  version) : nothing

    R2 = if !isnothing(addition)
        typeof(addition)
    else
        typeof(removal)
    end
    T2 = eltype(R2)
    return GreensFunction{T2, R2}(
        isnothing(addition) ? nothing : addition,
        isnothing(removal)  ? nothing : removal
    )
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# HDF5.jl natively supports Complex arrays (reads/writes as compound type
# with :r/:i fields). We just collect to a dense array before writing.

function _to_hdf5_array(A::AbstractArray)
    return collect(A)
end

function _from_hdf5_array(A::AbstractArray)
    return A
end

# Parse the eltype string back to a Julia type
function _parse_eltype(s::AbstractString)
    d = Dict(
        "Float64"       => Float64,
        "Float32"       => Float32,
        "ComplexF64"    => ComplexF64,
        "ComplexF32"    => ComplexF32,
        "Complex{Float64}" => ComplexF64,
        "Complex{Float32}" => ComplexF32,
    )
    haskey(d, s) && return d[s]
    throw(ArgumentError(
        "Responses I/O: unrecognized eltype string \"$s\""
    ))
end
