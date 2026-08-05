# =====================================================================
# Quanty wavefunction-dump reader
# =====================================================================
#
# Quanty's `print(psi)` produces blocks of the form:
#
#   WaveFunction: Wave Function
#   QComplex         =          0 (Real==0 or Complex==1)
#   N                =          2 (Number of basis functions used to discribe psi)
#   NFermionic modes =          4 (Number of fermions in the one particle basis)
#   NBosonic modes   =          0 (Number of bosons in the one particle basis)
#
#   #   pre-factor          Determinant
#   1   7.071067811865E-01  1100
#   2   7.071067811865E-01  0011
#
# For complex (QComplex=1), the data table has TWO pre-factor columns
# (real, imaginary):
#
#   #   pre-factor           pre-factor          Determinant
#   1   8.164965809277E-01   0.000000000000E+00  1100
#   2   0.000000000000E+00   4.082482904639E-01  0011
#
# The Determinant column is a length-NFermionic bit string with bit 0
# (MODE 0 in Quanty's 0-indexed convention) on the LEFT, mode NF-1 on
# the right. So "1100" = Quanty modes {0, 1} occupied.
#
# Convention bridge to MOAD:
# - Quanty mode index `i` (0-indexed) maps to MOAD's `(site_name, label)`
#   via the user-supplied `mode_map`, identical to the operator reader.
# - Each occupied mode is written into MOAD's packed-UInt64 encoding
#   via the basis's `EncodingMap`. The result is a length-`nwords`
#   `Vector{UInt64}` that we look up in the basis via `get_index`.

# --- Public API ----------------------------------------------------------

"""
    read_quanty_wavefunction(path, basis, mode_map; eltype=ComplexF64)
        -> Vector{eltype}

Read the FIRST `WaveFunction:` block from `path` and return a dense
MOAD-indexed coefficient vector aligned to `basis`. Coefficients of
basis states not appearing in the dump are zero.

`mode_map` follows the same convention as `read_quanty_operator`:
either a function `i -> (site_name, label)` or a `Dict{Int, ...}`
mapping Quanty's 0-indexed mode integers to MOAD addresses.

If Quanty's wavefunction is real (QComplex=0), the imaginary parts of
the returned vector are exactly zero. Use `eltype=Float64` when you
want a real return type.

Throws `ArgumentError` if a determinant in the dump lies outside
`basis` (which would indicate a sector mismatch — silent dropping
would mask a real error).
"""
function read_quanty_wavefunction(path::AbstractString, basis,
                                  mode_map; eltype = ComplexF64)
    open(path, "r") do io
        return _read_one_wavefunction(io, basis, mode_map, eltype)
    end
end

"""
    read_quanty_wavefunctions(path, basis, mode_map; eltype=ComplexF64)
        -> Vector{Vector{eltype}}

Read EVERY `WaveFunction:` block from `path` (typical use: an
`Eigensystem` dump that contains multiple eigenvectors). Returns a
vector of MOAD-indexed coefficient vectors in file order.
"""
function read_quanty_wavefunctions(path::AbstractString, basis,
                                   mode_map; eltype = ComplexF64)
    out = Vector{Vector{eltype}}()
    open(path, "r") do io
        while true
            ψ = _read_one_wavefunction(io, basis, mode_map, eltype;
                                       allow_eof = true)
            ψ === nothing && break
            push!(out, ψ)
        end
    end
    return out
end

"""
    read_quanty_eigenvalues(path) -> Vector{Float64}

Read a list of eigenvalues from a plain-text file with one number per
line (blank lines and `#`-comments ignored). Pairs naturally with
`read_quanty_wavefunctions` to form a `LinearAlgebra.Eigen` from a
Quanty `Eigensystem` dump.
"""
function read_quanty_eigenvalues(path::AbstractString)
    out = Float64[]
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            (isempty(line) || startswith(line, '#')) && continue
            push!(out, parse(Float64, line))
        end
    end
    return out
end

# --- Internals -----------------------------------------------------------

# Read one WaveFunction block. Returns the dense coefficient vector, or
# `nothing` if `allow_eof=true` and the file is exhausted before another
# block is found (used by `read_quanty_wavefunctions`).
function _read_one_wavefunction(io::IO, basis, mode_map, ::Type{T};
                                allow_eof::Bool = false) where {T}
    # Scan for the header.
    found = false
    while !eof(io)
        line = readline(io)
        if occursin("WaveFunction:", line)
            found = true
            break
        end
    end
    if !found
        allow_eof && return nothing
        throw(ArgumentError("No `WaveFunction:` block found in dump"))
    end

    # Parse the header fields.
    qcomplex = -1
    nbasis = -1
    nfermion = -1
    while !eof(io)
        line = readline(io)
        if (m = match(r"QComplex\s*=\s*(\d+)", line); m !== nothing)
            qcomplex = parse(Int, m.captures[1])
        elseif (m = match(r"^N\s*=\s*(\d+)", line); m !== nothing)
            nbasis = parse(Int, m.captures[1])
        elseif (m = match(r"NFermionic modes\s*=\s*(\d+)", line); m !== nothing)
            nfermion = parse(Int, m.captures[1])
        elseif occursin(r"^#+\s+pre-factor", line)
            break  # next is the data table (one or more `#` for the header)
        end
    end
    qcomplex == -1 && throw(ArgumentError("WaveFunction header missing QComplex"))
    nbasis == -1 && throw(ArgumentError("WaveFunction header missing N"))
    nfermion == -1 && throw(ArgumentError("WaveFunction header missing NFermionic modes"))
    qcomplex ∈ (0, 1) || throw(ArgumentError(
        "Unsupported QComplex=$qcomplex. Wavefunctions must be Real " *
        "(QComplex=0) or Complex (QComplex=1); other values like the " *
        "Mixed=2 format used for some operators are not supported by " *
        "this reader."))

    # Allocate and fill the dense coefficient vector.
    ψ = zeros(T, length(basis))
    nwords = basis.nwords
    buf = Vector{UInt64}(undef, nwords)

    for _ in 1:nbasis
        line = strip(readline(io))
        coef, bitstr = _parse_wavefunction_line(line, qcomplex)
        _bitstring_to_words!(buf, bitstr, basis.encoding, mode_map, nfermion)
        i = get_index(basis, buf)
        i == 0 && throw(ArgumentError(
            "Quanty determinant `$bitstr` is outside the basis sector. " *
            "Either the dump and basis describe different sectors, or " *
            "the mode_map is incorrect."))
        ψ[i] = T(coef)
    end
    return ψ
end

# Parse "1   7.071067811865E-01  1100" or
#       "1   8.164965809277E-01   0.000000000000E+00  1100"
# into (coefficient::ComplexF64, bitstring::String).
function _parse_wavefunction_line(line::AbstractString, qcomplex::Int)
    parts = split(line)
    if qcomplex == 0
        # parts = [index, coef_re, bitstring]
        length(parts) ≥ 3 || throw(ArgumentError(
            "Real wavefunction line malformed: `$line`"))
        re = parse(Float64, parts[2])
        return (ComplexF64(re, 0.0), String(parts[3]))
    else
        # parts = [index, coef_re, coef_im, bitstring]
        length(parts) ≥ 4 || throw(ArgumentError(
            "Complex wavefunction line malformed: `$line`"))
        re = parse(Float64, parts[2])
        im_ = parse(Float64, parts[3])
        return (ComplexF64(re, im_), String(parts[4]))
    end
end

# Encode a Quanty bit string ("1100") into MOAD's packed-UInt64 layout.
# Bit `i` (0-indexed) of the string corresponds to Quanty mode `i`; the
# user-supplied `mode_map` maps that to (site_name, label) in MOAD.
function _bitstring_to_words!(buf::AbstractVector{UInt64},
                              bitstr::AbstractString,
                              encoding,
                              mode_map,
                              nfermion::Int)
    fill!(buf, UInt64(0))
    length(bitstr) == nfermion || throw(ArgumentError(
        "Determinant `$bitstr` length $(length(bitstr)) ≠ NFermionic modes $nfermion"))
    @inbounds for (i, ch) in enumerate(bitstr)
        # Quanty mode index is 0-based; the i-th character (i in 1..nfermion)
        # corresponds to mode (i - 1).
        ch === '1' || ch === '0' ||
            throw(ArgumentError("Bit string contains non-binary char: `$bitstr`"))
        ch === '1' || continue
        qmode = i - 1
        site_name, label = _lookup_mode(mode_map, qmode)
        # Resolve the bit position in MOAD's encoding.
        entry = mode_entry(encoding, site_name, _splat(label))
        # FermionSite modes are 1-bit. Set that bit.
        # (Boson/spin sites would need set_span!; not needed for v0.1
        # since Quanty wavefunctions are over fermionic modes only.)
        buf[entry.word_idx] |= UInt64(1) << entry.bit_offset
    end
    return buf
end
