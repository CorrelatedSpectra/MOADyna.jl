# =====================================================================
# Eigensystem I/O — save / load eigenvalues + eigenvectors to HDF5
# =====================================================================
#
# Format (v"1"):
#   /version    attribute, "1"
#   /values     Vector{Float64}     (eigenvalues, sorted as produced by eigen)
#   /vectors    Matrix{ComplexF64}  (eigenvectors, columns are vectors)
#   /basis_id   optional, UInt or String  (token of the basis the vectors live in)
#
# Intentionally does NOT save the sparse Hamiltonian. Users reconstruct
# the matrix via `assemble(compile(H, basis), basis)` if needed; storing
# eigenvectors only keeps files small and decoupled from operator
# representation choices.
#
# `sys` is expected to expose `.values` (real-eltype vector) and
# `.vectors` (numeric-eltype matrix), which matches `LinearAlgebra.Eigen`
# returned by `eigen(H, basis)`.

const _EIGENSYSTEM_IO_VERSION = "1"

"""
    save_eigensystem(file::AbstractString, sys; basis_id = nothing)

Save eigenvalues + eigenvectors to an HDF5 file. The sparse Hamiltonian
is intentionally NOT saved; rebuild it via
`assemble(compile(H, basis), basis)` when required.

`sys` must expose `sys.values` (convertible to `Float64`) and
`sys.vectors` (convertible to `ComplexF64`); a `LinearAlgebra.Eigen`
returned by `eigen(H, basis)` qualifies. The optional `basis_id` keyword
is written under `/basis_id` if provided (typically the `UInt64` token
returned by `MOAD.Bases.basis_id(basis)`, but a `String` is also
accepted).

Round-trips with [`load_eigensystem`](@ref).
"""
function save_eigensystem(file::AbstractString, sys; basis_id = nothing)
    HDF5.h5open(file, "w") do f
        HDF5.attributes(f)["version"] = _EIGENSYSTEM_IO_VERSION
        f["values"]  = collect(Float64.(sys.values))
        f["vectors"] = collect(ComplexF64.(sys.vectors))
        if basis_id !== nothing
            f["basis_id"] = basis_id
        end
    end
    return file
end

"""
    load_eigensystem(file::AbstractString) -> NamedTuple{(:values, :vectors, :basis_id)}

Load an eigensystem written by [`save_eigensystem`](@ref). Returns a
`NamedTuple` with fields `values::Vector{Float64}`,
`vectors::Matrix{ComplexF64}`, and `basis_id` (the saved scalar/string
or `nothing` if the file omits it).

Raises `ArgumentError` if the file's `/version` attribute is
unrecognized.
"""
function load_eigensystem(file::AbstractString)
    HDF5.h5open(file, "r") do f
        version = read(HDF5.attributes(f)["version"])
        version == _EIGENSYSTEM_IO_VERSION || throw(ArgumentError(
            "eigensystem I/O: unrecognized version \"$version\" " *
            "(this MOAD supports \"$_EIGENSYSTEM_IO_VERSION\")"))

        values  = read(f["values"])
        vectors = read(f["vectors"])
        basis_id = haskey(f, "basis_id") ? read(f["basis_id"]) : nothing
        return (values = values, vectors = vectors, basis_id = basis_id)
    end
end
