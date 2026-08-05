"""
    AbstractResponse{T <: Number}

Abstract supertype for all matrix-of-ω response objects in `MOAD.Responses`.

Concrete subtypes:
- [`LanczosResponse{T}`](@ref) — block-Lanczos tridiagonal representation
- [`PoleResponse{T}`](@ref)    — pole + residue (spectral) representation
- [`GridResponse{T, N}`](@ref) — sampled on an ω-grid
- [`GreensFunction{T, R}`](@ref) — parametric single-particle Green's function
  wrapping an addition and/or removal channel
"""
abstract type AbstractResponse{T <: Number} end

# ---------------------------------------------------------------------------
# LanczosResponse{T}
# ---------------------------------------------------------------------------

"""
    LanczosResponse{T} <: AbstractResponse{T}

Block-Lanczos tridiagonal representation of a matrix-of-ω response.

# Fields
- `α::Vector{Matrix{T}}` — diagonal blocks (raw block-Lanczos H frame;
  `Eg` is applied at evaluation/conversion, not stored in the blocks).
- `β::Vector{Matrix{T}}` — subdiagonal blocks; `β[k]` has shape
  `(B_{k+1} × B_k)`. The (k+1, k) block of the block-tridiagonal T.
- `R::Matrix{T}` — rectangular initial-block QR factor, shape
  `(B_active × B_raw)`.
- `Eg::Float64` — energy reference (applied as `z = ω + Eg + iΓ/2`).
- `Γ::Float64` — broadening (FWHM).
- `sign::Int` — +1 for retarded/neutral; −1 reserved for removal channel
  (2c only).
- `prefactor::T` — explicit scalar multiplier (default `one(T)`).
- `converged::Bool` — `true` if the underlying block-Lanczos recurrence
  terminated before the iteration cap (Krylov-space exhaustion,
  rank-deflation early-exit, or tolerance soft-stop); `false` if the
  `krylovdim` cap was hit before any of those signals fired. Faithful
  pass-through of `block_lanczos`'s `converged` return field.
  Conversions to `PoleResponse` / `GridResponse` drop this flag — the
  converged concept lives at the Lanczos representation, not at the
  diagonalised / evaluated forms.

# Convention

The continued-fraction recurrence is evaluated in the (ω + Eg) frame:

    G_K = (z·I − α_K)⁻¹,    z = ω + Eg + iΓ/2
    G_k = (z·I − α_k − β_k' · G_{k+1} · β_k)⁻¹    (k = K−1, …, 1)

The user-basis correlator on the initial-block subspace is

    R(ω) = prefactor · R' · G₁(z) · R    (shape B_raw × B_raw)

See the **Responses** chapter of the manual for the β / cf convention.
"""
struct LanczosResponse{T} <: AbstractResponse{T}
    α::Vector{Matrix{T}}        # diagonal blocks
    β::Vector{Matrix{T}}        # subdiagonal blocks (B_{k+1} × B_k)
    R::Matrix{T}                # rectangular initial-block QR factor
    # Note: prefactor::T shares T with α/β/R deliberately. Scaling by a
    # complex scalar on a real LanczosResponse produces a different
    # parametric instance via promotion at the call site, not a
    # mixed-type field. If a future consumer needs decoupled
    # scalar-vs-block element types, add a second type parameter then.
    Eg::Float64                 # energy reference
    Γ::Float64                  # broadening (FWHM)
    sign::Int                   # +1 retarded / −1 removal-channel (2c only)
    prefactor::T                # explicit scalar multiplier
    converged::Bool             # block_lanczos soft-stop signal (true if
                                # residual fell below tol; false if krylovdim
                                # cap was hit)

    function LanczosResponse{T}(α::Vector{Matrix{T}}, β::Vector{Matrix{T}},
                                 R::Matrix{T}, Eg::Float64, Γ::Float64,
                                 sign::Int, prefactor::T,
                                 converged::Bool = true) where {T}
        # Non-empty Krylov chain
        if isempty(α)
            throw(ArgumentError(
                "LanczosResponse: α must be non-empty (at least one diagonal block)"
            ))
        end

        # Each α[k] must be square
        for (k, ak) in enumerate(α)
            if size(ak, 1) != size(ak, 2)
                throw(ArgumentError(
                    "LanczosResponse: α[$k] must be square; got size $(size(ak))"
                ))
            end
        end

        # β is one shorter than α
        if length(β) != length(α) - 1
            throw(ArgumentError(
                "LanczosResponse: length(β) = $(length(β)) must equal " *
                "length(α) - 1 = $(length(α) - 1)"
            ))
        end

        # Each β[k] connects α[k] to α[k+1]: shape (B_{k+1} × B_k)
        for k in 1:length(β)
            bk = β[k]
            expected_rows = size(α[k+1], 1)
            expected_cols = size(α[k], 2)
            if size(bk, 1) != expected_rows || size(bk, 2) != expected_cols
                throw(ArgumentError(
                    "LanczosResponse: β[$k] must have shape " *
                    "($(expected_rows) × $(expected_cols)) to connect α[$k] and α[$(k+1)]; " *
                    "got size $(size(bk))"
                ))
            end
        end

        # R rows must match the first α-block size (B_active)
        if size(R, 1) != size(α[1], 1)
            throw(ArgumentError(
                "LanczosResponse: size(R, 1) = $(size(R, 1)) must equal " *
                "size(α[1], 1) = $(size(α[1], 1)) (B_active)"
            ))
        end

        # sign must be +1 or -1
        if sign ∉ (+1, -1)
            throw(ArgumentError(
                "LanczosResponse: sign must be +1 or -1; got $sign"
            ))
        end

        return new{T}(α, β, R, Eg, Γ, sign, prefactor, converged)
    end
end

"""
    LanczosResponse(α, β, R, Eg, Γ, sign, prefactor, converged = true)

Unparameterized positional outer constructor; infers `T` from the element
type of `α` and forwards to the validated inner constructor. The
`converged` flag defaults to `true` for back-compat with callers that
predate the field.

Use this form when `α`, `β`, `R` are heterogeneous `AbstractMatrix`
subtypes (e.g., `Symmetric`, `Adjoint`, or mixed concrete types) and you
want automatic promotion to `Matrix{T}`. Use `LanczosResponse{T}(…)`
directly when all inputs are already `Matrix{T}` and you want to skip the
extra allocation.
"""
function LanczosResponse(α::Vector{<:AbstractMatrix{T}},
                          β::Vector{<:AbstractMatrix{T}},
                          R::AbstractMatrix{T},
                          Eg::Real, Γ::Real,
                          sign::Integer,
                          prefactor::T,
                          converged::Bool = true) where {T}
    return LanczosResponse{T}(
        Vector{Matrix{T}}(α),
        Vector{Matrix{T}}(β),
        Matrix{T}(R),
        Float64(Eg), Float64(Γ),
        Int(sign),
        prefactor,
        converged
    )
end

Base.eltype(::LanczosResponse{T}) where {T} = T
Base.eltype(::Type{LanczosResponse{T}}) where {T} = T

"""
    Base.size(L::LanczosResponse) -> Tuple{Int, Int}

Return the output matrix shape `(B_raw, B_raw)` where `B_raw = size(L.R, 2)`.
"""
Base.size(L::LanczosResponse) = (size(L.R, 2), size(L.R, 2))

function Base.show(io::IO, L::LanczosResponse{T}) where {T}
    K        = length(L.α)
    B_active = size(L.R, 1)
    B_raw    = size(L.R, 2)
    print(io, "LanczosResponse{$T}(K=$K levels, B_active=$B_active, B_raw=$B_raw, " *
              "Eg=$(L.Eg), Γ=$(L.Γ), sign=$(L.sign), converged=$(L.converged))")
end

# ---------------------------------------------------------------------------
# PoleResponse{T}
# ---------------------------------------------------------------------------

"""
    PoleResponse{T} <: AbstractResponse{T}

Pole + residue (spectral) representation of a matrix-of-ω response.

# Fields
- `a0::Matrix{T}` — constant offset; nonzero for self-energies. Shape
  `(n_rows × n_cols)`.
- `poles::Vector{Float64}` — pole positions in the `(ω − Eg)` frame, i.e.
  excitation energies when `Eg = E_GS`.
- `residues::Vector{Matrix{T}}` — residue matrices, each shape `(n_rows × n_cols)`.
- `Eg::Float64` — energy reference; default `0.0` (absolute frame). Spectroscopy
  wrappers pass `Eg = E_GS` so stored poles are excitation energies.
- `Γ::Float64` — broadening (FWHM).

# Evaluation

    R(ω) = a0 + Σ_n residues[n] / (ω − poles[n] + iΓ/2)

where ω is in the same frame as `Eg`.
"""
struct PoleResponse{T} <: AbstractResponse{T}
    a0::Matrix{T}               # constant offset
    poles::Vector{Float64}      # pole positions in the (ω − Eg) frame
    residues::Vector{Matrix{T}} # residue matrices
    Eg::Float64                 # energy reference (default 0.0 = absolute)
    Γ::Float64                  # broadening (FWHM)

    function PoleResponse{T}(a0::Matrix{T}, poles::Vector{Float64},
                             residues::Vector{Matrix{T}},
                             Eg::Float64, Γ::Float64) where {T}
        # Validate length consistency
        if length(poles) != length(residues)
            throw(ArgumentError(
                "PoleResponse: length(poles) = $(length(poles)) must equal " *
                "length(residues) = $(length(residues))"
            ))
        end

        # Validate uniform residue shape
        if length(residues) >= 2
            ref_size = size(residues[1])
            for (i, r) in enumerate(residues)
                if size(r) != ref_size
                    throw(ArgumentError(
                        "PoleResponse: all residues must have the same size; " *
                        "residues[1] has size $ref_size but residues[$i] has size $(size(r))"
                    ))
                end
            end
        end

        # Validate a0 shape against residues when non-empty
        if !isempty(residues) && size(a0) != size(first(residues))
            throw(ArgumentError(
                "PoleResponse: size(a0) = $(size(a0)) must equal " *
                "size(residues[1]) = $(size(first(residues)))"
            ))
        end

        new{T}(a0, poles, residues, Eg, Γ)
    end
end

"""
    PoleResponse(poles, residues; a0=zeros(T, n, n), Eg=0.0, Γ)

Convenience constructor with defaults `a0 = zeros(T, n_rows, n_cols)` and
`Eg = 0.0`.

Validates (via the inner constructor):
- `length(poles) == length(residues)`
- All `residues[i]` have the same `size`
- `size(a0) == size(residues[1])` when residues is non-empty
- When `residues` is empty and no explicit `a0` is supplied, raises
  `ArgumentError` (cannot infer shape); empty poles with explicit `a0`
  is a valid constant-only response.
"""
function PoleResponse(
    poles::AbstractVector{<:Real},
    residues::AbstractVector{<:AbstractMatrix{T}};
    a0::Union{AbstractMatrix{T}, Nothing} = nothing,
    Eg::Real = 0.0,
    Γ::Real
) where {T}
    residues_concrete = Vector{Matrix{T}}(residues)
    a0_resolved = if a0 === nothing
        if isempty(residues_concrete)
            throw(ArgumentError(
                "PoleResponse: cannot infer a0 dimensions from empty residues; " *
                "supply a0 explicitly"
            ))
        end
        zeros(T, size(first(residues_concrete), 1), size(first(residues_concrete), 2))
    else
        Matrix{T}(a0)
    end

    return PoleResponse{T}(a0_resolved, Vector{Float64}(poles), residues_concrete,
                           Float64(Eg), Float64(Γ))
end

"""
    PoleResponse(a0, poles, residues, Eg, Γ)

Unparameterized positional field-order outer constructor; infers `T` from
`a0` and forwards to the validated inner constructor. Intended for use by
HDF5 loaders.
"""
function PoleResponse(a0::AbstractMatrix{T},
                      poles::AbstractVector{<:Real},
                      residues::AbstractVector{<:AbstractMatrix{T}},
                      Eg::Real, Γ::Real) where {T}
    return PoleResponse{T}(
        Matrix{T}(a0),
        Vector{Float64}(poles),
        Vector{Matrix{T}}(residues),
        Float64(Eg), Float64(Γ)
    )
end

Base.eltype(::PoleResponse{T}) where {T} = T
Base.eltype(::Type{PoleResponse{T}}) where {T} = T

function Base.show(io::IO, P::PoleResponse{T}) where {T}
    n_poles = length(P.poles)
    sz = isempty(P.residues) ? size(P.a0) : size(first(P.residues))
    print(io, "PoleResponse{$T}(n_poles=$n_poles, size=$sz, Eg=$(P.Eg), Γ=$(P.Γ))")
end

# ---------------------------------------------------------------------------
# GridResponse{T, N}
# ---------------------------------------------------------------------------

"""
    GridResponse{T, N} <: AbstractResponse{T}

Matrix-of-ω response sampled on a discrete ω-grid. The last axis of `data`
indexes the frequency grid.

# Fields
- `data::Array{T, N}` — sampled response; last axis = ω.
- `ω::Vector{Float64}` — frequency grid in the `(ω − Eg)` frame (i.e.
  excitation energies when `Eg = E_GS`).
- `Eg::Float64` — energy reference; default `0.0` (absolute frame).
- `Γ::Float64` — broadening (FWHM).

# Constraint

`size(data, ndims(data)) == length(ω)` is enforced by the constructor.

# Constructor

    GridResponse(data, ω; Eg=0.0, Γ)          # ω::AbstractVector{<:Real}
    GridResponse{T, N}(data, ω, Eg, Γ)        # ω::Vector{Float64} (validates same constraint)
"""
struct GridResponse{T, N} <: AbstractResponse{T}
    data::Array{T, N}           # last axis = ω
    ω::Vector{Float64}          # in the (ω − Eg) frame
    Eg::Float64                 # energy reference (default 0.0 = absolute)
    Γ::Float64                  # broadening (FWHM)

    function GridResponse{T, N}(data::Array{T, N}, ω::Vector{Float64},
                                Eg::Float64, Γ::Float64) where {T, N}
        if size(data, ndims(data)) != length(ω)
            throw(ArgumentError(
                "GridResponse: size(data, ndims(data)) = $(size(data, ndims(data))) " *
                "must equal length(ω) = $(length(ω))"
            ))
        end
        new{T, N}(data, ω, Eg, Γ)
    end
end

"""
    GridResponse(data::Array{T,N}, ω::AbstractVector{<:Real}; Eg=0.0, Γ) where {T,N}

Outer constructor with keyword arguments; forwards to the inner constructor
which validates `size(data, ndims(data)) == length(ω)`.
Raises `ArgumentError` if the constraint is violated.
"""
function GridResponse(data::Array{T, N}, ω::AbstractVector{<:Real};
                      Eg::Real = 0.0, Γ::Real) where {T, N}
    return GridResponse{T, N}(data, Vector{Float64}(ω), Float64(Eg), Float64(Γ))
end

"""
    GridResponse(data, ω, Eg, Γ)

Unparameterized positional field-order outer constructor; infers `T` and `N`
from `data` and forwards to the validated inner constructor. Intended for use
by HDF5 loaders.
"""
function GridResponse(data::Array{T, N}, ω::AbstractVector{<:Real},
                      Eg::Real, Γ::Real) where {T, N}
    return GridResponse{T, N}(data, Vector{Float64}(ω), Float64(Eg), Float64(Γ))
end

Base.eltype(::GridResponse{T}) where {T} = T
Base.eltype(::Type{GridResponse{T, N}}) where {T, N} = T

function Base.show(io::IO, G::GridResponse{T, N}) where {T, N}
    nω      = length(G.ω)
    ω_range = nω > 0 ? "[$(G.ω[1]), $(G.ω[end])]" : "[]"
    print(io, "GridResponse{$T,$N}(size=$(size(G.data)), ω=$ω_range ($nω pts), " *
              "Eg=$(G.Eg), Γ=$(G.Γ))")
end

# ---------------------------------------------------------------------------
# GreensFunction{T, R}
# ---------------------------------------------------------------------------

"""
    GreensFunction{T, R <: AbstractResponse{T}} <: AbstractResponse{T}

Parametric single-particle Green's function wrapping an addition channel
and/or a removal channel.

# Fields
- `addition::Union{Nothing, R}` — particle-addition (particle) channel.
- `removal::Union{Nothing, R}` — particle-removal (hole) channel.

At least one channel must be populated; the constructor raises
`ArgumentError` if both are `nothing`.

# Notes

The inner representation type `R` is fixed at construction time. Combining
channels with different inner representations is done through the conversion
helpers (`to_pole`, `to_grid`) before constructing the `GreensFunction`.

# Constructor

    GreensFunction(addition::Union{Nothing, R}, removal::Union{Nothing, R})
    GreensFunction{T, R}(addition, removal)   # explicit parametric form
"""
struct GreensFunction{T, R <: AbstractResponse{T}} <: AbstractResponse{T}
    addition::Union{Nothing, R}
    removal::Union{Nothing, R}

    function GreensFunction{T, R}(addition::Union{Nothing, R},
                                  removal::Union{Nothing, R}) where {T, R <: AbstractResponse{T}}
        if isnothing(addition) && isnothing(removal)
            throw(ArgumentError(
                "GreensFunction: at least one channel must be populated " *
                "(both `addition` and `removal` are `nothing`)"
            ))
        end
        return new{T, R}(addition, removal)
    end
end

"""
    GreensFunction(addition::Union{Nothing, R}, removal::Union{Nothing, R}) where {T, R <: AbstractResponse{T}}

Outer constructor; forwards to the inner constructor which validates that
at least one channel is populated. Raises `ArgumentError` if both channels
are `nothing`.

Typical call patterns:
- `GreensFunction(addition, nothing)` — addition (particle) channel only.
- `GreensFunction(nothing, removal)` — removal (hole) channel only.
- `GreensFunction(addition, removal)` — both channels; `to_grid` sums them.

Both channels must share the same inner representation type `R`. Convert
to a common type (e.g., via `to_pole`) before constructing if they differ.
`GreensFunction(nothing, nothing)` is an `ArgumentError`.
"""
function GreensFunction(addition::Union{Nothing, R},
                        removal::Union{Nothing, R}) where {T, R <: AbstractResponse{T}}
    return GreensFunction{T, R}(addition, removal)
end

# Specialised method for the both-nothing case: T and R can't be inferred,
# so dispatch a clean ArgumentError instead of UndefVarError(:T, :static_parameter).
function GreensFunction(::Nothing, ::Nothing)
    throw(ArgumentError(
        "GreensFunction: at least one channel (addition or removal) must be populated"
    ))
end

Base.eltype(::GreensFunction{T}) where {T} = T
Base.eltype(::Type{GreensFunction{T, R}}) where {T, R} = T

function Base.show(io::IO, GF::GreensFunction{T, R}) where {T, R}
    channels = String[]
    isnothing(GF.addition) || push!(channels, "addition")
    isnothing(GF.removal)  || push!(channels, "removal")
    channel_str = join(channels, ", ")
    print(io, "GreensFunction{$T,$R}(channels=$channel_str)")
end
