# =====================================================================
# LanczosChunk + SpectraTensor — spectroscopy result types
# =====================================================================
#
# Data carriers passed between the spectroscopy entry points (`xas`,
# `rixs`, `fluorescence_yield`) and the post-processing helpers
# (`re_broaden`, `polarise`, `poles`, `evaluate_on_grid`,
# `save_spectra`, …).

"""
    LanczosChunk{T<:Number}

One block-Lanczos run, stored in a form that lets the cf machinery
re-evaluate the user-basis correlator at any frequency / Γ without
re-running matvecs.

Fields:

- `α::Vector{Matrix{T}}` — block-α stack, one matrix per Lanczos
  iteration; `α[k]` is `B_active × B_active`.
- `β::Vector{Matrix{T}}` — block-β stack, one per inner-recurrence step.
- `R::Matrix{T}` — initial QR factor of shape `B_active × B_raw`.
  `X_raw = V₁ · R` with `V₁` orthonormal. The user-basis correlator
  reads `R† · G₁(z) · R`. When the starting block has linearly
  dependent columns (symmetry-induced channel deflation), `B_active <
  B_raw` and `R` is rectangular; the output tensor still keeps the
  user's `B_raw` dimensions, recovered via the `R` sandwich.
- `raw_block_size::Int` — `B_raw`, the user's input dimensionality.
- `ψ_index::Int` — 1-based index of the input ψ this chunk corresponds
  to.
- `ω_in_index::Int` — 1-based index of the incident-energy point this
  chunk corresponds to (0 for XAS / FY which have no `ω_in`).
- `n_iter::Int` — actual number of block-Lanczos iterations reached
  (after deflation / convergence).
- `converged::Bool` — whether the soft-stop convergence condition
  fired before `n_iter` (i.e., `‖β‖_F / ‖β_1‖_F` dropped below `tol`
  for ≥ `min_iter` consecutive steps).

In all three spectroscopy pipelines `T = Complex{Real}` (the cf
machinery carries `iΓ/2`), regardless of whether the eventual
`SpectraTensor` element type is real (FY) or complex (XAS / RIXS).
"""
struct LanczosChunk{T<:Number}
    α::Vector{Matrix{T}}
    β::Vector{Matrix{T}}
    R::Matrix{T}
    raw_block_size::Int
    ψ_index::Int
    ω_in_index::Int
    n_iter::Int
    converged::Bool
end

"""
    SpectraTensor{S<:Number, T<:Real, N, F}

Result of an `xas`, `rixs`, or `fluorescence_yield` call. Parametric in
four type parameters:

- `S` — element type of the spectral tensor. `Complex{T}` for XAS / RIXS
  (the Green's-function-valued correlator carries phase information for
  e.g. circular polarisation); real `T` for fluorescence yield (the
  analytic ω_out integral has already extracted the real spectral
  weight).
- `T` — underlying real precision (typically `Float64`).
- `N` — number of dimensions of `tensor`. Set by input dispatch:
  scalar XAS → `N = 1`; XAS with operator vector → `N = 3`; RIXS
  scalar → `N = 2`; RIXS with operator vectors → `N = 6`; FY
  scalar → `N = 1`. List-of-ψ inputs add one leading axis.
- `F` — type of the `ω_grid` field. `AbstractRange{T}` for XAS / FY,
  `NTuple{2, AbstractRange{T}}` for RIXS.

Fields:

- `tensor::Array{S, N}` — the spectral tensor, shape per the dispatch
  table.
- `chunks::Union{Nothing, Vector{LanczosChunk{Complex{T}}}}` — list of
  block-Lanczos chunks underlying the tensor. Populated for fresh
  spectroscopy results; set to `nothing` for algebra-derived results
  (sums, scalar multiples, polarisation contractions) which no longer
  correspond to a single Lanczos run.
- `ω_grid::F` — the frequency grid(s) along which the tensor is
  sampled. `AbstractRange` for single-axis spectra (XAS / FY), tuple
  `(ω_in_grid, ω_out_grid)` for RIXS.
- `Eg::Float64` — the ground-state energy used as the zero-frequency
  reference in the Green's-function denominator
  `(ω + Eg + iΓ/2) · I − H`.
- `block_size::Int` — `B` at the outer-Lanczos level (`B_raw` for the
  outer block, useful for downstream sanity checks).
- `metadata::Dict{Symbol,Any}` — non-numerical context: operator names,
  Γ values, edge label, convergence flag, restrictions, auto-range
  diagnostics, MOAD version, timestamp, etc.
"""
struct SpectraTensor{S<:Number, T<:Real, N, F}
    tensor::Array{S, N}
    chunks::Union{Nothing, Vector{LanczosChunk{Complex{T}}}}
    ω_grid::F
    Eg::Float64
    block_size::Int
    metadata::Dict{Symbol,Any}
end

# --- show -------------------------------------------------------------------

function Base.show(io::IO, c::LanczosChunk{T}) where {T}
    print(io, "LanczosChunk{", T, "}(",
          "n_iter=", c.n_iter, ", ",
          "block=", size(c.R, 1), "×", c.raw_block_size,
          c.ω_in_index == 0 ? "" : ", ω_in_index=$(c.ω_in_index)",
          ", ψ_index=", c.ψ_index,
          ", converged=", c.converged, ")")
end

function Base.show(io::IO, st::SpectraTensor{S,T,N}) where {S,T,N}
    print(io, "SpectraTensor{", S, ", ", T, "}(")
    print(io, "tensor=Array{", S, ",", N, "} of size ", size(st.tensor))
    if st.chunks !== nothing
        print(io, ", chunks=", length(st.chunks))
    else
        print(io, ", chunks=nothing")
    end
    print(io, ", Eg=", st.Eg, ", block_size=", st.block_size, ")")
end
