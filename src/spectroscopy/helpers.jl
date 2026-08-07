# =====================================================================
# Post-processing helpers
# =====================================================================
#
# `re_broaden`, `polarise`, `poles`, `find_chunk`, `tridiagonal`,
# `restrict_to_window`, `average`, `weighted_sum`, `plot_range`, plus
# spectrum algebra (`+`, `-`, `*scalar`).
#
# All operate on the small per-chunk Lanczos data (block tridiagonal at
# most `B·K_fin × B·K_fin`) or on the precomputed tensor — no Lanczos
# rerun, no large-Hilbert-space operations.
#
# These are not re-exported at the umbrella `MOADyna` namespace; access as
# `MOADyna.Spectroscopy.re_broaden(...)`, etc.
#
# Note: these helpers all take `SpectraTensor` or `LanczosChunk` as
# primary arguments — Spectroscopy-domain types. The corresponding
# `LanczosResponse` / `PoleResponse` / `GridResponse` evaluation paths
# live in `MOADyna.Responses`; pulling these wrappers into Responses
# would create a Responses → Spectroscopy → Responses cycle.

# ---------------------------------------------------------------------
# find_chunk — convenience lookup
# ---------------------------------------------------------------------

"""
    find_chunk(result; ψ_index = 1, ω_in_index = 0) -> LanczosChunk

Locate the chunk corresponding to `(ψ_index, ω_in_index)` in
`result.chunks`. `ω_in_index = 0` for XAS and FY (no incident axis).
Linear scan; trivial cost.
"""
function find_chunk(result::SpectraTensor;
                    ψ_index::Int = 1, ω_in_index::Int = 0)
    result.chunks === nothing &&
        throw(ArgumentError("find_chunk: result has no chunks (algebra-derived)"))
    for c in result.chunks
        c.ψ_index == ψ_index && c.ω_in_index == ω_in_index && return c
    end
    throw(ArgumentError(
        "find_chunk: no chunk with ψ_index=$ψ_index, ω_in_index=$ω_in_index"))
end

# ---------------------------------------------------------------------
# tridiagonal — wrap (α, β, R) for serialisation / interop
# ---------------------------------------------------------------------

"""
    BlockTriDiagonal{T}

Container for `(α, β, R)` of one block-Lanczos chunk in tridiagonal-
Green's-function form. Used by `tridiagonal(chunk)` for serialisation
and for cross-package interop. The user-basis spectrum is reconstructed
by `R† · G₁(z) · R` where `G₁(z)` is the upper-left
`B_active × B_active` block of `(zI − T_K)⁻¹`.
"""
struct BlockTriDiagonal{T<:Number}
    α::Vector{Matrix{T}}
    β::Vector{Matrix{T}}
    R::Matrix{T}
end

"""
    tridiagonal(chunk::LanczosChunk) -> BlockTriDiagonal

Wrap a chunk's `(α, β, R)` data as a `BlockTriDiagonal`.
"""
tridiagonal(chunk::LanczosChunk{T}) where {T} =
    BlockTriDiagonal{T}(chunk.α, chunk.β, chunk.R)

# ---------------------------------------------------------------------
# poles — Lehmann decomposition in the user's transition-operator basis
# ---------------------------------------------------------------------

"""
    poles(result; ψ_index = 1, ω_in_index = 0)
        -> Vector{NamedTuple{(:E, :weight), Tuple{Float64, Matrix}}}

Eigendecompose the chunk's block-tridiagonal `T_K` and return the
Lehmann form of the user-basis correlator in the chunk:

```
χ(z) = Σ_n weight_n / (z − E_n)        z = ω + Eg + iΓ/2
weight_n = (R† |n⟩_top) · (R† |n⟩_top)†
```

where `|n⟩_top` is the top `B_active` block of the n-th Ritz vector of
`T_K`. Each `weight_n` has shape `B_raw × B_raw`. For scalar input
(`B_raw = 1`), `dropdims(weight_n; dims = (1, 2))` recovers a scalar
residue.

Cost: one dense Hermitian eigenproblem on the `B_active · K`-sized
T_K (~ms to ~s per chunk depending on K).
"""
function poles(result::SpectraTensor;
               ψ_index::Int = 1, ω_in_index::Int = 0)
    chunk = find_chunk(result; ψ_index = ψ_index, ω_in_index = ω_in_index)
    return _chunk_lehmann(chunk)
end

function _chunk_lehmann(chunk::LanczosChunk{T}) where {T<:Number}
    T_K = _build_T_K(chunk.α, chunk.β)
    F   = eigen(Hermitian(T_K))
    Es  = F.values                  # real
    Vs  = F.vectors                 # T_K dim × T_K dim
    B_active = size(chunk.R, 1)
    B_raw    = chunk.raw_block_size
    R        = chunk.R
    Rdag     = adjoint(R)
    out = NamedTuple{(:E, :weight), Tuple{Float64, Matrix{T}}}[]
    @inbounds for n in 1:length(Es)
        # |n⟩_top = first B_active rows of Vs[:, n]
        n_top  = Vs[1:B_active, n]
        # weight_n = R† · |n_top⟩ · ⟨n_top|† · R = (R† n_top)·(R† n_top)†
        v      = Rdag * n_top                  # B_raw × 1
        w      = v * adjoint(v)                # B_raw × B_raw
        push!(out, (E = Float64(real(Es[n])), weight = Matrix{T}(w)))
    end
    return out
end

# ---------------------------------------------------------------------
# re_broaden — re-evaluate cf at new Γ (output-axis only)
# ---------------------------------------------------------------------

"""
    re_broaden(result; Γ) -> SpectraTensor
    re_broaden(result; Γ, σ_gauss) -> SpectraTensor   # Voigt (Γ Lorentzian + σ Gaussian)

Re-evaluate the cf using each chunk's α/β/R at the new broadening Γ.
No Lanczos rerun. Returns a fresh `SpectraTensor` with the same chunks
(the recipe to evaluate them) and the new tensor.

Accepted aliases:

- `Γ` (canonical) for XAS and FY (no `Γ_intermediate` re-broadening
  see the note below).
- `Γ_final` is accepted as the canonical name when the result is a
  RIXS spectrum (matches the function signature of `rixs(...)`).
- ASCII Latin: `Gamma`, `Gamma_final`.
- For RIXS, `Γ_intermediate` re-broadening is rejected with a clear
  ArgumentError directing the user to rerun `rixs(...; Γ_intermediate)`.

`σ_gauss` adds a Gaussian convolution after the cf re-evaluation
(Voigt profile). Off by default. (The Lorentzian-only path is the
typical use; Voigt is for instrumental Gaussian broadening on top.)
"""
function re_broaden(result::SpectraTensor;
                    Γ                 = nothing,
                    Gamma             = nothing,
                    Γ_final           = nothing,
                    Gamma_final       = nothing,
                    Γ_intermediate    = nothing,
                    Gamma_intermediate = nothing,
                    σ_gauss::Real     = 0.0,
                    sigma_gauss       = nothing)
    fn_sym = Symbol(result.metadata[:function])
    is_rixs = fn_sym === :rixs
    is_fy   = fn_sym === :fluorescence_yield

    # FY refuses re-broadening — final dynamics is integrated out.
    is_fy && (Γ !== nothing || Gamma !== nothing ||
              Γ_final !== nothing || Gamma_final !== nothing) &&
        throw(ArgumentError(
            "re_broaden: FY result has no output Γ to re-broaden " *
            "(final-state dynamics analytic-integrated; rerun fluorescence_yield(...; Γ_intermediate) instead)"))

    # Γ_intermediate not supported by re_broaden (would require re-running outer Lanczos).
    (Γ_intermediate !== nothing || Gamma_intermediate !== nothing) &&
        throw(ArgumentError(
            "re_broaden: Γ_intermediate re-broadening would require " *
            "re-running the inner / per-ω_in outer Lanczos pass; " *
            "rerun rixs(...; Γ_intermediate = ...) or " *
            "fluorescence_yield(...; Γ_intermediate = ...) instead."))

    # Resolve Γ pair (canonical or final) and ASCII aliases.
    Γ_user = if is_rixs
        # RIXS: canonical name for output is Γ_final; both Γ and Γ_final accepted.
        candidates = [Γ, Γ_final, Gamma, Gamma_final]
        n_set = count(!isnothing, candidates)
        n_set ≤ 1 || throw(ArgumentError(
            "re_broaden (RIXS): specify Γ_final (or alias Gamma_final) only once; " *
            "Γ and Γ_final are aliases here"))
        n_set == 0 && throw(ArgumentError(
            "re_broaden: must specify Γ (or Γ_final / Gamma / Gamma_final)"))
        first(filter(!isnothing, candidates))
    else
        # XAS: canonical Γ; ASCII Gamma. Γ_final is rejected.
        (Γ_final !== nothing || Gamma_final !== nothing) &&
            throw(ArgumentError("re_broaden: Γ_final does not apply to XAS; use Γ"))
        _resolve_pair(Γ, Gamma; default = nothing, name = "Γ")
    end
    Γ_user === nothing && throw(ArgumentError("re_broaden: must specify Γ"))

    σ_use = sigma_gauss === nothing ? σ_gauss : sigma_gauss
    σ_use ≥ 0 || throw(ArgumentError("re_broaden: σ_gauss must be ≥ 0"))

    # Re-evaluate each chunk on the result's existing ω_grid with new Γ.
    if is_rixs
        return _rixs_rebroaden(result, Float64(Γ_user), Float64(σ_use))
    else
        return _xas_rebroaden(result, Float64(Γ_user), Float64(σ_use))
    end
end

# Internal: rebuild XAS / FY-shaped tensor with new Γ over result's ω_grid.
function _xas_rebroaden(result::SpectraTensor, Γ_new::Real, σ_gauss::Real)
    chunks = result.chunks
    chunks === nothing &&
        throw(ArgumentError("re_broaden: result has no chunks (algebra-derived)"))
    Eg     = result.Eg
    ω_grid = result.ω_grid
    n_ω    = length(ω_grid)
    χs = [evaluate_on_grid(c, ω_grid; Γ = Γ_new, Eg = Eg) for c in chunks]

    T_is_vector = result.metadata[:T_is_vector_input]::Bool
    ψ_is_list   = result.metadata[:ψ_is_list_input]::Bool
    N_T = result.metadata[:n_T]::Int
    N_ψ = result.metadata[:n_ψ]::Int

    new_tensor = _xas_pack(χs, T_is_vector, ψ_is_list, N_T, N_ψ, n_ω)

    if σ_gauss > 0
        new_tensor = _convolve_gaussian(new_tensor, ω_grid, σ_gauss)
    end

    new_meta = copy(result.metadata)
    new_meta[:Γ] = Γ_new
    σ_gauss > 0 && (new_meta[:σ_gauss] = σ_gauss)
    new_meta[:rebroadened_from] = (Γ = result.metadata[:Γ],)

    F = typeof(ω_grid)
    N = ndims(new_tensor)
    return SpectraTensor{eltype(new_tensor), Float64, N, F}(
        new_tensor, chunks, ω_grid, Eg, result.block_size, new_meta)
end

function _rixs_rebroaden(result::SpectraTensor, Γ_new::Real, σ_gauss::Real)
    chunks = result.chunks
    chunks === nothing &&
        throw(ArgumentError("re_broaden: result has no chunks (algebra-derived)"))
    Eg = result.Eg
    ω_in_grid, ω_out_grid = result.ω_grid::Tuple
    n_in  = length(ω_in_grid)
    n_out = length(ω_out_grid)

    md = result.metadata
    N_in   = md[:n_T_in]::Int
    N_out  = md[:n_T_out]::Int
    N_ψ    = md[:n_ψ]::Int
    T_in_vec  = md[:T_in_is_vector_input]::Bool
    T_out_vec = md[:T_out_is_vector_input]::Bool
    ψ_is_list = md[:ψ_is_list_input]::Bool
    full6d = T_in_vec || T_out_vec

    # Allocate output tensor matching input shape. RIXS 6-D layout is
    # (N_in, N_out, N_out, N_in, n_in, n_out) (Kramers-Heisenberg axes
    # i, j, k, l: i, l index T_in, j, k index T_out).
    new_tensor = if full6d && ψ_is_list
        zeros(ComplexF64, N_in, N_out, N_out, N_in, N_ψ, n_in, n_out)
    elseif full6d
        zeros(ComplexF64, N_in, N_out, N_out, N_in, n_in, n_out)
    elseif ψ_is_list
        zeros(ComplexF64, N_ψ, n_in, n_out)
    else
        zeros(ComplexF64, n_in, n_out)
    end

    for chunk in chunks
        χ̄ = evaluate_on_grid(chunk, ω_out_grid; Γ = Γ_new, Eg = Eg)
        ψ_idx    = chunk.ψ_index
        ω_in_idx = chunk.ω_in_index
        for ω_idx in 1:n_out
            slab = χ̄[:, :, ω_idx]
            if full6d
                χ_jikl = reshape(slab, N_out, N_in, N_out, N_in)
                χ_user = permutedims(χ_jikl, (2, 1, 3, 4))
                if ψ_is_list
                    new_tensor[:, :, :, :, ψ_idx, ω_in_idx, ω_idx] .= χ_user
                else
                    new_tensor[:, :, :, :, ω_in_idx, ω_idx] .= χ_user
                end
            else
                # Both T_in and T_out scalar → slab is 1×1.
                if ψ_is_list
                    new_tensor[ψ_idx, ω_in_idx, ω_idx] = slab[1, 1]
                else
                    new_tensor[ω_in_idx, ω_idx] = slab[1, 1]
                end
            end
        end
    end

    if σ_gauss > 0
        # Convolve along ω_out (the last axis).
        new_tensor = _convolve_gaussian(new_tensor, ω_out_grid, σ_gauss)
    end

    new_meta = copy(md)
    new_meta[:Γ_final] = Γ_new
    σ_gauss > 0 && (new_meta[:σ_gauss] = σ_gauss)
    new_meta[:rebroadened_from] = (Γ_final = md[:Γ_final],)

    F = typeof(result.ω_grid)
    N = ndims(new_tensor)
    return SpectraTensor{eltype(new_tensor), Float64, N, F}(
        new_tensor, chunks, result.ω_grid, Eg, result.block_size, new_meta)
end

# Gaussian convolution along the last axis (the ω axis). Real-space
# convolution; cheap for typical n_ω (≤ 10⁴).
function _convolve_gaussian(tensor::AbstractArray, ω_grid::AbstractRange,
                            σ::Real)
    n = length(ω_grid)
    Δω = step(ω_grid)
    half_window = ceil(Int, 5 * σ / Δω)
    kernel = [exp(-(k * Δω)^2 / (2 * σ^2)) for k in -half_window:half_window]
    kernel ./= sum(kernel)
    out = similar(tensor)
    fill!(out, 0)
    last_axis = ndims(tensor)
    # Convolve along the last axis.
    if last_axis == 1
        @inbounds for i in 1:n, k in eachindex(kernel)
            j = i + (k - half_window - 1)
            j ≥ 1 && j ≤ n && (out[i] += kernel[k] * tensor[j])
        end
    else
        slicedims = ntuple(d -> :, last_axis - 1)
        @inbounds for i in 1:n, k in eachindex(kernel)
            j = i + (k - half_window - 1)
            j ≥ 1 && j ≤ n &&
                (out[slicedims..., i] .+= kernel[k] .* tensor[slicedims..., j])
        end
    end
    return out
end

# ---------------------------------------------------------------------
# polarise — contract the polarisation tensor; return real intensity
# ---------------------------------------------------------------------

"""
    polarise(result::SpectraTensor) -> Array{Float64}
    polarise(result::SpectraTensor, ε::AbstractVector) -> Array{Float64}

Convert a raw `SpectraTensor` to a real observable by taking `−Im` of
the response tensor, optionally after contracting with polarisation
vectors.

**No `ε` argument** — scalar-operator case (XAS computed with a single
transition operator, RIXS with scalar `T_in`/`T_out`, or FY):
- Returns `-Im(tensor)` as a plain `Array{Float64}`.
- For FY results (which are already real), returns the tensor
  unchanged.
- Return shape mirrors the tensor layout: `(n_ω,)` for XAS,
  `(n_in, n_out)` for RIXS, `(n_ω_in,)` for FY — or with a leading
  `N_ψ` axis when multiple initial states were passed.

**Single `ε` argument** — polarised XAS (XAS computed with a vector of
transition operators):
- Computes `−Im(Σ_{ab} ε*_a χ_{ab}(ω) ε_b)`.
- `ε` must have length `N_T` (the number of operators passed to `xas`).
- Return shape: `(n_ω,)` for a single initial state, `(N_ψ, n_ω)`
  for a list of initial states.

Raises `ArgumentError` if the method/ε combination does not match the
result type.

See also `polarise(result, ε_in, ε_out)` for RIXS polarisation.
"""
function polarise(result::SpectraTensor)
    if Symbol(result.metadata[:function]) === :fluorescence_yield
        return Array(result.tensor)         # already real
    end
    # XAS/RIXS scalar case: tensor shape (n_ω,) for XAS or (n_in, n_out) for RIXS.
    is_complex = eltype(result.tensor) <: Complex
    return is_complex ? -imag.(result.tensor) : Array(result.tensor)
end

function polarise(result::SpectraTensor, ε::AbstractVector)
    md = result.metadata
    Symbol(md[:function]) === :xas ||
        throw(ArgumentError("polarise(result, ε): result is not XAS; use polarise(result, ε_in, ε_out) for RIXS"))
    md[:T_is_vector_input]::Bool ||
        throw(ArgumentError("polarise(result, ε): result was computed with a scalar T; call polarise(result) instead"))
    N_T = md[:n_T]::Int
    length(ε) == N_T ||
        throw(DimensionMismatch("polarise: ε has length $(length(ε)); expected $N_T (the operator-pair axis size)"))

    tensor = result.tensor
    ε_c = Vector{eltype(tensor)}(ε)
    ε_star = conj.(ε_c)

    # tensor shape:
    #   (N_T, N_T, n_ω)         when ψ is single
    #   (N_T, N_T, N_ψ, n_ω)    when ψ is a list
    if ndims(tensor) == 3
        n_ω = size(tensor, 3)
        out = Vector{Float64}(undef, n_ω)
        @inbounds for i in 1:n_ω
            s = zero(eltype(tensor))
            for a in 1:N_T, b in 1:N_T
                s += ε_star[a] * tensor[a, b, i] * ε_c[b]
            end
            out[i] = -imag(s)
        end
        return out
    elseif ndims(tensor) == 4
        N_ψ, n_ω = size(tensor, 3), size(tensor, 4)
        out = Matrix{Float64}(undef, N_ψ, n_ω)
        @inbounds for ψ_idx in 1:N_ψ, i in 1:n_ω
            s = zero(eltype(tensor))
            for a in 1:N_T, b in 1:N_T
                s += ε_star[a] * tensor[a, b, ψ_idx, i] * ε_c[b]
            end
            out[ψ_idx, i] = -imag(s)
        end
        return out
    else
        throw(ArgumentError("polarise: unexpected XAS tensor shape $(size(tensor))"))
    end
end

"""
    polarise(result::SpectraTensor, ε_in::AbstractVector,
             ε_out::AbstractVector) -> Array{Float64}

Polarisation contraction for a RIXS `SpectraTensor` (Kramers-Heisenberg
form):

```
σ(ω_in, ω_out) = −Im Σ_{ijkl} ε*_{in,i} ε_{out,j}
                              · χ_{ijkl}(ω_in, ω_out)
                              · ε*_{out,k} ε_{in,l}
```

`ε_in` and `ε_out` must have lengths matching the `N_T_in` and `N_T_out`
axes of the tensor (i.e. the number of in/out operators passed to `rixs`).

Return shape:
- Single initial state: `(n_ω_in, n_ω_out)` real.
- List of initial states: `(N_ψ, n_ω_in, n_ω_out)` real.

If RIXS was computed with scalar `T_in` and scalar `T_out` (no
polarisation axes), call `polarise(result)` instead — this method will
raise an `ArgumentError`.
"""
function polarise(result::SpectraTensor,
                  ε_in::AbstractVector, ε_out::AbstractVector)
    md = result.metadata
    Symbol(md[:function]) === :rixs ||
        throw(ArgumentError("polarise(result, ε_in, ε_out): result is not RIXS"))
    T_in_vec  = md[:T_in_is_vector_input]::Bool
    T_out_vec = md[:T_out_is_vector_input]::Bool
    (T_in_vec || T_out_vec) ||
        throw(ArgumentError(
            "polarise(result, ε_in, ε_out): RIXS result was computed with both " *
            "T_in and T_out scalar (no polarisation axes); call polarise(result) instead"))

    N_in  = md[:n_T_in]::Int
    N_out = md[:n_T_out]::Int
    length(ε_in) == N_in ||
        throw(DimensionMismatch("polarise: ε_in length $(length(ε_in)) ≠ N_in = $N_in"))
    length(ε_out) == N_out ||
        throw(DimensionMismatch("polarise: ε_out length $(length(ε_out)) ≠ N_out = $N_out"))

    tensor = result.tensor
    Tc = eltype(tensor)
    ε_in_c     = Vector{Tc}(ε_in)
    ε_in_star  = conj.(ε_in_c)
    ε_out_c    = Vector{Tc}(ε_out)
    ε_out_star = conj.(ε_out_c)

    if ndims(tensor) == 6
        # Single ψ: (N_in, N_out, N_out, N_in, n_in, n_out)
        n_in  = size(tensor, 5)
        n_out = size(tensor, 6)
        out = Matrix{Float64}(undef, n_in, n_out)
        @inbounds for ω_in_idx in 1:n_in, ω_out_idx in 1:n_out
            s = zero(Tc)
            for i in 1:N_in, j in 1:N_out, k in 1:N_out, l in 1:N_in
                s += ε_in_star[i] * ε_out_c[j] *
                     tensor[i, j, k, l, ω_in_idx, ω_out_idx] *
                     ε_out_star[k] * ε_in_c[l]
            end
            out[ω_in_idx, ω_out_idx] = -imag(s)
        end
        return out
    elseif ndims(tensor) == 7
        # List ψ: (N_in, N_out, N_out, N_in, N_ψ, n_in, n_out)
        N_ψ   = size(tensor, 5)
        n_in  = size(tensor, 6)
        n_out = size(tensor, 7)
        out = Array{Float64,3}(undef, N_ψ, n_in, n_out)
        @inbounds for ψ_idx in 1:N_ψ, ω_in_idx in 1:n_in, ω_out_idx in 1:n_out
            s = zero(Tc)
            for i in 1:N_in, j in 1:N_out, k in 1:N_out, l in 1:N_in
                s += ε_in_star[i] * ε_out_c[j] *
                     tensor[i, j, k, l, ψ_idx, ω_in_idx, ω_out_idx] *
                     ε_out_star[k] * ε_in_c[l]
            end
            out[ψ_idx, ω_in_idx, ω_out_idx] = -imag(s)
        end
        return out
    else
        throw(ArgumentError("polarise: unexpected RIXS tensor shape $(size(tensor))"))
    end
end

# ---------------------------------------------------------------------
# Spectrum algebra: +, -, *scalar, average, weighted_sum
# ---------------------------------------------------------------------

# `+`/`-`/`*` produce a SpectraTensor with `chunks = nothing` (the
# algebraic combination is no longer a single Lanczos run).

function Base.:+(a::SpectraTensor, b::SpectraTensor)
    _check_alg_compat(a, b, "+")
    new_tensor = a.tensor .+ b.tensor
    return _alg_result(new_tensor, a)
end

function Base.:-(a::SpectraTensor, b::SpectraTensor)
    _check_alg_compat(a, b, "-")
    new_tensor = a.tensor .- b.tensor
    return _alg_result(new_tensor, a)
end

Base.:*(α::Number, a::SpectraTensor) = _scalar_mul(α, a)
Base.:*(a::SpectraTensor, α::Number) = _scalar_mul(α, a)
Base.:/(a::SpectraTensor, α::Number) = _scalar_mul(inv(α), a)

function _scalar_mul(α::Number, a::SpectraTensor)
    new_tensor = α .* a.tensor
    return _alg_result(new_tensor, a)
end

function _check_alg_compat(a::SpectraTensor, b::SpectraTensor, op::String)
    a.ω_grid == b.ω_grid ||
        throw(ArgumentError("SpectraTensor $op: ω_grid mismatch"))
    size(a.tensor) == size(b.tensor) ||
        throw(DimensionMismatch("SpectraTensor $op: tensor shape mismatch ($(size(a.tensor)) vs $(size(b.tensor)))"))
end

function _alg_result(tensor, prototype::SpectraTensor)
    new_meta = Dict{Symbol,Any}(:function => :algebraic_combination,
                                 :Γ => get(prototype.metadata, :Γ, nothing),
                                 :Eg => prototype.Eg)
    F = typeof(prototype.ω_grid)
    N = ndims(tensor)
    return SpectraTensor{eltype(tensor), Float64, N, F}(
        tensor, nothing, prototype.ω_grid, prototype.Eg,
        prototype.block_size, new_meta)
end

"""
    average(results::AbstractVector{<:SpectraTensor}) -> SpectraTensor

Equal-weight mean of a collection of `SpectraTensor` results.

All inputs must share the same `ω_grid` and tensor shape; a mismatch
raises an error. The returned `SpectraTensor` has `chunks = nothing`,
so `re_broaden` is unavailable on the result — use the original
per-component results if you need to rebroad.

Typical use: averaging spectra computed for degenerate initial states
before comparing to experiment.

See also `weighted_sum` for Boltzmann-weighted averages.
"""
function average(results::AbstractVector{<:SpectraTensor})
    isempty(results) && throw(ArgumentError("average: empty input"))
    n = length(results)
    return weighted_sum(results, fill(1 / n, n))
end

"""
    weighted_sum(results::AbstractVector{<:SpectraTensor},
                 weights::AbstractVector{<:Number}) -> SpectraTensor

Compute the linear combination `Σ_i weights[i] * results[i]`.

All inputs must share the same `ω_grid` and tensor shape; a mismatch
raises an error. The returned `SpectraTensor` has `chunks = nothing`,
so `re_broaden` is unavailable on the result.

Typical use: thermal (Boltzmann) average over low-lying states, where
`weights[i] = exp(-β * E_i) / Z` and `results[i]` is the spectrum
computed from initial state `i`.

See also `average` for the equal-weight special case.
"""
function weighted_sum(results::AbstractVector{<:SpectraTensor},
                      weights::AbstractVector{<:Number})
    length(results) == length(weights) ||
        throw(DimensionMismatch("weighted_sum: length(results) ≠ length(weights)"))
    isempty(results) && throw(ArgumentError("weighted_sum: empty input"))
    out = weights[1] * results[1]
    for i in 2:length(results)
        out = out + weights[i] * results[i]
    end
    return out
end

# ---------------------------------------------------------------------
# restrict_to_window — crop ω axis (no recomputation)
# ---------------------------------------------------------------------

"""
    restrict_to_window(result, ω_window::Tuple{<:Real,<:Real}) -> SpectraTensor

Return a fresh `SpectraTensor` with the ω axis (last axis) cropped to
`ω_min ≤ ω ≤ ω_max`. Pure tensor slice; chunks are dropped to
`nothing` because the cropped grid is no longer the natural one for
re-evaluation.
"""
function restrict_to_window(result::SpectraTensor,
                            ω_window::Tuple{<:Real,<:Real})
    ωmin, ωmax = ω_window

    # Pick the axis to crop. RIXS has a tuple ω_grid (ω_in, ω_out); we
    # crop ω_out (the last tensor axis) and keep ω_in unchanged. XAS / FY
    # have a single ω_grid that maps to the last axis.
    if result.ω_grid isa Tuple
        ω_in_grid, ω_out_grid = result.ω_grid
        keep = findall(ω -> ωmin ≤ ω ≤ ωmax, ω_out_grid)
        isempty(keep) &&
            throw(ArgumentError("restrict_to_window: empty intersection with ω_out grid"))
        cropped_ω_out = ω_out_grid[keep[1]:keep[end]]
        new_grid = (ω_in_grid, cropped_ω_out)
    else
        keep = findall(ω -> ωmin ≤ ω ≤ ωmax, result.ω_grid)
        isempty(keep) &&
            throw(ArgumentError("restrict_to_window: empty intersection with ω_grid"))
        new_grid = result.ω_grid[keep[1]:keep[end]]
    end

    tensor = result.tensor
    last_axis = ndims(tensor)
    sl_pre = ntuple(d -> :, last_axis - 1)
    new_tensor = tensor[sl_pre..., keep[1]:keep[end]]

    new_meta = copy(result.metadata)
    new_meta[:restrict_window] = ω_window

    F = typeof(new_grid)
    N = ndims(new_tensor)
    return SpectraTensor{eltype(new_tensor), Float64, N, F}(
        new_tensor, nothing, new_grid, result.Eg,
        result.block_size, new_meta)
end

# ---------------------------------------------------------------------
# plot_range — auto-window for plotting
# ---------------------------------------------------------------------

"""
    plot_range(result; threshold = 0.01) -> Tuple{Float64, Float64}
    plot_range(result; rule = :integrated, q = 0.05) -> Tuple{Float64, Float64}

Suggest an `(ω_min, ω_max)` plot window covering the spectrum's
significant intensity. Two rules:

- `:peak` (default, via `threshold`): the smallest contiguous window
  containing all ω where the (real-valued) spectrum exceeds
  `threshold * max`. Default `threshold = 0.01`.
- `:integrated` (via `q`): the `[q, 1-q]` quantile window of the
  cumulative intensity.
"""
function plot_range(result::SpectraTensor;
                    threshold::Real = 0.01,
                    rule::Symbol    = :peak,
                    q::Real         = 0.05)
    s = polarise(result)
    spec = s isa AbstractMatrix ? vec(sum(s; dims = 1)) :  # collapse ψ axis if any
           s isa AbstractArray   ? vec(s) :
           s
    ω_grid = result.ω_grid isa Tuple ? last(result.ω_grid) : result.ω_grid

    if rule === :peak
        peak = maximum(spec)
        cutoff = threshold * peak
        idx = findall(>(cutoff), spec)
        isempty(idx) && return (Float64(first(ω_grid)), Float64(last(ω_grid)))
        return (Float64(ω_grid[idx[1]]), Float64(ω_grid[idx[end]]))
    elseif rule === :integrated
        cum = cumsum(spec)
        total = cum[end]
        total > 0 || return (Float64(first(ω_grid)), Float64(last(ω_grid)))
        lo_idx = findfirst(c -> c ≥ q * total, cum)
        hi_idx = findlast(c -> c ≤ (1 - q) * total, cum)
        return (Float64(ω_grid[lo_idx]), Float64(ω_grid[hi_idx]))
    else
        throw(ArgumentError("plot_range: rule must be :peak or :integrated; got :$rule"))
    end
end
