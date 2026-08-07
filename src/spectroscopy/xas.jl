# =====================================================================
# XAS — rank-2 absorption susceptibility
# =====================================================================
#
#   χ_{ab}(ω) = ⟨ψ_g| T_a† G(ω) T_b |ψ_g⟩,
#   G(ω)      = ((ω + Eg + iΓ/2)·I − H)⁻¹.
#
# Scalar T + scalar ψ → 1-D spectrum (n_ω). Vector inputs spread the
# operator-pair axes (a, b) and the ψ axis as documented in `xas`'s
# docstring. The polarisation tensor is the natural output; the user
# contracts to a real intensity via `polarise`.

"""
    xas(H, basis, T, ψ; kwargs...) -> SpectraTensor
    xas(H, basis, source::AbstractVector; kwargs...) -> SpectraTensor

Compute an X-ray absorption spectrum for the Hamiltonian `H` with
respect to dipole-like transition operator(s) `T` and ground state(s)
`ψ`.

The 3-argument form skips the internal `T·ψ` step and treats `source`
as the pre-applied Lanczos starting vector (the user has already
applied the dipole-like operator, possibly across bases). Mirrors
PyQuanty's `CreateSpectra(psi=..., H=...)` pattern. `Eg` is required
in this form; `source` must have `length == length(basis)`.

# Positional arguments

- `H::AbstractMatrix`        — Hamiltonian on the Hilbert space spanned by `basis`.
                               `SparseMatrixCSC` typical.
- `basis::AbstractBasis`     — the basis `H`, `T`, `ψ` all live on.
- `T`                        — `OperatorSum` (single transition operator) or
                               `AbstractVector{<:OperatorSum}` (cartesian / multi-channel).
                               Each operator is internally compiled and assembled
                               on `basis`; cache and reuse if you call `xas` many
                               times with the same operator set.
- `ψ`                        — `AbstractVector` (single ground state),
                               `AbstractVector{<:AbstractVector}` (list of independent
                               ground states; output gets a leading `N_ψ` axis), or
                               `LinearAlgebra.Eigen` (extracts the lowest-eigenvalue
                               eigenvector and uses `Eg = minimum(eigenvalues)`).

# Keyword arguments

ASCII Latin aliases are accepted for every Greek-named kwarg
(`Γ ↔ Gamma`, `ω_grid ↔ omega_grid`); specifying both forms raises
`ArgumentError`. Defaults route through `MOADyna.Spectroscopy.DEFAULTS`.

| kwarg | default | role |
|---|---|---|
| `ω_grid` | `:auto` | `AbstractRange` or `:auto`. `:auto` builds a pretty-step-snapped grid covering the H_eigenvalue range with `padding · Γ` margins (`auto_grid`). |
| `Γ` | `DEFAULTS.Γ_xas` | FWHM Lorentzian broadening (eV) |
| `Eg` | `nothing` | ground-state energy. Required for bare-vector ψ; forbidden when ψ is `Eigen`. |
| `krylovdim` | `DEFAULTS.krylovdim` | block-Lanczos iterations |
| `reorth` | `DEFAULTS.reorth` | `:none` or `:full` |
| `restrictions` | `nothing` | layer-2 `Restriction` for projected dynamics |
| `tol` | `DEFAULTS.tol` | block-Lanczos soft-stop on ‖β_k‖ |
| `temperature` | `nothing` | finite-T (`k_B = 1`). Only with an `Eigen` ψ: Boltzmann-averages the spectrum over the thermally populated initial states, `χ_T(ω) = Σ_m ρ_m χ_m(ω)`. `nothing` ⇒ T=0 (pick the ground state). |
| `N_states` | `nothing` | finite-T only: cap on the number of lowest `Eigen` states summed (degeneracy-complete; manifold-splitting raises). |
| `degen_tol` | `1e-8` | finite-T only: degeneracy tolerance for the ground-manifold / truncation logic. |

**Finite temperature.** Pass `ψ::Eigen` together with `temperature = …` to get
the Boltzmann ensemble average over initial states (each referenced to its own
`E_m`). The ensemble is the lowest `N_states` eigenstates of the supplied `Eigen`
(in MOADyna's single-`H` formulation these are the initial multiplet); the user is
responsible that they are. An explicit `ω_grid` is **required** (per-state `:auto`
windows differ), and the result is grid-only (`chunks = nothing` ⇒ no
`re_broaden`). The returned `SpectraTensor` carries `Eg = E₀` and metadata
`:temperature, :ensemble_energies, :ensemble_weights, :N_kept`.

# Return

A `SpectraTensor{Complex{Float64}, Float64, N, F}` whose `tensor`
shape follows the dispatch table:

| Input shape | `tensor` shape (`N`) |
|---|---|
| single T, single ψ | `(n_ω,)` (1) |
| vector T, single ψ | `(N_T, N_T, n_ω)` (3) |
| single T, list ψ | `(N_ψ, n_ω)` (2) |
| vector T, list ψ | `(N_T, N_T, N_ψ, n_ω)` (4) |

Real intensity `-Im` is taken at contraction time by `polarise`. The
complex tensor preserves phase for arbitrary polarisations (notably
circular).
"""
function xas end

# --- Eigen overload: extract GS, route to bare-vector path ----------------

function xas(H, basis::AbstractBasis, T, E::LinearAlgebra.Eigen;
             Eg = nothing, temperature = nothing,
             N_states = nothing, degen_tol = 1e-8, kwargs...)
    Eg === nothing ||
        throw(ArgumentError("xas: Eg cannot be specified when ψ is an Eigen"))
    if temperature === nothing
        # T = 0 ground-state path (unchanged). Thermal-only kwargs must not leak.
        (N_states === nothing && degen_tol == 1e-8) || throw(ArgumentError(
            "xas: N_states / degen_tol are finite-T-only kwargs and require " *
            "temperature = ...; got temperature = nothing"))
        k = argmin(real.(E.values))
        return xas(H, basis, T, E.vectors[:, k];
                   Eg = real(E.values[k]), kwargs...)
    end
    return _xas_thermal(H, basis, T, E; temperature = temperature,
                        N_states = N_states, degen_tol = degen_tol, kwargs...)
end

# --- Single ψ::AbstractVector{<:Number} -----------------------------------

function xas(H, basis::AbstractBasis, T, ψ::AbstractVector{<:Number};
             ω_grid           = nothing,
             omega_grid       = nothing,
             Γ                = nothing,
             Gamma            = nothing,
             Eg::Union{Real,Nothing} = nothing,
             krylovdim::Union{Int,Nothing}    = nothing,
             reorth::Union{Symbol,Nothing}    = nothing,
             tol::Union{Real,Nothing}         = nothing,
             restrictions     = nothing)
    Eg === nothing &&
        throw(ArgumentError("xas: Eg is required when ψ is a bare vector " *
                            "(pass `Eg = ...` or call with an Eigen)"))
    return _xas_run(H, basis, _to_T_list(T), [ψ], _is_T_vector(T), false;
                    Γ_use         = _resolve_pair(Γ, Gamma; default = DEFAULTS.Γ_xas, name = "Γ"),
                    ω_grid_use    = _resolve_pair(ω_grid, omega_grid; default = :auto, name = "ω_grid"),
                    krylovdim_use = something(krylovdim, DEFAULTS.krylovdim),
                    reorth_use    = something(reorth,    DEFAULTS.reorth),
                    tol_use       = something(tol,       DEFAULTS.tol),
                    Eg            = Float64(Eg),
                    restrictions  = restrictions)
end

# --- Pre-applied source vector (skip internal T·ψ) ------------------------
#
# Mirrors PyQuanty's `CreateSpectra(psi=..., H=...)` pattern: the user has
# already applied the dipole-like operator T to ψ (often across bases —
# embed → matvec → down-project) and just wants the Lanczos + continued-
# fraction pipeline to run on the supplied source. No T or ψ is touched
# inside this method, so the caller carries full responsibility for sign,
# normalisation and basis choice of `source`.

function xas(H, basis::AbstractBasis, source::AbstractVector{<:Number};
             ω_grid           = nothing,
             omega_grid       = nothing,
             Γ                = nothing,
             Gamma            = nothing,
             Eg::Union{Real,Nothing} = nothing,
             krylovdim::Union{Int,Nothing}    = nothing,
             reorth::Union{Symbol,Nothing}    = nothing,
             tol::Union{Real,Nothing}         = nothing,
             restrictions     = nothing)
    Eg === nothing &&
        throw(ArgumentError("xas: Eg is required when calling with a pre-applied " *
                            "source vector (no ψ from which to extract it)"))
    N_states = length(basis)
    length(source) == N_states ||
        throw(DimensionMismatch("xas: source has length $(length(source)); " *
                                "basis has $N_states"))
    return _xas_run_from_sources(H, basis,
                    [reshape(ComplexF64.(source), :, 1)], false, false;
                    Γ_use         = _resolve_pair(Γ, Gamma; default = DEFAULTS.Γ_xas, name = "Γ"),
                    ω_grid_use    = _resolve_pair(ω_grid, omega_grid; default = :auto, name = "ω_grid"),
                    krylovdim_use = something(krylovdim, DEFAULTS.krylovdim),
                    reorth_use    = something(reorth,    DEFAULTS.reorth),
                    tol_use       = something(tol,       DEFAULTS.tol),
                    Eg            = Float64(Eg),
                    restrictions  = restrictions)
end

# --- List of ψ's -----------------------------------------------------------

function xas(H, basis::AbstractBasis, T,
             ψs::AbstractVector{<:AbstractVector{<:Number}};
             ω_grid           = nothing,
             omega_grid       = nothing,
             Γ                = nothing,
             Gamma            = nothing,
             Eg::Union{Real,AbstractVector{<:Real},Nothing} = nothing,
             krylovdim::Union{Int,Nothing}    = nothing,
             reorth::Union{Symbol,Nothing}    = nothing,
             tol::Union{Real,Nothing}         = nothing,
             restrictions     = nothing)
    # Multi-ψ contract: a single shared `Eg::Real`. The list of ψ's is
    # interpreted as several ground-state vectors that share a common
    # zero-frequency reference (typical use case: a list of degenerate
    # GS multiplet partners). Per-ψ vectors of distinct Eg's are
    # rejected to avoid silently mixing different reference energies.
    Eg === nothing &&
        throw(ArgumentError("xas: Eg is required when ψ is a bare vector list"))
    Eg isa Real ||
        throw(ArgumentError("xas: Eg must be a single Real shared across all ψ; " *
                            "per-ψ Eg lists are not supported"))
    return _xas_run(H, basis, _to_T_list(T), collect(ψs), _is_T_vector(T), true;
                    Γ_use         = _resolve_pair(Γ, Gamma; default = DEFAULTS.Γ_xas, name = "Γ"),
                    ω_grid_use    = _resolve_pair(ω_grid, omega_grid; default = :auto, name = "ω_grid"),
                    krylovdim_use = something(krylovdim, DEFAULTS.krylovdim),
                    reorth_use    = something(reorth,    DEFAULTS.reorth),
                    tol_use       = something(tol,       DEFAULTS.tol),
                    Eg            = Float64(Eg),
                    restrictions  = restrictions)
end

# --- Internal helpers ------------------------------------------------------

_to_T_list(T::OperatorSum) = OperatorSum[T]
_to_T_list(Ts::AbstractVector{<:OperatorSum}) = collect(Ts)
_is_T_vector(::OperatorSum) = false
_is_T_vector(::AbstractVector{<:OperatorSum}) = true

# The actual driver. All public methods normalise their inputs to a
# `T_list::Vector{<:OperatorSum}` and a `ψ_list::Vector{<:AbstractVector}`
# and route here.
function _xas_run(H, basis::AbstractBasis,
                  T_list::Vector{<:OperatorSum},
                  ψ_list::Vector{<:AbstractVector},
                  T_is_vector::Bool, ψ_is_list::Bool;
                  Γ_use, ω_grid_use, krylovdim_use, reorth_use,
                  tol_use, Eg, restrictions)
    isempty(T_list) && throw(ArgumentError("xas: T must be non-empty"))
    isempty(ψ_list) && throw(ArgumentError("xas: ψ must be non-empty"))
    N_T = length(T_list)
    N_ψ = length(ψ_list)
    N_states = length(basis)
    for ψ in ψ_list
        length(ψ) == N_states ||
            throw(DimensionMismatch("xas: ψ has length $(length(ψ)); basis has $N_states"))
    end

    # Compile + assemble each transition operator once on the shared basis,
    # then build the source blocks T·ψ that drive Lanczos.
    T_assembled = SparseMatrixCSC[]
    for t in T_list
        try
            push!(T_assembled, assemble(compile(t, basis), basis))
        catch err
            throw(ArgumentError("xas: failed to compile/assemble transition operator: $err"))
        end
    end

    source_blocks = Matrix{ComplexF64}[]
    for ψ in ψ_list
        X = Matrix{ComplexF64}(undef, N_states, N_T)
        for k in 1:N_T
            X[:, k] = T_assembled[k] * ψ
        end
        push!(source_blocks, X)
    end

    return _xas_run_from_sources(H, basis, source_blocks,
                                 T_is_vector, ψ_is_list;
                                 Γ_use         = Γ_use,
                                 ω_grid_use    = ω_grid_use,
                                 krylovdim_use = krylovdim_use,
                                 reorth_use    = reorth_use,
                                 tol_use       = tol_use,
                                 Eg            = Eg,
                                 restrictions  = restrictions)
end

# Pipeline starting from already-built source blocks. `source_blocks` is a
# vector (one entry per ψ-equivalent) of N_states × N_T ComplexF64 matrices —
# each column is a pre-applied T_a · ψ. The pre-applied-source public method
# wraps a single column into a 1-element list. The standard 4-arg form
# normalises T_list × ψ_list into the same shape and routes here.
#
# The compute path delegates to `MOADyna.Responses.correlator(...)` in
# source_block mode (we have already done the T·ψ application up front,
# so we feed the pre-built block straight to correlator). For each ψ we
# obtain a `LanczosResponse`, harvest its α/β/R into a `LanczosChunk`
# (preserving the existing SpectraTensor.chunks layout for HDF5 round-
# trip and re_broaden), and call `to_grid(L, ω; Γ)` to materialise the
# spectral data — `.data` slots into the existing `_xas_pack` step
# unchanged.
#
# In source_block mode, correlator does not consult As / Bs beyond the
# `As === Bs` identity check. We pass an empty `OperatorSum[]` reused
# across calls so identity is satisfied; the empty vectors are never
# assembled or applied.
function _xas_run_from_sources(H, basis::AbstractBasis,
                               source_blocks::Vector{<:AbstractMatrix},
                               T_is_vector::Bool, ψ_is_list::Bool;
                               Γ_use, ω_grid_use, krylovdim_use, reorth_use,
                               tol_use, Eg, restrictions)
    isempty(source_blocks) && throw(ArgumentError("xas: source must be non-empty"))
    N_ψ = length(source_blocks)
    N_T = size(source_blocks[1], 2)
    N_states = length(basis)
    for X in source_blocks
        size(X, 1) == N_states ||
            throw(DimensionMismatch("xas: source row count $(size(X,1)); basis has $N_states"))
        size(X, 2) == N_T ||
            throw(DimensionMismatch("xas: ragged source blocks (got $(size(X,2)) cols, " *
                                    "expected $N_T)"))
    end

    # Empty As === Bs (reused across all correlator calls). Never
    # actually assembled or applied in source_block mode; only used to
    # satisfy correlator's `As === Bs` identity check (Bs may be empty
    # in source_block mode).
    empty_ops = OperatorSum[]

    # Forward block-Lanczos tuning kwargs. `restrictions` requires `basis`
    # alongside it — correlator auto-fills basis when restrictions is set.
    common_kwargs = (krylovdim = krylovdim_use,
                     reorth    = reorth_use,
                     tol       = tol_use,
                     retain_basis = false)
    restr_kwargs = restrictions === nothing ?
        common_kwargs :
        (; common_kwargs..., restrictions = restrictions, basis = basis)

    chunks = LanczosChunk{ComplexF64}[]
    responses = Responses.LanczosResponse{ComplexF64}[]
    n_iter_results = Int[]
    converged_results = Bool[]
    for (ψ_idx, X) in enumerate(source_blocks)
        L = Responses.correlator(H, basis, empty_ops, empty_ops;
                                 source_block = X,
                                 state        = nothing,
                                 Eg           = Eg,
                                 Γ            = Γ_use,
                                 channel      = :neutral,
                                 form         = :lanczos,
                                 restr_kwargs...)::Responses.LanczosResponse{ComplexF64}

        # Harvest α/β/R from the LanczosResponse into a LanczosChunk
        # (Spectroscopy-domain type, kept for HDF5 round-trip and
        # re_broaden). The α/β/R themselves are already at correlator's
        # promoted Complex type; α is the iteration count.
        α_c = Vector{Matrix{ComplexF64}}(L.α)
        β_c = Vector{Matrix{ComplexF64}}(L.β)
        R_c = Matrix{ComplexF64}(L.R)
        # n_iter = K; converged passes through faithfully from the
        # underlying block_lanczos via correlator → LanczosResponse.
        n_iter_val    = length(L.α)
        converged_val = L.converged
        chunk = LanczosChunk{ComplexF64}(α_c, β_c, R_c, N_T,
                                          ψ_idx, 0, n_iter_val, converged_val)
        push!(chunks, chunk)
        push!(responses, L)
        push!(n_iter_results, n_iter_val)
        push!(converged_results, converged_val)
    end

    ω_grid_resolved = ω_grid_use === :auto ?
        _xas_auto_grid(chunks, Eg, Γ_use) : ω_grid_use
    n_ω = length(ω_grid_resolved)

    # Materialise on the resolved grid via Responses.to_grid; .data is
    # shape (B_raw, B_raw, n_ω), exactly what the v0.1 evaluate_on_grid
    # returned, so `_xas_pack` consumes it unchanged.
    χs = Array{ComplexF64,3}[]
    for L in responses
        G = Responses.to_grid(L, ω_grid_resolved; Γ = Γ_use)
        push!(χs, Array{ComplexF64,3}(G.data))
    end

    tensor = _xas_pack(χs, T_is_vector, ψ_is_list, N_T, N_ψ, n_ω)

    metadata = Dict{Symbol,Any}(
        :function           => :xas,
        :Γ                  => Γ_use,
        :Eg                 => Eg,
        :krylovdim          => krylovdim_use,
        :reorth             => reorth_use,
        :tol                => tol_use,
        :restrictions       => restrictions,
        :converged          => all(converged_results),
        :n_iter_per_ψ       => n_iter_results,
        :n_T                => N_T,
        :n_ψ                => N_ψ,
        :T_is_vector_input  => T_is_vector,
        :ψ_is_list_input    => ψ_is_list,
        :auto_range         => (ω_grid_use === :auto),
        :basis_id           => basis_id(basis),
    )
    if ω_grid_use === :auto
        metadata[:auto_range_bandwidth] = (first(ω_grid_resolved), last(ω_grid_resolved))
    end

    block_size = N_T

    F = typeof(ω_grid_resolved)
    N = ndims(tensor)
    return SpectraTensor{ComplexF64, Float64, N, F}(
        tensor, chunks, ω_grid_resolved, Float64(Eg), block_size, metadata)
end

function _xas_pack(χs::Vector{Array{ComplexF64,3}},
                   T_is_vector::Bool, ψ_is_list::Bool,
                   N_T::Int, N_ψ::Int, n_ω::Int)
    if !T_is_vector && !ψ_is_list
        # Single ψ, single T → (n_ω,). χs[1] is (1, 1, n_ω); squeeze.
        out = Vector{ComplexF64}(undef, n_ω)
        @inbounds for i in 1:n_ω
            out[i] = χs[1][1, 1, i]
        end
        return out
    elseif T_is_vector && !ψ_is_list
        # (N_T, N_T, n_ω) — exactly what evaluate_on_grid returns for ψ=1.
        return χs[1]                          # already (N_T, N_T, n_ω)
    elseif !T_is_vector && ψ_is_list
        # Single T, multi ψ → (N_ψ, n_ω).
        out = Matrix{ComplexF64}(undef, N_ψ, n_ω)
        @inbounds for ψ_idx in 1:N_ψ, i in 1:n_ω
            out[ψ_idx, i] = χs[ψ_idx][1, 1, i]
        end
        return out
    else
        # Multi T, multi ψ → (N_T, N_T, N_ψ, n_ω).
        out = Array{ComplexF64,4}(undef, N_T, N_T, N_ψ, n_ω)
        @inbounds for ψ_idx in 1:N_ψ, i in 1:n_ω, b in 1:N_T, a in 1:N_T
            out[a, b, ψ_idx, i] = χs[ψ_idx][a, b, i]
        end
        return out
    end
end

# Build dense block tridiagonal T_K from α/β stacks (ragged-block aware).
function _build_T_K(α::Vector{<:AbstractMatrix}, β::Vector{<:AbstractMatrix})
    K = length(α)
    K ≥ 1 || throw(ArgumentError("_build_T_K: empty α stack"))
    Tc = eltype(α[1])
    sizes = Int[size(α[k], 1) for k in 1:K]
    offs  = cumsum(vcat(0, sizes))
    M = sum(sizes)
    T_K = zeros(Tc, M, M)
    @inbounds for k in 1:K
        rng = (offs[k]+1):offs[k+1]
        T_K[rng, rng] .= α[k]
    end
    @inbounds for k in 1:min(K-1, length(β))
        rng_k    = (offs[k]+1):offs[k+1]
        rng_kp1  = (offs[k+1]+1):offs[k+2]
        # β[k] has shape (sizes[k+1] × sizes[k]); sits below diagonal.
        T_K[rng_kp1, rng_k] .= β[k]
        T_K[rng_k, rng_kp1] .= adjoint(β[k])
    end
    return T_K
end

# Pole structure (eigenvalues of T_K). Internal — used by the auto-grid
# helper and by `poles`.
function _chunk_poles(chunk::LanczosChunk)
    T_K = _build_T_K(chunk.α, chunk.β)
    # T_K is Hermitian by construction. Hermitian wrapper enables LAPACK's
    # symmetric eigensolver and guarantees real eigenvalues.
    return real.(eigvals(Hermitian(T_K)))
end

function _xas_auto_grid(chunks::Vector{<:LanczosChunk}, Eg::Real, Γ::Real)
    e_min, e_max = Inf, -Inf
    for c in chunks
        ps = _chunk_poles(c)
        e_min = min(e_min, minimum(ps) - Eg)
        e_max = max(e_max, maximum(ps) - Eg)
    end
    return auto_grid(e_min, e_max, Γ)
end
