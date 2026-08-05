# =====================================================================
# Fluorescence yield — direct algorithm via analytic ω_out integral
# =====================================================================
#
# The integral over ω_out of the RIXS map collapses analytically:
#
#   FY(ω_in) = ∫dω_out (-Im) Σ_kl χ_{kkll}(ω_in, ω_out)
#            = π · Σ_{k,l} ‖T_decay_k† · G_int(ω_in) · T_excite_l |ψ⟩‖².
#
# So FY needs only the inner Lanczos (and decay-channel matvecs); no
# outer Lanczos, no H_final, no Γ_final, no ω_out grid.

"""
    fluorescence_yield(H_intermediate, basis, T_excite, T_decay, ψ; kwargs...)
        -> SpectraTensor

Compute the total fluorescence yield

```
FY(ω_in) = ∫dω_out (-Im) Σ_{k,l} χ_{kkll}(ω_in, ω_out)
         = π · Σ_{k,l} ‖T_decay_k† · G_int(ω_in) · T_excite_l · ψ‖²
```

via the analytic ω_out integral of the RIXS map (the second equality
follows from the Lehmann form of `G(ω_out)`). This is *not* a post-hoc
numerical integration of `rixs(...)` — no outer Lanczos runs, and the
result is real-valued from the start.

Output shape:
- single ψ → `(n_ω_in,)` of `Float64`
- list ψ   → `(N_ψ, n_ω_in)` of `Float64`

Vector inputs for `T_excite` / `T_decay` are interpreted as **channels
to sum** in the FY formula above — `(k, l)` runs over the cross product
of the two lists. This differs from `xas` and `rixs`, where multi-
operator inputs expand into tensor axes; for partial yields restricted
to specific (k, l) channels, call `fluorescence_yield` separately on
each pair.

**Finite temperature.** Pass `ψ::Eigen` with `temperature = …` (`k_B = 1`,
optional `N_states`, `degen_tol`) for the Boltzmann ensemble average over initial
states, `FY_T = Σ_m ρ_m FY_m`, each referenced to its own `E_m`. An explicit
`ω_in_grid` is required (per-state `:auto` grids differ); the result is grid-only
and carries `Eg = E₀` plus `:temperature`, `:ensemble_energies`,
`:ensemble_weights`, `:N_kept` metadata.
"""
function fluorescence_yield end

# Eigen overload.
function fluorescence_yield(H_i, basis::AbstractBasis, T_excite, T_decay,
                            E::LinearAlgebra.Eigen; Eg = nothing, temperature = nothing,
                            N_states = nothing, degen_tol = 1e-8, kwargs...)
    Eg === nothing ||
        throw(ArgumentError("fluorescence_yield: Eg cannot be specified when ψ is an Eigen"))
    if temperature === nothing
        (N_states === nothing && degen_tol == 1e-8) || throw(ArgumentError(
            "fluorescence_yield: N_states / degen_tol are finite-T-only kwargs and " *
            "require temperature = ...; got temperature = nothing"))
        k = argmin(real.(E.values))
        return fluorescence_yield(H_i, basis, T_excite, T_decay, E.vectors[:, k];
                                  Eg = real(E.values[k]), kwargs...)
    end
    return _fy_thermal(H_i, basis, T_excite, T_decay, E; temperature = temperature,
                       N_states = N_states, degen_tol = degen_tol, kwargs...)
end

# Single ψ.
function fluorescence_yield(H_i, basis::AbstractBasis, T_excite, T_decay,
                            ψ::AbstractVector{<:Number};
                            ω_in_grid           = nothing,
                            omega_in_grid       = nothing,
                            Γ_intermediate      = nothing,
                            Gamma_intermediate  = nothing,
                            Eg::Union{Real,Nothing} = nothing,
                            krylovdim::Union{Int,Nothing}    = nothing,
                            reorth::Union{Symbol,Nothing}    = nothing,
                            tol::Union{Real,Nothing}         = nothing,
                            restrictions_intermediate        = nothing)
    Eg === nothing &&
        throw(ArgumentError("fluorescence_yield: Eg required for bare vector ψ"))
    return _fy_run(H_i, basis, _to_T_list(T_excite), _to_T_list(T_decay),
                   [ψ], false;
                   Γ_int_use = _resolve_pair(Γ_intermediate, Gamma_intermediate;
                                              default = DEFAULTS.Γ_intermediate,
                                              name = "Γ_intermediate"),
                   ω_in_use  = _resolve_pair(ω_in_grid, omega_in_grid;
                                              default = :auto, name = "ω_in_grid"),
                   krylovdim_use = something(krylovdim, DEFAULTS.krylovdim),
                   reorth_use    = something(reorth,    DEFAULTS.reorth),
                   tol_use       = something(tol,       DEFAULTS.tol),
                   Eg            = Float64(Eg),
                   restrictions_intermediate = restrictions_intermediate)
end

# Multi-ψ.
function fluorescence_yield(H_i, basis::AbstractBasis, T_excite, T_decay,
                            ψs::AbstractVector{<:AbstractVector{<:Number}};
                            ω_in_grid = nothing, omega_in_grid = nothing,
                            Γ_intermediate = nothing, Gamma_intermediate = nothing,
                            Eg::Union{Real,Nothing} = nothing,
                            krylovdim::Union{Int,Nothing} = nothing,
                            reorth::Union{Symbol,Nothing} = nothing,
                            tol::Union{Real,Nothing} = nothing,
                            restrictions_intermediate = nothing)
    Eg === nothing &&
        throw(ArgumentError("fluorescence_yield: Eg required for bare vector list ψ"))
    Eg isa Real ||
        throw(ArgumentError("fluorescence_yield: per-ψ Eg vectors not supported; pass a single Real Eg"))
    return _fy_run(H_i, basis, _to_T_list(T_excite), _to_T_list(T_decay),
                   collect(ψs), true;
                   Γ_int_use = _resolve_pair(Γ_intermediate, Gamma_intermediate;
                                              default = DEFAULTS.Γ_intermediate,
                                              name = "Γ_intermediate"),
                   ω_in_use  = _resolve_pair(ω_in_grid, omega_in_grid;
                                              default = :auto, name = "ω_in_grid"),
                   krylovdim_use = something(krylovdim, DEFAULTS.krylovdim),
                   reorth_use    = something(reorth,    DEFAULTS.reorth),
                   tol_use       = something(tol,       DEFAULTS.tol),
                   Eg            = Float64(Eg),
                   restrictions_intermediate = restrictions_intermediate)
end

# --- Driver ---

function _fy_run(H_i, basis::AbstractBasis,
                 T_excite_list::Vector{<:OperatorSum},
                 T_decay_list::Vector{<:OperatorSum},
                 ψ_list::Vector{<:AbstractVector},
                 ψ_is_list::Bool;
                 Γ_int_use, ω_in_use, krylovdim_use, reorth_use, tol_use,
                 Eg, restrictions_intermediate)
    isempty(T_excite_list) && throw(ArgumentError("fluorescence_yield: T_excite must be non-empty"))
    isempty(T_decay_list)  && throw(ArgumentError("fluorescence_yield: T_decay must be non-empty"))
    isempty(ψ_list)        && throw(ArgumentError("fluorescence_yield: ψ must be non-empty"))

    N_states = length(basis)
    N_ex     = length(T_excite_list)
    N_dec    = length(T_decay_list)
    N_ψ      = length(ψ_list)
    for ψ in ψ_list
        length(ψ) == N_states ||
            throw(DimensionMismatch("fluorescence_yield: ψ length ≠ basis size"))
    end

    T_ex_assembled  = SparseMatrixCSC[assemble(compile(t, basis), basis) for t in T_excite_list]
    T_dec_assembled = SparseMatrixCSC[assemble(compile(t, basis), basis) for t in T_decay_list]

    # Per-ψ inner Lanczos pipeline (same machinery as RIXS step 1-2).
    chunks = LanczosChunk{ComplexF64}[]
    fy_per_ψ = Vector{Float64}[]    # one Vector{Float64} of length n_ω_in per ψ
    converged_list = Bool[]

    # ---- ω_in :auto needs ALL ψ's inner T_K eigenvalues unioned -------
    # Same 2-pass scheme as rixs.jl: pass 1 runs inner Lanczos per ψ
    # with retain_basis=false to harvest T_int eigenvalues without
    # holding V_int simultaneously across all ψ; pass 2 (the main loop
    # below) re-runs inner with retain_basis=true and continues.
    ω_in_resolved::AbstractRange = if ω_in_use === :auto
        e_min, e_max = Inf, -Inf
        for ψ in ψ_list
            X_int_p1 = Matrix{ComplexF64}(undef, N_states, N_ex)
            for k in 1:N_ex
                X_int_p1[:, k] = T_ex_assembled[k] * ψ
            end
            inner_p1 = Responses.block_lanczos(H_i, X_int_p1;
                                     krylovdim    = krylovdim_use,
                                     reorth       = reorth_use,
                                     tol          = tol_use,
                                     restrictions = restrictions_intermediate,
                                     basis        = restrictions_intermediate === nothing ? nothing : basis,
                                     retain_basis = false)
            T_int_p1 = _build_T_K(inner_p1.α, inner_p1.β)
            es = real.(eigvals(Hermitian(T_int_p1)))
            e_min = min(e_min, minimum(es) - Eg)
            e_max = max(e_max, maximum(es) - Eg)
        end
        auto_grid(e_min, e_max, Γ_int_use)
    else
        ω_in_use
    end

    for (ψ_idx, ψ) in enumerate(ψ_list)
        # Inner starting block: N_ex columns.
        X_int = Matrix{ComplexF64}(undef, N_states, N_ex)
        for k in 1:N_ex
            X_int[:, k] = T_ex_assembled[k] * ψ
        end
        inner = Responses.block_lanczos(H_i, X_int;
                              krylovdim    = krylovdim_use,
                              reorth       = reorth_use,
                              tol          = tol_use,
                              restrictions = restrictions_intermediate,
                              basis        = restrictions_intermediate === nothing ? nothing : basis,
                              retain_basis = true)
        # Drop the trailing V_{K+1} block: block_lanczos retains K+1
        # blocks but cf's T_K only uses the first K. See rixs.jl for the
        # detailed comment.
        V_int = hcat(inner.V_basis[1:length(inner.α)]...)::Matrix{ComplexF64}
        T_int = _build_T_K(inner.α, inner.β)
        M = size(T_int, 1)

        # Save inner chunk (diagnostic — FY has no outer Lanczos chunks
        # per ω_in, so these inner chunks are what `poles(fy_result)`
        # introspects).
        push!(chunks, LanczosChunk{ComplexF64}(
            inner.α, inner.β, inner.R, N_ex, ψ_idx, 0,
            inner.n_iter, inner.converged))
        push!(converged_list, inner.converged)

        # Pad ζ.
        B_active_inner = size(inner.R, 1)
        ζ = zeros(ComplexF64, M, N_ex)
        ζ[1:B_active_inner, :] .= inner.R

        n_ω_in = length(ω_in_resolved)
        fy = zeros(Float64, n_ω_in)
        for (i, ω_in) in enumerate(ω_in_resolved)
            z_in = ComplexF64(ω_in + Eg + im * Γ_int_use / 2)
            W = (z_in * I - T_int) \ ζ        # M × N_ex
            Y = V_int * W                      # N × N_ex
            # FY = π · Σ_{k, l} ‖T_decay_k† · Y[:, l]‖²
            acc = 0.0
            for l in 1:N_ex
                yl = @view Y[:, l]
                for k in 1:N_dec
                    z_kl = adjoint(T_dec_assembled[k]) * yl
                    acc += sum(abs2, z_kl)
                end
            end
            fy[i] = π * acc
        end
        push!(fy_per_ψ, fy)
    end

    # Pack output tensor.
    n_ω_in = length(ω_in_resolved)
    if ψ_is_list
        tensor = Matrix{Float64}(undef, N_ψ, n_ω_in)
        for ψ_idx in 1:N_ψ
            tensor[ψ_idx, :] .= fy_per_ψ[ψ_idx]
        end
    else
        tensor = fy_per_ψ[1]
    end

    metadata = Dict{Symbol,Any}(
        :function       => :fluorescence_yield,
        :Γ_intermediate => Γ_int_use,
        :Eg             => Eg,
        :krylovdim      => krylovdim_use,
        :reorth         => reorth_use,
        :tol            => tol_use,
        :restrictions_intermediate => restrictions_intermediate,
        :converged      => all(converged_list),
        :n_T_excite     => N_ex,
        :n_T_decay      => N_dec,
        :n_ψ            => N_ψ,
        :ψ_is_list_input => ψ_is_list,
        :auto_range      => (ω_in_use === :auto),
        :basis_id        => basis_id(basis),
    )

    F = typeof(ω_in_resolved)
    N = ndims(tensor)
    return SpectraTensor{Float64, Float64, N, F}(
        tensor, chunks, ω_in_resolved, Float64(Eg), N_ex, metadata)
end
