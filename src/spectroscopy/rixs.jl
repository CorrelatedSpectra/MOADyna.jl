# =====================================================================
# RIXS — Kramers-Heisenberg cross-correlator on two Hamiltonians
# =====================================================================
#
# Two-level Lanczos pipeline:
#   1. ONE inner block-Lanczos on `H_intermediate` (block size `N_in`),
#      retaining the Krylov basis so `G_int(ω_in) · T_in_l |ψ⟩` can be
#      recovered cheaply at any incident energy.
#   2. PER `ω_in` outer block-Lanczos on `H_final` with starting block
#      `T_out_k† · G_int(ω_in) · T_in_l |ψ⟩` (size `N_in · N_out`).
#   3. Continued-fraction evaluation, deferred until the `ω_out` grid
#      is resolved so that `ω_out_grid = :auto` can union the
#      final-state poles across all per-ω_in chunks.
#
# Lanczos delegation:
#   - Outer/final-state Lanczos: `correlator(...; source_block, form=:lanczos)`.
#     `source_block` mode (not `state`) because rixs has already built
#     X_after_inner = T_out† · G_int · T_in · ψ; passing through `state`
#     would re-apply T_out. `form=:lanczos` because the auto-`ω_out`
#     grid unions final-state poles across ALL chunks, requiring the
#     Lanczos form up front; each chunk is later materialised via
#     `to_grid(L, ω_out_resolved; Γ)`.
#   - Inner/intermediate-state Lanczos: `block_lanczos(...; retain_basis=true)`
#     directly. Apply-to-vector path (forms `Y = V_int · W`); not a
#     matrix-of-ω response, so `correlator` is the wrong abstraction.

"""
    rixs(H_final, H_intermediate, basis, T_in, T_out, ψ; kwargs...)
        -> SpectraTensor

Compute the RIXS Kramers-Heisenberg susceptibility tensor

```
χ(ω_in, ω_out) = ⟨ψ| R†(ω_in) G_final(ω_out) R(ω_in) |ψ⟩,
R(ω_in) = T_out† · G_int(ω_in) · T_in.
```

`G_int = ((ω_in + Eg + iΓ_intermediate/2)·I − H_intermediate)⁻¹` is the
intermediate-state propagator (typically a core-hole Hamiltonian);
`G_final = ((ω_out + Eg + iΓ_final/2)·I − H_final)⁻¹` is the final-state
propagator.

Output shape:

| Input | `tensor` shape (N) |
|---|---|
| scalar T_in, scalar T_out, single ψ | `(n_ω_in, n_ω_out)` (2) |
| vector T_in, vector T_out, single ψ | `(N_in, N_out, N_out, N_in, n_ω_in, n_ω_out)` (6) |

A list of ψ values adds a leading axis (matching `xas`'s convention).
Mixed scalar / vector inputs for T_in vs T_out promote to the full
6-D form internally; the result collapses to 2-D when both are scalar.

Greek-letter keyword arguments (`Γ_intermediate`, `Γ_final`,
`ω_in_grid`, `ω_out_grid`) accept ASCII Latin aliases
(`Gamma_intermediate`, `Gamma_final`, `omega_in_grid`, `omega_out_grid`);
mixing the two forms raises `ArgumentError`.

**Finite temperature.** Pass `ψ::Eigen` with `temperature = …` (`k_B = 1`,
optional `N_states`, `degen_tol`) for the Boltzmann ensemble average over initial
states, `χ_T = Σ_m ρ_m χ_m`, each referenced to its own `E_m`. Explicit
`ω_in_grid` **and** `ω_out_grid` are required (per-state `:auto` grids differ);
the result is grid-only and carries `Eg = E₀` plus `:temperature`,
`:ensemble_energies`, `:ensemble_weights`, `:N_kept` metadata.
"""
function rixs end

# Eigen overload — extract GS from full eigendecomposition.
function rixs(H_f, H_i, basis::AbstractBasis, T_in, T_out,
              E::LinearAlgebra.Eigen; Eg = nothing, temperature = nothing,
              N_states = nothing, degen_tol = 1e-8, kwargs...)
    Eg === nothing ||
        throw(ArgumentError("rixs: Eg cannot be specified when ψ is an Eigen"))
    if temperature === nothing
        (N_states === nothing && degen_tol == 1e-8) || throw(ArgumentError(
            "rixs: N_states / degen_tol are finite-T-only kwargs and require " *
            "temperature = ...; got temperature = nothing"))
        k = argmin(real.(E.values))
        return rixs(H_f, H_i, basis, T_in, T_out, E.vectors[:, k];
                    Eg = real(E.values[k]), kwargs...)
    end
    return _rixs_thermal(H_f, H_i, basis, T_in, T_out, E; temperature = temperature,
                         N_states = N_states, degen_tol = degen_tol, kwargs...)
end

# Single ψ vector.
function rixs(H_f, H_i, basis::AbstractBasis, T_in, T_out,
              ψ::AbstractVector{<:Number};
              ω_in_grid                = nothing,
              omega_in_grid            = nothing,
              ω_out_grid               = nothing,
              omega_out_grid           = nothing,
              Γ_intermediate           = nothing,
              Gamma_intermediate       = nothing,
              Γ_final                  = nothing,
              Gamma_final              = nothing,
              Eg::Union{Real,Nothing}  = nothing,
              krylovdim::Union{Int,Nothing}    = nothing,
              reorth::Union{Symbol,Nothing}    = nothing,
              tol::Union{Real,Nothing}         = nothing,
              restrictions_intermediate = nothing,
              restrictions_final        = nothing)
    Eg === nothing &&
        throw(ArgumentError("rixs: Eg required when ψ is a bare vector"))
    return _rixs_run(H_f, H_i, basis, _to_T_list(T_in), _to_T_list(T_out),
                     [ψ], _is_T_vector(T_in), _is_T_vector(T_out), false;
                     Γ_int_use   = _resolve_pair(Γ_intermediate, Gamma_intermediate;
                                                 default = DEFAULTS.Γ_intermediate,
                                                 name    = "Γ_intermediate"),
                     Γ_fin_use   = _resolve_pair(Γ_final, Gamma_final;
                                                 default = DEFAULTS.Γ_final,
                                                 name    = "Γ_final"),
                     ω_in_use    = _resolve_pair(ω_in_grid, omega_in_grid;
                                                 default = :auto, name = "ω_in_grid"),
                     ω_out_use   = _resolve_pair(ω_out_grid, omega_out_grid;
                                                 default = :auto, name = "ω_out_grid"),
                     krylovdim_use = something(krylovdim, DEFAULTS.krylovdim),
                     reorth_use    = something(reorth,    DEFAULTS.reorth),
                     tol_use       = something(tol,       DEFAULTS.tol),
                     Eg            = Float64(Eg),
                     restrictions_intermediate = restrictions_intermediate,
                     restrictions_final        = restrictions_final)
end

# Multi-ψ.
function rixs(H_f, H_i, basis::AbstractBasis, T_in, T_out,
              ψs::AbstractVector{<:AbstractVector{<:Number}};
              ω_in_grid = nothing, omega_in_grid = nothing,
              ω_out_grid = nothing, omega_out_grid = nothing,
              Γ_intermediate = nothing, Gamma_intermediate = nothing,
              Γ_final = nothing, Gamma_final = nothing,
              Eg::Union{Real,Nothing} = nothing,
              krylovdim::Union{Int,Nothing} = nothing,
              reorth::Union{Symbol,Nothing} = nothing,
              tol::Union{Real,Nothing} = nothing,
              restrictions_intermediate = nothing,
              restrictions_final        = nothing)
    Eg === nothing &&
        throw(ArgumentError("rixs: Eg required when ψ is a bare vector list"))
    Eg isa Real ||
        throw(ArgumentError("rixs: per-ψ Eg vectors not supported; pass a single Real Eg"))
    return _rixs_run(H_f, H_i, basis, _to_T_list(T_in), _to_T_list(T_out),
                     collect(ψs), _is_T_vector(T_in), _is_T_vector(T_out), true;
                     Γ_int_use = _resolve_pair(Γ_intermediate, Gamma_intermediate;
                                                default = DEFAULTS.Γ_intermediate,
                                                name = "Γ_intermediate"),
                     Γ_fin_use = _resolve_pair(Γ_final, Gamma_final;
                                                default = DEFAULTS.Γ_final, name = "Γ_final"),
                     ω_in_use  = _resolve_pair(ω_in_grid, omega_in_grid;
                                                default = :auto, name = "ω_in_grid"),
                     ω_out_use = _resolve_pair(ω_out_grid, omega_out_grid;
                                                default = :auto, name = "ω_out_grid"),
                     krylovdim_use = something(krylovdim, DEFAULTS.krylovdim),
                     reorth_use    = something(reorth,    DEFAULTS.reorth),
                     tol_use       = something(tol,       DEFAULTS.tol),
                     Eg            = Float64(Eg),
                     restrictions_intermediate = restrictions_intermediate,
                     restrictions_final        = restrictions_final)
end

# --- Driver ----------------------------------------------------------------

function _rixs_run(H_f, H_i, basis::AbstractBasis,
                   T_in_list::Vector{<:OperatorSum},
                   T_out_list::Vector{<:OperatorSum},
                   ψ_list::Vector{<:AbstractVector},
                   T_in_is_vector::Bool, T_out_is_vector::Bool,
                   ψ_is_list::Bool;
                   Γ_int_use, Γ_fin_use, ω_in_use, ω_out_use,
                   krylovdim_use, reorth_use, tol_use, Eg,
                   restrictions_intermediate, restrictions_final)
    isempty(T_in_list)  && throw(ArgumentError("rixs: T_in must be non-empty"))
    isempty(T_out_list) && throw(ArgumentError("rixs: T_out must be non-empty"))
    isempty(ψ_list)     && throw(ArgumentError("rixs: ψ must be non-empty"))

    N_states = length(basis)
    N_in     = length(T_in_list)
    N_out    = length(T_out_list)
    N_ψ      = length(ψ_list)
    B_raw_outer = N_in * N_out

    for ψ in ψ_list
        length(ψ) == N_states ||
            throw(DimensionMismatch("rixs: ψ length $(length(ψ)) ≠ basis size $N_states"))
    end

    # Compile + assemble all operators on shared basis.
    T_in_assembled  = SparseMatrixCSC[assemble(compile(t, basis), basis) for t in T_in_list]
    T_out_assembled = SparseMatrixCSC[assemble(compile(t, basis), basis) for t in T_out_list]

    # Per-ψ pipeline. We collect both LanczosChunk{ComplexF64} (for
    # SpectraTensor's `chunks` field + `_chunk_poles` auto-grid) and the
    # parallel `Responses.LanczosResponse` returned by `correlator(...)`
    # (for `to_grid(L, ω_out_resolved; Γ)` once the grid is resolved).
    # The two carry the same numerical content (same α, β, R) — chunks
    # exist for SpectraTensor invariants, LanczosResponses for the
    # Responses-layer evaluation path.
    chunks            = LanczosChunk{ComplexF64}[]
    outer_responses   = Responses.LanczosResponse{ComplexF64}[]

    # Empty As === Bs (reused across all outer correlator calls). Never
    # consulted in source_block mode beyond the `As === Bs` identity check.
    empty_outer_ops = OperatorSum[]

    # ---- ω_in :auto needs ALL ψ's inner T_K eigenvalues unioned --------
    # Two-pass approach when auto: pass 1 runs inner Lanczos per ψ with
    # retain_basis=false (V_int discarded) just to harvest T_int → poles.
    # Pass 2 (the main loop below) re-runs inner Lanczos with
    # retain_basis=true and continues to outer Lanczos. The 2× inner cost
    # is small relative to outer (which dominates wall time) and
    # avoids holding N_ψ V_int blocks in memory simultaneously.
    ω_in_resolved::AbstractRange = if ω_in_use === :auto
        e_min, e_max = Inf, -Inf
        for ψ in ψ_list
            X_int_p1 = Matrix{ComplexF64}(undef, N_states, N_in)
            for k in 1:N_in
                X_int_p1[:, k] = T_in_assembled[k] * ψ
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
        # ---- Step 1+2: build inner starting block + run inner Lanczos ----
        # Inner Lanczos uses `Responses.block_lanczos` directly because we
        # need the retained Krylov basis (V_int) to form Y = V_int · W —
        # an apply-to-vector path, not a matrix-of-ω return value, so
        # `correlator(...)` is the wrong abstraction here.
        X_int = Matrix{ComplexF64}(undef, N_states, N_in)
        for k in 1:N_in
            X_int[:, k] = T_in_assembled[k] * ψ
        end
        inner = Responses.block_lanczos(H_i, X_int;
                              krylovdim    = krylovdim_use,
                              reorth       = reorth_use,
                              tol          = tol_use,
                              restrictions = restrictions_intermediate,
                              basis        = restrictions_intermediate === nothing ? nothing : basis,
                              retain_basis = true)
        # block_lanczos with retain_basis=true stores K+1 blocks
        # [V_1, V_2, …, V_{K+1}], where V_{K+1} is the next block produced
        # at iteration K but never used inside the K-step Krylov
        # representation. The cf machinery's T_K is K × B_active, so V_int
        # must carry the FIRST K blocks only (drop V_{K+1}); otherwise
        # `V_int * W` is dimension-mismatched when block_lanczos didn't
        # soft-stop at the last iteration.
        V_int = hcat(inner.V_basis[1:length(inner.α)]...)::Matrix{ComplexF64}
        T_int_dense = _build_T_K(inner.α, inner.β)
        M = size(T_int_dense, 1)

        # ---- Step 3: per-ω_in outer Lanczos ----
        # Pad R_int columns to length M with zeros: ζ ∈ M × N_in.
        B_active_inner = size(inner.R, 1)
        ζ = zeros(ComplexF64, M, N_in)
        ζ[1:B_active_inner, :] .= inner.R         # B_active × N_in into top rows

        n_ω_in = length(ω_in_resolved)
        for (ω_in_idx, ω_in) in enumerate(ω_in_resolved)
            z_in = ComplexF64(ω_in + Eg + im * Γ_int_use / 2)
            # Solve (z_in I − T_int) · W = ζ in one banded LU.
            W = (z_in * I - T_int_dense) \ ζ      # M × N_in
            # Y = V_int · W, batched matmul.
            Y = V_int * W                          # N × N_in
            # Build outer starting block: column α(k, l) = (l-1)*N_out + k.
            X_out = Matrix{ComplexF64}(undef, N_states, B_raw_outer)
            for l in 1:N_in, k in 1:N_out
                col = (l - 1) * N_out + k
                X_out[:, col] = adjoint(T_out_assembled[k]) * Y[:, l]
            end

            # Outer Lanczos via correlator(...) in `source_block` mode
            # (X_out has already had T_out applied; passing it as `state`
            # would re-apply T_out and break shape). In source_block
            # mode correlator does not consult As / Bs beyond the
            # `As === Bs` identity check, so we pass an empty
            # `OperatorSum[]` reused across the call. `form = :lanczos`
            # because auto-`ω_out` resolution below needs the
            # LanczosResponse to harvest final-state poles across ALL
            # chunks; `to_grid(L, ω_out; Γ)` materialises the grid
            # response after the grid is settled.
            outer_response = Responses.correlator(
                H_f, basis, empty_outer_ops, empty_outer_ops;
                Γ            = Γ_fin_use,
                Eg           = Eg,
                state        = nothing,
                source_block = X_out,
                channel      = :neutral,
                form         = :lanczos,
                krylovdim    = krylovdim_use,
                reorth       = reorth_use,
                tol          = tol_use,
                restrictions = restrictions_final,
            )
            push!(outer_responses, outer_response)

            # Build the parallel LanczosChunk for SpectraTensor's `chunks`
            # field + `_chunk_poles` auto-grid. n_iter = length(α);
            # converged passes through faithfully from the underlying
            # block_lanczos via correlator → LanczosResponse.
            n_iter_outer    = length(outer_response.α)
            converged_outer = outer_response.converged
            push!(chunks, LanczosChunk{ComplexF64}(
                outer_response.α, outer_response.β,
                Matrix{ComplexF64}(outer_response.R),
                B_raw_outer, ψ_idx, ω_in_idx,
                n_iter_outer, converged_outer))
        end
    end

    # ---- Step 3.5: resolve ω_out_grid ----
    if ω_out_use === :auto
        e_min, e_max = Inf, -Inf
        for c in chunks
            es = real.(eigvals(Hermitian(_build_T_K(c.α, c.β))))
            e_min = min(e_min, minimum(es) - Eg)
            e_max = max(e_max, maximum(es) - Eg)
        end
        ω_out_resolved = auto_grid(e_min, e_max, Γ_fin_use)
    else
        ω_out_resolved = ω_out_use
    end
    n_ω_out = length(ω_out_resolved)
    n_ω_in  = length(ω_in_resolved)

    # ---- Step 4: cf evaluation per chunk + reshape + permutedims ----
    # Output tensor:
    #   if T_in_is_vector || T_out_is_vector → 6-D (N_in, N_out, N_out, N_in, n_ω_in, n_ω_out)
    #     [for now we always produce 6-D when ANY operator is vectorial; see below]
    #   else                                 → 2-D (n_ω_in, n_ω_out) for each ψ
    full6d = T_in_is_vector || T_out_is_vector
    converged_flag = all(c -> c.converged, chunks)

    # Per-(ψ, ω_in) GridResponse: extract `.data` from each outer
    # LanczosResponse via `to_grid(L, ω_out_resolved; Γ)`. Shape is
    # (B_raw_outer, B_raw_outer, n_ω_out) — identical to what
    # `evaluate_on_grid(chunk, …)` produced in v0.1, since the cf-recurrence
    # math is shared (Responses.cf_block under the hood).
    if full6d || ψ_is_list
        # 6-D shape (N_in, N_out, N_out, N_in, n_ω_in, n_ω_out): axes are
        # (i, j, k, l, ω_in, ω_out) — i, l index T_in (size N_in), j, k
        # index T_out (size N_out), matching the Kramers-Heisenberg
        # χ_{ijkl}(ω_in, ω) ∝ ε*_{in,i} ε_{out,j} χ_{ijkl} ε*_{out,k} ε_{in,l}
        # contraction.
        if ψ_is_list
            tensor = zeros(ComplexF64, N_in, N_out, N_out, N_in, N_ψ, n_ω_in, n_ω_out)
        else
            tensor = zeros(ComplexF64, N_in, N_out, N_out, N_in, n_ω_in, n_ω_out)
        end
        for (chunk, L_outer) in zip(chunks, outer_responses)
            ψ_idx    = chunk.ψ_index
            ω_in_idx = chunk.ω_in_index
            grid_response = Responses.to_grid(L_outer, ω_out_resolved;
                                              Γ = Γ_fin_use)
            χ̄ = grid_response.data                          # B_raw × B_raw × n_ω_out
            for ω_idx in 1:n_ω_out
                slab = χ̄[:, :, ω_idx]                        # B_raw × B_raw
                χ_jikl = reshape(slab, N_out, N_in, N_out, N_in)  # axes (j, i, k, l)
                χ_user = permutedims(χ_jikl, (2, 1, 3, 4))         # → (i, j, k, l)
                if ψ_is_list
                    tensor[:, :, :, :, ψ_idx, ω_in_idx, ω_idx] .= χ_user
                else
                    tensor[:, :, :, :, ω_in_idx, ω_idx] .= χ_user
                end
            end
        end
        # If both T_in and T_out are scalar (N_in = N_out = 1) but the user
        # passed lists of length 1, we still keep the 6-D form. The
        # collapse-to-2D path is only taken for true scalar inputs (handled
        # in the `else` branch below).
        if !full6d
            tensor = dropdims(tensor; dims = (1, 2, 3, 4))     # (N_ψ, n_ω_in, n_ω_out)
        end
    else
        # Scalar T_in, scalar T_out, single ψ: 2-D (n_ω_in, n_ω_out).
        tensor = zeros(ComplexF64, n_ω_in, n_ω_out)
        for (chunk, L_outer) in zip(chunks, outer_responses)
            ω_in_idx = chunk.ω_in_index
            grid_response = Responses.to_grid(L_outer, ω_out_resolved;
                                              Γ = Γ_fin_use)
            χ̄ = grid_response.data                          # 1×1×n_ω_out
            for ω_idx in 1:n_ω_out
                tensor[ω_in_idx, ω_idx] = χ̄[1, 1, ω_idx]
            end
        end
    end

    metadata = Dict{Symbol,Any}(
        :function       => :rixs,
        :Γ_intermediate => Γ_int_use,
        :Γ_final        => Γ_fin_use,
        :Eg             => Eg,
        :krylovdim      => krylovdim_use,
        :reorth         => reorth_use,
        :tol            => tol_use,
        :restrictions_intermediate => restrictions_intermediate,
        :restrictions_final        => restrictions_final,
        :converged      => converged_flag,
        :n_T_in         => N_in,
        :n_T_out        => N_out,
        :n_ψ            => N_ψ,
        :T_in_is_vector_input  => T_in_is_vector,
        :T_out_is_vector_input => T_out_is_vector,
        :ψ_is_list_input       => ψ_is_list,
        :auto_range            => (ω_in_use === :auto || ω_out_use === :auto),
        :basis_id              => basis_id(basis),
    )

    grid_pair = (ω_in_resolved, ω_out_resolved)
    F = typeof(grid_pair)
    N = ndims(tensor)
    return SpectraTensor{ComplexF64, Float64, N, F}(
        tensor, chunks, grid_pair, Float64(Eg), B_raw_outer, metadata)
end
