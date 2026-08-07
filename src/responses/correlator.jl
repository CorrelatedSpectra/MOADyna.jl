# =====================================================================
# MOADyna.Responses — correlator(...) constructor
# =====================================================================
#
# High-level constructor: takes `(H, basis, As, Bs; ...)`, builds the
# initial Krylov block, runs `block_lanczos`, and wraps the output as
# a `LanczosResponse` / `PoleResponse` / `GridResponse` per the `form`
# kwarg.
#
# Convention pins:
#
#   - **Autocorrelator-only contract:** `As === Bs` (object identity)
#     is validated. Cross-correlator (As ≠ Bs) belongs to a later
#     sub-phase; the signature accepts the broader shape for
#     forward-compat with joint-source-block + slice extensions.
#   - **Channel kwarg.** All four channels are implemented:
#     `:neutral` (default), `:addition`, `:removal`, `:both`. The
#     `:both` branch requires explicit `addition_indices` and/or
#     `removal_indices` (no silent default) — see chapter §3 for the
#     locked partition contract.
#   - **Source-input modes (mutually exclusive):**
#       (a) `state = :ground_state`, `source_block = nothing`: kernel
#           runs `eigen(H, basis; n=1)` to get ψ₀ + Eg.
#       (b) `state::AbstractVector` + `Eg::Real` + `source_block = nothing`:
#           user supplies state and Eg.
#       (c) `source_block::AbstractMatrix` + `Eg::Real` + `state = nothing`:
#           pre-built initial block; Bs unused.
#     Any other combination raises `ArgumentError`.
#   - **Sector-spanning is DOCUMENTED USER RESPONSIBILITY.** The kernel
#     does NOT validate; a nonzero-overlap heuristic can't distinguish
#     "operator gives zero physically" from "basis truncated away the
#     target sector".

"""
    correlator(H, basis, As::Vector{<:OperatorSum}, Bs::Vector{<:OperatorSum};
               ω                = nothing,
               Γ::Real,
               Eg               = nothing,
               state            = :ground_state,
               source_block     = nothing,
               channel          = :neutral,
               addition_indices = nothing,
               removal_indices  = nothing,
               form             = :lanczos,
               kwargs...) -> AbstractResponse

Build the matrix-of-ω response

    χ_{ab}(ω) = ⟨ψ₀ | A_a† G(ω) B_b | ψ₀⟩

using the unified resolvent convention `z = sign * (ω + iΓ/2) + Eg`
with leading `sign` factor (see the Conventions appendix for the
per-channel `sign` value),
then wrap the result as one of `LanczosResponse`, `PoleResponse`, or
`GridResponse` depending on the `form` kwarg.

# Positional arguments

- `H::AbstractMatrix` — Hamiltonian on `basis`. Typically `SparseMatrixCSC`.
- `basis::AbstractBasis` — the basis on which `H`, `As`, `Bs`, and
  `state` (or `source_block`) all live. **Sector-spanning is the
  caller's responsibility** (see Notes).
- `As::Vector{<:OperatorSum}` — left operators. **Must be `===` equal to
  `Bs`** (autocorrelator-only contract; cross-correlator deferred to v0.5+).
- `Bs::Vector{<:OperatorSum}` — right operators applied to `state` to
  build the initial Krylov block. **In source_block mode, `As` and `Bs`
  are not consulted** beyond the `As === Bs` identity check — pass them
  as `OperatorSum[]` (empty Vector) if you have no operators handy and
  the caller has already pre-built the initial block.

# Keyword arguments

| kwarg          | default          | role |
|----------------|------------------|------|
| `ω`            | `nothing`        | required when `form = :grid`; ignored otherwise |
| `Γ::Real`      | required         | FWHM Lorentzian broadening |
| `Eg`           | `nothing`        | required for state-vector / source-block modes; computed (and forbidden to supply) when `state = :ground_state` |
| `state`        | `:ground_state`  | `:ground_state` (compute ψ₀+Eg; Eg must be `nothing`) / `AbstractVector` (user ψ₀) / `nothing` (with `source_block`) |
| `source_block` | `nothing`        | pre-built initial block; mutually exclusive with non-`nothing` `state`; rejected for `channel = :both` |
| `channel`      | `:neutral`       | `:neutral` / `:addition` / `:removal` / `:both` — see chapter §3 for the partition contract on `:both` |
| `addition_indices` | `nothing`    | for `:addition`: defaults to `eachindex(Bs)` if `nothing`; for `:both`: explicit partition (may be empty/`nothing` on one side — at least one non-empty side required) |
| `removal_indices`  | `nothing`    | for `:removal`: defaults to `eachindex(Bs)` if `nothing`; for `:both`: explicit partition (may be empty/`nothing` on one side — at least one non-empty side required) |
| `form`         | `:lanczos`       | output type: `:lanczos`, `:pole`, or `:grid` |
| `T`            | `nothing`        | finite-T temperature in H's energy units (k_B=1); `nothing` = T=0 path. `form ∈ (:pole,:grid)`. With no `ensemble_states`, the ensemble is auto-computed from H on `basis` and is `:neutral`-only; supply `ensemble_states` for charged channels. |
| `N_states`     | `nothing`        | thermal-only: cap on number of lowest states summed (requires T) |
| `degen_tol`    | `1e-8`           | thermal-only: degeneracy tolerance for manifold-completeness (requires T) |
| `ensemble_states` | `nothing`     | thermal-only: explicit initial ensemble (`Vector` of vectors on `basis`), unlocking `:addition`/`:removal`/`:both` at finite T. For charged GF these are the N-sector eigenstates embedded in the N±1-spanning `basis`. Requires `ensemble_energies`. |
| `ensemble_energies` | `nothing`  | thermal-only: the per-state energies `E_m` aligned with `ensemble_states`. |

Any additional kwargs (e.g. `krylovdim`, `reorth`, `tol`, `restrictions`) are
forwarded to the internal `block_lanczos` kernel (`MOADyna.Responses.block_lanczos`).

# Return

- `channel = :neutral` / `:addition` (single channel):
  - `form = :lanczos` → `LanczosResponse{T}` with `sign = +1`.
  - `form = :pole`    → `PoleResponse{T}` (via `to_pole`).
  - `form = :grid`    → `GridResponse{Tc, 3}` materialised on `ω`.
- `channel = :removal` (single channel): same as above but inner `LanczosResponse`
  has `sign = -1`.
- `channel = :both`: returns `GreensFunction{T, R}` where `R` is the inner
  concrete type per `form`. Either field may be `nothing` if the
  corresponding index set is empty.

# Notes — sector-spanning (USER RESPONSIBILITY)

`basis` must span every conserved-quantity sector reached by
`Bs[j] · state` (or by `source_block`). The kernel does NOT validate
this — applying `Bs[j]` on a basis that doesn't cover the image sector
silently projects to zero, which can't be distinguished from "operator
truly gives zero physically". For `:neutral`, basis = same particle-
number sector as `state` is the canonical choice when `Bs[j]` are
particle-number-conserving. For non-conserving operators (e.g. dipole
in XAS), basis must include the sector(s) the operator reaches —
xas exemplifies the non-trivial case via `basis_xas` (n_2p ∈ 5:6).

If all `Bs[j] · state` columns are zero (rank-zero initial block),
`block_lanczos` raises its standard rank-zero error — a real error
path the kernel detects, distinct from sector-coverage diagnosis.

See the **Responses** chapter of the manual for the full contract and
source-input semantics.
"""
function correlator(H::AbstractMatrix,
                    basis::AbstractBasis,
                    As::Vector{<:OperatorSum},
                    Bs::Vector{<:OperatorSum};
                    ω                  = nothing,
                    Γ::Real,
                    Eg                 = nothing,
                    state              = :ground_state,
                    source_block       = nothing,
                    channel::Symbol    = :neutral,
                    addition_indices   = nothing,
                    removal_indices    = nothing,
                    form::Symbol       = :lanczos,
                    T                  = nothing,
                    N_states           = nothing,
                    degen_tol          = 1e-8,
                    ensemble_states    = nothing,
                    ensemble_energies  = nothing,
                    kwargs...)
    # --- 1. Autocorrelator validation ----------------------------------
    if !(As === Bs)
        throw(ArgumentError(
            "correlator: cross-correlator (As ≠ Bs) deferred to v0.5+; " *
            "current contract is autocorrelator-only (As === Bs)."
        ))
    end

    # --- 1b. Finite-T gate (LOCKED: precedes ALL channel dispatch) ---------
    #
    # When T === nothing the function behaves exactly as before (T=0 path),
    # EXCEPT that the thermal-only kwargs N_states / degen_tol must not be
    # supplied (guards against silent API drift outside thermal mode).
    # When T !== nothing we validate the finite-T contract (2d is :neutral-only,
    # same-sector) and dispatch to the thermal path.
    if T === nothing
        N_states === nothing || throw(ArgumentError(
            "correlator: N_states is a thermal-only kwarg and requires T (got T = nothing)"))
        degen_tol == 1e-8 || throw(ArgumentError(
            "correlator: degen_tol is a thermal-only kwarg and requires T (got T = nothing)"))
        (ensemble_states === nothing && ensemble_energies === nothing) || throw(ArgumentError(
            "correlator: ensemble_states / ensemble_energies are thermal-only kwargs and " *
            "require T (got T = nothing)"))
    else
        (T isa Real && T ≥ 0) || throw(ArgumentError(
            "correlator: T must be a real number ≥ 0; got T = $(repr(T))"))

        # A user-supplied initial ensemble (states + energies, both living on
        # `basis`) unlocks the CHARGED finite-T channels. The N-sector thermal
        # ensemble cannot be obtained by diagonalizing the N±1-spanning
        # propagation basis (its global ground state may live in the wrong
        # sector), so for charged GF the caller supplies the N-sector eigenstates
        # embedded in `basis`. With no ensemble supplied, the ensemble is
        # auto-computed from H on basis — well-defined only for the same-sector
        # :neutral channel.
        supplied = ensemble_states !== nothing || ensemble_energies !== nothing

        form ∈ (:pole, :grid) || throw(ArgumentError(
            "correlator: finite-T requires form = :pole or :grid (a thermal sum over many " *
            "states has no single Lanczos representation); got form = :$form"))
        # form = :grid needs ω; the thermal gate returns before the common
        # form/ω check below, so validate here to keep the ArgumentError contract
        # (otherwise to_grid(folded, nothing) would leak a MethodError).
        if form === :grid && ω === nothing
            throw(ArgumentError("form = :grid requires ω kwarg"))
        end
        Eg === nothing || throw(ArgumentError(
            "correlator: finite-T takes Eg per ensemble state (the per-state E_m); " *
            "do not supply Eg"))
        # degen_tol must be a positive finite real: degen_tol ≤ 0 disables
        # exact-degeneracy detection and, at T = 0, can make g = 0 (NaN weights).
        (degen_tol isa Real && isfinite(degen_tol) && degen_tol > 0) || throw(ArgumentError(
            "correlator: degen_tol must be a positive finite real; got degen_tol = $(repr(degen_tol))"))
        if N_states !== nothing
            # Bool <: Integer in Julia; exclude it so N_states = true isn't silently 1.
            (N_states isa Integer && !(N_states isa Bool) && N_states ≥ 1) || throw(ArgumentError(
                "correlator: N_states must be an Integer ≥ 1; got N_states = $(repr(N_states))"))
        end

        if supplied
            (ensemble_states !== nothing && ensemble_energies !== nothing) || throw(ArgumentError(
                "correlator: supply BOTH ensemble_states and ensemble_energies (got only one)"))
            length(ensemble_states) == length(ensemble_energies) || throw(DimensionMismatch(
                "correlator: ensemble_states ($(length(ensemble_states))) and " *
                "ensemble_energies ($(length(ensemble_energies))) length mismatch"))
            isempty(ensemble_states) &&
                throw(ArgumentError("correlator: ensemble_states is empty"))
            channel ∈ (:neutral, :addition, :removal, :both) || throw(ArgumentError(
                "correlator: channel must be one of :neutral, :addition, :removal, :both; " *
                "got :$channel"))
            state === :ground_state || throw(ArgumentError(
                "correlator: finite-T with a supplied ensemble takes the states from " *
                "ensemble_states; do not also pass state = $(repr(state))"))
            source_block === nothing || throw(ArgumentError(
                "correlator: finite-T is incompatible with source_block"))
            # Dispatch to the supplied-ensemble thermal path (any channel).
            return _thermal_response(H, basis, As, Bs;
                                     T = T, N_states = N_states, degen_tol = degen_tol,
                                     Γ = Γ, form = form, ω = ω,
                                     channel = channel,
                                     addition_indices = addition_indices,
                                     removal_indices = removal_indices,
                                     ensemble_states = ensemble_states,
                                     ensemble_energies = ensemble_energies,
                                     kwargs...)
        else
            # Auto-computed ensemble: same-sector :neutral only.
            channel === :neutral || throw(ArgumentError(
                "correlator: finite-T without a supplied ensemble is :neutral-only (the " *
                "ensemble is auto-computed from H on basis, well-defined only for the same " *
                "particle-number sector). For charged channels (single-particle GF / XPS) " *
                "pass ensemble_states + ensemble_energies — the N-sector thermal states " *
                "embedded in the N±1-spanning basis. Got channel = :$channel"))
            state === :ground_state || throw(ArgumentError(
                "correlator: finite-T requires state = :ground_state (the thermal ensemble is " *
                "computed from H's spectrum on basis, not supplied); got state = $(repr(state))"))
            source_block === nothing || throw(ArgumentError(
                "correlator: finite-T is incompatible with source_block (the ensemble is " *
                "computed from H's spectrum on basis)"))
            # Dispatch to the auto-ensemble thermal path (2d Task 3).
            return _thermal_response(H, basis, As, Bs;
                                     T = T, N_states = N_states, degen_tol = degen_tol,
                                     Γ = Γ, form = form, ω = ω, kwargs...)
        end
    end

    # --- 2. Channel validation -----------------------------------------
    if channel === :both
        # ----------------------------------------------------------------
        # :both branch — explicit partition contract (LOCKED)
        # ----------------------------------------------------------------

        # 2a. source_block is single-channel only
        if source_block !== nothing
            throw(ArgumentError(
                "correlator(:both): source_block is single-channel only — pass separate " *
                "addition_source_block / removal_source_block (deferred to v0.5+) or use " *
                "state= mode for :both"
            ))
        end

        # 2b. form validation (needed before sub-calls can proceed)
        form ∈ (:lanczos, :pole, :grid) ||
            throw(ArgumentError(
                "form must be one of :lanczos, :pole, :grid; got :$form"
            ))
        if form == :grid && ω === nothing
            throw(ArgumentError("form = :grid requires ω kwarg"))
        end

        # 2c. Parse addition_indices / removal_indices per the LOCKED partition contract
        #     Both nothing → error (no silent default to all-addition)
        if addition_indices === nothing && removal_indices === nothing
            throw(ArgumentError(
                "correlator(:both): pass addition_indices and/or removal_indices " *
                "explicitly — :both does not silently default to all-addition"
            ))
        end
        # One provided, other nothing → default the unspecified one to empty Int[]
        add_idx = addition_indices === nothing ? Int[] : addition_indices
        rem_idx = removal_indices  === nothing ? Int[] : removal_indices
        # Both empty after defaulting → same error
        if isempty(add_idx) && isempty(rem_idx)
            throw(ArgumentError(
                "correlator(:both): pass addition_indices and/or removal_indices " *
                "explicitly — :both does not silently default to all-addition"
            ))
        end

        # 2d. Validate individual index lists (reuse the helper below — but it
        #     is defined later as a closure inside this function, so we inline the
        #     validation here using the same logic)
        function _validate_both_indices(idxs, ch_name)
            idxs isa AbstractVector{<:Integer} ||
                throw(ArgumentError(
                    "correlator(:both): $(ch_name)_indices must be an " *
                    "AbstractVector{<:Integer}; got $(typeof(idxs))"
                ))
            for i in idxs
                i ∈ eachindex(Bs) ||
                    throw(ArgumentError(
                        "correlator(:both): $(ch_name)_indices has out-of-bounds " *
                        "index $i (Bs has $(length(Bs)) elements)"
                    ))
            end
            length(unique(idxs)) == length(idxs) ||
                throw(ArgumentError(
                    "correlator(:both): $(ch_name)_indices has duplicate entries"
                ))
        end
        !isempty(add_idx) && _validate_both_indices(add_idx, :addition)
        !isempty(rem_idx) && _validate_both_indices(rem_idx, :removal)

        # 2e. No overlap between addition and removal index sets
        if !isempty(intersect(add_idx, rem_idx))
            throw(ArgumentError(
                "correlator(:both): addition_indices and removal_indices overlap — " *
                "exotic self-adjoint operators not supported in v0.2"
            ))
        end

        # 2f. Length-equality check (when both channels populated)
        if !isempty(add_idx) && !isempty(rem_idx) && length(add_idx) != length(rem_idx)
            throw(ArgumentError(
                "correlator(:both): addition_indices and removal_indices have unequal " *
                "length; addition + removal sum requires aligned matrices"
            ))
        end

        # 2g. Recursive sub-calls — slice then pass nothing for indices so each
        #     sub-call uses all of its sliced operators (autocorrelator contract preserved).
        #     Forward: ω, Γ, Eg, state, form, and any additional kwargs (krylovdim,
        #     reorth, tol, etc.) captured in kwargs... Exclude source_block, channel,
        #     addition_indices, removal_indices (all handled at this level).
        addition_response = if !isempty(add_idx)
            ops_add = Bs[add_idx]
            correlator(H, basis, ops_add, ops_add;
                       ω                = ω,
                       Γ                = Γ,
                       Eg               = Eg,
                       state            = state,
                       source_block     = nothing,
                       channel          = :addition,
                       addition_indices = nothing,
                       removal_indices  = nothing,
                       form             = form,
                       kwargs...)
        else
            nothing
        end

        removal_response = if !isempty(rem_idx)
            ops_rem = Bs[rem_idx]
            correlator(H, basis, ops_rem, ops_rem;
                       ω                = ω,
                       Γ                = Γ,
                       Eg               = Eg,
                       state            = state,
                       source_block     = nothing,
                       channel          = :removal,
                       addition_indices = nothing,
                       removal_indices  = nothing,
                       form             = form,
                       kwargs...)
        else
            nothing
        end

        # 2h. Determine T and R from whichever sub-call returned a non-nothing result.
        #     Both sub-calls used the same `form`, so their inner types match.
        _non_nothing = addition_response !== nothing ? addition_response : removal_response
        T_inner = eltype(_non_nothing)
        R_inner = typeof(_non_nothing)

        return GreensFunction{T_inner, R_inner}(addition_response, removal_response)

    elseif channel ∉ (:neutral, :addition, :removal)
        throw(ArgumentError(
            "channel must be one of :neutral, :addition, :removal, :both; got :$channel"
        ))
    end

    # --- 3. form validation --------------------------------------------
    form ∈ (:lanczos, :pole, :grid) ||
        throw(ArgumentError(
            "form must be one of :lanczos, :pole, :grid; got :$form"
        ))
    if form == :grid && ω === nothing
        throw(ArgumentError("form = :grid requires ω kwarg"))
    end

    # --- 4. Source-input mode: validate mutually-exclusive kwargs ------
    #
    # Three valid combinations (per Convention pin):
    #   (a) state = :ground_state, source_block = nothing
    #   (b) state::AbstractVector + Eg::Real, source_block = nothing
    #   (c) source_block::AbstractMatrix + Eg::Real, state = nothing
    #
    # `Bs` non-emptiness is enforced only in state modes (a)/(b), where Bs
    # actually drives the initial-block construction. In source_block mode
    # (c), Bs is unused (the caller has already pre-built the initial
    # block); empty Bs is permitted there.

    if source_block !== nothing
        # mode (c)
        if state !== nothing
            throw(ArgumentError(
                "correlator: when source_block is provided, state must be `nothing` " *
                "(got state = $(repr(state))). source_block and state are mutually exclusive."
            ))
        end
        if Eg === nothing
            throw(ArgumentError(
                "correlator: source_block requires explicit Eg"
            ))
        end
        source_block isa AbstractMatrix || throw(ArgumentError(
            "correlator: source_block must be an AbstractMatrix; got $(typeof(source_block))"
        ))
    elseif state === :ground_state
        # mode (a) — Eg is computed automatically via eigen(H, basis; n=1).
        # Supplying Eg here violates the mode-disjoint contract: if you want
        # to override Eg, supply an explicit state::AbstractVector + Eg.
        if Eg !== nothing
            throw(ArgumentError(
                "correlator: Eg is computed automatically when state = :ground_state; " *
                "if you want to override, use state::Vector + Eg explicitly"
            ))
        end
        isempty(Bs) && throw(ArgumentError(
            "correlator: Bs cannot be empty in state mode (no operators to apply " *
            "to the ground state)"
        ))
    elseif state isa AbstractVector
        # mode (b)
        if Eg === nothing
            throw(ArgumentError(
                "correlator: state::AbstractVector requires explicit Eg"
            ))
        end
        isempty(Bs) && throw(ArgumentError(
            "correlator: Bs cannot be empty in state mode (no operators to apply " *
            "to the supplied state)"
        ))
    elseif state === nothing
        # state = nothing without source_block is invalid
        throw(ArgumentError(
            "correlator: at least one of `state` or `source_block` must be set " *
            "(got both nothing). Use state = :ground_state for the default."
        ))
    else
        throw(ArgumentError(
            "correlator: state must be :ground_state, an AbstractVector, or nothing " *
            "(when paired with source_block); got $(repr(state))"
        ))
    end

    # --- 4b. Resolve channel-specific operator indices -----------------
    #
    # For :addition/:removal, default to all Bs; for :neutral, indices are
    # unused (the full Bs is used directly in step 5). Validate when explicit.
    #
    # Helper: validate an explicit index vector against Bs.
    function _validate_channel_indices(idxs, ch_name)
        idxs isa AbstractVector{<:Integer} ||
            throw(ArgumentError(
                "correlator(:$ch_name): $(ch_name)_indices must be an " *
                "AbstractVector{<:Integer}; got $(typeof(idxs))"
            ))
        isempty(idxs) &&
            throw(ArgumentError(
                "correlator(:$ch_name): $(ch_name)_indices cannot be empty " *
                "for single-channel mode (use channel = :both if one side " *
                "should be empty)"
            ))
        for i in idxs
            i ∈ eachindex(Bs) ||
                throw(ArgumentError(
                    "correlator(:$ch_name): $(ch_name)_indices has out-of-bounds " *
                    "index $i (Bs has $(length(Bs)) elements)"
                ))
        end
        length(unique(idxs)) == length(idxs) ||
            throw(ArgumentError(
                "correlator(:$ch_name): $(ch_name)_indices has duplicate entries"
            ))
    end

    ops_active = if channel === :addition
        idx = addition_indices === nothing ? eachindex(Bs) : addition_indices
        if addition_indices !== nothing
            _validate_channel_indices(addition_indices, :addition)
        end
        Bs[idx]
    elseif channel === :removal
        idx = removal_indices === nothing ? eachindex(Bs) : removal_indices
        if removal_indices !== nothing
            _validate_channel_indices(removal_indices, :removal)
        end
        Bs[idx]
    else
        # :neutral — use all Bs
        Bs
    end

    # --- 5. Build the initial block X_raw ------------------------------
    N_basis = length(basis)
    X_raw, Eg_resolved = if source_block !== nothing
        # mode (c): pre-built block
        size(source_block, 1) == N_basis ||
            throw(DimensionMismatch(
                "correlator: source_block has $(size(source_block, 1)) rows; " *
                "basis has $N_basis"
            ))
        (Matrix(source_block), Float64(Eg))
    else
        # modes (a) and (b): build from ops_active · ψ₀
        ψ₀, Eg_use = if state === :ground_state
            E = eigen(H, basis; n = 1)
            k = argmin(real.(E.values))
            ψ_gs = E.vectors[:, k]
            Eg_computed = real(E.values[k])
            (ψ_gs, Eg_computed)
        else
            # state::AbstractVector
            length(state) == N_basis ||
                throw(DimensionMismatch(
                    "correlator: state has length $(length(state)); basis has $N_basis"
                ))
            (state, Float64(Eg))
        end

        # Compile + assemble each active B operator on the basis, then form
        # X[:, j] = B_j · ψ₀.
        N_B = length(ops_active)

        # Choose element type: complex at the product's natural precision.
        # Promote ψ₀'s eltype with H's eltype so that, e.g., a Float32
        # state paired with a Float64 Hamiltonian gives ComplexF64 (full
        # product precision) rather than ComplexF32 (state precision only).
        Tstate = eltype(ψ₀)
        THam   = eltype(H)
        Tprod  = promote_type(Tstate, THam)
        Tcol   = Tprod <: Complex ? Tprod : Complex{Tprod}
        X = Matrix{Tcol}(undef, N_basis, N_B)
        for j in 1:N_B
            B_assembled = assemble(compile(ops_active[j], basis), basis)
            X[:, j] = B_assembled * ψ₀
        end
        (X, Eg_use)
    end

    # --- 6. Run the kernel ---------------------------------------------
    #
    # Forward any user-supplied tuning kwargs (krylovdim, reorth, tol,
    # min_iter, max_iter, deflate_tol, restrictions, basis, retain_basis)
    # via `kwargs...`. Defaults come from `block_lanczos` itself.
    #
    # `restrictions` requires `basis` — if the user passes restrictions
    # without basis, forward our basis automatically (consistent with
    # spectroscopy wrappers).  Splat kwargs directly (no Dict allocation;
    # type-stable) with basis auto-filled when restrictions is present.
    #
    # For :addition, wrap in try/catch to provide a channel-specific
    # rank-zero hint (basis may not span the (N+1) sector).
    _run_lanczos(X_in) = if haskey(kwargs, :restrictions) && !haskey(kwargs, :basis)
        block_lanczos(H, X_in; basis = basis, kwargs...)
    else
        block_lanczos(H, X_in; kwargs...)
    end

    result = if channel === :addition
        try
            _run_lanczos(X_raw)
        catch err
            if err isa ArgumentError && occursin("rank zero", err.msg)
                throw(ArgumentError(
                    "correlator(:addition): initial block has rank zero — this often " *
                    "means the basis does not span the (N+1) sector reached by the " *
                    "selected source operators, or the operators have zero overlap " *
                    "with the state. (Underlying: $(err.msg))"
                ))
            else
                rethrow(err)
            end
        end
    elseif channel === :removal
        try
            _run_lanczos(X_raw)
        catch err
            if err isa ArgumentError && occursin("rank zero", err.msg)
                throw(ArgumentError(
                    "correlator(:removal): initial block has rank zero — this often means " *
                    "the basis does not span the (N−1) sector reached by the selected source " *
                    "operators, or the operators have zero overlap with the state. " *
                    "(Underlying: $(err.msg))"
                ))
            else
                rethrow(err)
            end
        end
    else
        _run_lanczos(X_raw)
    end

    # --- 7. Wrap as LanczosResponse ------------------------------------
    #
    # block_lanczos may return `length(β) == length(α)` when the recurrence
    # hits the krylovdim cap without exhausting the Krylov subspace — the
    # trailing β_K connects α[K] to a never-stored α[K+1]. The cf-recurrence
    # consumers (cf.jl, _build_T_K) ignore this trailing β, but
    # LanczosResponse's validator requires `length(β) == length(α) - 1`,
    # so we drop it here. (Same convention as `_build_T_K`'s
    # `min(K-1, length(β))` cap.)
    Teltype = eltype(X_raw)
    α_raw = Vector{Matrix{Teltype}}(result.α)
    β_raw = Vector{Matrix{Teltype}}(result.β)
    α = α_raw
    β = length(β_raw) == length(α) ? β_raw[1:end-1] : β_raw
    R = Matrix{Teltype}(result.R)

    sign_val = channel === :removal ? -1 : +1
    L = LanczosResponse{Teltype}(α, β, R, Float64(Eg_resolved), Float64(Γ),
                                 sign_val, one(Teltype), Bool(result.converged))

    # --- 8. Convert to requested form ----------------------------------
    if form === :lanczos
        return L
    elseif form === :pole
        return to_pole(L)
    else  # :grid
        return to_grid(L, ω; Γ = Float64(Γ))
    end
end

# =====================================================================
# Finite-T thermal helpers (internal; 2d Tasks 2–3)
# =====================================================================
#
# Physics (locked derivation, finite-temperature thermal average):
#
#   I_{ab}(ω,T) = Σ_m ρ_m ⟨m| A_a† (ω + E_m − H + iΓ/2)⁻¹ B_b |m⟩,
#   ρ_m = e^{−βE_m}/Z,   Z = Σ_m e^{−βE_m},   k_B = 1.
#
# Each summand is EXACTLY the T=0 kernel correlator(:neutral; state=ψ_m,
# Eg=E_m), so the thermal path reuses the single-state kernel per state,
# re-labels each result's Eg → E₀ (a metadata-only change — PoleResponse
# evaluation is Eg-independent; the physical pole positions live in
# `poles`), weights by ρ_m, and folds via `+`.

"""
    _thermal_states(H, basis, T, N_states, degen_tol)
        -> (states, energies, weights, E0)

Compute the lowest eigenpairs of `H` on `basis` and their Boltzmann
weights for the finite-T thermal average. Internal helper; assumes a
validated contract (T isa Real ≥ 0; N_states is `nothing` or `Integer ≥ 1`).

Returns:
- `states`   — `Vector` of eigenvectors (columns of `eigen`), aligned with
- `energies::Vector{Float64}` — ascending eigenvalues (`E_m`),
- `weights::Vector{Float64}`  — 1/Z-normalised ρ_m, and
- `E0::Float64`               — the ground-state energy (lowest E_m).

Zero-weight states (e.g. the T=0 excited states) are dropped before
returning, so the caller never runs the kernel on them.
"""
function _thermal_states(H, basis, T::Real, N_states, degen_tol)
    # Full-spectrum guard: the full trace is only tractable on a small
    # basis (the eigen dense-path threshold dense_below = 1024).
    if N_states === nothing && length(basis) > 1024
        throw(ArgumentError(
            "correlator: full-spectrum finite-T trace (N_states = nothing) is " *
            "only tractable on small bases (length(basis) ≤ 1024); pass an " *
            "explicit N_states"))
    end

    # Request one buffer state beyond N_states to test manifold completeness.
    n_req = N_states === nothing ? length(basis) : min(N_states + 1, length(basis))
    E = eigen(H, basis; n = n_req)
    return _boltzmann_ensemble(E.values, E.vectors, T, N_states, degen_tol)
end

"""
    _boltzmann_ensemble(energies, vectors, T, N_states, degen_tol)
        -> (states, energies_kept, weights, E0)

Truncate a precomputed spectrum to its lowest `N_states` eigenpairs
(degeneracy-complete), form the 1/Z-normalised Boltzmann weights, and drop
zero-weight states. Shared by `_thermal_states` (which supplies `eigen(H, basis)`)
and the Spectroscopy finite-T wrappers (which supply a user `Eigen` of the
*initial* spectrum, whose eigenvectors live in the propagation `basis`).

`energies` / `vectors` need NOT be pre-sorted — they are sorted here (carrying
the eigenvectors), so the caller never has to trust the input `Eigen` ordering.
The contract is only that the lowest kept states are the intended initial
manifold. `T == 0` gives equal weight `1/g` across the degenerate ground
manifold (within `degen_tol`); `T > 0` gives `ρ_m = e^{-βE_m}/Z`. The same P8
degeneracy gate (truncation must not split a manifold) and P10 under-convergence
warning apply.
"""
function _boltzmann_ensemble(energies_in::AbstractVector, vectors_in::AbstractMatrix,
                             T::Real, N_states, degen_tol)
    # Sort ascending, carrying eigenvectors (do NOT trust input ordering).
    energies_all = Float64.(real.(energies_in))
    perm = sortperm(energies_all)
    energies_all = energies_all[perm]
    vectors_all = vectors_in[:, perm]
    E0 = energies_all[1]

    # First-excluded eigenvalue (if a buffer state exists beyond N_states),
    # captured BEFORE truncation for the degeneracy gate (P8) and the
    # under-convergence warning (P10).
    has_buffer = N_states !== nothing && length(energies_all) ≥ N_states + 1
    E_excluded = has_buffer ? energies_all[N_states + 1] : nothing

    # Degeneracy gate (P8): truncating at N_states must not split a
    # degenerate manifold.
    if has_buffer && abs(energies_all[N_states + 1] - energies_all[N_states]) < degen_tol
        throw(ArgumentError(
            "correlator: truncation at N_states=$N_states splits a degenerate " *
            "manifold (E[$N_states] and E[$(N_states + 1)] differ by < " *
            "degen_tol=$degen_tol); increase N_states to include the full manifold"))
    end

    # Keep the first min(N_states, length) states/energies.
    n_keep = N_states === nothing ? length(energies_all) :
             min(N_states, length(energies_all))
    energies = energies_all[1:n_keep]
    vectors  = vectors_all[:, 1:n_keep]

    # Boltzmann weights.
    weights = if T == 0
        # Degeneracy-correct zero-T limit: equal weight 1/g across the
        # (possibly degenerate) ground manifold, zero elsewhere.
        w = [abs(e - E0) < degen_tol ? 1.0 : 0.0 for e in energies]
        g = count(>(0.0), w)
        w ./ g
    else
        β = 1 / T
        raw = exp.(-β .* (energies .- E0))
        raw ./ sum(raw)
    end

    # Under-convergence warning (P10): only when T > 0 AND a buffer existed
    # — the lowest EXCLUDED state still carries appreciable Boltzmann weight.
    if T > 0 && has_buffer
        β = 1 / T
        if exp(-β * (E_excluded - E0)) > 1e-3
            @warn "correlator: finite-T thermal sum may be under-converged; the " *
                  "lowest excluded state has Boltzmann weight > 1e-3. Increase N_states."
        end
    end

    # Drop zero-weight states (e.g. T=0 excited states) so the caller never
    # runs the kernel on them; keep states/energies/weights aligned.
    keep = findall(w -> w != 0, weights)
    states_out = [vectors[:, j] for j in keep]
    return (states_out, energies[keep], weights[keep], E0)
end

# Re-frame a per-state response's energy reference Eg → E₀ so that responses
# computed from different thermal initial states share a common frame and the
# `+` overloads accept them. Stored poles are physical excitation frequencies
# (addition `E_n^{N+1}−E_m`, removal `E_m−E_n^{N-1}` via `sign = -1`, neutral
# `E_n−E_m`), independent of the reference, so this is a metadata-only relabel.
_reframe_eg(P::PoleResponse, Eg) =
    PoleResponse{eltype(P)}(P.a0, P.poles, P.residues, Float64(Eg), P.Γ)

# Recurse into both GreensFunction channels (charged `:both`).
function _reframe_eg(GF::GreensFunction, Eg)
    add = GF.addition === nothing ? nothing : _reframe_eg(GF.addition, Eg)
    rem = GF.removal  === nothing ? nothing : _reframe_eg(GF.removal,  Eg)
    return GreensFunction(add, rem)
end

"""
    _thermal_response(H, basis, As, Bs; T, N_states, degen_tol, Γ, form, ω,
                      channel = :neutral, addition_indices = nothing,
                      removal_indices = nothing, ensemble_states = nothing,
                      ensemble_energies = nothing, kwargs...)
        -> PoleResponse | GridResponse | GreensFunction

Finite-T thermal average `A_T = Σ_m ρ_m A_m`. Internal helper; assumes a
validated contract (see the finite-T gate in `correlator`).

The ensemble is either auto-computed from `H` on `basis` (`ensemble_states ===
nothing`, same-sector `:neutral` only) or supplied explicitly via
`ensemble_states` (each a vector on `basis`) + `ensemble_energies` (the
charged-channel path: the N-sector thermal states embedded in the N±1-spanning
basis). Each per-state response is built in **pole form** (so it is linearly
combinable; `:both` yields a `GreensFunction{_,PoleResponse}`), re-labelled
`Eg → E₀` (recursively for `GreensFunction`), weighted by `ρ_m`, folded via `+`,
then dispatched on `form`.
"""
function _thermal_response(H, basis, As, Bs;
                           T, N_states, degen_tol, Γ, form, ω,
                           channel = :neutral,
                           addition_indices = nothing,
                           removal_indices = nothing,
                           ensemble_states = nothing,
                           ensemble_energies = nothing,
                           kwargs...)
    states, energies, weights, E0 = if ensemble_states === nothing
        _thermal_states(H, basis, T, N_states, degen_tol)
    else
        # Supplied ensemble: run the same truncation/degeneracy/weight logic on
        # the user-provided (energies, states). hcat the state vectors into the
        # matrix `_boltzmann_ensemble` expects.
        V = reduce(hcat, ensemble_states)
        _boltzmann_ensemble(ensemble_energies, V, T, N_states, degen_tol)
    end

    weighted = map(eachindex(states)) do m
        per_state = correlator(H, basis, As, Bs;
                               Γ = Γ, Eg = energies[m], state = states[m],
                               source_block = nothing, channel = channel,
                               addition_indices = addition_indices,
                               removal_indices = removal_indices,
                               form = :pole, kwargs...)   # NO T — single-state path
        weights[m] * _reframe_eg(per_state, E0)
    end

    folded = reduce(+, weighted)   # Eg = E₀; PoleResponse, or GreensFunction for :both

    form === :pole && return folded
    # form === :grid: materialise on ω. For :both, grid each channel and keep
    # the GreensFunction wrapper (consistent with the T=0 :both grid path),
    # rather than summing the channels.
    return _thermal_to_grid(folded, ω, Float64(Γ))
end

_thermal_to_grid(R::AbstractResponse, ω, Γ) = to_grid(R, ω; Γ = Γ)
function _thermal_to_grid(GF::GreensFunction, ω, Γ)
    add = GF.addition === nothing ? nothing : to_grid(GF.addition, ω; Γ = Γ)
    rem = GF.removal  === nothing ? nothing : to_grid(GF.removal,  ω; Γ = Γ)
    return GreensFunction(add, rem)
end
