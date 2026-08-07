# =====================================================================
# Block Lanczos — hand-rolled, with rank-revealing initial QR + ragged-
# block deflation + optional FRO
# =====================================================================
#
# Hand-rolled rather
# than KrylovKit-wrapped because we need:
#   - per-iteration α/β block matrices kept around for cf re-evaluation
#   - the rectangular initial QR factor `R` (B_active × B_raw) for the
#     `R† G₁ R` user-basis sandwich
#   - ragged-block deflation (`B_k` may shrink at any iteration)
#   - in-loop projection for the `P · H · P` restriction semantics
#
# KrylovKit's block Lanczos doesn't expose any of those, so we maintain
# our own primitive.

"""
    block_lanczos(H, X_raw; kwargs...) -> NamedTuple

Run a block-Lanczos recurrence on `H` with starting block `X_raw`. The
initial column-pivoted QR ortho-normalises the starting block,
producing the rectangular factor `R` of shape `B_active × B_raw`
(rank-deflated when `X_raw` is not full column rank). The recurrence
then produces block-α and block-β stacks. Ragged-block deflation drops
columns of subsequent `V_{k+1}` blocks whose pivot falls below
`deflate_tol · ||X_pivot_max||`, shrinking the active block size on the
fly.

# Arguments

- `H::AbstractMatrix` — the Hamiltonian. Anything supporting `H * V` for
  `V::Matrix`. Typically `SparseMatrixCSC`.
- `X_raw::AbstractMatrix` — the `N × B_raw` starting block. Eltype is
  preserved through the recurrence; for cf evaluation downstream the
  caller is expected to pass `Complex{T}` (the cf machinery is complex
  because `z = ω + Eg + iΓ/2`).

# Keyword arguments

Defaults shown here are the kernel-level literals used by
`MOADyna.Responses.block_lanczos`. The `MOADyna.Spectroscopy` wrappers
(`xas`, `rixs`, `fluorescence_yield`) resolve domain defaults via
`Spectroscopy.DEFAULTS` before calling this kernel.

| kwarg | default | meaning |
|---|---|---|
| `krylovdim::Int` | `200` | target number of block-Lanczos iterations |
| `reorth::Symbol` | `:none` | `:none` (3-term) or `:full` (FRO) |
| `tol::Real` | `1e-10` | soft-stop threshold on `‖β_k‖_F / ‖β_1‖_F` |
| `min_iter::Int` | `5` | consecutive sub-tol iters → soft-stop |
| `max_iter::Int` | `500` | hard cap (in case user pushes `krylovdim` very high) |
| `deflate_tol::Real` | `1e-12` | rank-revealing QR pivot threshold (relative) |
| `restrictions` | `nothing` | `Restriction` / `Vector{<:Restriction}` / `nothing` |
| `basis` | `nothing` | required when `restrictions !== nothing` |
| `retain_basis::Bool` | `false` | keep the concatenated Krylov basis (used by `rixs` and `fluorescence_yield` for the inner-Lanczos pass) |

# Returns

A `NamedTuple` with fields:

- `α::Vector{Matrix{T}}` — block-α stack; `α[k]` is `B_k × B_k`.
- `β::Vector{Matrix{T}}` — block-β stack of `length(β) == length(α) - 1`
  (or possibly fewer when soft-stop cuts the recurrence early); each
  `β[k]` is `B_{k+1} × B_k` (rectangular when deflation occurred).
- `R::Matrix{T}` — initial QR factor, `B_active × B_raw`. Always present.
- `V_basis::Union{Nothing, Vector{Matrix{T}}}` — when
  `retain_basis = true`, the per-iteration orthonormal blocks; otherwise
  `nothing`. **The first `length(α)` blocks `V_1, …, V_K` are the
  Krylov basis paired with the block-tridiagonal `T_K`** (built from
  `α` and `β`); these are the blocks that `cf_block` /
  `evaluate_on_grid` and the `R†·G₁·R` sandwich operate against. When
  the recurrence does NOT exhaust the Krylov subspace, an additional
  trailing block `V_{K+1}` is also stored — it is the next orthonormal
  block produced by iteration `K`'s QR but **never enters the cf
  representation**. Code that builds `V_int = hcat(V_basis...)` for
  the cf sandwich must take the first `length(α)` blocks only;
  including the trailing block dimension-mismatches against `T_K`.
- `n_iter::Int` — number of α blocks produced (i.e. `K`).
- `converged::Bool` — whether the soft-stop fired.

# Restrictions semantics

When `restrictions !== nothing`:

1. The starting block `X_raw` is projected first: `X_proj = P · X_raw`.
   The QR is on `X_proj`; the returned `R` relates `V_1` to `X_proj`,
   not `X_raw`. This is the standard projected-dynamics treatment —
   any portion of `X_raw` outside the projector's range is physically
   meaningless under `P · H · P`.
2. After every matvec `W = H · V_k`, the projector is reapplied
   `W ← P · W`. This is the sole place `restrictions` enters the
   recurrence; subsequent residual subtractions
   (`W -= V_k α_k`, `W -= V_{k-1} β_{k-1}†`) remain in the projected
   subspace because each `V_k` already lives there.

This is what `MOADyna.Bases.apply_restriction!` is for; the same projector
is reused by all three spectroscopy entry points when their respective
`restrictions*` kwargs are non-`nothing`.
"""
function block_lanczos(H, X_raw::AbstractMatrix;
                       krylovdim::Int      = 200,
                       reorth::Symbol      = :none,
                       tol::Real           = 1e-10,
                       min_iter::Int       = 5,
                       max_iter::Int       = 500,
                       deflate_tol::Real   = 1e-12,
                       restrictions        = nothing,
                       basis               = nothing,
                       retain_basis::Bool  = false)
    T = eltype(X_raw)
    N, B_raw = size(X_raw)

    # --- Validate inputs --------------------------------------------------
    reorth in (:none, :full) ||
        throw(ArgumentError("block_lanczos: reorth must be :none or :full; got :$reorth"))
    krylovdim ≥ 1 ||
        throw(ArgumentError("block_lanczos: krylovdim must be ≥ 1; got $krylovdim"))
    max_iter ≥ 1 ||
        throw(ArgumentError("block_lanczos: max_iter must be ≥ 1; got $max_iter"))
    if restrictions !== nothing && basis === nothing
        throw(ArgumentError("block_lanczos: restrictions require basis; pass `basis = ...`"))
    end
    size(H, 1) == size(H, 2) == N ||
        throw(DimensionMismatch("H ($(size(H, 1))×$(size(H, 2))) does not match starting block N=$N"))

    # --- Step 0: project and initial QR ----------------------------------
    X = restrictions !== nothing ?
            _project_block(X_raw, restrictions, basis) :
            Matrix{T}(X_raw)             # promote to dense Matrix{T}; may be a copy
    F        = qr(X, ColumnNorm())
    pivots   = abs.(diag(F.R))
    pivots[1] > 0 ||
        throw(ArgumentError("block_lanczos: starting block has rank zero" *
                            (restrictions === nothing ? "" :
                             " (restriction may have annihilated all columns)")))
    init_threshold = deflate_tol * pivots[1]
    B_active = something(findfirst(p -> p < init_threshold, pivots), B_raw + 1) - 1
    B_active ≥ 1 ||
        throw(ArgumentError("block_lanczos: deflated to rank zero at initial QR"))

    V_curr  = Matrix{T}(F.Q)[:, 1:B_active]                     # N × B_active, orthonormal
    R_init  = Matrix{T}(F.R[1:B_active, :])[:, invperm(F.p)]    # B_active × B_raw

    # --- Step 1: storage --------------------------------------------------
    α_blocks  = Matrix{T}[]
    β_blocks  = Matrix{T}[]
    V_blocks  = retain_basis || reorth === :full ? Matrix{T}[copy(V_curr)] : Matrix{T}[]
    V_prev    = nothing
    β_prev    = nothing                 # β_{k-1} of shape B_curr × B_prev (or nothing for k=1)
    β1_fro    = NaN
    consec    = 0
    converged = false
    n_iter    = 0

    # --- Step 2: 3-term recurrence ---------------------------------------
    upper = min(krylovdim, max_iter)
    for k in 1:upper
        # W = H · V_curr; project for restrictions.
        W = H * V_curr                                  # N × B_curr
        if restrictions !== nothing
            _apply_to_columns!(W, restrictions, basis)
        end

        # α_k = V_curr† · W; subtract.
        α_k = adjoint(V_curr) * W                       # B_curr × B_curr
        push!(α_blocks, α_k)
        W .-= V_curr * α_k

        # W -= V_prev · β_prev†   (k ≥ 2 only)
        if V_prev !== nothing && β_prev !== nothing
            W .-= V_prev * adjoint(β_prev)              # N × B_curr
        end

        # FRO: orthogonalise W against ALL stored V blocks (CGS-2 outside
        # the immediate predecessors). Memory grows O(B·K) but kills ghosts.
        if reorth === :full
            for V_old in V_blocks
                coeff = adjoint(V_old) * W
                W .-= V_old * coeff
            end
            for V_old in V_blocks    # second pass for stability
                coeff = adjoint(V_old) * W
                W .-= V_old * coeff
            end
        end

        n_iter = k

        # On the last allowed iteration we don't compute a new V_{k+1};
        # but we DO need β_k for cf evaluation. So always run the QR
        # except when residual is numerically zero.
        residual_norm = norm(W)
        if residual_norm < deflate_tol * pivots[1]
            # Krylov subspace is exhausted.
            converged = true
            break
        end

        # QR of W → V_{k+1}, β_k. Rank-reveal locally.
        F2        = qr(W, ColumnNorm())
        pivots2   = abs.(diag(F2.R))
        β_thresh  = deflate_tol * (isnan(β1_fro) ? pivots2[1] : β1_fro)
        B_next    = something(findfirst(p -> p < β_thresh, pivots2), length(pivots2) + 1) - 1

        if B_next == 0
            # Full block deflated — soft-stop. β_k effectively zero.
            converged = true
            break
        end

        V_next  = Matrix{T}(F2.Q)[:, 1:B_next]                       # N × B_next
        β_k     = Matrix{T}(F2.R[1:B_next, :])[:, invperm(F2.p)]     # B_next × B_curr
        push!(β_blocks, β_k)

        β_k_fro = norm(β_k)
        if k == 1
            β1_fro = β_k_fro
        end

        # Convergence test.
        if β_k_fro < tol * β1_fro
            consec += 1
            if consec ≥ min_iter
                converged = true
                # We've already pushed α_k and β_k; advance one more α at
                # next loop? No — the chapter says soft-stop right here.
                break
            end
        else
            consec = 0
        end

        # Advance.
        V_prev = V_curr
        β_prev = β_k
        V_curr = V_next
        if retain_basis || reorth === :full
            push!(V_blocks, copy(V_curr))
        end
    end

    return (α        = α_blocks,
            β        = β_blocks,
            R        = R_init,
            V_basis  = retain_basis ? V_blocks : nothing,
            n_iter   = n_iter,
            converged = converged)
end

# --- Internal helpers ------------------------------------------------------

function _project_block(X::AbstractMatrix, restrictions, basis)
    T = eltype(X)
    out = Matrix{T}(X)                  # owns its memory; safe to mutate
    _apply_to_columns!(out, restrictions, basis)
    return out
end

function _apply_to_columns!(M::AbstractMatrix, restrictions, basis)
    @inbounds for c in 1:size(M, 2)
        apply_restriction!(view(M, :, c), restrictions, basis)
    end
    return M
end
