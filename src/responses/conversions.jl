# =====================================================================
# MOAD.Responses — conversions: to_pole, to_grid
# =====================================================================
#
# Convention pins:
#   - to_pole: block-tridiag eigen, top-block selector,
#              poles[n] = sign * (λ_n - Eg)  (unified for sign ∈ {±1})
#   - to_grid(LanczosResponse|PoleResponse, ω; Γ): Γ override allowed
#   - to_grid(GridResponse, ω; Γ): requires ω == R.ω AND Γ == R.Γ

"""
    to_pole(L::LanczosResponse{T}) -> PoleResponse{T}

Convert a `LanczosResponse` to the exact `PoleResponse` representation by
diagonalising the block-tridiagonal matrix `T_K`.

# Algorithm (locked Convention pin)

1. Assemble the block-tridiagonal `T_K` of size `M × M` where
   `M = Σ_k size(L.α[k], 1)`.
2. Diagonalise: `λ_n, V = eigen(Hermitian(T_K))`.
3. For each eigenvector n:
   - `v_top_n = V[1:B_active, n]` — top-block selector (B_active rows).
   - `v_n = L.R' * v_top_n` — length `B_raw` vector.
   - `residues[n] = L.prefactor * v_n * v_n'` — shape `B_raw × B_raw`.
4. `poles[n] = L.sign * (λ_n - L.Eg)` — unified for `sign ∈ {±1}`.
   Retarded removal GF is `-cf_block(...; z = Eg − ω − iΓ/2)` (unified
   pole structure).
5. `a0 = zeros(T, B_raw, B_raw)`, `Eg = L.Eg`, `Γ = L.Γ`.

The `L.converged` flag is intentionally NOT carried over: `PoleResponse`
IS the diagonalised representation of whatever the Lanczos recurrence
converged (or didn't converge) to. The convergence concept lives at the
Lanczos layer; consumers needing it should query the source
`LanczosResponse` directly before calling `to_pole`.
"""
function to_pole(L::LanczosResponse{T}) where {T}
    K = length(L.α)
    # Block sizes: B_k = size(α[k], 1)
    block_sizes = [size(L.α[k], 1) for k in 1:K]
    M = sum(block_sizes)
    offsets = [sum(block_sizes[1:k-1]) for k in 1:K]  # start index - 1 for each block

    # Assemble block-tridiagonal T_K preserving the full element type T.
    # For real T this gives a real-symmetric matrix; for complex T (e.g.,
    # Hamiltonians with SOC or B-field) this gives a Hermitian matrix with
    # non-trivial imaginary blocks. Using real.(·) would discard those entries
    # and produce wrong eigenvalues.
    Tk = zeros(T, M, M)

    for k in 1:K
        r = offsets[k]+1 : offsets[k]+block_sizes[k]
        Tk[r, r] = L.α[k]
    end
    for k in 1:K-1
        # β[k] has shape (B_{k+1} × B_k)
        # off-diagonal (k, k+1) block in T_K
        r_k1 = offsets[k+1]+1 : offsets[k+1]+block_sizes[k+1]  # rows for block k+1
        c_k  = offsets[k]+1   : offsets[k]+block_sizes[k]        # cols for block k
        Tk[r_k1, c_k] = L.β[k]       # (k+1, k) block
        Tk[c_k, r_k1] = L.β[k]'      # (k, k+1) block = adjoint
    end

    # Diagonalise using Hermitian wrapper — works for both real-symmetric and
    # complex-Hermitian T_K. eigen(Hermitian(·)) always returns real eigenvalues.
    F    = eigen(Hermitian(Tk))
    λ    = F.values         # M eigenvalues, sorted ascending
    V    = F.vectors        # M × M eigenvector matrix

    B_active = size(L.R, 1)
    B_raw    = size(L.R, 2)

    poles    = Float64.(L.sign .* (λ .- L.Eg))
    residues = Vector{Matrix{T}}(undef, M)

    for n in 1:M
        v_top_n   = V[1:B_active, n]              # B_active-length vector
        v_n       = L.R' * v_top_n                # B_raw-length vector
        residues[n] = L.prefactor * (v_n * v_n')  # B_raw × B_raw
    end

    a0 = zeros(T, B_raw, B_raw)
    return PoleResponse{T}(a0, poles, residues, L.Eg, L.Γ)
end

# ---------------------------------------------------------------------------
# to_grid — materialise a response onto a frequency grid
# ---------------------------------------------------------------------------

"""
    to_grid(R::LanczosResponse, ω::AbstractVector{<:Real}; Γ = R.Γ) -> GridResponse

Materialise a `LanczosResponse` onto the frequency grid `ω`. The `Γ`
keyword allows overriding the broadening; the returned `GridResponse`
carries `Γ_override` and inherits `Eg = R.Eg`.

Each ω point is evaluated via `cf_block` (exact continued-fraction).
"""
function to_grid(L::LanczosResponse{T}, ω::AbstractVector{<:Real};
                 Γ::Real = L.Γ) where {T}
    ω_vec  = Vector{Float64}(ω)
    B_raw  = size(L.R, 2)
    nω     = length(ω_vec)

    # Preserve the source element precision: Float32 → ComplexF32,
    # Float64 → ComplexF64, ComplexF32 → ComplexF32, etc.
    Tc = T <: Complex ? T : Complex{T}

    # data has shape (B_raw, B_raw, nω)
    data   = Array{Tc, 3}(undef, B_raw, B_raw, nω)

    # Temporarily build a LanczosResponse with the (possibly overridden) Γ.
    # `converged` is preserved (Γ override is a broadening change, not a
    # convergence event), but downstream the GridResponse drops it anyway —
    # convergence semantics live at the Lanczos representation only.
    L_eval = if Float64(Γ) == L.Γ
        L
    else
        LanczosResponse{T}(L.α, L.β, L.R, L.Eg, Float64(Γ), L.sign,
                            L.prefactor, L.converged)
    end

    for (iω, ω_pt) in enumerate(ω_vec)
        data[:, :, iω] = L_eval(ω_pt)
    end

    return GridResponse{Tc, 3}(data, ω_vec, L.Eg, Float64(Γ))
end

"""
    to_grid(P::PoleResponse, ω::AbstractVector{<:Real}; Γ = P.Γ) -> GridResponse

Materialise a `PoleResponse` onto the frequency grid `ω`. The `Γ`
keyword allows overriding the broadening; the returned `GridResponse`
carries `Γ_override` and inherits `Eg = P.Eg`.
"""
function to_grid(P::PoleResponse{T}, ω::AbstractVector{<:Real};
                 Γ::Real = P.Γ) where {T}
    ω_vec  = Vector{Float64}(ω)
    nω     = length(ω_vec)
    sz     = size(P.a0)  # (n_rows, n_cols)

    # Preserve the source element precision (same pattern as to_grid(LanczosResponse)).
    Tc = T <: Complex ? T : Complex{T}

    data   = Array{Tc, 3}(undef, sz[1], sz[2], nω)

    # Build a PoleResponse with the (possibly overridden) Γ for evaluation
    P_eval = if Float64(Γ) == P.Γ
        P
    else
        PoleResponse{T}(P.a0, P.poles, P.residues, P.Eg, Float64(Γ))
    end

    for (iω, ω_pt) in enumerate(ω_vec)
        data[:, :, iω] = P_eval(ω_pt)
    end

    return GridResponse{Tc, 3}(data, ω_vec, P.Eg, Float64(Γ))
end

"""
    to_grid(G::GridResponse, ω::AbstractVector{<:Real}; Γ = G.Γ) -> GridResponse

Return `G` itself if `ω == G.ω` AND `Γ == G.Γ`. Otherwise raises
`ArgumentError` — re-evaluating from a sampled grid would require lossy
fitting (deferred to v0.5+).

This is an identity pass-through, not a re-grid or re-broaden. To evaluate
on a different ω grid or with a different Γ, first convert back to a pole
representation via `to_pole`, then call `to_grid` on the resulting
`PoleResponse`. Note that `to_pole` requires a `LanczosResponse`; if you
only have a `GridResponse`, the Lanczos source must be retained separately.
"""
function to_grid(G::GridResponse{T, N}, ω::AbstractVector{<:Real};
                 Γ::Real = G.Γ) where {T, N}
    ω_vec = Vector{Float64}(ω)
    if ω_vec != G.ω
        throw(ArgumentError(
            "to_grid(GridResponse): ω mismatch — re-evaluation from a sampled " *
            "grid requires lossy fitting, deferred to v0.5+"
        ))
    end
    if Float64(Γ) != G.Γ
        throw(ArgumentError(
            "to_grid(GridResponse): Γ mismatch (requested Γ=$(Float64(Γ)), stored " *
            "Γ=$(G.Γ)) — re-broadening a GridResponse deferred to v0.5+"
        ))
    end
    return G
end

"""
    to_grid(GF::GreensFunction, ω::AbstractVector{<:Real}; Γ = ...) -> GridResponse

Materialise a `GreensFunction` onto the frequency grid `ω` by summing
populated channels (each converted to grid form first). The `Γ` kwarg
defaults to the broadening of the first populated channel.
"""
function to_grid(GF::GreensFunction, ω::AbstractVector{<:Real}; Γ::Real = _gf_γ(GF))
    if !isnothing(GF.addition) && !isnothing(GF.removal)
        g_add = to_grid(GF.addition, ω; Γ = Γ)
        g_rem = to_grid(GF.removal,  ω; Γ = Γ)
        return g_add + g_rem
    elseif !isnothing(GF.addition)
        return to_grid(GF.addition, ω; Γ = Γ)
    else
        return to_grid(GF.removal,  ω; Γ = Γ)
    end
end

# helper to get Γ from the first populated channel
function _gf_γ(GF::GreensFunction)
    ch = isnothing(GF.addition) ? GF.removal : GF.addition
    return ch.Γ
end

