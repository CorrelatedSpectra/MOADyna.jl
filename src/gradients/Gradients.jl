# src/gradients/Gradients.jl
#
# MOAD.Gradients — the v0.3 differentiable forward model (T=0 only).
#
# Makes MOAD's `(physical parameters θ → XAS spectrum)` map differentiable in θ,
# forward-mode + analytic-resolvent — NOT reverse-AD through the eigensolver. The
# whole layer is built on an AFFINE Hamiltonian model
#
#     H(θ) = Σ_i c_i(θ) · M_i ,        dH(θ̇) = Σ_i (∂c_i/∂θ · θ̇)_i · M_i ,
#
# with term matrices M_i assembled ONCE and combined numerically as θ varies; the
# symbolic builders (coulomb, Akm, …) are never re-run inside the differentiated path
# (their `iszero`/`chop` branches are non-smooth and change sparsity).
#
# Contents (built and finite-difference-validated in seven steps):
#   • Coefficient maps + `AffineModel` (`hamiltonian`/`dhamiltonian`); `zsa_onsite`.
#   • Ground-state derivatives: `groundstate` (non-degenerate gate), `grad_E0`
#     (Hellmann–Feynman), `sternheimer_dψ0` (projected Sternheimer).
#   • Resolvent JVP: `response_C`/`response_jvp` (fixed source) and the full coupled
#     `xas_response_C`/`xas_response_jvp` (θ in both H_g and H_f).
#   • Degeneracy/sector awareness: `groundstate_manifold`, `manifold_gate`,
#     `lowlying_spectrum`, and the opt-in `classify_groundstate` (the ONLY symbol that
#     reaches into the physics modules; never on a differentiated path).
#   • Spectrum VJP/pullback: `XASGradientModel`, `spectrum`, `spectrum_and_jvp`,
#     `jacobian`, `spectrum_with_pullback` (S = −Im C/π, cotangent W = −iλ/π); plus a
#     degeneracy-clustered spectral measure (`freeze_windows`, `spectral_clusters`).
#   • `fit_spectrum`: a deterministic direct-fit baseline (the `Optim` weakdep
#     extension `MOADOptimExt`).
#
# Scope: T=0 only (finite-T dropped by design). Parameter INFERENCE (observation
# model, NPE/HMC/SBC/OOD) lives in a separate sibling package that depends on MOAD;
# MOAD never hard-depends on the ML stack.

module Gradients

using LinearAlgebra
using SparseArrays
using KrylovKit: linsolve
using ..PointGroups: classify_state   # spatial irrep labeling (build-step 5 bridge)

export AbstractCoefficientMap, AffineMap, coeffs, coeff_jacobian
export AffineModel, hamiltonian, dhamiltonian, n_params, n_terms
export zsa_onsite
export groundstate, grad_E0, sternheimer_dψ0
export response_C, response_jvp
export xas_response_C, xas_response_jvp
export GroundState, SectorReport, GroundStateReport
export groundstate_manifold, manifold_gate, lowlying_spectrum, classify_groundstate
export XASGradientModel, spectrum, spectrum_and_jvp, jacobian, spectrum_with_pullback
export SpectralCluster, FrozenWindow, freeze_windows, spectral_clusters, spectral_clusters_jvp
export fit_spectrum

# --- Coefficient maps  θ ↦ (c_1, …, c_nterms) ------------------------------
#
# Interface: `coeffs(map, θ)` and `coeff_jacobian(map, θ)`. For the v0.3 anchor
# the map is affine (constant Jacobian), but the interface is kept general so a
# later nonlinear coefficient map is simply a different concrete subtype.

abstract type AbstractCoefficientMap end

"""
    coeffs(map::AbstractCoefficientMap, θ::AbstractVector) -> Vector

Term coefficients `c_i(θ)`, one per preassembled term matrix.
"""
function coeffs end

"""
    coeff_jacobian(map::AbstractCoefficientMap, θ::AbstractVector) -> Matrix

Jacobian `∂c_i/∂θ_k` (size `nterms × nparams`), evaluated at `θ`.
"""
function coeff_jacobian end

"""
    AffineMap(c0::AbstractVector, J::AbstractMatrix)

Affine coefficient map `c(θ) = c0 + J·θ` with constant Jacobian `J`
(`nterms × nparams`).
"""
struct AffineMap <: AbstractCoefficientMap
    c0::Vector{Float64}
    J::Matrix{Float64}
    function AffineMap(c0::AbstractVector, J::AbstractMatrix)
        size(J, 1) == length(c0) || throw(DimensionMismatch(
            "AffineMap: size(J,1)=$(size(J,1)) must equal length(c0)=$(length(c0))."))
        return new(collect(float.(c0)), collect(float.(J)))
    end
end

coeffs(m::AffineMap, θ::AbstractVector) = m.c0 .+ m.J * θ
coeff_jacobian(m::AffineMap, θ::AbstractVector) = m.J

# --- Affine Hamiltonian model ----------------------------------------------

"""
    AffineModel(terms, cmap, names)

Preassembled-term Hamiltonian model. `terms[i]` are fixed matrices `M_i`
(assembled once; never rebuilt as `θ` varies), `cmap::AbstractCoefficientMap`
maps `θ` to the per-term coefficients, and `names` are the parameter symbols
(`length(names) == nparams`).

    H(θ)   = Σ_i coeffs(cmap, θ)[i] · M_i
    dH(θ̇) = Σ_i (coeff_jacobian(cmap, θ)·θ̇)[i] · M_i
"""
struct AffineModel{M<:AbstractMatrix, C<:AbstractCoefficientMap}
    terms::Vector{M}
    cmap::C
    names::Vector{Symbol}
    function AffineModel(terms::Vector{M}, cmap::C,
                         names::Vector{Symbol}) where {M<:AbstractMatrix,
                                                       C<:AbstractCoefficientMap}
        isempty(terms) && throw(ArgumentError("AffineModel: `terms` is empty."))
        sz = size(first(terms))
        for (i, t) in enumerate(terms)
            size(t) == sz || throw(DimensionMismatch(
                "AffineModel: term $i has size $(size(t)) ≠ $(sz); all terms " *
                "must be assembled on the same basis."))
        end
        np = length(names)
        c = coeffs(cmap, zeros(np))
        length(c) == length(terms) || throw(DimensionMismatch(
            "AffineModel: coefficient map returns $(length(c)) coefficients for " *
            "$(length(terms)) terms."))
        Jc = coeff_jacobian(cmap, zeros(np))
        size(Jc) == (length(terms), np) || throw(DimensionMismatch(
            "AffineModel: coeff_jacobian has size $(size(Jc)); expected " *
            "($(length(terms)), $(np)) = (nterms, nparams)."))
        return new{M,C}(terms, cmap, names)
    end
end

n_params(model::AffineModel) = length(model.names)
n_terms(model::AffineModel) = length(model.terms)

"""
    hamiltonian(model::AffineModel, θ::AbstractVector) -> AbstractMatrix

Assemble `H(θ) = Σ_i c_i(θ) M_i` by numeric combination of the preassembled terms.
"""
function hamiltonian(model::AffineModel, θ::AbstractVector)
    length(θ) == n_params(model) || throw(DimensionMismatch(
        "hamiltonian: length(θ)=$(length(θ)) ≠ nparams=$(n_params(model))."))
    c = coeffs(model.cmap, θ)
    H = c[1] * model.terms[1]
    for i in 2:length(model.terms)
        H += c[i] * model.terms[i]
    end
    return H
end

"""
    dhamiltonian(model::AffineModel, θ::AbstractVector, θ̇::AbstractVector) -> AbstractMatrix

Directional derivative `dH = Σ_i (J(θ)·θ̇)_i M_i` along `θ̇`. (`θ` is used only
through `coeff_jacobian`; for the affine map the result is θ-independent.)
"""
function dhamiltonian(model::AffineModel, θ::AbstractVector, θ̇::AbstractVector)
    length(θ) == n_params(model) || throw(DimensionMismatch(
        "dhamiltonian: length(θ)=$(length(θ)) ≠ nparams=$(n_params(model))."))
    length(θ̇) == n_params(model) || throw(DimensionMismatch(
        "dhamiltonian: length(θ̇)=$(length(θ̇)) ≠ nparams=$(n_params(model))."))
    dc = coeff_jacobian(model.cmap, θ) * θ̇
    dH = dc[1] * model.terms[1]
    for i in 2:length(model.terms)
        dH += dc[i] * model.terms[i]
    end
    return dH
end

# --- ZSA onsite-energy Jacobian --------------------------------------------
#
# Mirrors `Shells.onsite_energies` (src/shells/onsite_energies.jl) but ALSO
# returns the Jacobian ∂ε/∂(anchor energies) — the constant linear map that
# makes the charge-transfer Δ (an anchor energy) an affine knob. Adds the
# SVD rank/conditioning guard the design requires (`onsite_energies` checks
# the anchor count but not the rank of the ZSA matrix).

"""
    zsa_onsite(target_shells, anchors; U=NamedTuple(), pairs=(), rank_rtol=1e-8)
        -> (ε, dε_dEanchor, A)

Solve the ZSA onsite-energy system `ε = A \\ b` for the shells in
`target_shells` and return `ε` (Vector, ordered as `target_shells`), the
constant Jacobian `dε_dEanchor` (`n_shells × n_anchors`, equal to `A⁺` since `b`
depends on each anchor energy with unit coefficient), and the ZSA matrix `A`.

`anchors` is a vector of `config::NamedTuple => energy::Real`. The matrix `A` is
built from the integer configuration occupations only (θ-independent), so
`dε_dEanchor` is a constant; differentiating ε w.r.t. an anchor energy (e.g. Δ)
is just the corresponding column.

Guards that `A` has full column rank via its singular values
(`σ_min > rank_rtol·σ_max`); a rank-deficient anchor set is rejected, since it
yields a deterministic but physically ill-identified onsite map.
"""
function zsa_onsite(target_shells::AbstractVector{Symbol},
                    anchors::AbstractVector;
                    U = NamedTuple(), pairs = (), rank_rtol::Real = 1e-8)
    n_shells  = length(target_shells)
    n_anchors = length(anchors)
    n_anchors ≥ n_shells || throw(ArgumentError(
        "zsa_onsite: under-determined system ($(n_anchors) anchors, " *
        "$(n_shells) shells)."))
    (rank_rtol > 0 && isfinite(rank_rtol)) || throw(ArgumentError(
        "zsa_onsite: rank_rtol must be positive and finite; got $(rank_rtol)."))
    target_set = Set(target_shells)

    A = zeros(Float64, n_anchors, n_shells)
    b = zeros(Float64, n_anchors)
    for (i, anchor) in enumerate(anchors)
        config, E_anchor = anchor
        for k in keys(config)
            k in target_set || throw(ArgumentError(
                "zsa_onsite: anchor #$(i) mentions shell `$(k)` which is not in " *
                "target_shells $(target_shells)."))
        end
        resid = float(E_anchor)
        for (j, s) in enumerate(target_shells)
            occ = float(get(config, s, 0))
            A[i, j] = occ
            resid -= occ * (occ - 1) / 2 * float(get(U, s, 0))
        end
        for pair in pairs
            (sa, sb), Up = pair
            resid -= float(get(config, sa, 0)) * float(get(config, sb, 0)) * float(Up)
        end
        b[i] = resid
    end

    # One SVD drives both the rank guard and the pseudoinverse, so the
    # acceptance tolerance and A⁺ are always consistent.
    F = svd(A)                  # thin SVD: A = U·Diagonal(S)·V'
    if isempty(F.S) || minimum(F.S) ≤ rank_rtol * maximum(F.S)
        throw(ArgumentError(
            "zsa_onsite: ZSA matrix A is rank-deficient / ill-conditioned " *
            "(σ_min ≤ $(rank_rtol)·σ_max); the anchor set does not identify the " *
            "onsite energies. Add independent anchors."))
    end

    Apinv = F.V * Diagonal(inv.(F.S)) * F.U'    # A⁺ (= A⁻¹ for square full-rank)
    ε = Apinv * b
    dε_dEanchor = Apinv         # ∂b/∂E_anchor_i = e_i ⇒ ∂ε/∂E_anchor = A⁺
    return ε, dε_dEanchor, A
end

# --- Ground-state derivatives (build-step 2) -------------------------------
#
# Non-degenerate ground state. dE0 via Hellmann–Feynman; dψ0 via a projected
# (shifted full-rank) Sternheimer solve. Operates directly on the assembled
# matrix (AffineModel carries no basis). The projector uses the conjugate
# transpose `ψ0'`, so it is correct for complex ψ0 (SOC / complex Hamiltonians).

"""
    groundstate(model::AffineModel, θ; gap_tol=1e-6, dense_max=4096) -> (E0, ψ0, gap)

Lowest eigenpair of `H(θ)` with a non-degeneracy gate. Dense path
`eigen(Hermitian(Matrix(H)))` — the fixtures are small and ≥2 eigenvalues are
needed to measure the gap. Errors if `gap = E1 − E0 ≤ gap_tol`: a
(near-)degenerate ground state makes `dψ0` ill-conditioned / undefined. For a
genuinely degenerate ground manifold use [`groundstate_manifold`](@ref) (build-step
5), which returns the orthonormal manifold basis and gates on the external gap.
`ψ0` is normalized.
"""
function groundstate(model::AffineModel, θ::AbstractVector;
                     gap_tol::Real = 1e-6, dense_max::Integer = 4096)
    (gap_tol > 0 && isfinite(gap_tol)) || throw(ArgumentError(
        "groundstate: gap_tol must be positive and finite; got $(gap_tol)."))
    H = hamiltonian(model, θ)
    d = size(H, 1)
    d ≥ 2 || throw(ArgumentError(
        "groundstate: Hilbert dimension $(d) < 2; cannot measure a gap."))
    d ≤ dense_max || throw(ArgumentError(
        "groundstate: dimension $(d) exceeds dense_max=$(dense_max). This is the " *
        "dense path (full eigen); a Krylov/eigsolve path for large bases is a " *
        "later build-step. Raise dense_max to override deliberately."))
    F = eigen(Hermitian(Matrix(H)))
    E0, E1 = F.values[1], F.values[2]
    gap = E1 - E0
    gap > gap_tol || throw(ArgumentError(
        "groundstate: (near-)degenerate ground state (gap = $(gap) ≤ " *
        "gap_tol = $(gap_tol)); dψ0 is ill-conditioned. Use a degeneracy-aware " *
        "treatment (build-step 5)."))
    ψ0 = F.vectors[:, 1]
    return E0, ψ0 / norm(ψ0), gap
end

"""
    grad_E0(model::AffineModel, θ, ψ0) -> Vector{Float64}

Hellmann–Feynman energy gradient `∂E0/∂θ_k = Re⟨ψ0| ∂H/∂θ_k |ψ0⟩` for the
non-degenerate ground state `ψ0` at `θ`.
"""
function grad_E0(model::AffineModel, θ::AbstractVector, ψ0::AbstractVector)
    np = n_params(model)
    g = zeros(Float64, np)
    for k in 1:np
        ek = zeros(np); ek[k] = 1.0
        g[k] = real(dot(ψ0, dhamiltonian(model, θ, ek) * ψ0))
    end
    return g
end

"""
    sternheimer_dψ0(model::AffineModel, θ, E0, ψ0, θ̇; rtol=1e-12) -> dψ0

First-order ground-state response along `θ̇` from the projected Sternheimer
equation `(H − E0) dψ0 = −P · (∂H/∂θ·θ̇) · ψ0`, `P = I − ψ0ψ0†`,
`⟨ψ0|dψ0⟩ = 0`. Solved on the shifted Hermitian-positive-definite operator
`A = (H − E0 I) + ψ0ψ0†` (matrix-free), whose solution coincides with the
perpendicular Sternheimer response.
"""
function sternheimer_dψ0(model::AffineModel, θ::AbstractVector, E0::Real,
                         ψ0::AbstractVector, θ̇::AbstractVector; rtol::Real = 1e-12)
    (rtol > 0 && isfinite(rtol)) || throw(ArgumentError(
        "sternheimer_dψ0: rtol must be positive and finite; got $(rtol)."))
    H = hamiltonian(model, θ)
    proj(v) = v .- ψ0 .* (ψ0' * v)                  # P v = v − ψ0 ⟨ψ0|v⟩
    rhs = -proj(dhamiltonian(model, θ, θ̇) * ψ0)
    Aop(v) = H * v .- E0 .* v .+ ψ0 .* (ψ0' * v)    # (H − E0) v + ψ0 ⟨ψ0|v⟩
    x, info = linsolve(Aop, rhs; isposdef = true, rtol = rtol)
    info.converged == 1 || throw(ErrorException(
        "sternheimer_dψ0: linsolve did not converge (normres = $(info.normres)). " *
        "A gradient layer must not continue with an unconverged solve."))
    return proj(x)                                  # clean any roundoff ψ0-component
end

# --- Resolvent JVP (build-step 3) ------------------------------------------
#
# Scalar response C(ω) = X† G(ω) X and its directional derivative, at FIXED
# source X and FIXED reference E0 (the moving E0(θ) and X(θ) couplings are
# build-step 4). G(ω) = [(ω + E0 + iΓ/2)·I − H(θ)]⁻¹, so with z fixed,
# dG = G·dH·G and dC = X† G dH G X — evaluated by two shifted solves:
# R = G X, dR = G (dH R), dC = X† dR.

function _check_resolvent_inputs(model::AffineModel, X::AbstractVector,
                                 ω_grid::AbstractVector, Γ::Real, rtol::Real)
    (Γ > 0 && isfinite(Γ)) || throw(ArgumentError(
        "response: Γ must be positive and finite; got $(Γ)."))
    (rtol > 0 && isfinite(rtol)) || throw(ArgumentError(
        "response: rtol must be positive and finite; got $(rtol)."))
    all(isfinite, ω_grid) || throw(ArgumentError("response: ω_grid must be finite."))
    N = size(first(model.terms), 1)
    length(X) == N || throw(DimensionMismatch(
        "response: length(X)=$(length(X)) ≠ Hilbert dimension $(N)."))
    return nothing
end

"""
    _resolvent_solve(H, z, v; rtol=1e-12) -> x

Solve `(z·I − H) x = v` for a complex shift `z` (the operator is non-Hermitian)
via GMRES. Throws on non-convergence.
"""
function _resolvent_solve(H::AbstractMatrix, z::Number, v::AbstractVector;
                          rtol::Real = 1e-12)
    Aop(x) = z .* x .- H * x
    x, info = linsolve(Aop, v; rtol = rtol)
    info.converged == 1 || throw(ErrorException(
        "_resolvent_solve: GMRES did not converge at z=$(z) " *
        "(normres = $(info.normres))."))
    return x
end

"""
    response_C(model::AffineModel, θ, X, ω_grid; E0, Γ, rtol=1e-12) -> Vector{ComplexF64}

Scalar response `C(ω) = X† G(ω) X` at fixed source `X` and fixed reference `E0`,
with `G(ω) = [(ω + E0 + iΓ/2)·I − H(θ)]⁻¹`. (Build-step 3: `X` and `E0` fixed;
the moving-state couplings are build-step 4.)
"""
function response_C(model::AffineModel, θ::AbstractVector, X::AbstractVector,
                    ω_grid::AbstractVector; E0::Real, Γ::Real, rtol::Real = 1e-12)
    _check_resolvent_inputs(model, X, ω_grid, Γ, rtol)
    isfinite(E0) || throw(ArgumentError("response_C: E0 must be finite; got $(E0)."))
    H = hamiltonian(model, θ)
    C = Vector{ComplexF64}(undef, length(ω_grid))
    for (i, ω) in enumerate(ω_grid)
        z = ω + E0 + im * Γ / 2
        R = _resolvent_solve(H, z, X; rtol = rtol)
        C[i] = dot(X, R)                 # X† R
    end
    return C
end

"""
    response_jvp(model::AffineModel, θ, θ̇, X, ω_grid; E0, Γ, rtol=1e-12) -> Vector{ComplexF64}

Directional derivative `dC(ω) = X† G(ω) dH G(ω) X` along `θ̇`, at fixed `X` and
`E0`. Two shifted solves per `ω`: `R = G X`, `dR = G (dH R)`, `dC = X† dR`.
"""
function response_jvp(model::AffineModel, θ::AbstractVector, θ̇::AbstractVector,
                      X::AbstractVector, ω_grid::AbstractVector;
                      E0::Real, Γ::Real, rtol::Real = 1e-12)
    _check_resolvent_inputs(model, X, ω_grid, Γ, rtol)
    isfinite(E0) || throw(ArgumentError("response_jvp: E0 must be finite; got $(E0)."))
    H  = hamiltonian(model, θ)
    dH = dhamiltonian(model, θ, θ̇)
    dC = Vector{ComplexF64}(undef, length(ω_grid))
    for (i, ω) in enumerate(ω_grid)
        z  = ω + E0 + im * Γ / 2
        R  = _resolvent_solve(H, z, X; rtol = rtol)
        dR = _resolvent_solve(H, z, dH * R; rtol = rtol)
        dC[i] = dot(X, dR)               # X† G dH G X
    end
    return dC
end

# --- Full XAS JVP: couple H_g and H_f (build-step 4) -----------------------
#
# C_AB(ω) = X_A† G X_B, X_A = A·E·ψ0, X_B = B·E·ψ0, G = [(ω+E0+iΓ/2)I − H_f]⁻¹,
# with (E0, ψ0) the ground state of H_g(θ). θ enters BOTH H_g and H_f, so the JVP
# threads dE0 (Hellmann–Feynman), dψ0 (Sternheimer) → dX, and dH_f:
#   dC = dX_A†·R + L†·dX_B + L†(dH_f − dE0·I)·R,   R = G X_B,  L = G† X_A.
# A, B are tuples of N_f×N_f transition operators (polarisation components); E is a
# fixed N_f×N_g embedding. GS-only gate — the resolvent handles final-state crossings.

function _check_xas_inputs(g_model::AffineModel, f_model::AffineModel, A, B,
                           E::AbstractMatrix, ω_grid::AbstractVector,
                           Γ::Real, rtol::Real, embedding_tol::Real)
    g_model.names == f_model.names || throw(ArgumentError(
        "xas response: g_model and f_model must share the parameter schema " *
        "(names AND order); got $(g_model.names) vs $(f_model.names)."))
    Nf = size(first(f_model.terms), 1)
    Ng = size(first(g_model.terms), 1)
    size(E) == (Nf, Ng) || throw(DimensionMismatch(
        "xas response: E must be N_f×N_g = $(Nf)×$(Ng); got $(size(E))."))
    (embedding_tol > 0 && isfinite(embedding_tol)) || throw(ArgumentError(
        "xas response: embedding_tol must be positive and finite; got $(embedding_tol)."))
    norm(E' * E - I) ≤ embedding_tol || throw(ArgumentError(
        "xas response: E must be an isometry (‖E'E − I‖ = $(norm(E' * E - I)) > " *
        "embedding_tol = $(embedding_tol)); pass a column-orthonormal embedding."))
    for (lbl, ops) in (("A", A), ("B", B))
        isempty(ops) && throw(ArgumentError("xas response: $(lbl) is empty."))
        for (i, op) in enumerate(ops)
            size(op) == (Nf, Nf) || throw(DimensionMismatch(
                "xas response: $(lbl)[$(i)] must be N_f×N_f = $(Nf)×$(Nf); got $(size(op))."))
        end
    end
    (Γ > 0 && isfinite(Γ)) || throw(ArgumentError(
        "xas response: Γ must be positive and finite; got $(Γ)."))
    (rtol > 0 && isfinite(rtol)) || throw(ArgumentError(
        "xas response: rtol must be positive and finite; got $(rtol)."))
    all(isfinite, ω_grid) || throw(ArgumentError("xas response: ω_grid must be finite."))
    return nothing
end

# Source block X = hcat(op·v for op in ops), shape length(v) × length(ops).
function _source_block(ops, v::AbstractVector)
    X = Matrix{ComplexF64}(undef, length(v), length(ops))
    for (j, op) in enumerate(ops)
        X[:, j] = op * v
    end
    return X
end

# Column-by-column resolvent solve for a matrix RHS (KrylovKit linsolve is per-vector).
function _resolvent_solve_block(H::AbstractMatrix, z::Number, V::AbstractMatrix;
                                rtol::Real = 1e-12)
    X = Matrix{ComplexF64}(undef, size(V))
    for j in axes(V, 2)
        X[:, j] = _resolvent_solve(H, z, V[:, j]; rtol = rtol)
    end
    return X
end

"""
    xas_response_C(g_model, f_model, A, B, E, θ, ω_grid; Γ, gap_tol=1e-6, rtol=1e-12)
        -> Array{ComplexF64,3}   (nA × nB × nω)

Coupled XAS response `C_AB(ω) = X_A† G(ω) X_B`, with `X_A = A·E·ψ0`,
`X_B = B·E·ψ0`, `G = [(ω+E0+iΓ/2)I − H_f(θ)]⁻¹`, and `(E0,ψ0)` the ground state of
`H_g(θ)`. `A`,`B` are tuples of `N_f×N_f` transition operators; `E` is the
`N_f×N_g` embedding. GS-only gate (via `groundstate`); the resolvent handles
final-state crossings.
"""
function xas_response_C(g_model::AffineModel, f_model::AffineModel, A, B,
                        E::AbstractMatrix, θ::AbstractVector, ω_grid::AbstractVector;
                        Γ::Real, gap_tol::Real = 1e-6, rtol::Real = 1e-12,
                        embedding_tol::Real = 1e-8)
    _check_xas_inputs(g_model, f_model, A, B, E, ω_grid, Γ, rtol, embedding_tol)
    E0, ψ0, _ = groundstate(g_model, θ; gap_tol = gap_tol)
    Eψ0 = E * ψ0
    X_A = _source_block(A, Eψ0)
    X_B = _source_block(B, Eψ0)
    Hf = hamiltonian(f_model, θ)
    C = Array{ComplexF64,3}(undef, size(X_A, 2), size(X_B, 2), length(ω_grid))
    for (i, ω) in enumerate(ω_grid)
        z = ω + E0 + im * Γ / 2
        R = _resolvent_solve_block(Hf, z, X_B; rtol = rtol)
        C[:, :, i] = X_A' * R
    end
    return C
end

"""
    xas_response_jvp(g_model, f_model, A, B, E, θ, θ̇, ω_grid; Γ, gap_tol=1e-6, rtol=1e-12)
        -> Array{ComplexF64,3}   (nA × nB × nω)

Directional derivative of the coupled XAS response along `θ̇`, threading the
ground-state energy/state response and the final-Hamiltonian derivative:
`dC = dX_A†·R + L†·dX_B + L†(dH_f − dE0·I)·R`, `R = G X_B`, `L = G† X_A` (left
solve at the conjugate shift). `dE0` is Hellmann–Feynman, `dψ0` the projected
Sternheimer response, `dX = A/B·E·dψ0` (transition operators / embedding fixed).
"""
function xas_response_jvp(g_model::AffineModel, f_model::AffineModel, A, B,
                          E::AbstractMatrix, θ::AbstractVector, θ̇::AbstractVector,
                          ω_grid::AbstractVector;
                          Γ::Real, gap_tol::Real = 1e-6, rtol::Real = 1e-12,
                          embedding_tol::Real = 1e-8)
    _check_xas_inputs(g_model, f_model, A, B, E, ω_grid, Γ, rtol, embedding_tol)
    E0, ψ0, _ = groundstate(g_model, θ; gap_tol = gap_tol)
    dE0 = real(dot(ψ0, dhamiltonian(g_model, θ, θ̇) * ψ0))      # Hellmann–Feynman
    dψ0 = sternheimer_dψ0(g_model, θ, E0, ψ0, θ̇; rtol = rtol)
    Eψ0, Edψ0 = E * ψ0, E * dψ0
    X_A,  X_B  = _source_block(A, Eψ0),  _source_block(B, Eψ0)
    dX_A, dX_B = _source_block(A, Edψ0), _source_block(B, Edψ0)
    Hf  = hamiltonian(f_model, θ)
    dHf = dhamiltonian(f_model, θ, θ̇)
    dC = Array{ComplexF64,3}(undef, size(X_A, 2), size(X_B, 2), length(ω_grid))
    for (i, ω) in enumerate(ω_grid)
        z = ω + E0 + im * Γ / 2
        R = _resolvent_solve_block(Hf, z, X_B; rtol = rtol)         # G X_B
        L = _resolvent_solve_block(Hf, conj(z), X_A; rtol = rtol)   # G† X_A (conjugate shift)
        dC[:, :, i] = dX_A' * R + L' * dX_B + L' * (dHf * R - dE0 * R)
    end
    return dC
end

# --- Build-step 5: degeneracy gate + sector-aware ground state -------------
#
# Make the T=0 ground-state step degeneracy-aware (a g-fold manifold rather than a
# single state) and, via an opt-in physics bridge, sector-aware (spatial irrep +
# ⟨S²⟩). Everything a differentiated path touches (`groundstate_manifold`,
# `manifold_gate`) is pure linear algebra — no physics-module deps; only the opt-in
# `classify_groundstate` reaches into `classify_state`, and it is NEVER on a
# differentiated path. The manifold-summed *derivative* is build-step 6.

"""
    GroundState

Degeneracy-resolved T=0 ground manifold of `H(θ)`.

# Fields
- `E0::Float64` — ground energy (the `g`-fold-degenerate value).
- `U::Matrix{ComplexF64}` — `d × g` orthonormal basis of the ground manifold; the
  columns span it but the intra-manifold gauge is arbitrary.
- `g::Int` — manifold degeneracy (`size(U, 2)`).
- `gap_external::Float64` — `E_g − E0`, the gap to the first state *above* the
  manifold (zero-based: the manifold occupies `E0 … E_{g−1}`).
"""
struct GroundState
    E0::Float64
    U::Matrix{ComplexF64}
    g::Int
    gap_external::Float64
end

"""
    SectorReport

One labeled energy level (a degenerate cluster) with its physics labels.

# Fields
- `energy::Float64` — cluster energy.
- `gap_above_E0::Float64` — `energy − E0` (`0.0` for the ground cluster).
- `degeneracy::Int` — cluster size.
- `irrep::Symbol` — dominant spatial Mulliken irrep from the **cluster-averaged**
  per-irrep weights (`Tr(V'·P_irrep·V)/deg`), gauge-invariant within the cluster.
- `irrep_weight::Float64` — **purity diagnostic**: the averaged dominant-irrep
  weight. A low value flags a reducible / accidental degeneracy or a non-symmetry
  Hamiltonian — treat the cluster as *mixed*; do not read `irrep` alone as a clean
  sector label when this is small.
- `S2::Float64` — ⟨S²⟩ averaged over the cluster.
- `spin::Float64` — `S` from `S(S+1)=⟨S²⟩`, rounded to the nearest half-integer.
- `multiplicity::Int` — `2S+1` (rounded).
"""
struct SectorReport
    energy::Float64
    gap_above_E0::Float64
    degeneracy::Int
    irrep::Symbol
    irrep_weight::Float64
    S2::Float64
    spin::Float64
    multiplicity::Int
end

"""
    GroundStateReport

Output of [`classify_groundstate`](@ref): the ground sector plus competing low-lying
sectors within an energy window.

# Fields
- `ground::SectorReport` — the ground manifold's labels.
- `competitors::Vector{SectorReport}` — labeled clusters with energy in
  `(E0, E0 + window]`, ascending in energy.
- `gs::GroundState` — the underlying matrix-level ground manifold.
"""
struct GroundStateReport
    ground::SectorReport
    competitors::Vector{SectorReport}
    gs::GroundState
end

# Dense eigen-decomposition grouped into degenerate clusters. Clustering is by
# distance from each cluster's FIRST eigenvalue (`λ − λ_start ≤ degen_tol`), NOT by
# chained consecutive gaps — so a cluster can never grow wider than `degen_tol`.
# Returns `(values, vectors, ranges)` with `ranges` a partition of `1:d` into
# ascending-energy clusters. Shared by `groundstate_manifold`, `lowlying_spectrum`,
# and `classify_groundstate` so the three never disagree on cluster boundaries.
function _eigen_clusters(H::AbstractMatrix; degen_tol::Real, dense_max::Integer)
    (degen_tol > 0 && isfinite(degen_tol)) || throw(ArgumentError(
        "_eigen_clusters: degen_tol must be positive and finite; got $(degen_tol)."))
    d = size(H, 1)
    d ≥ 1 || throw(ArgumentError("_eigen_clusters: empty matrix."))
    d ≤ dense_max || throw(ArgumentError(
        "_eigen_clusters: dimension $(d) exceeds dense_max=$(dense_max); the dense " *
        "path (full eigen) is intended for the small step-5 fixtures. Raise dense_max " *
        "to override deliberately."))
    F = eigen(Hermitian(Matrix(H)))
    λ = F.values
    ranges = UnitRange{Int}[]
    i = 1
    while i ≤ d
        j = i
        while j < d && λ[j + 1] - λ[i] ≤ degen_tol   # distance from the cluster start
            j += 1
        end
        push!(ranges, i:j)
        i = j + 1
    end
    return F.values, F.vectors, ranges
end

"""
    groundstate_manifold(model, θ; gap_tol=1e-6, degen_tol=1e-8, dense_max=4096)
        -> GroundState

Lowest eigenvalue of `H(θ)` and the orthonormal basis `U` of its (possibly
degenerate) ground manifold. Eigenvalues within `degen_tol` of the minimum form the
manifold (`g` of them). Errors if the **external** gap `E_g − E0 ≤ gap_tol` — the
manifold is not separated from the rest of the spectrum (a level-crossing seam or a
too-tight `degen_tol`); requires `degen_tol < gap_tol`. Dense path
(`eigen(Hermitian(Matrix(H)))`).

Note an *exact* two-sector crossing that is externally isolated returns `g == 2`
(healthy `gap_external` to the third state) — it is the non-degenerate
[`groundstate`](@ref) wrapper and [`manifold_gate`](@ref) that reject the crossing,
not this gate.
"""
function groundstate_manifold(model::AffineModel, θ::AbstractVector;
                              gap_tol::Real = 1e-6, degen_tol::Real = 1e-8,
                              dense_max::Integer = 4096)
    (gap_tol > 0 && isfinite(gap_tol)) || throw(ArgumentError(
        "groundstate_manifold: gap_tol must be positive and finite; got $(gap_tol)."))
    (degen_tol > 0 && isfinite(degen_tol)) || throw(ArgumentError(
        "groundstate_manifold: degen_tol must be positive and finite; got $(degen_tol)."))
    degen_tol < gap_tol || throw(ArgumentError(
        "groundstate_manifold: require degen_tol < gap_tol (got degen_tol=$(degen_tol), " *
        "gap_tol=$(gap_tol)); otherwise the clustering and the external-gap gate overlap."))
    H = hamiltonian(model, θ)
    d = size(H, 1)
    d ≥ 2 || throw(ArgumentError(
        "groundstate_manifold: Hilbert dimension $(d) < 2; cannot measure an external gap."))
    λ, V, ranges = _eigen_clusters(H; degen_tol = degen_tol, dense_max = dense_max)
    g = length(ranges[1])
    E0 = λ[1]
    g < d || throw(ArgumentError(
        "groundstate_manifold: the whole spectrum (d=$(d)) is degenerate within " *
        "degen_tol=$(degen_tol); no external state to gate against."))
    gap_external = λ[g + 1] - E0
    gap_external > gap_tol || throw(ArgumentError(
        "groundstate_manifold: ground manifold (g=$(g)) is not externally isolated " *
        "(E_g − E0 = $(gap_external) ≤ gap_tol=$(gap_tol)); the cluster is unresolved " *
        "(a level-crossing seam or a too-tight degen_tol)."))
    return GroundState(E0, Matrix{ComplexF64}(V[:, ranges[1]]), g, gap_external)
end

"""
    manifold_gate(model, θ, U, θ̇; split_atol=1e-8, split_rtol=1e-6, orth_tol=1e-8)
        -> (ok::Bool, α::Float64, split::Float64)

Test whether the direction `θ̇` preserves the ground manifold `U` to first order, i.e.
whether `B = U'·dH(θ̇)·U` is a scalar multiple of the identity. Returns `α =
real(tr(B))/g`, `split = opnorm(B − α·I)`, and `ok` iff
`split ≤ split_atol + split_rtol·max(opnorm(B), |α|)` (absolute + relative; no hidden
energy unit, so it handles `α≈0` pure-splitting cleanly). A direction with `!ok`
*splits* the manifold and is non-differentiable for the single-state / equal-weight
spectrum: the caller must refuse to return a gradient along it.
"""
function manifold_gate(model::AffineModel, θ::AbstractVector, U::AbstractMatrix,
                       θ̇::AbstractVector; split_atol::Real = 1e-8,
                       split_rtol::Real = 1e-6, orth_tol::Real = 1e-8)
    (split_atol ≥ 0 && isfinite(split_atol)) || throw(ArgumentError(
        "manifold_gate: split_atol must be nonnegative and finite; got $(split_atol)."))
    (split_rtol ≥ 0 && isfinite(split_rtol)) || throw(ArgumentError(
        "manifold_gate: split_rtol must be nonnegative and finite; got $(split_rtol)."))
    (orth_tol > 0 && isfinite(orth_tol)) || throw(ArgumentError(
        "manifold_gate: orth_tol must be positive and finite; got $(orth_tol)."))
    N = size(first(model.terms), 1)
    size(U, 1) == N || throw(DimensionMismatch(
        "manifold_gate: size(U,1)=$(size(U, 1)) ≠ Hilbert dimension $(N)."))
    g = size(U, 2)
    g ≥ 1 || throw(ArgumentError("manifold_gate: U has no columns."))
    norm(U' * U - I) ≤ orth_tol || throw(ArgumentError(
        "manifold_gate: U columns are not orthonormal (‖U†U − I‖ > orth_tol=$(orth_tol))."))
    B = Matrix{ComplexF64}(U' * (dhamiltonian(model, θ, θ̇) * U))
    α = real(tr(B)) / g
    split = opnorm(B - α * I)
    ok = split ≤ split_atol + split_rtol * max(opnorm(B), abs(α))
    return (ok = ok, α = α, split = split)
end

"""
    lowlying_spectrum(model, θ; window, degen_tol=1e-8, dense_max=4096)
        -> Vector{Tuple{Float64, Matrix{ComplexF64}}}

Energy clusters with energy in `[E0, E0 + window]`, each `(energy, V)` where `V` is
the `d × deg` orthonormal basis of that degenerate cluster (clustered by distance
from the cluster's first eigenvalue). The first entry is the ground manifold,
consistent with [`groundstate_manifold`](@ref). `window > 0` is required.
"""
function lowlying_spectrum(model::AffineModel, θ::AbstractVector; window::Real,
                           degen_tol::Real = 1e-8, dense_max::Integer = 4096)
    (window > 0 && isfinite(window)) || throw(ArgumentError(
        "lowlying_spectrum: window must be positive and finite; got $(window)."))
    H = hamiltonian(model, θ)
    λ, V, ranges = _eigen_clusters(H; degen_tol = degen_tol, dense_max = dense_max)
    E0 = λ[1]
    out = Tuple{Float64, Matrix{ComplexF64}}[]
    for r in ranges
        energy = λ[first(r)]
        energy - E0 ≤ window || break
        push!(out, (energy, Matrix{ComplexF64}(V[:, r])))
    end
    return out
end

# Cluster-averaged, gauge-invariant labels for one degenerate cluster `V` (d×deg):
# the dominant spatial irrep from averaged per-irrep weights and the cluster-mean
# ⟨S²⟩ → spin. Returns (irrep, irrep_weight, S2, spin, multiplicity).
function _label_cluster(V::AbstractMatrix, basis, m, G, S2::AbstractMatrix, tol::Real)
    deg = size(V, 2)
    # accumulate per-irrep weights, preserving the irrep ORDER from classify_state's
    # `weights` vector (the same fixed order for every column of a given group) so the
    # dominant pick is deterministic — including exact mixed-cluster ties (first wins).
    order = Symbol[]
    acc = Dict{Symbol, Float64}()
    for j in 1:deg
        w, _, _ = classify_state(V[:, j], basis, m, G; tol = tol)   # weights::Vector{Pair}
        for (ir, wt) in w
            haskey(acc, ir) || push!(order, ir)
            acc[ir] = get(acc, ir, 0.0) + wt
        end
    end
    dom_ir = :none; dom_w = -1.0
    for ir in order
        avg = acc[ir] / deg
        if avg > dom_w
            dom_w = avg; dom_ir = ir
        end
    end
    S2_val = real(tr(V' * (S2 * V))) / deg                          # gauge-invariant trace
    S = (-1 + sqrt(max(0.0, 1 + 4 * S2_val))) / 2
    spin = round(2S) / 2
    mult = round(Int, 2 * spin + 1)
    return dom_ir, dom_w, S2_val, spin, mult
end

"""
    classify_groundstate(model, θ; basis, m, G, S2, window,
                         gap_tol=1e-6, degen_tol=1e-8, dense_max=4096, tol=1e-6)
        -> GroundStateReport

Sector-aware T=0 ground-state report (opt-in physics bridge — never on a
differentiated path). Labels the ground manifold and every competing cluster within
`window` above `E0` by (a) spatial Mulliken irrep via
`classify_state(ψ, basis, m, G)` on each orthonormal column, averaged per irrep over
the cluster (gauge-invariant), and (b) spin from ⟨S²⟩ using the caller-supplied
compiled `S²` matrix `S2` (`d×d`, e.g. `compile(Ssqr(m, shells…), basis)`); spin
follows `S(S+1)=⟨S²⟩`.

`window > 0` is required (no universal default): 1–3 eV typically captures the lowest
crystal-field / multiplet competitors in 3d transition-metal-oxide L-edges. Applies
the same external-gap gate as [`groundstate_manifold`](@ref) on the ground cluster.

# Contract
`model`'s term matrices must be assembled in the **same** `basis` (same mode
ordering) as `basis`/`S2`. This is the caller's responsibility and cannot be checked
from the bare matrices. `classify_state` is a spin-fixed *spatial* diagnostic: under
spin–orbit coupling / a magnetic field the irrep label degrades (and `irrep_weight`
drops — read it as a purity flag), while ⟨S²⟩ remains meaningful.
"""
function classify_groundstate(model::AffineModel, θ::AbstractVector;
                              basis, m, G, S2::AbstractMatrix, window::Real,
                              gap_tol::Real = 1e-6, degen_tol::Real = 1e-8,
                              dense_max::Integer = 4096, tol::Real = 1e-6)
    (window > 0 && isfinite(window)) || throw(ArgumentError(
        "classify_groundstate: window must be positive and finite; got $(window)."))
    (gap_tol > 0 && isfinite(gap_tol)) || throw(ArgumentError(
        "classify_groundstate: gap_tol must be positive and finite; got $(gap_tol)."))
    degen_tol < gap_tol || throw(ArgumentError(
        "classify_groundstate: require degen_tol < gap_tol (got degen_tol=$(degen_tol), " *
        "gap_tol=$(gap_tol))."))
    H = hamiltonian(model, θ)
    N = size(H, 1)
    size(S2) == (N, N) || throw(DimensionMismatch(
        "classify_groundstate: S2 is $(size(S2)), expected ($(N), $(N)) " *
        "(it must be compiled in the same basis as the model)."))
    λ, V, ranges = _eigen_clusters(H; degen_tol = degen_tol, dense_max = dense_max)
    E0 = λ[1]
    g = length(ranges[1])
    g < N || throw(ArgumentError(
        "classify_groundstate: the whole spectrum is degenerate within " *
        "degen_tol=$(degen_tol); no external state to gate against."))
    gap_external = λ[g + 1] - E0
    gap_external > gap_tol || throw(ArgumentError(
        "classify_groundstate: ground manifold (g=$(g)) is not externally isolated " *
        "(E_g − E0 = $(gap_external) ≤ gap_tol=$(gap_tol)); level-crossing seam or " *
        "too-tight degen_tol."))
    reports = SectorReport[]
    for r in ranges
        energy = λ[first(r)]
        energy - E0 ≤ window || break
        Vr = Matrix{ComplexF64}(V[:, r])
        ir, irw, s2, spin, mult = _label_cluster(Vr, basis, m, G, S2, tol)
        push!(reports, SectorReport(energy, energy - E0, size(Vr, 2), ir, irw, s2, spin, mult))
    end
    gs = GroundState(E0, Matrix{ComplexF64}(V[:, ranges[1]]), g, gap_external)
    return GroundStateReport(reports[1], reports[2:end], gs)
end

# --- Build-step 6: spectrum VJP/pullback + clustered spectral measure ------
#
# Steps 3–4 give the complex correlator C_AB(ω) and its JVP dC. Step 6 turns those
# into the real intensity S = −Im C/π, its dense Jacobian and pullback (the VJP the
# sibling inference layer calls), and an optional degeneracy-clustered spectral
# measure (gauge-invariant moments of S over FROZEN windows). The grid response is
# the primary differentiable object; the clustered measure is a structured view on
# top — no per-pole residue derivatives (poles are used ONCE, in freeze_windows, to
# DEFINE windows, never differentiated).

"""
    XASGradientModel(g_model, f_model, A, B, embedding, ω_grid;
                     Γ, gap_tol=1e-6, rtol=1e-12, embedding_tol=1e-8)

Bundle the fixed pieces of the coupled-XAS forward map so the spectrum / Jacobian /
pullback API takes only `θ`. `A`,`B` are tuples of `N_f×N_f` transition operators;
`embedding` is the `N_f×N_g` isometry; `ω_grid`, `Γ` define the resolvent. The
constructor validates the static pieces (shared schema, embedding isometry, A/B
shapes, `Γ`/`rtol`/`gap_tol`/`embedding_tol` positive, `ω_grid` finite and **strictly
increasing**) so a malformed model fails immediately; θ-dependent checks remain in the
step-4 path on use.
"""
struct XASGradientModel
    g_model::AffineModel
    f_model::AffineModel
    A::Tuple
    B::Tuple
    embedding::Matrix{ComplexF64}
    ω_grid::Vector{Float64}
    Γ::Float64
    gap_tol::Float64
    rtol::Float64
    embedding_tol::Float64
    function XASGradientModel(g_model::AffineModel, f_model::AffineModel, A, B,
                              embedding::AbstractMatrix, ω_grid::AbstractVector;
                              Γ::Real, gap_tol::Real = 1e-6, rtol::Real = 1e-12,
                              embedding_tol::Real = 1e-8)
        At = Tuple(A); Bt = Tuple(B)
        ωv = collect(Float64, ω_grid)
        Emb = Matrix{ComplexF64}(embedding)
        _check_xas_inputs(g_model, f_model, At, Bt, Emb, ωv, Γ, rtol, embedding_tol)
        (gap_tol > 0 && isfinite(gap_tol)) || throw(ArgumentError(
            "XASGradientModel: gap_tol must be positive and finite; got $(gap_tol)."))
        all(i -> ωv[i] < ωv[i + 1], 1:length(ωv) - 1) || throw(ArgumentError(
            "XASGradientModel: ω_grid must be strictly increasing (needed for trapezoid " *
            "weights and contiguous window membership)."))
        return new(g_model, f_model, At, Bt, Emb, ωv, Float64(Γ), Float64(gap_tol),
                   Float64(rtol), Float64(embedding_tol))
    end
end

n_params(model::XASGradientModel) = n_params(model.g_model)

# Forward complex correlator on the model's grid (delegates to the step-4 path).
_xas_C(model::XASGradientModel, θ) = xas_response_C(model.g_model, model.f_model,
    model.A, model.B, model.embedding, θ, model.ω_grid; Γ = model.Γ,
    gap_tol = model.gap_tol, rtol = model.rtol, embedding_tol = model.embedding_tol)

_xas_dC(model::XASGradientModel, θ, θ̇) = xas_response_jvp(model.g_model, model.f_model,
    model.A, model.B, model.embedding, θ, θ̇, model.ω_grid; Γ = model.Γ,
    gap_tol = model.gap_tol, rtol = model.rtol, embedding_tol = model.embedding_tol)

"""
    spectrum(model::XASGradientModel, θ) -> Array{Float64,3}   # nA × nB × nω

XAS intensity `S = −Im C_AB(ω)/π` on the model's ω-grid (the common scalar case is
`nA=nB=1`). Inherits the non-degeneracy ground-state gate (via `groundstate`): a
(near-)degenerate ground state errors — use [`groundstate_manifold`](@ref) /
[`classify_groundstate`](@ref) to inspect a degenerate manifold.
"""
spectrum(model::XASGradientModel, θ::AbstractVector) = (-1 / π) .* imag(_xas_C(model, θ))

"""
    spectrum_and_jvp(model, θ, θ̇) -> (S, dS)   # each nA × nB × nω, real

Spectrum and its directional derivative `dS = −Im(dC)/π` along `θ̇`.
"""
function spectrum_and_jvp(model::XASGradientModel, θ::AbstractVector, θ̇::AbstractVector)
    S  = (-1 / π) .* imag(_xas_C(model, θ))
    dS = (-1 / π) .* imag(_xas_dC(model, θ, θ̇))
    return S, dS
end

"""
    jacobian(model, θ) -> Array{Float64,4}   # nA × nB × nω × Nθ

Dense spectrum Jacobian; column `k` is `spectrum_and_jvp(model, θ, e_k)[2]`. Cost =
`Nθ` JVPs (Fisher / diagnostics; `spectrum_with_pullback` avoids materializing it).
"""
function jacobian(model::XASGradientModel, θ::AbstractVector)
    np = n_params(model)
    S = spectrum(model, θ)
    J = Array{Float64,4}(undef, size(S, 1), size(S, 2), size(S, 3), np)
    for k in 1:np
        ek = zeros(np); ek[k] = 1.0
        J[:, :, :, k] = (-1 / π) .* imag(_xas_dC(model, θ, ek))
    end
    return J
end

"""
    spectrum_with_pullback(model, θ) -> (S, pullback)

`S` and a closure `pullback(λ::Array{Float64,3}) -> g::Vector{Float64}` mapping a real
cotangent on `S` (shape `nA×nB×nω`) to the parameter gradient `g_θ` (length `Nθ`) via
`g[k] = Σ λ .* dS_k` — equivalently `Σ Re(conj(W).*dC_k)` with `W = −iλ/π`. The `Nθ`
forward JVP columns are materialized eagerly (each its own immutable array), so the
ground-state solve happens once per parameter, and `pullback` is then a cheap
contraction. The pullback API is stable if the internals later become true adjoint
accumulation.
"""
function spectrum_with_pullback(model::XASGradientModel, θ::AbstractVector)
    np = n_params(model)
    S = spectrum(model, θ)
    cols = Vector{Array{Float64,3}}(undef, np)
    for k in 1:np
        ek = zeros(np); ek[k] = 1.0
        cols[k] = (-1 / π) .* imag(_xas_dC(model, θ, ek))   # fresh array per k
    end
    function pullback(λ::AbstractArray)
        size(λ) == size(S) || throw(DimensionMismatch(
            "spectrum_with_pullback: cotangent λ has size $(size(λ)), expected $(size(S))."))
        (eltype(λ) <: Real && all(isfinite, λ)) || throw(ArgumentError(
            "spectrum_with_pullback: cotangent λ must be real and finite (S = −Im C/π is " *
            "real; a complex λ would return a complex, ill-defined gradient)."))
        return [sum(λ .* cols[k]) for k in 1:np]
    end
    return S, pullback
end

# --- Clustered spectral measure (gauge-invariant moments over FROZEN windows) ---

"""
    FrozenWindow

A fixed energy window on the spectrum's ω-axis, with precomputed grid membership. The
membership is frozen at definition time so the windowed moments are a smooth fixed
linear functional of the grid response.

# Fields
- `lo`, `hi` — ω bounds; `center` — frozen centroid; `members::UnitRange{Int}` — the
  ω-grid indices inside `[lo, hi]`.
"""
struct FrozenWindow
    lo::Float64
    hi::Float64
    center::Float64
    members::UnitRange{Int}
end

"""
    SpectralCluster

A degeneracy-clustered, gauge-invariant spectral-measure view of one frozen window:
moments of the intensity `S = −Im C/π` (NOT the complex correlator). Singletons reduce
to ordinary per-line spectral weights.

# Fields
- `center::Float64` — frozen window centroid.
- `moment0::Matrix{Float64}` — `∫_window S dω` (nA×nB), the integrated spectral weight.
- `moment1::Matrix{Float64}` — `∫_window (ω − center)·S dω` (nA×nB), first central moment.
- `width::Float64` — `hi − lo`.
- `members::UnitRange{Int}` — frozen ω-grid indices (diagnostic).
"""
struct SpectralCluster
    center::Float64
    moment0::Matrix{Float64}
    moment1::Matrix{Float64}
    width::Float64
    members::UnitRange{Int}
end

# Trapezoid quadrature weights LOCAL to a contiguous member range `a:b` on a
# strictly-increasing grid `ω` — the composite-trapezoid rule over the sampled extent
# [ω[a], ω[b]]: endpoints get a half-interval to their single in-window neighbour, so
# the window does NOT borrow half-intervals from grid points outside it (a window-local
# integral, not a slice of the global one). A one-point range has zero measure.
function _window_weights(ω::AbstractVector, mem::UnitRange{Int})
    a, b = first(mem), last(mem)
    w = zeros(Float64, length(mem))
    a == b && return w                       # single sample ⇒ zero-width trapezoid
    for (k, i) in enumerate(mem)
        lo = i == a ? ω[a] : ω[i - 1]
        hi = i == b ? ω[b] : ω[i + 1]
        w[k] = (hi - lo) / 2
    end
    return w
end

"""
    freeze_windows(model, θ_ref; cluster_tol, pad=0.0) -> Vector{FrozenWindow}

Define frozen energy windows ONCE from the final-state excitation spectrum at the
reference `θ_ref`. The excitation axis matches the spectrum: with `z = ω + E0 + iΓ/2`
poles sit at `ω = E_n^f − E0`, so windows use `εf = eigvals(H_f(θ_ref)) − E0` with
`E0 = groundstate(g_model, θ_ref)[1]`. Eigenvalues are clustered within `cluster_tol`
(distance-from-cluster-start, the step-5 rule); each cluster `[ε_lo, ε_hi]` becomes a
window `[ε_lo − pad, ε_hi + pad]` with precomputed ω-grid members. This one-off pole
computation only DEFINES windows; it is never differentiated.
"""
function freeze_windows(model::XASGradientModel, θ_ref::AbstractVector;
                        cluster_tol::Real, pad::Real = 0.0)
    (cluster_tol > 0 && isfinite(cluster_tol)) || throw(ArgumentError(
        "freeze_windows: cluster_tol must be positive and finite; got $(cluster_tol)."))
    (pad ≥ 0 && isfinite(pad)) || throw(ArgumentError(
        "freeze_windows: pad must be nonnegative and finite; got $(pad)."))
    E0 = groundstate(model.g_model, θ_ref; gap_tol = model.gap_tol)[1]
    εf = eigvals(Hermitian(Matrix(hamiltonian(model.f_model, θ_ref)))) .- E0
    ω = model.ω_grid
    windows = FrozenWindow[]
    i = 1
    while i ≤ length(εf)
        j = i
        while j < length(εf) && εf[j + 1] - εf[i] ≤ cluster_tol   # distance from start
            j += 1
        end
        lo = εf[i] - pad; hi = εf[j] + pad
        idx = findall(x -> lo ≤ x ≤ hi, ω)
        length(idx) ≥ 2 || throw(ArgumentError(
            "freeze_windows: window [$(lo), $(hi)] (excitation cluster $(i):$(j)) spans " *
            "$(length(idx)) ω_grid point(s); the trapezoid rule needs ≥ 2. Widen pad or " *
            "refine the grid."))
        (idx == collect(first(idx):last(idx))) || throw(ErrorException(
            "freeze_windows: window members $(idx) are not contiguous on a strictly " *
            "increasing ω_grid — internal invariant violated."))
        members = first(idx):last(idx)
        center = sum(@view ω[members]) / length(members)
        push!(windows, FrozenWindow(lo, hi, center, members))
        i = j + 1
    end
    # Windows must be DISJOINT in grid membership — overlapping pads would
    # double-count a shared grid point across clusters (over-counting spectral
    # weight). Reject rather than silently double-count; the caller should reduce
    # `pad` or merge the clusters (raise `cluster_tol`).
    for k in 1:length(windows) - 1
        last(windows[k].members) < first(windows[k + 1].members) || throw(ArgumentError(
            "freeze_windows: windows $(k) and $(k + 1) share ω-grid points " *
            "($(windows[k].members) ∩ $(windows[k + 1].members) ≠ ∅); reduce pad or " *
            "raise cluster_tol so the windows are disjoint."))
    end
    return windows
end

# Windowed moments of an intensity array S (nA×nB×nω) over one frozen window, using
# WINDOW-LOCAL trapezoid weights (so an interior window does not pick up half-intervals
# outside it). Shared by value and JVP so the weighting is identical. Returns (m0, m1).
function _window_moments(S::AbstractArray{<:Real,3}, win::FrozenWindow, ω::AbstractVector)
    nA, nB = size(S, 1), size(S, 2)
    m0 = zeros(Float64, nA, nB)
    m1 = zeros(Float64, nA, nB)
    w = _window_weights(ω, win.members)
    for (k, i) in enumerate(win.members)
        m0 .+= w[k] .* @view S[:, :, i]
        m1 .+= (w[k] * (ω[i] - win.center)) .* @view S[:, :, i]
    end
    return m0, m1
end

"""
    spectral_clusters(model, θ, windows::Vector{FrozenWindow}) -> Vector{SpectralCluster}

Integrate the spectrum `S = −Im C/π` over each FROZEN window (trapezoid weights).
Membership is fixed by `windows` — a smooth fixed linear functional of the grid
response; nothing is re-clustered at this `θ`.
"""
function spectral_clusters(model::XASGradientModel, θ::AbstractVector,
                           windows::Vector{FrozenWindow})
    S = spectrum(model, θ)
    ω = model.ω_grid
    return [begin
                m0, m1 = _window_moments(S, win, ω)
                SpectralCluster(win.center, m0, m1, win.hi - win.lo, win.members)
            end for win in windows]
end

"""
    spectral_clusters_jvp(model, θ, θ̇, windows) -> Vector{NamedTuple{(:dmoment0,:dmoment1)}}

Directional derivative of the windowed moments along `θ̇`, integrating `dS = −Im(dC)/π`
over the same FROZEN windows. Because membership/weights are constants this is a linear
functional of `dS` and matches FD of `spectral_clusters` for all `θ` (including when a
physical pole crosses a window boundary — frozen membership).
"""
function spectral_clusters_jvp(model::XASGradientModel, θ::AbstractVector,
                               θ̇::AbstractVector, windows::Vector{FrozenWindow})
    dS = (-1 / π) .* imag(_xas_dC(model, θ, θ̇))
    ω = model.ω_grid
    return [begin
                d0, d1 = _window_moments(dS, win, ω)
                (dmoment0 = d0, dmoment1 = d1)
            end for win in windows]
end

# --- Build-step 7: deterministic direct-fit baseline (Optim weakdep) -------
#
# `fit_spectrum` is a gradient-driven least-squares baseline: minimize
# ½‖w∘(spectrum(model,θ) − target)‖² with the analytic VJP (step-6 pullback). It is a
# DIAGNOSTIC / sanity baseline — calibrated inference (NPE/HMC/SBC) lives in the sibling
# package, not here. The method is provided by the `MOADOptimExt` package extension; the
# variadic fallback below errors helpfully when `Optim` is not loaded (it is strictly
# less specific than the extension's 3-positional-argument method, so no collision).

"""
    fit_spectrum(model::XASGradientModel, target, θ0; optimizer, options, loss_weights)

Deterministic least-squares direct fit of `θ` to a `target` spectrum (same shape as
`spectrum(model, θ0)`), gradient-driven via the analytic pullback. Returns a NamedTuple
`(θ, loss, result)`. **Requires `Optim.jl`** — run `using Optim` to activate the method
(provided by `MOADOptimExt`). This is a baseline/diagnostic; calibrated parameter
inference belongs to the sibling inference package.
"""
function fit_spectrum end

fit_spectrum(::XASGradientModel, args...; kwargs...) = error(
    "fit_spectrum requires Optim.jl — run `using Optim` to load the MOADOptimExt extension.")

end # module Gradients
