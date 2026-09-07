# src/shells/slater.jl
#
# Single-shell Slater Coulomb interaction in standard normal-ordered form
# (Cowan eq 6.16). The emitted operator is
#
#     H_C = (1/2) Σ_k F^k Σ_{m1+m2=m3+m4} Σ_{s1,s2}
#           c^k(ℓ, m1; ℓ, m3) · c^k(ℓ, m4; ℓ, m2)
#           · cdag(s, m1, s1) cdag(s, m2, s2) c(s, m4, s2) c(s, m3, s1)
#
# with the Gaunt-like coefficient
#
#     c^k(l1, m1; l2, m2) = (-1)^m1 · sqrt((2 l1 + 1)(2 l2 + 1))
#                            · 3j(l1, k, l2; 0, 0, 0)
#                            · 3j(l1, k, l2; -m1, m1-m2, m2).
#
# Note on prefactor sign: Quanty's un-normal-ordered form has the inner
# pair `c(m3, s1) c(m4, s2)` with a leading `−1/2`. Standard normal order
# anticommutes these, flipping the sign and yielding the `+1/2` here with
# the pair ordering `c(m4, s2) c(m3, s1)`.
#
# F^0 is auto-derived from the spherical centroid U via
#     F^0 = U + Σ_{k > 0, even} a^k_intra(ℓ) · F^k
# with a^k_intra(ℓ) = (2ℓ+1)/(4ℓ+1) · 3j(ℓ, k, ℓ; 0, 0, 0)².
# Validated values:
#   p (ℓ=1):  a² = 2/25
#   d (ℓ=2):  a² = 2/63,   a⁴ = 2/63
#   f (ℓ=3):  a² = 4/195,  a⁴ = 2/143,  a⁶ = 100/5577

using WignerSymbols: wigner3j

"""
    _slater_ranks(x, sym::Symbol) -> Tuple

Normalise a Slater-integral keyword to a plain rank-ordered `Tuple`.

`F` and `G` are rank-*positional*: entry `i` is the i-th higher-rank integral in
ascending `k`. A `NamedTuple` is accepted because that is what
[`atomic_parameters`](@ref) returns, but its fields must be named
`F2`, `F4`, ... / `G1`, `G3`, ... in strictly ascending rank.
Field order in a `NamedTuple` is whatever the author wrote, so validating it is
what stops a mis-ordered `(F4 = ..., F2 = ...)` from being silently mapped onto
the wrong ranks and producing a wrong Hamiltonian with no error.
"""
_slater_ranks(x::Tuple, ::Symbol) = x

function _slater_ranks(x::NamedTuple, sym::Symbol)
    isempty(x) && return ()
    pat = Regex("^" * String(sym) * raw"(\d+)$")
    ranks = map(keys(x)) do k
        mt = match(pat, String(k))
        mt === nothing && throw(ArgumentError(
            "coulomb: when `$sym` is given as a NamedTuple its fields must be " *
            "named $(sym)<rank> (e.g. $(sym)2, $(sym)4); got :$k. " *
            "Pass a plain Tuple to supply ranks positionally."))
        parse(Int, mt.captures[1])
    end
    all(ranks[i] < ranks[i + 1] for i in 1:(length(ranks) - 1)) || throw(ArgumentError(
        "coulomb: `$sym` NamedTuple fields must be in strictly ascending rank; " *
        "got $(keys(x)). `$sym` is rank-positional, so an out-of-order " *
        "NamedTuple would assign the integrals to the wrong ranks."))
    return values(x)
end

"""
    coulomb(m::ShellModel, shell::Symbol; U, F) -> OperatorSum

Single-shell Slater Coulomb interaction in standard normal-ordered form
(Cowan eq 6.16). `F` is the tuple `(F^2, F^4, ...)` of higher-rank
Slater integrals in eV; `U` is the spherical centroid (Hubbard U). The
zeroth Slater integral is auto-derived as

    F^0 = U + Σ_{k > 0, even} a^k_intra(ℓ) · F^k,
    a^k_intra(ℓ) = (2ℓ+1)/(4ℓ+1) · 3j(ℓ, k, ℓ; 0, 0, 0)².

The emitted operator is

    H_C = (1/2) Σ_k F^k Σ_{m1+m2=m3+m4} Σ_{s1,s2}
          c^k(ℓ, m1; ℓ, m3) · c^k(ℓ, m4; ℓ, m2)
          · cdag(s, m1, s1) cdag(s, m2, s2) c(s, m4, s2) c(s, m3, s1)

with c^k(l1, m1; l2, m2) = (−1)^m1 · sqrt((2l1+1)(2l2+1))
                            · 3j(l1, k, l2; 0, 0, 0)
                            · 3j(l1, k, l2; −m1, m1−m2, m2).

# Arguments
- `m::ShellModel` — shell registry.
- `shell::Symbol` — shell tag, e.g. `:Ni_3d`.
- `U` — spherical centroid (eV).
- `F` — `(F^2, F^4, ...)` higher-rank Slater integrals (eV), as a plain
  `Tuple` or as a rank-ordered `NamedTuple` such as `atomic_parameters(...).Fdd`.

# Example
```julia
m = ShellModel([:Ni_3d])
H = coulomb(m, :Ni_3d; U = 7.3, F = (11.14, 6.87))
```
"""
function coulomb(m::ShellModel, shell::Symbol; U, F::Union{Tuple,NamedTuple})
    F = _slater_ranks(F, :F)
    site = site_of(m, shell)
    ell  = ell_of(m, shell)
    F0   = _compute_F0_intra(ell, U, F)
    F_full = (F0, F...)   # (F^0, F^2, F^4, ...)

    out = OperatorSum{ComplexF64}()

    for k_idx in eachindex(F_full)
        k   = 2 * (k_idx - 1)   # 0, 2, 4, ...
        F_k = F_full[k_idx]
        iszero(F_k) && continue
        # k must satisfy triangle inequality with l, l: 0 ≤ k ≤ 2ℓ.
        k > 2 * ell && continue
        threej_diag = Float64(wigner3j(Float64, ell, k, ell, 0, 0, 0))
        iszero(threej_diag) && continue
        pre_kk = (2 * ell + 1) * threej_diag    # (-1)^? handled in c^k
        for m1 in -ell:ell, m2 in -ell:ell
            # m1 + m2 = m3 + m4 ⇒ m4 = m1 + m2 - m3; m3 ∈ [-ell, ell] AND m4 ∈ [-ell, ell]
            m3_lo = max(-ell, m1 + m2 - ell)
            m3_hi = min( ell, m1 + m2 + ell)
            for m3 in m3_lo:m3_hi
                m4 = m1 + m2 - m3
                # 3j selection: |m_q| ≤ k for the middle slot.
                abs(m1 - m3) > k && continue
                abs(m4 - m2) > k && continue
                # c^k(ℓ, m1; ℓ, m3)
                w13 = Float64(wigner3j(Float64, ell, k, ell, -m1, m1 - m3, m3))
                iszero(w13) && continue
                c_k_13 = ((-1)^m1) * pre_kk * w13
                # c^k(ℓ, m4; ℓ, m2)
                w42 = Float64(wigner3j(Float64, ell, k, ell, -m4, m4 - m2, m2))
                iszero(w42) && continue
                c_k_42 = ((-1)^m4) * pre_kk * w42
                val = 0.5 * F_k * c_k_13 * c_k_42
                iszero(val) && continue
                d1, u1 = _orbital_mode_pair(ell, m1)
                d2, u2 = _orbital_mode_pair(ell, m2)
                d3, u3 = _orbital_mode_pair(ell, m3)
                d4, u4 = _orbital_mode_pair(ell, m4)
                # spin pairings: index 1 carries s1 (paired with index 3),
                # index 2 carries s2 (paired with index 4). Sum 4 spin combos.
                # Operator order: cdag(m1, s1) cdag(m2, s2) c(m4, s2) c(m3, s1).
                spin_modes = ((u1, u2, u4, u3),    # (s1, s2) = (↑, ↑)
                              (u1, d2, d4, u3),    # (s1, s2) = (↑, ↓)
                              (d1, u2, u4, d3),    # (s1, s2) = (↓, ↑)
                              (d1, d2, d4, d3))    # (s1, s2) = (↓, ↓)
                for (a1, a2, a4, a3) in spin_modes
                    out += val * cdag(site, a1) * cdag(site, a2) *
                                 c(site, a4) * c(site, a3)
                end
            end
        end
    end
    # Hermitisation+chop matches the pattern used in `Akm` (group_projected.jl):
    # symbolic accumulation can leave per-term roundoff that should cancel
    # between Hermitian-conjugate pairs. Symmetrise so the assembled matrix
    # passes `ishermitian`.
    return chop((out + out') / 2; tol = 1e-12)
end

# Internal helper. Computes F^0 = U + Σ_k a^k_intra(ℓ) · F^k from the
# spherical centroid and the higher Slater integrals.
function _compute_F0_intra(ell::Int, U::Real, F::Tuple)
    F0 = float(U)
    for (k_idx, F_k) in enumerate(F)
        k = 2 * k_idx   # F[1] = F², F[2] = F⁴, ...
        k > 2 * ell && continue
        w = Float64(wigner3j(Float64, ell, k, ell, 0, 0, 0))
        coeff = (2 * ell + 1) / (4 * ell + 1) * w^2
        F0 += coeff * F_k
    end
    return F0
end

"""
    coulomb(m::ShellModel, shellA::Symbol, shellB::Symbol; U, F=(), G=()) -> OperatorSum

Two-shell (cross-shell) Slater Coulomb interaction in standard
normal-ordered form. The emitted operator is

    H_C^{AB} = +Σ_k F^k Σ_{m1+m2=m3+m4} Σ_{s1,s2}
                c^k(ℓ_A, m1; ℓ_A, m3) · c^k(ℓ_B, m4; ℓ_B, m2)
                · cdag(A, m1, s1) cdag(B, m2, s2) c(B, m4, s2) c(A, m3, s1)
                                            (DIRECT,   k = 0, 2, ..., 2 min(ℓ_A, ℓ_B))

             − Σ_k G^k Σ_{m1+m2=m3+m4} Σ_{s1,s2}
               c^k(ℓ_A, m1; ℓ_B, m3) · c^k(ℓ_A, m4; ℓ_B, m2)
               · cdag(A, m1, s1) cdag(B, m2, s2) c(B, m3, s1) c(A, m4, s2)
                                            (EXCHANGE, k = |ℓ_A−ℓ_B|, ..., ℓ_A+ℓ_B step 2)

with c^k(l1, m1; l2, m2) = (−1)^m1 · sqrt((2l1+1)(2l2+1))
                            · 3j(l1, k, l2; 0, 0, 0)
                            · 3j(l1, k, l2; −m1, m1−m2, m2).

PREFACTORS (LOCKED, asymmetric direct vs exchange):
- DIRECT:   +1.  The single-shell case has +1/2 (m1↔m2 / m3↔m4 swap
  is a symmetry of the summand). With distinct A/B labels that swap is
  not a symmetry, so no 1/2 — net +1. Matches Quanty's
  `CreateOperatorCoulombTwoShellsConstantOccupation` direct loop after
  anticommuting the inner-pair (Quanty writes c(A) c(B); MOADyna writes
  c(B) c(A) → sign flip on canonicalization, then Quanty's −1 → MOADyna's +1).
- EXCHANGE: −1.  Quanty writes c(B) c(A) in the exchange loop — the
  SAME order as MOADyna — so canonicalization does NOT insert a sign flip.
  Quanty's −1 stays −1 in MOADyna. (This asymmetry vs the direct branch
  is a real consequence of the operator-order difference between
  Quanty's two loops, not a sign-pinning oversight.)

The asymmetry is regression-tested in `test/shells/test_slater.jl`
("coulomb two-shell — exchange-only centroid matches Quanty convention")
and end-to-end-tested by the NiO L₂,₃ XAS native acceptance
(`test/shells/validation/test_nio_xas_native.jl`, residuals at 1.5e-12
of peak vs the PyQuanty reference).

F^0(ℓ_A, ℓ_B) is auto-derived from the spherical centroid `U_AB`. Only
the EXCHANGE Gᵏ contributes (direct F^k>0 averages to zero in the
centroid):

    F^0 = U + (1/2) Σ_k 3j(ℓ_A, k, ℓ_B; 0, 0, 0)² · G^k,

with k = |ℓ_A − ℓ_B|, |ℓ_A − ℓ_B| + 2, ..., ℓ_A + ℓ_B.

# Arguments
- `m::ShellModel` — shell registry.
- `shellA`, `shellB` — distinct shell tags (e.g. `:Ni_2p`, `:Ni_3d`).
- `U` — cross-shell spherical centroid `U_AB` (eV).
- `F` — direct higher Slater integrals `(F^2, F^4, ...)` in eV (k > 0).
  Length up to `min(ℓ_A, ℓ_B)`. A plain `Tuple` supplies the ranks
  positionally; a rank-ordered `NamedTuple` such as
  `atomic_parameters(...).Fpd` is also accepted.
- `G` — exchange Slater integrals
  `(G^{|ℓ_A−ℓ_B|}, G^{|ℓ_A−ℓ_B|+2}, ...)` in eV. Length up to
  `(ℓ_A + ℓ_B − |ℓ_A − ℓ_B|)/2 + 1`. Same `Tuple` / `NamedTuple` rule as `F`.

# Example: NiO 2p–3d core–valence Coulomb
```julia
m = ShellModel([:Ni_2p, :Ni_3d])
H = coulomb(m, :Ni_2p, :Ni_3d; U = 8.5, F = (6.67,), G = (4.92, 2.80))
```
"""
function coulomb(m::ShellModel, shellA::Symbol, shellB::Symbol;
                 U, F::Union{Tuple,NamedTuple} = (), G::Union{Tuple,NamedTuple} = ())
    F = _slater_ranks(F, :F)
    G = _slater_ranks(G, :G)
    shellA == shellB && throw(ArgumentError(
        "coulomb two-shell: shellA and shellB must be distinct; got " *
        "$shellA == $shellB. Use the single-shell `coulomb(m, :s; U, F)`."))
    siteA = site_of(m, shellA); ellA = ell_of(m, shellA)
    siteB = site_of(m, shellB); ellB = ell_of(m, shellB)

    F0 = _compute_F0_inter(ellA, ellB, U, F, G)
    F_full = (F0, F...)   # (F^0, F^2, ..., user-supplied tail)

    out = OperatorSum{ComplexF64}()

    # --- Direct: k = 0, 2, ..., 2 min(ellA, ellB) ---
    k_direct_max = 2 * min(ellA, ellB)
    for k_idx in eachindex(F_full)
        k   = 2 * (k_idx - 1)
        k > k_direct_max && continue
        F_k = F_full[k_idx]
        iszero(F_k) && continue
        threej_diag_A = Float64(wigner3j(Float64, ellA, k, ellA, 0, 0, 0))
        threej_diag_B = Float64(wigner3j(Float64, ellB, k, ellB, 0, 0, 0))
        (iszero(threej_diag_A) || iszero(threej_diag_B)) && continue
        pre_A = (2 * ellA + 1) * threej_diag_A
        pre_B = (2 * ellB + 1) * threej_diag_B
        for m1 in -ellA:ellA, m2 in -ellB:ellB
            m3_lo = max(-ellA, m1 + m2 - ellB)
            m3_hi = min( ellA, m1 + m2 + ellB)
            for m3 in m3_lo:m3_hi
                m4 = m1 + m2 - m3
                # 3j middle-slot selection: |m_q| ≤ k.
                abs(m1 - m3) > k && continue
                abs(m4 - m2) > k && continue
                w_A = Float64(wigner3j(Float64, ellA, k, ellA, -m1, m1 - m3, m3))
                iszero(w_A) && continue
                w_B = Float64(wigner3j(Float64, ellB, k, ellB, -m4, m4 - m2, m2))
                iszero(w_B) && continue
                c_k_A = ((-1)^m1) * pre_A * w_A
                c_k_B = ((-1)^m4) * pre_B * w_B
                val = F_k * c_k_A * c_k_B   # NO 1/2 — distinct A/B labels
                iszero(val) && continue
                d_A1, u_A1 = _orbital_mode_pair(ellA, m1)
                d_A3, u_A3 = _orbital_mode_pair(ellA, m3)
                d_B2, u_B2 = _orbital_mode_pair(ellB, m2)
                d_B4, u_B4 = _orbital_mode_pair(ellB, m4)
                # Operator order: cdag(A, m1, s1) cdag(B, m2, s2)
                #                 c(B, m4, s2) c(A, m3, s1).
                # s1 pairs (m1, m3) on A; s2 pairs (m2, m4) on B.
                spin_combos = ((u_A1, u_B2, u_B4, u_A3),   # (s1,s2)=(↑,↑)
                               (u_A1, d_B2, d_B4, u_A3),   # (s1,s2)=(↑,↓)
                               (d_A1, u_B2, u_B4, d_A3),   # (s1,s2)=(↓,↑)
                               (d_A1, d_B2, d_B4, d_A3))   # (s1,s2)=(↓,↓)
                for (a1, b2, b4, a3) in spin_combos
                    out += val * cdag(siteA, a1) * cdag(siteB, b2) *
                                 c(siteB, b4) * c(siteA, a3)
                end
            end
        end
    end

    # --- Exchange: k = |ellA − ellB|, |ellA − ellB|+2, ..., ellA + ellB ---
    k_exch_min = abs(ellA - ellB)
    k_exch_max = ellA + ellB
    for (k_idx, G_k) in enumerate(G)
        k = k_exch_min + 2 * (k_idx - 1)
        k > k_exch_max && continue
        iszero(G_k) && continue
        threej_diag_AB = Float64(wigner3j(Float64, ellA, k, ellB, 0, 0, 0))
        iszero(threej_diag_AB) && continue
        pre_AB = sqrt((2 * ellA + 1) * (2 * ellB + 1)) * threej_diag_AB
        for m1 in -ellA:ellA, m2 in -ellB:ellB
            # Quanty case 2 ranges: m3 ∈ [-ellB, ellB], m4 = m1+m2−m3 ∈ [-ellA, ellA]
            m3_lo = max(-ellB, m1 + m2 - ellA)
            m3_hi = min( ellB, m1 + m2 + ellA)
            for m3 in m3_lo:m3_hi
                m4 = m1 + m2 - m3
                abs(m1 - m3) > k && continue
                abs(m4 - m2) > k && continue
                w_13 = Float64(wigner3j(Float64, ellA, k, ellB, -m1, m1 - m3, m3))
                iszero(w_13) && continue
                w_42 = Float64(wigner3j(Float64, ellA, k, ellB, -m4, m4 - m2, m2))
                iszero(w_42) && continue
                c_k_AB_13 = ((-1)^m1) * pre_AB * w_13
                c_k_AB_42 = ((-1)^m4) * pre_AB * w_42
                val = -G_k * c_k_AB_13 * c_k_AB_42   # NO 1/2; − because MOADyna's c(B) c(A) order matches Quanty's exchange order (no anticommute sign flip)
                iszero(val) && continue
                d_A1, u_A1 = _orbital_mode_pair(ellA, m1)
                d_A4, u_A4 = _orbital_mode_pair(ellA, m4)
                d_B2, u_B2 = _orbital_mode_pair(ellB, m2)
                d_B3, u_B3 = _orbital_mode_pair(ellB, m3)
                # Operator order: cdag(A, m1, s1) cdag(B, m2, s2)
                #                 c(B, m3, s1) c(A, m4, s2).
                # s1 pairs (m1 on A, m3 on B); s2 pairs (m2 on B, m4 on A).
                spin_combos = ((u_A1, u_B2, u_B3, u_A4),   # (s1,s2)=(↑,↑)
                               (u_A1, d_B2, u_B3, d_A4),   # (s1,s2)=(↑,↓)
                               (d_A1, u_B2, d_B3, u_A4),   # (s1,s2)=(↓,↑)
                               (d_A1, d_B2, d_B3, d_A4))   # (s1,s2)=(↓,↓)
                for (a1, b2, b3, a4) in spin_combos
                    out += val * cdag(siteA, a1) * cdag(siteB, b2) *
                                 c(siteB, b3) * c(siteA, a4)
                end
            end
        end
    end

    # Hermitisation+chop: floating-point cleanup, same pattern as
    # single-shell `coulomb` and `Akm`.
    return chop((out + out') / 2; tol = 1e-12)
end

# Internal helper. Computes
#   F^0(ℓ_A, ℓ_B) = U + (1/2) Σ_k 3j(ℓ_A, k, ℓ_B; 0, 0, 0)² · G^k
# with k = |ℓ_A − ℓ_B|, |ℓ_A − ℓ_B| + 2, ..., ℓ_A + ℓ_B. Validated
# coefficient table:
#   p–d (ℓ=1, ℓ'=2):  k=1 → 1/15;  k=3 → 3/70.
#   d–d' (ℓ=2, ℓ'=2): k=0 → 1/10;  k=2 → 1/35;  k=4 → 1/35.
#   p–f (ℓ=1, ℓ'=3):  k=2 → 3/70;  k=4 → 2/63.
function _compute_F0_inter(ellA::Int, ellB::Int, U::Real, F::Tuple, G::Tuple)
    F0 = float(U)
    k_min = abs(ellA - ellB)
    k_max = ellA + ellB
    for (k_idx, G_k) in enumerate(G)
        k = k_min + 2 * (k_idx - 1)
        k > k_max && continue
        w = Float64(wigner3j(Float64, ellA, k, ellB, 0, 0, 0))
        coeff = w^2 / 2
        F0 += coeff * G_k
    end
    return F0
end
