# src/shells/multipole.jl
#
# Raw rank-k Wigner-Eckart building blocks for the standard transition /
# moment operators (sub-phase 2h): the general electric multipole
# transition operator, the E2 quadrupole transition, the nIXS scattering
# operator, the charge-quadrupole moment Q, and (with spin coupling added
# in a later stage) the spin-dipole T.
#
# This file provides the two convention-locked primitives that every 2h
# operator is built from:
#
#   `_ckq_coeff(ℓ_b, ℓ_a, k, q, m_a)`  — the Racah C^k_q matrix element
#       ⟨ℓ_b, m_a+q | C^k_q | ℓ_a, m_a⟩, the rank-k cross-shell
#       generalization of `dipole.jl`'s inline k=1 coefficient.
#
#   `_ckm_contract(m, shellA, shellB, terms; spin)` — the *raw,
#       non-Hermitizing* contraction of a list of `(k, q, coeff)` triples
#       with the C^k_q matrix elements into a second-quantized
#       `OperatorSum`.
#
# Convention (verified bit-for-bit against Quanty
# `SlaterCoefficientC`, Core/BasicMath/BasicMath_Combinatorial.cpp:236-240):
#
#   ⟨ℓ_b m_b | C^k_q | ℓ_a m_a⟩ = (−1)^{m_b} √((2ℓ_a+1)(2ℓ_b+1))
#                                 · 3j(ℓ_b, k, ℓ_a; −m_b, q, m_a)
#                                 · 3j(ℓ_b, k, ℓ_a; 0, 0, 0),   m_b = m_a + q.
#
# Quanty's `(l1, m1)` is the created (bra) index = MOAD's `(ℓ_b, m_b)`;
# `(l2, m2)` the annihilated (ket) = `(ℓ_a, m_a)`; `q = m1 − m2 = m_b − m_a`.
# At k=1 this is identical to `dipole.jl`, so `dipole` is recoverable as the
# k=1 special case (refactored in Stage A).
#
# IMPORTANT — why NOT reuse `Akm` (group_projected.jl): `Akm` Hermitizes its
# result (`(out + out') / 2`), which is correct for a crystal-field
# *potential* but wrong here — Quanty's `Qxx`/`Tz` are built by raw addition,
# and the nIXS operator e^{iq·r} at a fixed momentum transfer **q** is
# generally non-Hermitian. `_ckm_contract` performs NO symmetrization.

using WignerSymbols: wigner3j

"""
    _ckq_coeff(ℓ_b::Integer, ℓ_a::Integer, k::Integer, q::Integer, m_a::Integer) -> Float64

The Racah-normalized rank-k spherical-tensor matrix element

    ⟨ℓ_b, m_a + q | C^k_q | ℓ_a, m_a⟩
        = (−1)^{m_b} √((2ℓ_a+1)(2ℓ_b+1))
          · 3j(ℓ_b, k, ℓ_a; −m_b, q, m_a) · 3j(ℓ_b, k, ℓ_a; 0, 0, 0),

with `m_b = m_a + q`. Returns `0.0` when `m_b` is out of range `[-ℓ_b, ℓ_b]`,
when `|q| > k`, or when the diagonal 3j vanishes (parity: `ℓ_a + ℓ_b + k`
must be even). This is the rank-k cross-shell generalization of the inline
coefficient in `dipole.jl` (the k=1 case) and matches Quanty's
`SlaterCoefficientC(k, ℓ_b, m_b, ℓ_a, m_a)` bit-for-bit.
"""
function _ckq_coeff(ℓ_b::Integer, ℓ_a::Integer, k::Integer,
                    q::Integer, m_a::Integer)
    abs(q) <= k || return 0.0
    m_b = m_a + q
    (m_b < -ℓ_b || m_b > ℓ_b) && return 0.0
    threej_diag = Float64(wigner3j(Float64, ℓ_b, k, ℓ_a, 0, 0, 0))
    iszero(threej_diag) && return 0.0
    w = Float64(wigner3j(Float64, ℓ_b, k, ℓ_a, -m_b, q, m_a))
    iszero(w) && return 0.0
    sign = iseven(m_b) ? 1.0 : -1.0
    return sign * sqrt((2 * ℓ_a + 1) * (2 * ℓ_b + 1)) * threej_diag * w
end

"""
    _ckm_contract(m::ShellModel, shellA::Symbol, shellB::Symbol,
                  terms; spin::Symbol = :both) -> OperatorSum{ComplexF64}

Raw (non-Hermitizing) contraction of rank-k spherical-tensor terms into a
second-quantized one-body operator. Each entry of `terms` is a triple
`(k, q, coeff)` (with `coeff` possibly complex), contributing

    coeff · Σ_{m_a, σ}  ⟨ℓ_b, m_a+q | C^k_q | ℓ_a, m_a⟩
                        c†(shellB, m_a+q, σ) c(shellA, m_a, σ),

summed over the requested spin channels. The operator maps shell A
(annihilated) to shell B (created), i.e. the `shellA => shellB` /
absorption direction, matching `dipole`'s convention. Same-shell
(`shellA == shellB`) is allowed (the moment-operator case, e.g. Q).

`spin`:
- `:both` (default) — sum the ↓ and ↑ channels with equal weight
  (spin-diagonal charge operators: multipole, E2, nIXS, Q).
- `:up` / `:dn` — a single spin channel (building block for spin-coupled
  operators such as the spin-dipole T).

No selection-rule guard or Hermitization is applied — this is the bare
primitive; public wrappers add the physical guards.
"""
function _ckm_contract(m::ShellModel, shellA::Symbol, shellB::Symbol,
                       terms; spin::Symbol = :both)
    spin in (:both, :up, :dn) || throw(ArgumentError(
        "spin must be :both, :up, or :dn; got :$spin"))
    siteA = site_of(m, shellA)
    siteB = site_of(m, shellB)
    ℓ_a = ell_of(m, shellA)
    ℓ_b = ell_of(m, shellB)

    out = OperatorSum{ComplexF64}()
    for (k, q, coeff) in terms
        iszero(coeff) && continue
        for m_a in -ℓ_a:ℓ_a
            m_b = m_a + q
            (m_b < -ℓ_b || m_b > ℓ_b) && continue
            ck = _ckq_coeff(ℓ_b, ℓ_a, k, q, m_a)
            iszero(ck) && continue
            w = coeff * ck
            d_a, u_a = _orbital_mode_pair(ℓ_a, m_a)
            d_b, u_b = _orbital_mode_pair(ℓ_b, m_b)
            if spin === :both
                out += w * (cdag(siteB, u_b) * c(siteA, u_a) +
                            cdag(siteB, d_b) * c(siteA, d_a))
            elseif spin === :up
                out += w * (cdag(siteB, u_b) * c(siteA, u_a))
            else # :dn
                out += w * (cdag(siteB, d_b) * c(siteA, d_a))
            end
        end
    end
    return out
end

"""
    multipole(m::ShellModel, shells::Pair{Symbol, Symbol}, k::Integer;
              radial = 1.0) -> Vector{OperatorSum}

General rank-`k` electric multipole transition operator between two shells,
via Wigner-Eckart on Racah's `C^k_q` tensor (radial integral `radial`,
default 1). The pair `shellA => shellB` denotes the absorption direction
`c†(b) c(a)` (initial shell A → final shell B). `k = 1` is the electric
dipole (E1); `k = 2` the electric quadrupole (E2); etc.

Returns the `2k+1` spherical components `[T^k_{-k}, …, T^k_{+k}]` (index `q`
at position `q + k + 1`), each

    T^k_q = radial · Σ_{m_a, σ} ⟨ℓ_b, m_a+q | C^k_q | ℓ_a, m_a⟩
                              c†(b, m_a+q, σ) c(a, m_a, σ),

built by the raw (non-Hermitizing) `_ckm_contract`.

# Selection rules
Rank `k` must satisfy the triangle inequality `|ℓ_a − ℓ_b| ≤ k ≤ ℓ_a + ℓ_b`
and the parity rule `ℓ_a + ℓ_b + k` even (else every `C^k_q` matrix element
vanishes). Both are enforced with `ArgumentError`.

# Example: E2 between Ni 1s and Ni 3d (K pre-edge quadrupole)
```julia
m = ShellModel([:Ni_1s, :Ni_3d])
T2 = multipole(m, :Ni_1s => :Ni_3d, 2)   # 5 spherical components
```
"""
function multipole(m::ShellModel, shells::Pair{Symbol, Symbol}, k::Integer;
                   radial::Number = 1.0)
    shellA, shellB = shells
    ellA = ell_of(m, shellA)
    ellB = ell_of(m, shellB)
    k >= 0 || throw(ArgumentError("multipole: rank k must be ≥ 0; got $k."))
    (abs(ellA - ellB) <= k <= ellA + ellB) || throw(ArgumentError(
        "multipole: rank k=$k violates the triangle inequality " *
        "|ℓ_a − ℓ_b| ≤ k ≤ ℓ_a + ℓ_b for $shellA (ℓ=$ellA) → " *
        "$shellB (ℓ=$ellB)."))
    isodd(ellA + ellB + k) && throw(ArgumentError(
        "multipole: parity rule requires (ℓ_a + ℓ_b + k) even; got " *
        "$shellA (ℓ=$ellA) → $shellB (ℓ=$ellB), k=$k " *
        "(sum = $(ellA + ellB + k))."))
    rc = ComplexF64(radial)
    return OperatorSum[_ckm_contract(m, shellA, shellB, [(k, q, rc)])
                       for q in -k:k]
end

"""
    _real_tesseral_components(comps::AbstractVector, k::Integer)
        -> (Vector{OperatorSum}, Vector{Tuple{Int, Char}})

Combine the `2k+1` complex spherical tensor components `comps`
(`comps[q+k+1] = T^k_q`) into the real tesseral set, using the
`C`-normalized-tensor convention

    real_c(m) = (T^k_{−m} + (−1)^m T^k_{+m}) / √2
    real_s(m) = i (T^k_{−m} − (−1)^m T^k_{+m}) / √2

(the `(−1)^m` from the Condon-Shortley phase distinguishes this from a
plain `Y_{ℓm}` tesseral transform). The `(−1)^m` factor reproduces the
dipole Cartesian assembly at `k=1`: `real_c(1) = (T_{−1} − T_{+1})/√2 = T_x`,
`real_s(1) = i(T_{−1} + T_{+1})/√2 = T_y`, `m=0` component `= T_z`.

Returns the components in order `[(0,'0'), (1,'c'), (1,'s'), (2,'c'),
(2,'s'), …]` and a parallel `(|m|, '0'|'c'|'s')` tag list.
"""
function _real_tesseral_components(comps::AbstractVector, k::Integer)
    idx(q) = q + k + 1
    out = OperatorSum[]
    tags = Tuple{Int, Char}[]
    push!(out, comps[idx(0)])
    push!(tags, (0, '0'))
    inv_sqrt2 = 1 / sqrt(2)
    for mm in 1:k
        cp = comps[idx(mm)]
        cm = comps[idx(-mm)]
        sgn = iseven(mm) ? 1 : -1
        push!(out, inv_sqrt2 * (cm + sgn * cp))
        push!(tags, (mm, 'c'))
        push!(out, (im * inv_sqrt2) * (cm - sgn * cp))
        push!(tags, (mm, 's'))
    end
    return out, tags
end

"""
    quadrupole_transition(m::ShellModel, shells::Pair{Symbol, Symbol};
                          radial = 1.0) -> Vector{OperatorSum}

Electric-quadrupole (E2) transition operator between two shells, as the
five real tesseral components in the documented order

    [ T_{z²}, T_{xz}, T_{yz}, T_{x²−y²}, T_{xy} ]

derived from the spherical `multipole(m, shells, 2; radial)` via
`_real_tesseral_components` (the same Condon-Shortley convention that gives
the dipole Cartesian components at `k=1`). The pair is the `shellA => shellB`
absorption direction. For the bare spherical components `T^2_q`, call
`multipole(m, shells, 2; radial)` directly.

# Selection rules
E2 requires `Δℓ = 0, ±2` with even parity (`ℓ_a + ℓ_b` even) and the
triangle inequality `|ℓ_a − ℓ_b| ≤ 2 ≤ ℓ_a + ℓ_b` — enforced by the
underlying `multipole(..., 2)`.

# Example: Ni 1s → 3d K pre-edge quadrupole
```julia
m = ShellModel([:Ni_1s, :Ni_3d])
Q = quadrupole_transition(m, :Ni_1s => :Ni_3d)
T_z2, T_xz, T_yz, T_x2y2, T_xy = Q
```
"""
function quadrupole_transition(m::ShellModel, shells::Pair{Symbol, Symbol};
                               radial::Number = 1.0)
    comps = multipole(m, shells, 2; radial = radial)
    tess, _tags = _real_tesseral_components(comps, 2)
    return tess
end

"""
    quadrupole(m::ShellModel, shell::Symbol; radial = 1.0) -> Vector{OperatorSum}

Charge (electric) quadrupole MOMENT operator `Q_ij = 3 r_i r_j − r² δ_ij` on a
single `shell`, as the six Cartesian components in the documented order

    [ Q_xx, Q_yy, Q_zz, Q_xy, Q_xz, Q_yz ].

DIMENSIONLESS by default (Quanty `CreateOperatorQ*`, `StandardOperators.cpp:3315-3786`,
are bare angular C² combinations with no radial moment); the optional `radial`
scale (e.g. `⟨r²⟩`, in the caller's units) multiplies all six components. This
is a same-shell expectation operator (NOT a transition); it is spin-summed and
Hermitian. The verified C²_m combinations (`r ≡ √(3/2)`) are

    Q_xx = −C²₀ + r(C²₋₂+C²₂),   Q_yy = −C²₀ − r(C²₋₂+C²₂),   Q_zz = 2·C²₀,
    Q_xy = i·r(C²₋₂−C²₂),        Q_xz = r(C²₋₁−C²₁),          Q_yz = i·r(C²₋₁+C²₁).

The set is traceless (`Q_xx+Q_yy+Q_zz = 0`). Requires `ℓ ≥ 1` (an s-shell has
no rank-2 moment).

# Example
```julia
m = ShellModel([:Ni_3d])
Qxx, Qyy, Qzz, Qxy, Qxz, Qyz = quadrupole(m, :Ni_3d)
```
"""
function quadrupole(m::ShellModel, shell::Symbol; radial::Number = 1.0)
    ell = ell_of(m, shell)
    ell >= 1 || throw(ArgumentError(
        "quadrupole: a rank-2 moment requires ℓ ≥ 1; shell $shell has ℓ=$ell."))
    s = ComplexF64(radial)
    r = sqrt(1.5) * s
    Qxx = _ckm_contract(m, shell, shell, [(2, 0, -s), (2, -2, r), (2, 2, r)])
    Qyy = _ckm_contract(m, shell, shell, [(2, 0, -s), (2, -2, -r), (2, 2, -r)])
    Qzz = _ckm_contract(m, shell, shell, [(2, 0, 2s)])
    Qxy = _ckm_contract(m, shell, shell, [(2, -2, im * r), (2, 2, -im * r)])
    Qxz = _ckm_contract(m, shell, shell, [(2, -1, r), (2, 1, -r)])
    Qyz = _ckm_contract(m, shell, shell, [(2, -1, im * r), (2, 1, im * r)])
    return OperatorSum[Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]
end

"""
    spin_dipole_T(m::ShellModel, shell::Symbol) -> Vector{OperatorSum}

The spin-dipole tensor operator `T = Σ_i [ s_i − 3 r̂_i (r̂_i · s_i) ]` on a
single `shell`, as the three Cartesian components `[T_x, T_y, T_z]`. This is
the operator that enters the XMCD spin sum rule — **NOT** the magnetic-dipole
(M1) operator `L + 2S`.

`T` is a rank-2 spatial tensor (C²) coupled to spin, built term-by-term to
match Quanty `CreateOperatorT{x,y,z}` (`StandardOperators.cpp:2600-2886`)
bit-for-bit. Writing `v_q(m) ≡ ⟨m+q|C²_q|m⟩` (the same-shell Racah C²; the
`_ckq_coeff` value), e.g.

    T_z = Σ_m [ −v₀(m)(c†_{m↑}c_{m↑} − c†_{m↓}c_{m↓})
                − (√6/2) v₋₁(m) c†_{m−1,↑}c_{m,↓}
                + (√6/2) v₊₁(m) c†_{m+1,↓}c_{m,↑} ],

with the C⁰ piece `= −2·C²₀·s_z`. Each component is Hermitian (the physical
observable); the operator is spin-summed/spin-coupled and raw. Requires `ℓ ≥ 1`.

# Example
```julia
m = ShellModel([:Ni_3d])
Tx, Ty, Tz = spin_dipole_T(m, :Ni_3d)
```
"""
function spin_dipole_T(m::ShellModel, shell::Symbol)
    ell = ell_of(m, shell)
    ell >= 1 || throw(ArgumentError(
        "spin_dipole_T: a rank-2 spatial tensor requires ℓ ≥ 1; " *
        "shell $shell has ℓ=$ell."))
    site = site_of(m, shell)
    s6 = sqrt(6.0)
    s32 = sqrt(1.5)

    Tx = OperatorSum{ComplexF64}()
    Ty = OperatorSum{ComplexF64}()
    Tz = OperatorSum{ComplexF64}()

    up(ml) = _orbital_mode_pair(ell, ml)[2]
    dn(ml) = _orbital_mode_pair(ell, ml)[1]
    inrange(ml) = -ell <= ml <= ell
    cval(q, ml) = _ckq_coeff(ell, ell, 2, q, ml)   # ⟨ml+q|C²_q|ml⟩

    for ml in -ell:ell
        v0 = cval(0, ml)

        # ── T_z ──
        if !iszero(v0)
            Tz += (-v0) * (cdag(site, up(ml)) * c(site, up(ml)))
            Tz += ( v0) * (cdag(site, dn(ml)) * c(site, dn(ml)))
        end
        if inrange(ml - 1)
            vm1 = cval(-1, ml)
            iszero(vm1) || (Tz += (-0.5 * s6 * vm1) * (cdag(site, up(ml - 1)) * c(site, dn(ml))))
        end
        if inrange(ml + 1)
            vp1 = cval(1, ml)
            iszero(vp1) || (Tz += (0.5 * s6 * vp1) * (cdag(site, dn(ml + 1)) * c(site, up(ml))))
        end

        # ── T_x, T_y ──  (Ty = i × the imaginary coefficients in Quanty)
        # C¹ s_z
        if inrange(ml + 1)
            v = cval(1, ml)
            if !iszero(v)
                Tx += ( 0.5 * s32 * v) * (cdag(site, up(ml + 1)) * c(site, up(ml)))
                Tx += (-0.5 * s32 * v) * (cdag(site, dn(ml + 1)) * c(site, dn(ml)))
                Ty += (-0.5 * s32 * v * im) * (cdag(site, up(ml + 1)) * c(site, up(ml)))
                Ty += ( 0.5 * s32 * v * im) * (cdag(site, dn(ml + 1)) * c(site, dn(ml)))
            end
        end
        # C⁰ s± (spin flip, same m)
        if !iszero(v0)
            Tx += (0.5 * v0) * (cdag(site, up(ml)) * c(site, dn(ml)))
            Tx += (0.5 * v0) * (cdag(site, dn(ml)) * c(site, up(ml)))
            Ty += (-0.5 * v0 * im) * (cdag(site, up(ml)) * c(site, dn(ml)))
            Ty += ( 0.5 * v0 * im) * (cdag(site, dn(ml)) * c(site, up(ml)))
        end
        # C² s−
        if inrange(ml + 2)
            v = cval(2, ml)
            if !iszero(v)
                Tx += (-s32 * v) * (cdag(site, dn(ml + 2)) * c(site, up(ml)))
                Ty += ( s32 * v * im) * (cdag(site, dn(ml + 2)) * c(site, up(ml)))
            end
        end
        # C⁻¹ s_z
        if inrange(ml - 1)
            v = cval(-1, ml)
            if !iszero(v)
                Tx += (-0.5 * s32 * v) * (cdag(site, up(ml - 1)) * c(site, up(ml)))
                Tx += ( 0.5 * s32 * v) * (cdag(site, dn(ml - 1)) * c(site, dn(ml)))
                Ty += (-0.5 * s32 * v * im) * (cdag(site, up(ml - 1)) * c(site, up(ml)))
                Ty += ( 0.5 * s32 * v * im) * (cdag(site, dn(ml - 1)) * c(site, dn(ml)))
            end
        end
        # C⁻² s+
        if inrange(ml - 2)
            v = cval(-2, ml)
            if !iszero(v)
                Tx += (-s32 * v) * (cdag(site, up(ml - 2)) * c(site, dn(ml)))
                Ty += (-s32 * v * im) * (cdag(site, up(ml - 2)) * c(site, dn(ml)))
            end
        end
    end
    return OperatorSum[Tx, Ty, Tz]
end
