# src/shells/nixs.jl
#
# Non-resonant inelastic X-ray scattering (nIXS) transition operator and
# the radial-integral / angular primitives it needs (sub-phase 2h, Stage B).
#
# The nIXS scattering operator is the second-quantized e^{i q·r}, expanded
# in multipoles via the plane-wave (Rayleigh) expansion. In Racah's
# C^k_q normalization:
#
#   e^{i q·r} = Σ_k i^k (2k+1) j_k(qr) Σ_m C^k_m(q̂)* C^k_m(r̂),
#
# obtained from 4π Σ_{km} i^k j_k(qr) Y*_{km}(q̂) Y_{km}(r̂) using
# C^k_m = √(4π/(2k+1)) Y_{km}. The angular factor on q̂ is the COMPLEX
# CONJUGATE C^k_m(q̂)* — locked by the Legendre addition theorem
# Σ_m C^k_m(q̂)* C^k_m(r̂) = P_k(q̂·r̂) (test in test_multipole.jl). The
# radial factor is the spherical-Bessel moment
#
#   Rj_k(q) = ⟨R_b(r) | j_k(qr) | R_a(r)⟩,
#
# a number per allowed k (BYO via `radial_integrals`, or computed by
# `radial_integral`). Allowed k are set by the triangle inequality +
# parity (d→d: k ∈ {0,2,4}; p→d: k ∈ {1,3}). The operator at fixed q is
# generally non-Hermitian, so it is built by the raw `_ckm_contract`.
#
# The nIXS *spectrum* is then obtained by handing this operator to the
# shipped `dynamical_structure_factor` (sub-phase 2e).

"""
    _assoc_legendre(l::Integer, m::Integer, x::Real) -> Float64

Associated Legendre function `P_l^m(x)` for `0 ≤ m ≤ l`, including the
Condon-Shortley phase `(−1)^m`. Standard upward recurrence (Numerical
Recipes `plgndr`). For `m = 0` this is the ordinary Legendre polynomial
`P_l(x)`.
"""
function _assoc_legendre(l::Integer, m::Integer, x::Real)
    (0 <= m <= l) || throw(ArgumentError("_assoc_legendre needs 0 ≤ m ≤ l; got l=$l, m=$m"))
    pmm = 1.0
    if m > 0
        somx2 = sqrt((1 - x) * (1 + x))
        fact = 1.0
        for _ in 1:m
            pmm *= -fact * somx2     # (−1)^m (2i−1)!! (1−x²)^{m/2}
            fact += 2.0
        end
    end
    l == m && return pmm
    pmmp1 = x * (2m + 1) * pmm
    l == m + 1 && return pmmp1
    pll = 0.0
    for ll in (m + 2):l
        pll = ((2ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    end
    return pll
end

"""
    _spherical_harmonic_C(k::Integer, m::Integer, θ::Real, φ::Real) -> ComplexF64

Racah-normalized spherical harmonic `C^k_m(θ, φ) = √(4π/(2k+1)) Y_{km}(θ, φ)`,
normalized so `C^k_0(0, 0) = 1`. For `m ≥ 0`,
`C^k_m = √((k−m)!/(k+m)!) P_k^m(cos θ) e^{imφ}` (Condon-Shortley phase in
`P_k^m`); for `m < 0`, `C^k_m = (−1)^{|m|} conj(C^k_{|m|})`.
"""
function _spherical_harmonic_C(k::Integer, m::Integer, θ::Real, φ::Real)
    am = abs(m)
    am <= k || return ComplexF64(0)
    norm = sqrt(factorial(k - am) / factorial(k + am))
    val = norm * _assoc_legendre(k, am, cos(θ)) * cis(am * φ)
    return m < 0 ? ComplexF64((-1)^am * conj(val)) : ComplexF64(val)
end

"""
    _double_factorial(n::Integer) -> Float64

Double factorial `n!! = n·(n−2)·(n−4)·…` (down to 1 or 2); `(-1)!! = 0!! = 1`.
"""
function _double_factorial(n::Integer)
    r = 1.0
    k = n
    while k > 1
        r *= k
        k -= 2
    end
    return r
end

"""
    _bessel_small_x_series(k::Integer, x::Real) -> Float64

Spherical Bessel `j_k(x)` from its convergent power series

    j_k(x) = x^k / (2k+1)!! · Σ_{n≥0} (−x²/2)^n / (n! · (2k+3)(2k+5)…(2k+2n+1)),

used for small `|x|`, where the upward recurrence in `_spherical_bessel_j`
is numerically unstable (`x ≪ k`).
"""
function _bessel_small_x_series(k::Integer, x::Real)
    x2half = x * x / 2
    a = 1.0          # ratio of term_n to term_0
    s = 1.0
    n = 0
    while abs(a) > 1e-18 * abs(s) + 1e-300 && n < 200
        n += 1
        a *= -x2half / (n * (2k + 2n + 1))
        s += a
    end
    return s * x^k / _double_factorial(2k + 1)
end

"""
    _spherical_bessel_j(k::Integer, x::Real) -> Float64

Spherical Bessel function of the first kind `j_k(x)`. For `k ≥ 2` and
`|x| < k` (where the upward recurrence is unstable) the convergent power
series `_bessel_small_x_series` is used; elsewhere `j_0 = sin x / x`,
`j_1 = sin x / x² − cos x / x`, and the upward recurrence
`j_{n+1} = (2n+1)/x · j_n − j_{n−1}` (stable for `|x| ≳ k`).
"""
function _spherical_bessel_j(k::Integer, x::Real)
    k >= 0 || throw(ArgumentError("spherical Bessel order k must be ≥ 0; got $k"))
    iszero(x) && return k == 0 ? 1.0 : 0.0
    (k >= 2 && abs(x) < k) && return _bessel_small_x_series(k, x)
    j0 = sin(x) / x
    k == 0 && return j0
    j1 = sin(x) / x^2 - cos(x) / x
    k == 1 && return j1
    jm1, jn = j0, j1
    jp1 = j1
    for n in 1:(k - 1)
        jp1 = (2n + 1) / x * jn - jm1
        jm1, jn = jn, jp1
    end
    return jp1
end

"""
    radial_integral(R_a, R_b, r, k; kind = :bessel, q = 0.0, weight = :physical) -> Float64

Trapezoidal radial integral on the grid `r` (possibly non-uniform):

    ∫ R_a(r) · 𝒦_k(r) · R_b(r) · w(r) dr,

the radial factor of a rank-`k` transition moment.

- `kind = :bessel` → `𝒦_k = j_k(q·r)` (spherical Bessel; the nIXS moment
  `Rj_k(q)`); `kind = :power` → `𝒦_k = r^k` (the E2/Eλ moment `⟨r^k⟩`).
- `weight = :physical` → `w = r²` (inputs are the physical radial
  wavefunctions `R(r)`); `weight = :reduced` → `w = 1` (inputs are the
  reduced radial functions `u(r) = r·R(r)`, so `∫u_a j_k u_b dr =
  ∫R_a j_k R_b r² dr` — this matches the tabulated radial files as
  integrated in Quanty's tutorial 28 `28_NIXS_dd.lua`).

The grid `r` must have `≥ 2` strictly increasing, finite points.
- `q` is the momentum transfer (only used for `:bessel`); units must match
  `r` (Quanty's atomic files use `q` per Bohr radius `a₀` on a Bohr grid).
"""
function radial_integral(R_a::AbstractVector, R_b::AbstractVector,
                         r::AbstractVector, k::Integer;
                         kind::Symbol = :bessel, q::Real = 0.0,
                         weight::Symbol = :physical)
    n = length(r)
    (length(R_a) == n && length(R_b) == n) || throw(ArgumentError(
        "radial_integral: R_a, R_b, r must have equal length; got " *
        "$(length(R_a)), $(length(R_b)), $n."))
    n >= 2 || throw(ArgumentError(
        "radial_integral: need ≥ 2 grid points; got $n."))
    kind in (:bessel, :power) || throw(ArgumentError(
        "radial_integral: kind must be :bessel or :power; got :$kind."))
    weight in (:physical, :reduced) || throw(ArgumentError(
        "radial_integral: weight must be :physical or :reduced; got :$weight."))
    all(isfinite, r) || throw(ArgumentError(
        "radial_integral: grid r has non-finite entries."))
    all(i -> r[i + 1] > r[i], 1:(n - 1)) || throw(ArgumentError(
        "radial_integral: grid r must be strictly increasing."))
    f = Vector{Float64}(undef, n)
    for i in 1:n
        kernel = kind === :bessel ? _spherical_bessel_j(k, q * r[i]) : float(r[i])^k
        w = weight === :physical ? float(r[i])^2 : 1.0
        f[i] = float(R_a[i]) * kernel * float(R_b[i]) * w
    end
    s = 0.0
    for i in 1:(n - 1)
        s += 0.5 * (f[i] + f[i + 1]) * (r[i + 1] - r[i])
    end
    return s
end

"""
    _nixs_allowed_ranks(ℓ_a::Integer, ℓ_b::Integer) -> Vector{Int}

The multipole ranks `k` allowed in the `ℓ_a → ℓ_b` nIXS expansion:
triangle `|ℓ_a − ℓ_b| ≤ k ≤ ℓ_a + ℓ_b` with even parity `ℓ_a + ℓ_b + k`.
(d→d: `[0, 2, 4]`; p→d: `[1, 3]`.)
"""
_nixs_allowed_ranks(ℓ_a::Integer, ℓ_b::Integer) =
    [k for k in abs(ℓ_a - ℓ_b):(ℓ_a + ℓ_b) if iseven(ℓ_a + ℓ_b + k)]

"""
    nixs(m::ShellModel, shells::Pair{Symbol, Symbol};
         theta::Real, phi::Real, radial_integrals) -> OperatorSum

The non-resonant inelastic X-ray scattering (nIXS) transition operator
`e^{i q·r}` between two shells, at a momentum transfer **q** whose
direction is `q̂ = (theta, phi)` (polar, azimuthal) and whose magnitude
enters only through the radial Bessel moments `radial_integrals`:

    T_nIXS = Σ_k i^k (2k+1) Rj_k(q) Σ_m C^k_m(q̂)* · (C^k_m transition op),

summed over the allowed ranks `k` (triangle + parity). `radial_integrals`
is an `AbstractDict{Int, <:Number}` mapping each allowed `k` to its
spherical-Bessel moment `Rj_k(q) = ⟨R_b|j_k(qr)|R_a⟩` (BYO — compute with
`radial_integral`, or supply your own). The pair is the `shellA => shellB`
direction. The operator is generally non-Hermitian (it is the bare
`e^{i q·r}`), built by the raw `_ckm_contract`; feed it to
`dynamical_structure_factor` for the spectrum `S(q, ω)`.

# Errors
`ArgumentError` if `radial_integrals` is empty or names a rank `k` not in
the allowed set for `ℓ_a → ℓ_b`.

# Example: NiO 3d-3d nIXS (d→d, k ∈ {0,2,4})
```julia
m  = ShellModel([:Ni_3d])
# Rj_k from a tabulated reduced radial function u_3d(r) on grid r:
Rj = Dict(k => radial_integral(u3d, u3d, r, k; kind=:bessel, q=4.5, weight=:reduced)
          for k in (0, 2, 4))
T  = nixs(m, :Ni_3d => :Ni_3d; theta=0.0, phi=0.0, radial_integrals=Rj)
```
"""
function nixs(m::ShellModel, shells::Pair{Symbol, Symbol};
              theta::Real, phi::Real,
              radial_integrals::AbstractDict{<:Integer, <:Number})
    shellA, shellB = shells
    ellA = ell_of(m, shellA)
    ellB = ell_of(m, shellB)
    allowed = _nixs_allowed_ranks(ellA, ellB)
    isempty(radial_integrals) && throw(ArgumentError(
        "nixs: radial_integrals is empty; supply Rj_k for k in $allowed."))

    out = OperatorSum{ComplexF64}()
    for (k, Rjk) in sort(collect(radial_integrals); by = first)
        k in allowed || throw(ArgumentError(
            "nixs: rank k=$k is not allowed for $shellA (ℓ=$ellA) → " *
            "$shellB (ℓ=$ellB); allowed ranks: $allowed."))
        iszero(Rjk) && continue
        prefac = (im^k) * (2k + 1) * Rjk
        terms = [(k, mq, prefac * conj(_spherical_harmonic_C(k, mq, theta, phi)))
                 for mq in -k:k]
        out += _ckm_contract(m, shellA, shellB, terms)
    end
    return out
end
