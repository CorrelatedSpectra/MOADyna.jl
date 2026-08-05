# src/shells/dipole.jl
#
# Cross-shell electric-dipole (E1) operator via Wigner-Eckart on Racah's
# normalized C^1_q tensor (radial integral set to 1).
#
# Spherical components, for q ∈ {−1, 0, +1}:
#
#   T_q = Σ_{m_a, σ} ⟨ℓ_b, m_a + q | C^1_q | ℓ_a, m_a⟩
#                   c†(b, m_a + q, σ) c(a, m_a, σ)
#
# with the Gaunt-like coefficient
#
#   ⟨ℓ_b, m_b | C^1_q | ℓ_a, m_a⟩ = (−1)^m_b · sqrt((2 ℓ_a + 1)(2 ℓ_b + 1))
#                                   · 3j(ℓ_b, 1, ℓ_a; −m_b, q, m_a)
#                                   · 3j(ℓ_b, 1, ℓ_a; 0, 0, 0).
#
# Cartesian conversion (standard):
#
#   T_x = (T_{−1} − T_{+1}) / √2
#   T_y =  i (T_{−1} + T_{+1}) / √2
#   T_z =  T_0
#
# Parity rule: (ℓ_a + ℓ_b) must be ODD — the diagonal 3j vanishes
# otherwise. Triangle inequality |ℓ_a − ℓ_b| ≤ 1 ≤ ℓ_a + ℓ_b combined
# with parity restricts Δℓ to ±1 strictly. Same-shell (ℓ_a = ℓ_b)
# calls and Δℓ ≠ ±1 calls (e.g. p → f) are rejected with
# `ArgumentError`.
#
# Hermiticity note: T_x and T_y are NOT individually Hermitian for a
# CROSS-shell dipole. T_x for `:Ni_2p => :Ni_3d` is the absorption
# (core → valence) operator — its Hermitian conjugate is the emission
# operator. They are the canonical absorption operators consistent with
# Quanty's TXAS convention.
#
# Implemented as the k=1 special case of `multipole` (see multipole.jl);
# the C^1_q matrix elements come from the shared `_ckq_coeff`.

"""
    dipole(m::ShellModel, shells::Pair{Symbol, Symbol}) -> Vector{OperatorSum}

Cartesian E1 dipole operator `[T_x, T_y, T_z]` between two shells, via
Wigner-Eckart with Racah's `C^1_q` normalisation (radial integral = 1).
The pair `shellA => shellB` denotes the absorption direction
``c†(b) c(a)`` (from initial shell A to final shell B).

Spherical components `T_q`, q ∈ {−1, 0, +1}, are built as

    T_q = Σ_{m_a, σ} ⟨ℓ_b, m_a + q | C^1_q | ℓ_a, m_a⟩
                    c†(b, m_a + q, σ) c(a, m_a, σ),

with the Gaunt-like coefficient

    ⟨ℓ_b, m_b | C^1_q | ℓ_a, m_a⟩ = (−1)^m_b · sqrt((2ℓ_a+1)(2ℓ_b+1))
                                    · 3j(ℓ_b, 1, ℓ_a; −m_b, q, m_a)
                                    · 3j(ℓ_b, 1, ℓ_a; 0, 0, 0).

Cartesian assembly:

    T_x = (T_{−1} − T_{+1}) / √2,  T_y = i (T_{−1} + T_{+1}) / √2,  T_z = T_0.

# Parity rule
`(ℓ_a + ℓ_b)` must be odd. Combined with the triangle inequality this
forces Δℓ = ±1 strictly. Same-shell calls (e.g. d→d) and Δℓ ≠ ±1
calls (e.g. p→f) are rejected with `ArgumentError`.

# Arguments
- `m::ShellModel` — shell registry.
- `shells::Pair{Symbol, Symbol}` — `shellA => shellB`; the operator maps
  shell A (initial) to shell B (final), i.e. `c†(b) c(a)`.

# Returns
- `Vector{OperatorSum}` of length 3: `[T_x, T_y, T_z]`.

# Example: NiO L-edge XAS dipole (Ni 2p → Ni 3d)
```julia
m = ShellModel([:Ni_2p, :Ni_3d])
T = dipole(m, :Ni_2p => :Ni_3d)
T_x, T_y, T_z = T
```

# Notes
For a cross-shell dipole, `T_x` and `T_y` are NOT individually Hermitian
— each is the absorption operator only. Its adjoint is the emission
operator. This matches the Quanty TXAS convention.
"""
function dipole(m::ShellModel, shells::Pair{Symbol, Symbol})
    shellA, shellB = shells
    ellA = ell_of(m, shellA)
    ellB = ell_of(m, shellB)

    # E1-specific guards (clearer messages than the generic `multipole`
    # triangle/parity errors, and exercised by the dipole tests). For E1
    # these are equivalent to `multipole(..., 1)`'s guards: parity
    # ℓ_a+ℓ_b+1 even ⇔ ℓ_a+ℓ_b odd, and the triangle then forces Δℓ = 1.
    iseven(ellA + ellB) && throw(ArgumentError(
        "dipole: parity rule requires (ℓ_a + ℓ_b) odd; got " *
        "$shellA (ℓ=$ellA) → $shellB (ℓ=$ellB), sum = $(ellA + ellB)."))
    abs(ellA - ellB) == 1 || throw(ArgumentError(
        "dipole: E1 selection rule requires Δℓ = ±1; got " *
        "$shellA (ℓ=$ellA) → $shellB (ℓ=$ellB), Δℓ = $(abs(ellA - ellB))."))

    # E1 = rank-1 multipole. Spherical components T_{-1}, T_0, T_{+1} then
    # the standard Cartesian assembly (kept explicit so this is bit-for-bit
    # identical to the pre-refactor dipole; verified p→d and d→p).
    Tm1, T0, Tp1 = multipole(m, shells, 1)
    inv_sqrt2 = 1 / sqrt(2)
    Tx = inv_sqrt2 * (Tm1 - Tp1)
    Ty = (im * inv_sqrt2) * (Tm1 + Tp1)
    Tz = T0
    return OperatorSum[Tx, Ty, Tz]
end
