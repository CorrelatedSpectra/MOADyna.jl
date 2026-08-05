# src/shells/angular_momentum.jl
#
# Shell-keyed orbital angular momentum operators. Diagonal in the
# (ℓ, m_ℓ) basis used by the m-major + dn-then-up mode layout, so
# Lz reduces to a number-operator sum:
#
#     Lz = Σ_{m_ℓ = -ℓ}^{ℓ}  m_ℓ · ( n(m_ℓ, ↑) + n(m_ℓ, ↓) )
#
# The ladder operators couple adjacent m_ℓ on each spin channel:
#
#     L⁺ = Σ_{m_ℓ = -ℓ}^{ℓ-1} sqrt(ℓ(ℓ+1) − m_ℓ(m_ℓ+1)) ·
#               ( c†(m_ℓ+1, ↑) c(m_ℓ, ↑) + c†(m_ℓ+1, ↓) c(m_ℓ, ↓) )
#     L⁻ = (L⁺)†

"""
    Lz(m::ShellModel, shell::Symbol) -> OperatorSum

Orbital angular momentum z-component on a shell. In the (ℓ, m_ℓ)
basis underlying the m-major + dn-then-up mode layout, `Lz` is
diagonal:

    Lz = Σ_{m_ℓ = -ℓ}^{ℓ}  m_ℓ · ( n(m_ℓ, ↑) + n(m_ℓ, ↓) )

independent of spin. Returns an `OperatorSum` over the modes of the
shell's `FermionSite`.
"""
function Lz(m::ShellModel, shell::Symbol)
    site = site_of(m, shell)
    ell = ell_of(m, shell)
    m_values = _m_values(ell)
    nonzero = filter(!iszero, m_values)
    isempty(nonzero) && return 0 * n(site, 1)
    return sum(
        let (dn_idx, up_idx) = _orbital_mode_pair(ell, m_l)
            m_l * (n(site, up_idx) + n(site, dn_idx))
        end
        for m_l in nonzero
    )
end

"""
    Lplus(m::ShellModel, shell::Symbol) -> OperatorSum

Raising operator for orbital angular momentum on a shell:

    L⁺ = Σ_{m_ℓ = -ℓ}^{ℓ-1} sqrt(ℓ(ℓ+1) − m_ℓ(m_ℓ+1)) ·
             ( c†(m_ℓ+1, ↑) c(m_ℓ, ↑) + c†(m_ℓ+1, ↓) c(m_ℓ, ↓) )

Acts as `L⁺ |ℓ, m_ℓ, σ⟩ = sqrt(ℓ(ℓ+1) − m_ℓ(m_ℓ+1)) |ℓ, m_ℓ+1, σ⟩` on
each spin channel σ. For an s-shell (ℓ=0) the operator is identically
zero (returned as an empty `OperatorSum` of the right type).
"""
function Lplus(m::ShellModel, shell::Symbol)
    site = site_of(m, shell)
    ell = ell_of(m, shell)
    if ell == 0
        return 0 * n(site, 1)
    end
    return sum(
        let (d_lo, u_lo) = _orbital_mode_pair(ell, m_l),
            (d_hi, u_hi) = _orbital_mode_pair(ell, m_l + 1),
            coeff = sqrt(ell * (ell + 1) - m_l * (m_l + 1))
            coeff * (cdag(site, u_hi) * c(site, u_lo) +
                     cdag(site, d_hi) * c(site, d_lo))
        end
        for m_l in -ell:(ell - 1)
    )
end

"""
    Lminus(m::ShellModel, shell::Symbol) -> OperatorSum

Lowering operator for orbital angular momentum on a shell, `L⁻ = (L⁺)†`:

    L⁻ = Σ_{m_ℓ = -ℓ+1}^{ℓ} sqrt(ℓ(ℓ+1) − m_ℓ(m_ℓ−1)) ·
             ( c†(m_ℓ−1, ↑) c(m_ℓ, ↑) + c†(m_ℓ−1, ↓) c(m_ℓ, ↓) )

Acts as `L⁻ |ℓ, m_ℓ, σ⟩ = sqrt(ℓ(ℓ+1) − m_ℓ(m_ℓ−1)) |ℓ, m_ℓ−1, σ⟩` on
each spin channel σ. Implemented as `adjoint(Lplus(m, shell))`, so the
return type matches `Lplus`. For an s-shell (ℓ=0) the operator is
identically zero (empty `OperatorSum` of the right type), because
`Lplus` is zero there and `adjoint` preserves that.

See also: `Lplus`, `Lx`, `Ly`.
"""
Lminus(m::ShellModel, shell::Symbol) = adjoint(Lplus(m, shell))

"""
    Lx(m::ShellModel, shell::Symbol) -> OperatorSum

Cartesian x-component of orbital angular momentum on a shell,
`Lx = (L⁺ + L⁻) / 2`. Returns an `OperatorSum` built from
`Lplus` and `Lminus` for the shell's modes. For an s-shell (ℓ=0)
both ladder operators are identically zero, so `Lx` is also zero
(empty `OperatorSum` of the right type).

See also: `Ly`, `Lz`, `Lplus`, `Lminus`.
"""
Lx(m::ShellModel, shell::Symbol) = (1//2) * (Lplus(m, shell) + Lminus(m, shell))

"""
    Ly(m::ShellModel, shell::Symbol) -> OperatorSum

Cartesian y-component of orbital angular momentum on a shell,
`Ly = (L⁺ − L⁻) / (2i)`. Returns a complex-coefficient `OperatorSum`
built from `Lplus` and `Lminus` for the shell's modes. For an s-shell
(ℓ=0) both ladder operators are identically zero, so `Ly` is also zero
(empty `OperatorSum` of the right type).

See also: `Lx`, `Lz`, `Lplus`, `Lminus`.
"""
Ly(m::ShellModel, shell::Symbol) = (-im / 2) * (Lplus(m, shell) - Lminus(m, shell))

"""
    Lsqr(m::ShellModel, shell::Symbol) -> OperatorSum

Magnitude squared of orbital angular momentum: `L² = Lx² + Ly² + Lz²`.
On a one-electron `|ℓ, m_ℓ, σ⟩` state this evaluates to ℓ(ℓ+1).
"""
function Lsqr(m::ShellModel, shell::Symbol)
    lx = Lx(m, shell)
    ly = Ly(m, shell)
    lz = Lz(m, shell)
    return lx*lx + ly*ly + lz*lz
end

# --- total angular momentum J = L + S ---
#
# Each component delegates to the orbital (L) and spin (S) shell-keyed
# operators already in scope; no new modes or coefficients are introduced.

"""
    Jx(m::ShellModel, shell::Symbol) -> OperatorSum

Cartesian x-component of total angular momentum on a shell,
`Jx = Lx + Sx`, returned as an `OperatorSum`. For an s-shell (ℓ=0)
the orbital part `Lx` is identically zero, so `Jx` reduces to the
spin operator `Sx` alone.

See also: `Jy`, `Jz`, `Jplus`, `Jminus`, `Lx`, `Sx`.
"""
Jx(m::ShellModel, shell::Symbol) = Lx(m, shell) + Sx(m, shell)

"""
    Jy(m::ShellModel, shell::Symbol) -> OperatorSum

Cartesian y-component of total angular momentum on a shell,
`Jy = Ly + Sy`, returned as an `OperatorSum`. For an s-shell (ℓ=0)
the orbital part `Ly` is identically zero, so `Jy` reduces to the
spin operator `Sy` alone.

See also: `Jx`, `Jz`, `Jplus`, `Jminus`, `Ly`, `Sy`.
"""
Jy(m::ShellModel, shell::Symbol) = Ly(m, shell) + Sy(m, shell)

"""
    Jz(m::ShellModel, shell::Symbol) -> OperatorSum

z-component of total angular momentum on a shell, `Jz = Lz + Sz`,
returned as an `OperatorSum`. For an s-shell (ℓ=0) the orbital part
`Lz` is identically zero (the only m_ℓ value is 0, which has a
vanishing coefficient), so `Jz` reduces to the spin operator `Sz`
alone.

See also: `Jx`, `Jy`, `Jplus`, `Jminus`, `Lz`, `Sz`.
"""
Jz(m::ShellModel, shell::Symbol) = Lz(m, shell) + Sz(m, shell)

"""
    Jplus(m::ShellModel, shell::Symbol) -> OperatorSum

Raising operator for total angular momentum on a shell,
`J⁺ = L⁺ + S⁺`, returned as an `OperatorSum`. For an s-shell (ℓ=0)
the orbital part `Lplus` is identically zero, so `Jplus` reduces to
the spin raising operator `Splus` alone.

See also: `Jminus`, `Jx`, `Jy`, `Jz`, `Lplus`, `Splus`.
"""
Jplus(m::ShellModel, shell::Symbol) = Lplus(m, shell) + Splus(m, shell)

"""
    Jminus(m::ShellModel, shell::Symbol) -> OperatorSum

Lowering operator for total angular momentum on a shell,
`J⁻ = L⁻ + S⁻ = (J⁺)†`, returned as an `OperatorSum`. For an
s-shell (ℓ=0) the orbital part `Lminus` is identically zero, so
`Jminus` reduces to the spin lowering operator `Sminus` alone.

See also: `Jplus`, `Jx`, `Jy`, `Jz`, `Lminus`, `Sminus`.
"""
Jminus(m::ShellModel, shell::Symbol) = Lminus(m, shell) + Sminus(m, shell)

"""
    Jsqr(m::ShellModel, shell::Symbol) -> OperatorSum

Magnitude squared of total angular momentum: `J² = Jx² + Jy² + Jz²`.
On a one-electron `|ℓ, m_ℓ, σ⟩` shell with spin-1/2 this evaluates to
the j(j+1) eigenvalues for j = ℓ ± 1/2.
"""
function Jsqr(m::ShellModel, shell::Symbol)
    jx = Jx(m, shell)
    jy = Jy(m, shell)
    jz = Jz(m, shell)
    return jx*jx + jy*jy + jz*jz
end

# --- atomic spin-orbit (one-body, single-particle l·s) ---
#
# H_SO = ζ Σ_i l_i · s_i is built from single-particle matrix elements,
# NOT as the product of total operators L · S — the latter generates
# spurious two-body cross terms (Σ_{i≠j} l^i · s^j). In the (m_ℓ, σ)
# basis the decomposition is:
#
#     l·s = lz·sz + (1/2)(l⁺ s⁻ + l⁻ s⁺)
#
# with one-particle matrix elements
#
#     ⟨m, σ|lz·sz|m, σ⟩ = m · σ_z/2
#     ⟨m+1, ↓|(1/2) l⁺ s⁻|m, ↑⟩ = (1/2) sqrt(ℓ(ℓ+1) − m(m+1))
#
# and the (1/2) l⁻ s⁺ piece is the adjoint of (1/2) l⁺ s⁻.

"""
    LS(m::ShellModel, shell::Symbol) -> OperatorSum

ONE-BODY atomic spin-orbit on a single shell:

    H_SO = Σ_i (l_i · s_i) = Σ_{ab} ⟨a|l·s|b⟩ c†(a) c(b)

This is **not** the product of total angular-momentum operators
`L · S` — that would generate spurious two-body terms
`Σ_{i≠j} l_i · s_j`. Decomposition used:

    H_SO = Σ_m  m/2 · ( n(m, ↑) − n(m, ↓) )                                  (lz·sz)
         + (1/2) Σ_m sqrt(ℓ(ℓ+1) − m(m+1)) · c†(m+1, ↓) c(m, ↑) + h.c.        (l⁺ s⁻ + l⁻ s⁺)

Returns an `OperatorSum` whose every term has at most two ladder
entries (a one-body operator). For an s-shell (ℓ=0) the operator is
identically zero (returned as an empty `OperatorSum` of the right type).
"""
function LS(m::ShellModel, shell::Symbol)
    site = site_of(m, shell)
    ell  = ell_of(m, shell)

    # lz·sz: diagonal in (m_ℓ, σ). Skip m_ℓ = 0 since the coefficient vanishes.
    diag = 0 * n(site, 1)
    for m_l in -ell:ell
        m_l == 0 && continue
        d_idx, u_idx = _orbital_mode_pair(ell, m_l)
        diag += (m_l / 2) * (n(site, u_idx) - n(site, d_idx))
    end

    # (1/2) l⁺ s⁻: |m, ↑⟩ → (1/2) sqrt(ℓ(ℓ+1) − m(m+1)) |m+1, ↓⟩.
    # For ℓ=0 the loop is empty; mirror Lplus's idiom for a type-stable empty sum.
    if ell == 0
        return diag
    end
    raising = 0 * cdag(site, 1) * c(site, 1)
    for m_l in -ell:(ell - 1)
        coeff = 0.5 * sqrt(ell * (ell + 1) - m_l * (m_l + 1))
        d_hi, _ = _orbital_mode_pair(ell, m_l + 1)
        _, u_lo = _orbital_mode_pair(ell, m_l)
        raising += coeff * cdag(site, d_hi) * c(site, u_lo)
    end
    # (1/2) l⁻ s⁺ is the adjoint of (1/2) l⁺ s⁻.
    return diag + raising + adjoint(raising)
end
