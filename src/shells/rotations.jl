# src/shells/rotations.jl
#
# Basis-change matrices for one shell of a `ShellModel`. Provides:
#   * `to_real(m, shell)` — (ℓm) → real cubic harmonic transform for
#     ℓ ∈ {0, 1, 2}. ℓ = 3 (f-shell) is deferred.
#   * `to_jlmj(m, shell)` — (ℓm, σ) → (j, m_j) j-coupled basis transform
#     via Clebsch-Gordan coefficients.
#
# Convention recap (matches `MOAD.Algebra.rotate`'s Sakurai convention):
#   columns of the returned `U` = new basis vectors expressed in the
#   old basis. Equivalently, `U[j, i]` = ⟨e_old_j | e_new_i⟩.
#
# Ordering (locked):
#   * Orbital: m-ordered (Questaal) — both old (ℓm) and new (real K) axes
#     run m = -ℓ, …, +ℓ. Real-K names per shell:
#       p: (p_y, p_z, p_x)
#       d: (d_xy, d_yz, d_{3z²-r²}, d_xz, d_{x²-y²})
#   * Spin layout: m-major + dn-then-up (spin FAST). For `to_real`, the
#     full per-shell 2(2ℓ+1) × 2(2ℓ+1) matrix is `kron(U_orbital, I_2)`.
#
# Phase convention: Condon-Shortley standard tesseral (real harmonics) /
# Condon-Shortley Clebsch-Gordan (j-coupling).

using WignerSymbols: clebschgordan

"""
    to_real(m::ShellModel, shell::Symbol) -> Matrix{ComplexF64}

Build the (ℓm) → real-cubic-harmonic basis-change matrix for one shell of
the model. Shape is `2(2ℓ+1) × 2(2ℓ+1)`; columns are the new real-K basis
vectors expressed in the old (ℓm) basis (Sakurai convention, matching
`MOAD.Algebra.rotate`).

Real-K column ordering (m-ordered, Questaal):
  * ℓ=0 (s): trivial 2×2 identity (spin block only).
  * ℓ=1 (p): `(p_y, p_z, p_x)` indexed by m_K = -1, 0, +1.
  * ℓ=2 (d): `(d_xy, d_yz, d_{3z²-r²}, d_xz, d_{x²-y²})` indexed by
    m_K = -2, -1, 0, +1, +2.

Spin layout: m-major + dn-then-up (spin FAST), so the full matrix factors
as `kron(U_orbital, I_2)`.

Phase convention: Condon-Shortley standard tesseral combinations.

Throws `ArgumentError` for ℓ = 3 (f-shell, deferred) and ℓ ≥ 4.
"""
function to_real(m::ShellModel, shell::Symbol)
    ell = ell_of(m, shell)
    if ell == 0
        return ComplexF64[1 0; 0 1]
    elseif ell == 1
        U_orb = _to_real_p()
    elseif ell == 2
        U_orb = _to_real_d()
    elseif ell == 3
        throw(ArgumentError(
            "to_real for f-shell (ℓ=3) deferred to a future task"))
    else
        throw(ArgumentError("to_real: unsupported ℓ = $ell"))
    end
    return kron(U_orb, ComplexF64[1 0; 0 1])
end

# --- helpers: hand-written orbital blocks, Condon-Shortley standard tesseral ---

function _to_real_p()
    # Rows: m_old = -1, 0, +1. Cols: m_K = -1, 0, +1 → (p_y, p_z, p_x).
    inv_sqrt2 = 1 / sqrt(2)
    U = zeros(ComplexF64, 3, 3)
    # col 1 (p_y, m_K = -1): (i/√2)(|m=-1⟩ + |m=+1⟩)
    U[1, 1] = im * inv_sqrt2
    U[3, 1] = im * inv_sqrt2
    # col 2 (p_z, m_K = 0): |m=0⟩
    U[2, 2] = 1
    # col 3 (p_x, m_K = +1): (1/√2)(|m=-1⟩ - |m=+1⟩)
    U[1, 3] = inv_sqrt2
    U[3, 3] = -inv_sqrt2
    return U
end

"""
    to_jlmj(m::ShellModel, shell::Symbol) -> Matrix{ComplexF64}

Build the (ℓ m_ℓ, σ) → (ℓ j m_j) basis-change matrix for one shell of
the model. Shape is `2(2ℓ+1) × 2(2ℓ+1)`; columns are the new
`|ℓ, j, m_j⟩` basis vectors expressed in the old `|ℓ, m_ℓ, σ⟩` basis
(Sakurai convention, matching `MOAD.Algebra.rotate`):

    U[ old=(m_ℓ, σ), new=(j, m_j) ] = ⟨ℓ, m_ℓ; 1/2, σ | j, m_j⟩

(standard Condon-Shortley Clebsch-Gordan).

Old-basis (row) ordering: m-major + dn-then-up, **spin FAST** —
`(m_ℓ=-ℓ, σ=-1//2), (m_ℓ=-ℓ, σ=+1//2), (m_ℓ=-ℓ+1, σ=-1//2), …`.

New-basis (column) ordering: j ascending then m_j ascending; j ranges
over {ℓ-1/2, ℓ+1/2} (the j=ℓ-1/2 block is omitted for ℓ=0).
  * ℓ=0 (s): `(1/2, ±1/2)`.
  * ℓ=1 (p): `(1/2,-1/2), (1/2,+1/2), (3/2,-3/2), …, (3/2,+3/2)`.
  * ℓ=2 (d): `(3/2,-3/2), …, (3/2,+3/2), (5/2,-5/2), …, (5/2,+5/2)`.

`Rational{Int}` half-integer literals (`1//2`, `3//2`) are required by
`WignerSymbols.clebschgordan`.
"""
function to_jlmj(m::ShellModel, shell::Symbol)
    ell = ell_of(m, shell)
    if ell == 0
        # s-shell: only j = 1/2. CG ⟨0,0; 1/2, σ | 1/2, m_j⟩ = δ_{σ, m_j}.
        return ComplexF64[1 0; 0 1]
    end
    # New-basis columns: j ascending, then m_j ascending.
    j_values = (ell - 1//2, ell + 1//2)
    cols_jmj = Tuple{Rational{Int}, Rational{Int}}[]
    for j in j_values
        for m_j in (-j):j
            push!(cols_jmj, (j, m_j))
        end
    end
    n_dim = 2 * (2 * ell + 1)
    @assert length(cols_jmj) == n_dim
    U = zeros(ComplexF64, n_dim, n_dim)
    # CG conservation: m_ℓ + σ = m_j ⇒ at most two nonzero σ per column.
    ell_r = Rational{Int}(ell)
    for (col_idx, (j, m_j)) in enumerate(cols_jmj)
        for σ in (-1//2, +1//2)
            m_l = m_j - σ                  # forced by CG conservation
            (m_l < -ell_r || m_l > ell_r) && continue
            cg = clebschgordan(ell_r, m_l, 1//2, σ, j, m_j)
            iszero(cg) && continue
            # Old-basis row index: m-major + dn-then-up (spin FAST).
            m_l_int = Int(m_l)             # m_l is integer for this CG
            σ_idx = (σ == -1//2) ? 1 : 2
            row_idx = 2 * (m_l_int + ell) + σ_idx
            U[row_idx, col_idx] = ComplexF64(Float64(cg))
        end
    end
    return U
end

function _to_real_d()
    # Rows: m_old = -2, -1, 0, +1, +2. Cols: m_K = -2, -1, 0, +1, +2.
    inv_sqrt2 = 1 / sqrt(2)
    U = zeros(ComplexF64, 5, 5)
    # col 1 (d_xy, m_K = -2): (i/√2)(|m=-2⟩ - |m=+2⟩)
    U[1, 1] = im * inv_sqrt2
    U[5, 1] = -im * inv_sqrt2
    # col 2 (d_yz, m_K = -1): (i/√2)(|m=-1⟩ + |m=+1⟩)
    U[2, 2] = im * inv_sqrt2
    U[4, 2] = im * inv_sqrt2
    # col 3 (d_{3z²-r²}, m_K = 0): |m=0⟩
    U[3, 3] = 1
    # col 4 (d_xz, m_K = +1): (1/√2)(|m=-1⟩ - |m=+1⟩)
    U[2, 4] = inv_sqrt2
    U[4, 4] = -inv_sqrt2
    # col 5 (d_{x²-y²}, m_K = +2): (1/√2)(|m=-2⟩ + |m=+2⟩)
    U[1, 5] = inv_sqrt2
    U[5, 5] = inv_sqrt2
    return U
end
