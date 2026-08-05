# src/shells/group_projected.jl
#
# Group-projected crystal-field operator on a shell.
#
# Wraps `MOAD.PointGroups.expand_clm(G, ℓ, irrep_coeffs)` — which returns
# the rank-k spherical-tensor expansion `[(k=k, m=m, coeff=A_km), ...]` —
# and contracts it with the shell's `C^k_m` matrix elements (Wigner-Eckart)
# to produce the second-quantized form
#
#     V = Σ_{k,m} A_{km}  Σ_{m_a, m_b, σ}  ⟨ℓ m_b | C^k_m | ℓ m_a⟩
#                          c†(b, m_b, σ) c(b, m_a, σ).
#
# We re-use `MOAD.PointGroups.Bkm_matrix(ℓ, k, m)` for the matrix
# elements rather than re-deriving the Wigner-Eckart formula here. This
# guarantees the C^k_m convention agrees with the one `expand_clm` uses
# to *define* the A_km, so the round-trip is convention-consistent.
#
# Note on cross-module dependency: MOAD.PointGroups is included AFTER
# MOAD.Shells in `src/MOAD.jl`, so we cannot `using ..PointGroups` here
# at module-load time. Resolution via `MOAD.PointGroups.<name>` happens
# inside the function body, which runs at call time when the
# PointGroups submodule is fully loaded.

"""
    Akm(m::ShellModel, shell::Symbol, group::Symbol,
        irrep_coeffs::AbstractVector;
        experimental::Bool=false) -> OperatorSum

Crystal-field operator on a single shell, projected onto the irreducible
representations of a point group.

Builds the rank-k spherical-tensor expansion via
`MOAD.PointGroups.expand_clm(G, ℓ, irrep_coeffs)` (with
`G = pointgroup(group)`), then contracts each `(k, m, A_km)` triple with
the shell's `C^k_m` matrix elements `⟨ℓ m_b | C^k_m | ℓ m_a⟩` to produce
the second-quantized operator

    V = Σ_{k,m} A_{km}  Σ_{m_a, m_b, σ}
                ⟨ℓ m_b | C^k_m | ℓ m_a⟩ c†(b, m_b, σ) c(b, m_a, σ).

The shell modes are the (m_l, σ) fermionic modes laid out by
`_orbital_mode_pair`; both spin channels are summed.

# Arguments
- `m::ShellModel` — the shell registry.
- `shell::Symbol` — the shell tag (e.g. `:Ni_3d`).
- `group::Symbol` — the point group label (e.g. `:Oh`, `:D4h`).
- `irrep_coeffs::AbstractVector` — packed real parameters in
  `subduce(G, ℓ)` order; same convention as
  `expand_clm(G, ℓ, irrep_coeffs)`. For multiplicity-free occurrences
  this is one scalar per occurring IR (the `expand_clm_central` form);
  with multiplicities, each IR contributes coordinates on its
  CF-reachable Hermitian basis — for real `⊗I` IRs the historical
  diagonal-then-symmetric-off-diagonal packing, for trigonal/pentagonal
  `⊗J` IRs additionally an imaginary direction (see `expand_clm`).
- `experimental::Bool` — forwarded to `expand_clm`. Default `false`
  enforces the strict gates (reference-labelled groups, `m_Γ ≤ 2`).

# Example: cubic 10Dq splitting on a 3d shell
```julia
m = ShellModel([:Ni_3d])
op = Akm(m, :Ni_3d, :Oh, [0.6, -0.4])
# subduce(Oh, 2) = Eg ⊕ T2g; this places the Eg pair at +0.6 and the
# T2g triplet at -0.4 in the single-particle spectrum.
```
"""
function Akm(m::ShellModel, shell::Symbol, group::Symbol,
             irrep_coeffs::AbstractVector;
             experimental::Bool=false)
    site = site_of(m, shell)
    ell  = ell_of(m, shell)

    # Late-bound resolution: PointGroups is included after Shells in
    # src/MOAD.jl, so we look up these names at call time.
    PG = getfield(parentmodule(@__MODULE__), :PointGroups)
    G = PG.pointgroup(group)
    clm_list = PG.expand_clm(G, ell, irrep_coeffs;
                             experimental=experimental)

    out = OperatorSum{ComplexF64}()
    for entry in clm_list
        k, m_q, A_km = entry.k, entry.m, entry.coeff
        iszero(A_km) && continue
        # ⟨ℓ m_b | C^k_m | ℓ m_a⟩ for all (m_a, m_b); rows = m_b, cols = m_a.
        B = PG.Bkm_matrix(ell, k, m_q)
        for (j, m_a) in enumerate(-ell:ell), (i, m_b) in enumerate(-ell:ell)
            ckme = B[i, j]
            iszero(ckme) && continue
            d_a, u_a = _orbital_mode_pair(ell, m_a)
            d_b, u_b = _orbital_mode_pair(ell, m_b)
            coeff = A_km * ckme
            out += coeff * (cdag(site, u_b) * c(site, u_a) +
                            cdag(site, d_b) * c(site, d_a))
        end
    end
    # The Wigner-Eckart sum is Hermitian by construction (Hermitian H_Γ
    # blocks ⇒ Hermitian C^k_m contraction), but the per-(k,m) loop
    # accumulates roundoff in coefficients that should cancel in
    # Hermitian-conjugate pairs. Symmetrise + chop so the returned
    # operator is exactly equal to its adjoint and the assembled matrix
    # passes `ishermitian` for the eigen contract in `MOAD.ED`.
    return chop((out + out') / 2; tol = 1e-12)
end

"""
    hop(m::ShellModel, shellA::Symbol, shellB::Symbol, group::Symbol;
        irrep::Symbol) -> OperatorSum

Irrep-projected single-particle hybridization between two shells of
**equal ℓ**.

Given the IR projector `P_Γ` on the `(2ℓ+1)`-dim spherical-harmonic
space (from `MOAD.PointGroups.project(G, Γ, ℓ)`), build

    H_proj = Σ_σ Σ_{j,k}  P_Γ[j, k]  c†(A, j, σ) c(B, k, σ)  +  h.c.

with **unit overall amplitude** — the user multiplies by the physical
hybridization scalar (e.g. `V_eg`) at the call site. Both spin
channels are summed.

# Arguments
- `m::ShellModel` — the shell registry.
- `shellA, shellB::Symbol` — the two shell tags. Must satisfy
  `ell_of(m, shellA) == ell_of(m, shellB)`.
- `group::Symbol` — the point-group label (e.g. `:Oh`).
- `irrep::Symbol` — the Mulliken IR label, initial-caps as in the
  group's character table (`:Eg`, `:T2g` for `Oh`).

# Errors
- `ArgumentError` if `shellA` and `shellB` have different ℓ.
- `ArgumentError` if `irrep` does not appear in `subduce(G, ℓ)`.
- `ArgumentError` if the IR's multiplicity in `subduce(G, ℓ)` is `≥ 2`
  (multiplicity-disambiguated hop is not yet implemented).

# Example: Oh d-shell hybridization (eg channel)
```julia
m = ShellModel([:Ni_3d, :L_3d])
V_eg = 2.06   # eV (NiO canonical)
H_eg = V_eg * hop(m, :Ni_3d, :L_3d, :Oh; irrep=:Eg)
```
"""
function hop(m::ShellModel, shellA::Symbol, shellB::Symbol, group::Symbol;
             irrep::Symbol)
    ellA = ell_of(m, shellA)
    ellB = ell_of(m, shellB)
    ellA == ellB || throw(ArgumentError(
        "hop: shells must have the same ℓ; got $shellA (ℓ=$ellA) vs " *
        "$shellB (ℓ=$ellB)"))
    ell = ellA
    siteA = site_of(m, shellA)
    siteB = site_of(m, shellB)

    # Late-bound resolution; mirrors `Akm` above.
    PG = getfield(parentmodule(@__MODULE__), :PointGroups)
    G  = PG.pointgroup(group)

    # Multiplicity gate. Look up m_Γ via subduce.
    sub = PG.subduce(G, ell)
    idx = findfirst(p -> first(p) === irrep, sub)
    idx === nothing && throw(ArgumentError(
        "hop: irrep :$irrep does not appear in subduce($(G.name), $ell); " *
        "have " * join(string.(first.(sub)), ", ")))
    m_Γ = last(sub[idx])
    if m_Γ >= 2
        throw(ArgumentError(
            "hop: irrep :$irrep occurs $(m_Γ) times in subduce(" *
            "$(G.name), $ell); multiplicity-disambiguated hop is not yet " *
            "implemented."))
    end

    # P[j, k] on the (2ℓ+1) spherical-harmonic basis. Hermitian, idempotent.
    P = PG.project(G, irrep, ell)

    out = OperatorSum{ComplexF64}()
    n_orb = 2 * ell + 1
    for j in 1:n_orb, k in 1:n_orb
        coeff = P[j, k]
        iszero(coeff) && continue
        m_a = j - ell - 1   # row index → orbital m on shell A
        m_b = k - ell - 1   # col index → orbital m on shell B
        d_a, u_a = _orbital_mode_pair(ell, m_a)
        d_b, u_b = _orbital_mode_pair(ell, m_b)
        # Sum both spin channels.
        out += coeff * (cdag(siteA, u_a) * c(siteB, u_b) +
                        cdag(siteA, d_a) * c(siteB, d_b))
    end
    # Hermitize: H_proj = out + out†. Symmetrise + chop in the same
    # style as `Akm`, so the returned operator is exactly self-adjoint
    # and clean of roundoff terms below 1e-12.
    return chop(out + out'; tol = 1e-12)
end
