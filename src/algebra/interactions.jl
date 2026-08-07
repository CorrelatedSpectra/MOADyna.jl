# src/algebra/interactions.jl
#
# Algebra-level interaction primitives for fermionic multi-orbital sites.
# These take a FermionSite + a list of (dn, up) orbital-mode pairs as input.
# Shell-keyed forms in MOADyna.Shells are sugar that resolve a shell to its
# orbital pairs and delegate here.
#
# Covered: density_density, kanamori (Hubbard-Kanamori interactions),
#          and spin operators Sz, Splus, Sminus, Sx, Sy, Ssqr on
#          FermionSite (complements the SpinSite-based forms in operators.jl).

# =====================================================================
# Density-density interaction
# =====================================================================

"""
    density_density(site::FermionSite, orbital_pairs; U, J = 0, Up = U - 2J)

Density-density (Ising-like Hubbard-Kanamori) interaction over a list of
orbital mode pairs `[(dn_α, up_α), …]` on a single `FermionSite`.

Each pair `(dn, up)` identifies the two fermion-mode indices (within `site`)
that share an orbital — one for each spin. Coulomb structure:

    H_DD =  U      Σ_α        n_α↑ n_α↓
         + (U−3J)  Σ_{α<β,σ}  n_ασ n_βσ
         + (U−2J)  Σ_{α<β,σ}  n_ασ n_β,−σ

Default `Up = U − 2J` (cubic spin-rotation invariant relation); pass `Up=`
explicitly to override.

# Arguments

- `site` — the `FermionSite` the operators act on.
- `orbital_pairs` — a collection of `(dn_idx, up_idx)` tuples identifying
  the modes for each orbital. Indices are 1-based within `site`.
- `U` — intra-orbital Hubbard scale.
- `J` (default 0) — Hund's exchange.
- `Up` (default `U - 2J`) — inter-orbital opposite-spin coefficient.
"""
function density_density(site::FermionSite, orbital_pairs::AbstractVector;
                         U, J = 0, Up = U - 2J)
    pairs = collect(orbital_pairs)   # canonicalize to a Vector

    isempty(pairs) && error(
        "density_density: orbital_pairs must be non-empty (got an empty list)")

    # Seed accumulator: zero OperatorSum of the correct promoted type.
    out = 0 * (cdag(site, pairs[1][1]) * c(site, pairs[1][1]))

    # Intra-orbital: U n_α↑ n_α↓
    for (d_idx, u_idx) in pairs
        out += U * n(site, u_idx) * n(site, d_idx)
    end

    parallel    = Up - J   # = U − 3J by default
    antiparallel = Up       # = U − 2J by default

    for i in 1:length(pairs), j in (i+1):length(pairs)
        a_d, a_u = pairs[i]
        b_d, b_u = pairs[j]
        # Parallel spin: n_ασ n_βσ for σ = ↑ and σ = ↓
        out += parallel * n(site, a_u) * n(site, b_u)
        out += parallel * n(site, a_d) * n(site, b_d)
        # Opposite spin: n_ασ n_β,−σ — both orderings
        out += antiparallel * n(site, a_u) * n(site, b_d)
        out += antiparallel * n(site, a_d) * n(site, b_u)
    end
    return out
end

# =====================================================================
# Full Kanamori interaction
# =====================================================================

"""
    kanamori(site::FermionSite, orbital_pairs; U, J, Up = U - 2J)

Full (spin-rotation invariant) Kanamori interaction: density-density part
plus spin-flip and pair-hopping terms, over a list of orbital mode pairs on
a `FermionSite`.

    H = H_DD + H_SF + H_PH

with

    H_DD  — see `density_density`
    H_SF  = −J Σ_{α≠β} c†(α,↑) c(α,↓) c†(β,↓) c(β,↑)   # spin-flip
    H_PH  =  J Σ_{α≠β} c†(α,↑) c†(α,↓) c(β,↓) c(β,↑)   # pair-hop

# Arguments

- `site` — the `FermionSite` the operators act on.
- `orbital_pairs` — a collection of `(dn_idx, up_idx)` tuples.
- `U` — intra-orbital Hubbard scale.
- `J` — Hund's exchange.
- `Up` (default `U - 2J`) — inter-orbital opposite-spin coefficient.
"""
function kanamori(site::FermionSite, orbital_pairs::AbstractVector;
                  U, J, Up = U - 2J)
    pairs = collect(orbital_pairs)
    H = density_density(site, pairs; U = U, J = J, Up = Up)

    for i in 1:length(pairs), j in 1:length(pairs)
        i == j && continue
        a_d, a_u = pairs[i]
        b_d, b_u = pairs[j]
        # Spin-flip: −J · c†(α,↑) c(α,↓) c†(β,↓) c(β,↑)
        H += -J * cdag(site, a_u) * c(site, a_d) * cdag(site, b_d) * c(site, b_u)
        # Pair-hop: J · c†(α,↑) c†(α,↓) c(β,↓) c(β,↑)
        H += J * cdag(site, a_u) * cdag(site, a_d) * c(site, b_d) * c(site, b_u)
    end
    return H
end

# =====================================================================
# Spin operators on FermionSite — orbital-pair primitives
# =====================================================================
#
# These extend Sz, Splus, Sminus, Sx, Sy (already defined in operators.jl
# for SpinSite) with new methods for FermionSite + orbital-pair list.
# Ssqr is new (the SpinSite version is in operators.jl's S() bundle;
# a dedicated Ssqr function is introduced here).

"""
    Sz(site::FermionSite, orbital_pairs) -> OperatorSum

z-component of total spin on a `FermionSite` restricted to a list of
orbital mode pairs `[(dn, up), …]`:

    Sz = (1/2) Σ_α (n(α, up) − n(α, dn))
"""
function Sz(site::FermionSite, orbital_pairs::AbstractVector)
    pairs = collect(orbital_pairs)
    return sum((1//2) * (n(site, p[2]) - n(site, p[1])) for p in pairs)
end

"""
    Splus(site::FermionSite, orbital_pairs) -> OperatorSum

Raising operator S₊ = Σ_α c†(α, up) c(α, dn) on a `FermionSite`.
"""
function Splus(site::FermionSite, orbital_pairs::AbstractVector)
    pairs = collect(orbital_pairs)
    return sum(cdag(site, p[2]) * c(site, p[1]) for p in pairs)
end

"""
    Sminus(site::FermionSite, orbital_pairs) -> OperatorSum

Lowering operator S₋ = (S₊)† on a `FermionSite`.
"""
Sminus(site::FermionSite, orbital_pairs::AbstractVector) =
    adjoint(Splus(site, orbital_pairs))

"""
    Sx(site::FermionSite, orbital_pairs) -> OperatorSum

x-component of total spin: Sx = (1/2)(S₊ + S₋) on a `FermionSite`.
"""
Sx(site::FermionSite, orbital_pairs::AbstractVector) =
    (1//2) * (Splus(site, orbital_pairs) + Sminus(site, orbital_pairs))

"""
    Sy(site::FermionSite, orbital_pairs) -> OperatorSum

y-component of total spin: Sy = (−i/2)(S₊ − S₋) on a `FermionSite`.
"""
Sy(site::FermionSite, orbital_pairs::AbstractVector) =
    (-im/2) * (Splus(site, orbital_pairs) - Sminus(site, orbital_pairs))

"""
    Ssqr(site::FermionSite, orbital_pairs) -> OperatorSum

Total spin squared S² = Sx² + Sy² + Sz² on a `FermionSite`.
"""
function Ssqr(site::FermionSite, orbital_pairs::AbstractVector)
    sx = Sx(site, orbital_pairs)
    sy = Sy(site, orbital_pairs)
    sz = Sz(site, orbital_pairs)
    return sx*sx + sy*sy + sz*sz
end
