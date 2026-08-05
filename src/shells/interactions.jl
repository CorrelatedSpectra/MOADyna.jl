# src/shells/interactions.jl
#
# Shell-keyed sugar for density_density and kanamori. These extend the
# Algebra-level primitives (imported in Shells.jl via `import ..Algebra:
# density_density, kanamori`) with ShellModel-keyed dispatch that resolves
# a shell tag to its full orbital-pair list, then delegates to the Algebra
# primitive.
#
# Per user directive: no orbital-subset dispatch on shells (that would
# leak unused modes into the Hilbert). Only full-shell and vararg-of-shells
# forms are provided.

# --- density_density ---

"""
    density_density(m::ShellModel, shell::Symbol; U, J = 0, Up = U - 2J)
    -> OperatorSum

Density-density (Ising-like Hubbard-Kanamori) Coulomb interaction on one
shell. Resolves `shell` to its full orbital-pair list and delegates to
`density_density(site::FermionSite, orbital_pairs; U, J, Up)` in
MOAD.Algebra.

The conventional spin-rotation-invariant relation `Up = U − 2J` is used
by default; pass `Up=` explicitly to override.

    H_DD =  U      Σ_α       n_α↑ n_α↓
         + (U−3J) Σ_{α<β,σ} n_ασ n_βσ
         + (U−2J) Σ_{α<β,σ} n_ασ n_β,−σ

# Arguments

- `m::ShellModel` — the shell registry.
- `shell::Symbol` — the shell tag, e.g. `:Ni_3d`.
- `U` — intra-orbital Hubbard scale.
- `J` (default 0) — Hund's exchange.
- `Up` (default `U - 2J`) — inter-orbital opposite-spin coefficient.
"""
function density_density(m::ShellModel, shell::Symbol; U, J = 0, Up = U - 2J)
    site = site_of(m, shell)
    ell = ell_of(m, shell)
    pairs = [_orbital_mode_pair(ell, mv) for mv in _m_values(ell)]
    return density_density(site, pairs; U = U, J = J, Up = Up)
end

# --- kanamori ---

"""
    kanamori(m::ShellModel, shell::Symbol; U, J, Up = U - 2J) -> OperatorSum

Full (spin-rotation invariant) Kanamori interaction on one shell:
density-density + spin-flip + pair-hop. Resolves `shell` to its full
orbital-pair list and delegates to `kanamori(site::FermionSite, orbital_pairs;
U, J, Up)` in MOAD.Algebra.

    H = H_DD + H_SF + H_PH

Default `Up = U − 2J` (cubic spin-rotation-invariant relation).

# Arguments

- `m::ShellModel` — the shell registry.
- `shell::Symbol` — the shell tag, e.g. `:Ni_3d`.
- `U` — intra-orbital Hubbard scale.
- `J` — Hund's exchange.
- `Up` (default `U - 2J`) — inter-orbital opposite-spin coefficient.
"""
function kanamori(m::ShellModel, shell::Symbol; U, J, Up = U - 2J)
    site = site_of(m, shell)
    ell = ell_of(m, shell)
    pairs = [_orbital_mode_pair(ell, mv) for mv in _m_values(ell)]
    return kanamori(site, pairs; U = U, J = J, Up = Up)
end
