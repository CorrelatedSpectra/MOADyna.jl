# Multiplets & Standard Operators

```@meta
CurrentModule = MOADyna
DocTestSetup  = quote
    using MOADyna
end
```

`MOADyna.Shells` is the multiplet layer: a [`ShellModel`](@ref) registry that names the
atomic shells of a problem, plus shell-keyed builders for every standard operator a
ligand-field / multiplet calculation needs — Slater–Condon Coulomb, spin–orbit coupling,
crystal field, hybridization, angular momentum, and the electromagnetic transition /
moment operators. Each builder returns an [`OperatorSum`](@ref) (or a vector of them) in
the same symbolic algebra as the rest of MOADyna, so multiplet terms add directly to any
other part of a Hamiltonian and assemble through the same [`compile`](@ref) /
[`assemble`](@ref) path.

This chapter is task-oriented: it walks the construction of a transition-metal
multiplet Hamiltonian term by term, then the transition operators used for spectroscopy.
It covers the standard multiplet and transition operators currently implemented in MOADyna
v0.2.

## The shell registry

A [`ShellModel`](@ref) is built from a list of shell tags `:Element_nℓ`. Each tag fixes an
orbital angular momentum ``\ell`` (`s`, `p`, `d`, `f` ``\to`` 0, 1, 2, 3) and lays out its ``2(2\ell+1)`` spin–orbitals
on a [`FermionSite`](@ref MOADyna.Algebra.FermionSite); ligand ("bath") shells are named like any other shell.

```@example shells
using MOADyna

# Ni L-edge problem: a 2p core shell, the Ni 3d valence shell, and a ligand
# 3d-symmetry bath shell.
m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])

# orbital angular momenta ℓ of each shell (s,p,d,f → 0,1,2,3)
(ell_of(m, :Ni_2p), ell_of(m, :Ni_3d), ell_of(m, :L_3d))
```

`nshells(m, :Ni_3d)` returns the occupation-count *observable* for a shell — used to build
sector restrictions (`nshells(m, :Ni_3d) == 8`), as shown at the end of this chapter.

## Slater–Condon Coulomb

[`coulomb`](@ref) builds the normal-ordered two-body Coulomb interaction from Slater
integrals. The single-shell form takes the direct integrals `F` ``= (F^2, F^4, \ldots)``; the
two-shell form (e.g. a core–valence interaction) additionally takes the exchange
integrals `G`.

```@example shells
# d–d Coulomb on the Ni 3d shell (F⁰ folded into U).
H_dd = coulomb(m, :Ni_3d; U = 0.0, F = (11.14, 6.87))

# 2p–3d core–valence Coulomb (direct F² + exchange G¹, G³), as in L-edge XAS.
H_pd = coulomb(m, :Ni_2p, :Ni_3d; U = 0.0, F = (6.67,), G = (4.92, 2.80))

(H_dd isa OperatorSum, H_pd isa OperatorSum)
```

### Kanamori parametrisation

For general multi-orbital Hubbard / DMFT-style models, [`kanamori`](@ref) builds the **full
rotationally-invariant** on-shell interaction — intra/inter-orbital density–density, Hund
spin-flip, and pair-hopping — from `U`, the Hund coupling `J`, and (optionally) `Up`
(default ``U - 2J``). [`density_density`](@ref) is the Ising-like (density-only) reduction.

```@example shells
H_kan = kanamori(m, :Ni_3d; U = 7.3, J = 0.9)       # U′ = U − 2J by default
H_dd_only = density_density(m, :Ni_3d; U = 7.3, J = 0.9)
(H_kan isa OperatorSum, H_dd_only isa OperatorSum)
```

## Spin–orbit coupling

[`LS`](@ref) builds the one-body spin–orbit operator ``\sum_i \hat{l}_i\cdot\hat{s}_i`` on a
shell (the constant ``\zeta`` is *not* included); multiply by the atomic spin–orbit constant
``\zeta`` to form ``H_{\mathrm{SO}} = \zeta\sum_i \hat{l}_i\cdot\hat{s}_i``.

```@example shells
ζ_3d = 0.081      # eV
H_soc = ζ_3d * LS(m, :Ni_3d)
H_soc isa OperatorSum
```

## Crystal field

[`Akm`](@ref) builds the crystal-field potential on a shell, projected onto the
irreducible representations of a point group (see the [Point Groups](@ref) chapter). For
a cubic (``O_h``) ``d`` shell the parameters are the ``(e_g, t_{2g})`` splittings; the canonical
`10Dq` convention is `[0.6, -0.4] · 10Dq`.

```@example shells
tenDq = 0.56      # eV
H_cf  = tenDq * Akm(m, :Ni_3d, :Oh, [0.6, -0.4])
H_cf isa OperatorSum
```

## Hybridization / ligand field

[`hop`](@ref) builds the irrep-projected single-particle hybridization between two shells
of equal `ℓ` with unit amplitude; multiply by the physical hopping (e.g. `V_eg`, `V_t2g`).

```@example shells
V_eg, V_t2g = 2.06, 1.21      # eV
H_hyb = V_eg  * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :Eg) +
        V_t2g * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :T2g)
H_hyb isa OperatorSum
```

## Angular-momentum operators

The shell-keyed angular-momentum operators are available as Cartesian components and
ladder operators: [`Lz`](@ref)/`Lx`/`Ly`/`Lplus`/`Lminus`, the spin operators `Sz`/…, and
the total-`J` set [`Jz`](@ref)/…. These are used for Zeeman terms, expectation tables, and
sum rules.

```@example shells
Bz, Hz = 1.0e-6, 0.120        # eV (a tiny symmetry-breaking field + Zeeman)
H_zeeman = Bz * (2 * Sz(m, :Ni_3d) + Lz(m, :Ni_3d)) + Hz * Sz(m, :Ni_3d)
H_zeeman isa OperatorSum
```

### Single-particle basis transforms

MOADyna works internally in the spherical-harmonic ``|\ell m\sigma\rangle`` basis.
[`to_real`](@ref) and [`to_jlmj`](@ref) return the per-shell single-particle change-of-basis
matrices to the **real cubic harmonics** and to the **spin–orbit-coupled** ``|\ell, j, m_j\rangle``
basis respectively — for reading or rotating single-particle operators and states into whichever
basis a given analysis is natural in (real-cubic for crystal field, ``jj`` for strong spin–orbit).

```@example shells
U_real = to_real(m, :Ni_3d)     # |ℓmσ⟩ → real cubic harmonics
U_jmj  = to_jlmj(m, :Ni_3d)     # |ℓmσ⟩ → |j, m_j⟩
(size(U_real), size(U_jmj))     # 10×10 for a d shell (5 orbitals × 2 spins)
```

## Electromagnetic transition operators

MOADyna builds electric-multipole transition operators by Wigner–Eckart on Racah's
normalized ``C^{k}_{q}`` tensor. The radial integral defaults to 1 and can be supplied via
the `radial` keyword.

### Electric dipole (E1)

[`dipole`](@ref) returns the Cartesian components ``[T_x, T_y, T_z]`` for an electric-dipole
(Δℓ = ±1) transition, in the `shellA => shellB` absorption direction. This is the operator
used by [`xas`](@ref) and [`rixs`](@ref).

```@example shells
T = dipole(m, :Ni_2p => :Ni_3d)     # Ni L-edge 2p → 3d
length(T)
```

### General rank-`k` multipole and E2

[`multipole`](@ref) is the general rank-`k` engine, returning the ``2k+1`` spherical
components ``[T^k_{-k},\ldots,T^k_{+k}]`` (E1 is the `k = 1` case). [`quadrupole_transition`](@ref) is
the user-facing electric-quadrupole (E2) operator, returning the five real tesseral
components ``[z^2,\ xz,\ yz,\ x^2{-}y^2,\ xy]``.

```@example shells
mq = ShellModel([:Ni_1s, :Ni_3d])
Q_E2 = quadrupole_transition(mq, :Ni_1s => :Ni_3d)   # K pre-edge quadrupole
length(Q_E2)
```

### Charge-quadrupole moment and spin-dipole `T`

[`quadrupole`](@ref) is the charge-quadrupole *moment* ``Q_{ij} = 3 r_i r_j - r^2 \delta_{ij}`` (a same-shell
expectation operator, six dimensionless Cartesian components), and [`spin_dipole_T`](@ref)
is the spin-dipole tensor ``T = \sum_i [\,s_i - 3\hat{r}_i(\hat{r}_i\cdot s_i)\,]`` that enters the XMCD spin sum rule
(three Cartesian components) — *not* the magnetic-dipole operator `L + 2S`.

```@example shells
Qmom = quadrupole(m, :Ni_3d)        # [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]
Tspin = spin_dipole_T(m, :Ni_3d)    # [Tx, Ty, Tz]
(length(Qmom), length(Tspin))
```

### Non-resonant inelastic X-ray scattering (nIXS)

[`nixs`](@ref) builds the ``e^{i \mathbf{q}\cdot\mathbf{r}}`` scattering operator as the plane-wave multipole
expansion ``\sum_k i^k(2k+1) R^j_k(q) \sum_m C^k_m(\hat q)^* C^k_m(\hat r)``, summed over the allowed ranks
(d→d: `k ∈ {0,2,4}`). The radial Bessel moments ``R^j_k(q) = \langle R|j_k(qr)|R\rangle`` are supplied per
rank; [`radial_integral`](@ref) computes them (or the ``\langle r^k\rangle`` moments) from a tabulated
radial wavefunction. The resulting operator is handed to
[`dynamical_structure_factor`](@ref) for ``S(\mathbf{q},\omega)``; see `examples/08_nio_nixs.jl`.

```@example shells
Rj = Dict(0 => 0.070, 2 => 0.161, 4 => 0.086)   # ⟨R₃d|j_k(qr)|R₃d⟩ at q ≈ 4.5/a₀
T_q = nixs(m, :Ni_3d => :Ni_3d; theta = 0.0, phi = 0.0, radial_integrals = Rj)
T_q isa OperatorSum
```

## Atomic parameters

The Slater integrals (``F^k``/``G^k``), spin–orbit constants (ζ), and tabulated radial moments
(``\langle r^2\rangle``, ``\langle r^4\rangle``) used above can be looked up for transition-metal and lanthanide
configurations with [`atomic_parameters`](@ref) — see the [Atomic Parameters](@ref)
chapter. The nIXS Bessel moments ``R^j_k(q) = \langle R|j_k(qr)|R\rangle`` are a *different* quantity:
they are momentum-dependent and are computed from a radial wavefunction with
[`radial_integral`](@ref) (or supplied directly), not read from the atomic-parameter
table.

## Putting it together

A complete ground-state multiplet Hamiltonian is the sum of these terms; restrict to the
physical electron count and diagonalize. The end-to-end NiO workflow (ground state → XAS →
RIXS → nIXS) is the subject of the [NiO: XAS, RIXS, and nIXS](@ref) tutorial.

```@example shells
H_GS = coulomb(m, :Ni_3d; U = 0.0, F = (11.14, 6.87)) +
       0.081 * LS(m, :Ni_3d) +
       0.56  * Akm(m, :Ni_3d, :Oh, [0.6, -0.4])
b  = basis(m, nshells(m, :Ni_3d) == 8, nshells(m, :Ni_2p) == 6, nshells(m, :L_3d) == 10)
gs = eigen(assemble(compile(H_GS, b), b), b; n = 1)
round(gs.values[1]; digits = 4)
```

All `@example` blocks above are executed at build time against the real API, so every
snippet in this chapter is verified to run.
