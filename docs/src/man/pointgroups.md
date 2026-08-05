# Point Groups

```@meta
CurrentModule = MOAD
DocTestSetup  = quote
    using MOAD
end
```

`MOAD.PointGroups` turns a Schoenflies symbol into a fully computable group: it enumerates the
symmetry operations, builds the character table (cross-checked against curated reference
tables), and derives subduction, projectors, and crystal-field parameters from first
principles. It covers **42 reference-labelled groups** (32 crystallographic + 10
non-crystallographic molecular) plus any group you supply generators for. This chapter is the
task-oriented tour; the [Point-Group Theory Primer](@ref) appendix is the deeper reference.

## Constructing a group

[`pointgroup`](@ref) builds the group eagerly — all elements, conjugacy classes, and
irreducible representations are computed at construction:

```@example pg
using MOAD

G = pointgroup(:Oh)
(order = length(G.elements), nclasses = length(G.classes), nirreps = length(G.irreps))
```

[`reference_label_groups`](@ref) lists the groups with curated Mulliken labels, and
[`has_reference_labels`](@ref) checks a single one:

```@example pg
(n_reference_groups = length(reference_label_groups()), Oh = has_reference_labels(:Oh))
```

## Character table

[`character_table`](@ref) returns the classes, irreps, and the character matrix;
[`print_character_table`](@ref MOAD.PointGroups.print_character_table) pretty-prints it, and [`character_table_compare`](@ref) shows
the reference vs computed orderings side by side.

```@example pg
ct = character_table(G; source = :reference)
(ct.classes, ct.irreps)
```

```@example pg
print_character_table(G)
```

## Subduction: crystal-field splitting

When the full-rotation ``D^\ell`` shell is restricted to a point group it becomes reducible —
[`subduce`](@ref) returns the irrep content `Γ => multiplicity`, i.e. how the ``(2\ell+1)``-fold
degeneracy splits:

```@example pg
(d_shell = subduce(G, 2),     # ℓ=2 in Oh: e_g ⊕ t_2g
 f_shell = subduce(G, 3))     # ℓ=3 in Oh: a_2u ⊕ t_1u ⊕ t_2u
```

So a ``d`` electron in an octahedral field splits into a 2-fold ``E_g`` and a 3-fold ``T_{2g}``
level — the familiar crystal-field picture, here derived rather than assumed.

## Projectors

[`project`](@ref) builds the projector ``P_\Gamma`` onto an irrep subspace of the shell (a
Hermitian idempotent whose trace is the subspace dimension):

```@example pg
using LinearAlgebra
P = project(G, :T2g, 2)
(idempotent = round(norm(P^2 - P); sigdigits = 2), trace = round(real(tr(P)); digits = 3))
```

## Crystal-field parameters

[`nparams`](@ref) counts the independent crystal-field parameters, and
[`expand_clm_central`](@ref) converts per-irrep energies into the Racah ``A_{km}`` coefficients
— the same coefficients [`Akm`](@ref) uses to build the crystal-field operator in the
[Multiplets & Standard Operators](@ref) chapter:

```@example pg
nP  = nparams(G, 2)                                 # two energies (E_g, T_2g)
akm = expand_clm_central(G, 2, [0.6, -0.4])         # ε(E_g)=0.6, ε(T_2g)=-0.4 eV
(nparams = nP, Akm = [(a.k, a.m, round(real(a.coeff); digits = 4)) for a in akm])
```

For a cubic ``d`` shell only the ``k = 4`` terms survive, with the cubic-harmonic constraint
``A_{4,\pm4} = \sqrt{5/14}\,A_{4,0}`` enforced automatically. Lower-symmetry groups (e.g.
`:C4v`, `:D4h`, `:Td`) and multiplicity-aware ``f``-shell blocks via [`expand_clm`](@ref MOAD.PointGroups.expand_clm) are
worked in the [Point-Group Theory Primer](@ref).

## Classifying a subspace

[`classify_subspace`](@ref) decomposes a ``G``-invariant subspace into irreps from its
character ``\chi_S(g) = \operatorname{Tr}_S U(g)`` — the tool for assigning a **term symbol** to
a computed multiplet. With the full ``d`` shell as the subspace (character from the Wigner
matrices [`wignerd`](@ref)) it reproduces the subduction:

```@example pg
χ_d = ComplexF64[tr(wignerd(g, 2)) for g in G.elements]
classify_subspace(χ_d, G)
```

## Classifying a many-body state

To classify an actual *many-body* eigenstate, [`classify_state`](@ref)`(ψ, basis, m, G)` builds
the Fock-space representation ``\hat U(g)`` of each group element from the single-particle Wigner
matrices and returns the projector weights ``w_\Gamma = \langle\psi|P_\Gamma|\psi\rangle``. As a
worked example, the Ni²⁺ ``d^8`` ground state in a cubic field is the orbital singlet, spin
triplet ``{}^3A_{2g}`` — and the routine confirms its spatial irrep is ``A_{2g}``:

```@example pg
using MOAD
md  = ShellModel([:Ni_3d])
Hd  = coulomb(md, :Ni_3d; U = 0.0, F = (11.14, 6.87)) +
      0.56 * Akm(md, :Ni_3d, :Oh, [0.6, -0.4])
bd  = basis(md, nshells(md, :Ni_3d) == 8)
gsd = eigen(assemble(compile(Hd, bd), bd), bd; n = 1)
res = classify_state(gsd.vectors[:, 1], bd, md, pointgroup(:Oh))
res.dominant_IR
```

This is a **spatial** (orbital) point-group action with the spin label held fixed — a
diagnostic for the spatial irrep, not a double-group (spinor) representation. Apply it to a
spin-independent model (no spin–orbit coupling, no magnetic field), where ``\hat U(g)`` is a
genuine symmetry; the [NiO: XAS, RIXS, and nIXS](@ref) tutorial’s full model adds spin–orbit
coupling, so its term symbol carries the spin label separately.
