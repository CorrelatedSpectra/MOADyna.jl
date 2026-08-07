# Point-Group Theory Primer

This document introduces MOADyna's `PointGroups` module for readers who know
quantum mechanics but have not yet worked through a systematic group theory
course. The level is roughly that of Atkins–Child–Phillips "Tables for Group
Theory" or Cotton "Chemical Applications of Group Theory" — worked crystal-field
examples are the primary vehicle.

For the task-oriented walkthrough of the same engine, see the
[Point Groups](@ref) chapter; this primer is the longer, less formal companion.

---

## 1. What this layer does

A point group is a finite set of rotations and improper rotations (reflections,
inversion, rotoreflections) that leave a molecule or coordination polyhedron
unchanged. The word "point" indicates that at least one spatial point — the
centre — is fixed under every operation. In practice, for transition-metal
spectroscopy, the relevant group is determined by the coordination geometry:
octahedral (O_h), tetrahedral (T_d), square-planar (C_4v or D_4h), trigonal
bipyramidal (D_3h), and so on.

`MOADyna.PointGroups` converts a Schoenflies symbol into a computable object. Once
you call `pointgroup(:Oh)`, the module has done the following for you:

- enumerated all 48 symmetry operations (elements) of O_h by BFS closure from
  a minimal generator set stored in `data.jl`,
- divided those 48 elements into 10 conjugacy classes,
- computed the 10 irreducible representations (irreps) and their Mulliken labels
  (A1g, T2g, …) by a Burnside–Dixon diagonalisation in the regular
  representation, cross-validated against curated reference tables,
- stored Wigner D-matrix machinery so that any element can act on any
  angular-momentum shell.

Everything else in the module — subduction, projectors, crystal-field expansion,
term-symbol classification — is built from those four ingredients.

### Quick start

```@example primer
using MOADyna

G = pointgroup(:Oh)           # construct O_h; eager: all irreps computed now
print_character_table(G)      # pretty-prints the canonical 10×10 table

subduce(G, 2)                 # how D^2 (d-shell) decomposes into Oh irreps
```

```@example primer
expand_clm_central(G, 2, [0.6, -0.4])
# returns Akm coefficients for a d-shell in Oh with ε(Eg)=0.6, ε(T2g)=-0.4
```

For the full set of supported groups, call `reference_label_groups()`, which
returns all 42 groups that have curated Mulliken labels:

```@example primer
reference_label_groups()
```

---

## 2. Group construction

### From Schoenflies symbol to `PointGroup`

```@example primer
G = pointgroup(:Oh)
```

This single call runs a pipeline:

1. Look up the minimal Cartesian generator matrices for O_h in
   `POINTGROUP_GENERATORS` (stored in `src/pointgroups/data.jl`).
2. Run BFS closure: multiply generators pairwise until no new elements appear.
   For O_h this gives 48 orthogonal 3×3 matrices.
3. Build the multiplication table, partition elements into conjugacy classes.
4. Compute the complex character table via simultaneous diagonalisation of class
   operators in the regular representation (the Burnside–Dixon algorithm). This
   gives correct irrep dimensions even for degenerate classes.
5. Apply Frobenius–Schur folding to collapse complex-conjugate pairs of irreps
   into real Mulliken irreps (needed for cyclic-family groups like C_3, C_4v,
   …).
6. Match computed character rows against the curated reference table to assign
   canonical Mulliken labels (A1g, T2g, …). Any mismatch raises an error — the
   reference table and the Burnside computation serve as mutual oracles.

The result is a `PointGroup` struct that is fully populated at construction
time. Constructing O_h takes well under a second.

### Conjugacy classes

Two elements $g, h \in G$ are conjugate when there exists $k \in G$ such that
$h = k g k^{-1}$. The conjugacy class of $g$ is the full orbit under this
equivalence. Characters — the traces of representation matrices — are constant
on each class, which is why character tables have one column per class rather
than one column per element.

For reference-table groups the class labels follow the Schoenflies–Mulliken
convention: `E` (identity), `8C3` (eight three-fold rotations around body
diagonals), `6C2pr` (six two-fold rotations around face diagonals), etc. The
number prefix is the class size.

For groups not in the reference table (constructed from arbitrary generators or
with `experimental=true`), classes are named by BFS-discovery order:
`E`, `C3_111`, `C4z·C3_111`, … These names are deterministic and stable within
a MOADyna version but should not be treated as publication-quality Mulliken labels.
Use `has_reference_labels(:YourGroup)` to check which regime you are in.

### Irreducible representations and Frobenius–Schur folding

Every finite group has exactly as many distinct irreps (over the complex
numbers) as it has conjugacy classes. For groups with complex-conjugate pairs of
irreps — the cyclic-family groups — the displayed character table folds each pair
into a single real Mulliken label with doubled dimension. For example, C_3 has
complex irreps $e^{2\pi i/3}$ and $e^{-2\pi i/3}$ which appear together as the 2D
real irrep E. (The 2D E irreps of groups like C_4v are genuinely irreducible over
the reals — a single constituent, not a folded pair.)

Internally, MOADyna keeps track of `ComplexIR` objects (the absolutely irreducible
complex constituents) and the displayed `IRrep` (the Mulliken view). This
two-layer design is what makes projector formulas correct for both the simple
A1g/T2g cases and the folded E cases. Users do not normally need to touch
`ComplexIR` directly.

### Frame rotations

If you need the group realised in a non-standard Cartesian frame (e.g. with C4
along the y-axis rather than z), use `rotate`:

```@example primer
Ry = [0 0 1; 0 1 0; -1 0 0]     # rotate z → x
G_rotated = rotate(G, Ry)
```

The rotated group is a new `PointGroup` with the same abstract structure and the
same Mulliken labels, but different element matrices.

---

## 3. Character tables

The character table collects the trace of each irrep matrix evaluated at each
class representative. It encodes the entire representation theory of the group:
whether representations are real or complex, how they tensor-multiply, how many
times a given irrep occurs when a bigger representation is restricted to the
group.

### Accessing the character table

```@example primer
G = pointgroup(:Oh)

# Source :reference — canonical published ordering of classes and IRs
ct = character_table(G; source=:reference)
ct.classes   # [:E, :C3, :C2pr, :C4, :C2sq, :i, :S4, :S6, :σh, :σd]
ct.irreps    # [:A1g, :A2g, :Eg, :T1g, :T2g, :A1u, :A2u, :Eu, :T1u, :T2u]
ct.characters  # 10×10 Matrix{Float64}

# Source :computed — same algebra, BFS-discovery class ordering
ct2 = character_table(G; source=:computed)

# Human-readable comparison of both orderings
character_table_compare(G)
```

`print_character_table(G)` shows only the reference table (if available) or the
computed table (for groups without a reference entry), formatted for the REPL.

### Example: O_h character table

```@example primer
print_character_table(pointgroup(:Oh))
```

The key physical fact encoded here: T1g and T2g differ by the sign of their
character on the face-diagonal C2 class (`6C2pr`). T1g has $\chi(C_2') = -1$,
T2g has $\chi(C_2') = +1$. This is how MOADyna (following Cotton's convention)
distinguishes the two three-dimensional irreps of O_h. Readers sometimes confuse
the T1/T2 labelling — always check the character at `6C2pr` or `6C4`.

### Key property: character orthogonality

For any two irreps $\Gamma$, $\Lambda$ of $G$:

$$\frac{1}{|G|} \sum_{g \in G} \chi_\Gamma(g)^* \, \chi_\Lambda(g) = \delta_{\Gamma \Lambda}$$

This inner product is the workhorse of projection theory. MOADyna's `subduce` and
`classify_subspace` both evaluate exactly this sum.

---

## 4. Subduction and projectors

### Subduction D^ℓ ↓ G

An angular-momentum shell of quantum number $\ell$ carries a $(2\ell+1)$-dimensional
representation $D^\ell$ of the full rotation group SO(3). When we restrict to
the finite point group $G \subset \text{SO}(3)$ (or O(3) for groups with
improper elements), $D^\ell$ becomes reducible. It decomposes into a direct sum
of irreps:

$$D^\ell \downarrow G = \bigoplus_\Gamma m_\Gamma \, \Gamma$$

where $m_\Gamma$ is the multiplicity — the number of times $\Gamma$ appears.
The multiplicity is computed by the character-orthogonality inner product:

$$m_\Gamma = \frac{1}{|G|} \sum_{g \in G} \chi_\Gamma(g)^* \, \chi_{D^\ell}(g)$$

In MOADyna:

```@example primer
G = pointgroup(:Oh)
subduce(G, 2)   # d-shell (ℓ=2) restricted to Oh
```

```@example primer
subduce(G, 3)   # f-shell (ℓ=3)
```

```@example primer
subduce(pointgroup(:C4v), 2)
```

The physical meaning: an electron in a d-shell ($\ell=2$) placed in an
octahedral crystal field sees its five-fold degeneracy split into a 2D level
($E_g$) and a 3D level ($T_{2g}$). The $m_\Gamma = 1$ for both means there is no
multiplicity mixing in the Oh d-shell case.

### IR projectors

The projector onto the $\Gamma$ subspace of $V_\ell$ is:

$$P_\Gamma = \frac{d_\Gamma}{|G|} \sum_{g \in G} \chi_\Gamma(g)^* \, D^\ell(g)$$

where $d_\Gamma$ is the dimension of $\Gamma$. In MOADyna:

```@example primer
G = pointgroup(:Oh)
P_Eg  = project(G, :Eg,  2)   # 5×5 complex Hermitian matrix
P_T2g = project(G, :T2g, 2)   # 5×5 complex Hermitian matrix
```

Projectors satisfy $P_\Gamma^2 = P_\Gamma$ (idempotent) and
$P_\Gamma^\dagger = P_\Gamma$ (Hermitian). MOADyna enforces both at construction.

Quick proof of concept:

```@example primer
using MOADyna, LinearAlgebra
G = pointgroup(:Oh)
P = project(G, :T2g, 2)

# Idempotency: P² = P
@show norm(P^2 - P)

# Trace = dimension of the subspace = 3 (T2g is 3D)
@show real(tr(P))

# Apply to a tesseral harmonic vector e_{m=0}
v = zeros(ComplexF64, 5)
v[3] = 1.0   # m=0 component (index 3 in -2,-1,0,+1,+2 ordering)
Pv = P * v
@show norm(Pv);   # for T2g: the m=0 harmonic lies entirely in the Eg subspace
```

The last line encodes a known crystal-field fact: the $d_{z^2}$ orbital (m=0
component) is part of $E_g$, not $T_{2g}$, in octahedral symmetry.

---

## 5. Crystal-field expansion: `expand_clm`

This section covers the most physics-relevant part of the module. The functions
here answer: "given the energy splittings of the d (or f) orbitals in this
coordination environment, what are the $A_{km}$ crystal-field parameters?"

### Setup: what the CF Hamiltonian looks like

On a single $\ell$-shell, a $G$-invariant *crystal field* (a multiplicative
potential $\sum A_{km} C^k_m$) is block-diagonal over the distinct Mulliken
irreps $\Gamma$ occurring in $D^\ell \downarrow G$:

$$V = \bigoplus_\Gamma H_\Gamma$$

where $m_\Gamma$ is the multiplicity of $\Gamma$ and $d_\Gamma$ its dimension.
For most groups each block factorises as $H_\Gamma = M_\Gamma \otimes I_{d_\Gamma}$
with a real-symmetric multiplicity matrix $M_\Gamma$ (size $m_\Gamma \times m_\Gamma$)
— e.g. tetragonal/orthorhombic groups. For the **trigonal and pentagonal** $E$
irreps (and the no-symmetry C1/Ci case) the copies couple through a
row-endomorphism ($\otimes J$) instead, so $H_\Gamma$ is genuinely Hermitian on
the full $m_\Gamma d_\Gamma$-dimensional isotypic block and carries an *imaginary*
coupling (e.g. D3d's trigonal $A_{4,\pm3}$) that a real-symmetric $\otimes I$
matrix cannot represent. `expand_clm` parametrizes each block on its
*CF-reachable* Hermitian basis, which covers both cases.

The total number of real parameters is `nparams(G, ℓ)`:

$$N_\text{params} = \sum_\Gamma \dim\big(\text{CF-reachable}_\Gamma\big)$$

which equals $\sum_\Gamma m_\Gamma(m_\Gamma+1)/2$ for the $\otimes I$ irreps.
For most cubic-symmetry d-shell problems $m_\Gamma = 1$ for every $\Gamma$, so
each block is a single number — one energy per irrep ("multiplicity-free"). The
"multiplicity-aware" case appears for groups like $C_{4v}$ with an f-shell, where
the $E$ irrep occurs twice.

```@example primer
G = pointgroup(:Oh)
@show nparams(G, 2)   # (Eg and T2g, both m=1: two energies)
@show nparams(G, 3)   # (A2u, T1u, T2u all m=1; one 1×1 block each)

G4 = pointgroup(:C4v)
@show nparams(G4, 2)  # (A1, B1, B2, E; all m=1)
@show nparams(G4, 3); # (A1, B1, B2: m=1; E: m=2 → contributes 3)
```

### `expand_clm_central` — the multiplicity-free helper

When every multiplicity $m_\Gamma = 1$ (or when you want to set
$H_\Gamma = \varepsilon_\Gamma I_{m_\Gamma}$ for simplicity), use:

```julia
G = pointgroup(:Oh)
akm = expand_clm_central(G, 2, [ε_Eg, ε_T2g])
```

The energies are in the order returned by `subduce(G, 2)`:

```@example primer
subduce(G, 2)  # first entry: Eg, second: T2g
```

Example: octahedral d-shell with $\varepsilon_{E_g} = 0.6$ eV and
$\varepsilon_{T_{2g}} = -0.4$ eV (so 10Dq = 1.0 eV):

```@example primer
using MOADyna
G = pointgroup(:Oh)
akm = expand_clm_central(G, 2, [0.6, -0.4])
for a in akm
    println("A$(a.k)$(a.m) = ", round(real(a.coeff), digits=4))
end
```

For Oh symmetry with a d-shell only $k=4$ terms survive (Wigner–Eckart
requires even $k$ with $k \le 2\ell$, and Oh's selection rule kills $k=2$).
The relation $A_{4,\pm 4} = \sqrt{5/14} \, A_{4,0}$ is the famous cubic
harmonic constraint enforced automatically by MOADyna.

### `expand_clm` — the general multiplicity-aware interface

For the full multiplicity-aware case, pass one block matrix per irrep in
`subduce` order:

```@example primer
G = pointgroup(:Oh)
sub = subduce(G, 2)

# One 1×1 matrix per m=1 irrep:
blocks = [fill(0.6, 1, 1), fill(-0.4, 1, 1)]
akm = expand_clm(G, 2, blocks)
```

Flat-vector overload (packs diagonal then upper-triangle per block, in
`subduce` order):

```@example primer
expand_clm(G, 2, [0.6, -0.4])   # equivalent for the m=1 case
```

### Example: d-shell in C_{4v} (square-planar / C4-symmetric coordination)

Square-planar coordination (e.g. CuO square plane) has effective C_{4v} symmetry
at the metal site. The d-shell splits into four irreps:

```@example primer
G = pointgroup(:C4v)
subduce(G, 2)
```

All multiplicities are 1, so four energies suffice. A physically reasonable
parameter set (in eV, after subtracting the barycentre):

```@example primer
using MOADyna
G = pointgroup(:C4v)
# A1 ↔ d_z²,  B1 ↔ d_x²-y²,  B2 ↔ d_xy,  E ↔ {d_xz, d_yz}
ε = Dict(:A1 => -0.6, :B1 => 1.2, :B2 => 0.3, :E => -0.5)
sub = subduce(G, 2)
energies = [ε[label] for (label, _) in sub]
akm = expand_clm_central(G, 2, energies)
for a in akm
    println("A$(a.k)$(a.m) = ", round(real(a.coeff), digits=4))
end
```

The output will include both $k=2$ and $k=4$ terms, because C_{4v} allows $m \in
\{0, \pm 4\}$ but also has $k=2$ terms that Oh suppresses.

### Example: d-shell in D_{4h} (tetragonal distortion of octahedron)

D_{4h} = C_{4v} × {E, i}. The d-shell subduction mirrors C_{4v} with a
gerade/ungerade label:

```@example primer
G = pointgroup(:D4h)
subduce(G, 2)
```

Four gerade irreps (the d-shell has even parity), so again four parameters. The
A1g/B1g/B2g/Eg Cartan structure is the same as A1/B1/B2/E in C_{4v}; the
inversion label is added for free by the D_{4h} generator.

```@example primer
using MOADyna
G = pointgroup(:D4h)
sub = subduce(G, 2)
# Tetragonal splitting: elongated octahedron squashes Eg → {A1g, B1g} and splits T2g → {B2g, Eg}
ε_tet = Dict(:A1g => -0.5, :B1g => 1.0, :B2g => 0.2, :Eg => -0.4)
energies = [ε_tet[label] for (label, _) in sub]
akm = expand_clm_central(G, 2, energies)
for a in akm
    println("A$(a.k)$(a.m) = ", round(real(a.coeff), digits=4))
end
```

### Example: d-shell in T_d (tetrahedral coordination)

T_d has no inversion centre. The d-shell splits into E and T_2:

```@example primer
G = pointgroup(:Td)
subduce(G, 2)
```

Note the parameter order: E comes first in the reference table ordering. In a
tetrahedral field the sign of the crystal splitting relative to octahedral is
often discussed in terms of the ligand-field argument that 10Dq(Td) ≈ −(4/9)
10Dq(Oh). In MOADyna you simply supply the energies; the Akm coefficients carry
the sign automatically via the geometry embedded in the generator matrices:

```@example primer
using MOADyna
G_Oh = pointgroup(:Oh)
G_Td = pointgroup(:Td)

# Same energy splitting, different groups
akm_Oh = expand_clm_central(G_Oh, 2, [0.6, -0.4])   # Eg=0.6, T2g=-0.4
akm_Td = expand_clm_central(G_Td, 2, [-0.4, 0.6])   # E=-0.4, T2=+0.6

for a in akm_Oh; println("Oh A$(a.k)$(a.m) = $(round(real(a.coeff),digits=4))"); end
for a in akm_Td; println("Td A$(a.k)$(a.m) = $(round(real(a.coeff),digits=4))"); end
```

As the printed output shows, the Oh and Td results have $A_{40}$ (and the
paired $A_{4\pm 4}$) of equal magnitude but opposite sign. The sign reversal
reflects the opposite orientation of the crystal field in the two groups. (The
familiar ligand-field relation 10Dq(T_d) ≈ −(4/9)·10Dq(O_h) holds for
*identical* ligands at the same metal–ligand distance, not for this example,
which feeds the two groups different input splittings.)

### Example: f-shell in C_{4v} (multiplicity-aware)

The f-shell ($\ell=3$) in C_{4v} has a multiplicity-2 E block:

```@example primer
G = pointgroup(:C4v)
@show subduce(G, 3)
@show nparams(G, 3); # three 1×1 blocks + one 2×2 symmetric block = 3 + 3 = 6
```

The 2×2 E block has three independent real parameters: two diagonal energies
$\varepsilon_{E_1}$, $\varepsilon_{E_2}$, and one off-diagonal mixing $M_E$.

```@example primer
using MOADyna
G = pointgroup(:C4v)
sub = subduce(G, 3)

# One block per irrep (in sub order):
# A1: 1×1, B1: 1×1, B2: 1×1, E: 2×2 symmetric
blocks = [
    fill(1.0, 1, 1),         # A1: ε_A1 = 1.0
    fill(0.5, 1, 1),         # B1: ε_B1 = 0.5
    fill(-0.3, 1, 1),        # B2: ε_B2 = -0.3
    [1.2  0.4;               # E block: ε_E1=1.2, ε_E2=-0.8, M_E=0.4
     0.4 -0.8],
]
akm = expand_clm(G, 3, blocks)
for a in akm
    println("A$(a.k)$(a.m) = ", round(real(a.coeff), digits=4))
end
```

The off-diagonal entry $M_E = 0.4$ mixes the two E copies (the $|m|=1$-derived
and the $|m|=3$-derived f-orbital combinations). Setting it to zero gives the
central-projector limit.

### Production vs. experimental mode

By default, `expand_clm` and `expand_clm_central` require that the group have
curated reference Mulliken labels (`has_reference_labels(G.name) == true`). This
ensures that the parameter ordering in your call matches a canonical published
table, making results reproducible across MOADyna versions.

For groups outside `reference_label_groups()`, you can opt in with:

```julia
G_custom = pointgroup(:SomeGroup)   # auto-named labels
akm = expand_clm(G_custom, 2, energies; experimental=true)
```

The `experimental=true` flag is a contract: you acknowledge that MOADyna's internal
IR ordering for this group is deterministic but may not match any published
reference. Results are reproducible within a MOADyna version but should be labelled
as non-canonical in publications.

---

## 6. Term-symbol classification

### Classifying a subspace

Given a $G$-invariant subspace (for example, the ground-state multiplet of a
model Hamiltonian), you can decompose it into Mulliken irreps by supplying the
subspace character — the trace of the representation matrix at each element of
$G$.

$$\chi_S(g) = \mathrm{Tr}_S \, U(g)$$

```@example primer
using MOADyna, LinearAlgebra
G = pointgroup(:Oh)

# Compute D^2(g) for every element of Oh
Dl = [wignerd(e, 2) for e in G.elements]
chi_D2 = ComplexF64[tr(D) for D in Dl]

# Classify the full d-shell as a subspace of V_2
classify_subspace(chi_D2, G)
```

This reproduces the subduction result, as expected: the full d-shell breaks into
one $E_g$ copy and one $T_{2g}$ copy.

A more realistic application: after running an ED calculation, compute the
character of the degenerate ground-state subspace and classify it.

### Classifying a single state

For a single state $|\psi\rangle$, the relevant quantity is not a character
(which requires a class function) but the **projector weights**
$w_\Gamma = \langle\psi | P_\Gamma | \psi\rangle$.

```julia
classify_state(matrix_elements, G)
```

Here `matrix_elements[g]` = $\langle\psi | U(g) | \psi\rangle$, one per element
of $G$ in `G.elements` order. The return value is a named tuple
`(weights, dominant_IR, dominant_weight)`.

Note on current scope: computing the actual matrix elements
$\langle\psi | U(g) | \psi\rangle$ for a many-body wavefunction requires the
lifted Fock-space representation `(rep::LiftedRep)(g_idx, ψ)`; the single-particle
rotation wiring it needs is not yet exposed (it is on the roadmap). For now you
supply precomputed values (e.g. from `wignerd` for single-particle states, or from
an external code for many-body states).

### Worked example with precomputed character

Suppose you have diagonalised a Hamiltonian in the d-shell and found a
five-element degenerate ground state spanning all of $V_2$. Then
$\chi_S(g) = \mathrm{tr} \, D^2(g)$ for all $g$, and classification gives back
$D^2 \downarrow O_h = E_g \oplus T_{2g}$ as expected.

For a two-dimensional subspace (the $E_g$ sector, say), you would restrict the
character to those two basis states and get $\chi_S(g) = \mathrm{tr}_{E_g}
D^2(g)$, which classify_subspace would return as `[:Eg => 1]`.

---

## 7. Conventions and gotchas

Knowing these conventions prevents the most common sign errors when comparing
MOADyna's output to other codes or textbooks.

### Active rotation convention

MOADyna uses the **active** convention: symmetry operations rotate the physical
state while the coordinate axes are fixed. Under a rotation $R$, a state
$|\ell, m\rangle$ becomes $\sum_{m'} D^\ell_{m'm}(R) |\ell, m'\rangle$.
Some texts and codes (notably some Bilbao Crystallographic Server outputs) use
the **passive** convention (axes rotate, states are fixed), which gives the
transpose/inverse Wigner matrix. If you see an overall sign flip or imaginary
unit in a comparison, check this first.

### C^k_m: Racah-normalised spherical harmonics

The $A_{km}$ coefficients in MOADyna's `expand_clm` output are defined via the
Racah-normalised operators $C^k_m$ (also called "spherical tensor operators of
rank $k$"):

$$C^k_m(\hat{r}) = \sqrt{\frac{4\pi}{2k+1}} Y^k_m(\hat{r})$$

The key property is $C^k_0(z\text{-axis}) = 1$ (normalisation at the pole).
This differs from the Wybourne or Stevens conventions by numerical prefactors.
MOADyna's matrix-element formula for $\langle \ell m_1 | C^k_m | \ell m_2\rangle$
uses 3j symbols via `WignerSymbols.jl`.

The Hermiticity relation:

$$A_{k,-m} = (-1)^m A_{k,m}^*$$

means that for real crystal fields (the current scope), $A_{k,m}$ and
$A_{k,-m}$ are related by a sign and complex conjugation. The MOADyna output
respects this: you will see paired `m` and `-m` entries with the same real
magnitude in most cases.

### Wigner D^ℓ: zyz Euler convention

MOADyna decomposes every group element into Euler angles via the zyz convention
($R(\alpha, \beta, \gamma) = R_z(\alpha) R_y(\beta) R_z(\gamma)$) and evaluates
the Wigner small-d function. For improper rotations ($\det R = -1$), the parity
factor $(-1)^\ell$ multiplies the entire D-matrix:

$$D^\ell(R) = (-1)^\ell D^\ell(R_\text{proper}), \quad R_\text{proper} = -R$$

This means that for a d-shell ($\ell=2$) the inversion acts as $+1$ (even
parity), while for an f-shell ($\ell=3$) the inversion acts as $-1$ (odd
parity). This is the physical content of the gerade/ungerade labels.

### C_{4v} and D_{4h} selection rules

In both C_{4v} and D_{4h}, the surviving ``A_{km}`` terms satisfy
``m \in \{0, \pm 4, \pm 8, \ldots\}``. In particular, ``A_{20}`` and ``A_{40}``
both survive (from the uniaxial four-fold ``C_4`` axis), while ``A_{4\pm 2}``
and ``A_{2\pm 2}`` are forced to zero by the four-fold rotation. This is why
the C_{4v} d-shell has only ``A_{20}``, ``A_{40}``, and ``A_{4\pm 4}``
non-zero — even though C_{4v} has lower symmetry than O_h.

### Multiplicity-space gauge

For groups and shells with $m_\Gamma > 1$, the $m_\Gamma \times m_\Gamma$ block
$H_\Gamma$ is specified in MOADyna's canonical multiplicity-space basis (the
deterministic output of the §9.2 tesseral-pivot algorithm in the architecture
chapter). Other codes and standard published table parameterisations may use a
different basis for the multiplicity space, related to MOADyna's by an orthogonal
transform.

Practical consequence: for ``m_\Gamma = 1`` (almost all d-shell cases in common
point groups), there is no gauge freedom and no ambiguity. For ``m_\Gamma = 2``
(C_{4v} f-shell E block), the diagonal energies are unambiguous but the
off-diagonal mixing element ``M_E`` is gauge-dependent. If you need to compare
this to another code's parametrisation, you will need to determine the rotation
between the two conventions; this is not currently automated.

---

## 8. Further reading

These texts are suggested as background; they are not cited inline in the MOADyna
documentation.

- Cotton, *Chemical Applications of Group Theory*, 3rd ed. — the standard
  chemistry reference; Cotton's IR labelling conventions are the ones MOADyna
  follows for most groups.
- Atkins, Child, and Phillips, *Tables for Group Theory* — concise Oxford
  pamphlet with character tables and direct-product tables for all common point
  groups.
- Tinkham, *Group Theory and Quantum Mechanics* — more rigorous treatment,
  includes Wigner–Eckart theorem and Wigner D-matrices; the level closest to
  MOADyna's internal formalism.
- Bradley and Cracknell, *The Mathematical Theory of Symmetry in Solids* —
  authoritative reference for space groups and double groups; overkill for most
  crystal-field work but essential if you go beyond point groups.
- Griffith, *The Theory of Transition-Metal Ions* — classic multiplet theory
  text; uses a related but slightly different convention for $A_{km}$.
