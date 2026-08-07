# Validation against Quanty

Empirical correctness checks for the operator-algebra layer
(`MOADyna.Algebra`), comparing the Tier-2 canonicalized `OperatorSum` output
against Quanty's normal-ordered operator dump.

## Convention bridge

Both codes apply Tier-2 canonicalization, but use different ordering
conventions within each statistics group:

| | Mode indexing | Within-cdag sort | Within-c sort |
|---|---|---|---|
| Quanty | 0-indexed | descending | descending |
| MOADyna   | 1-indexed (or whatever the user assigns) | ascending | ascending |

The two conventions are algebraically equivalent — flipping the sort
direction means an even number of swaps for any chain that matches
between cdag's and c's, so the sign cancels and the coefficient is the
same.

## What's verified

`test_print_hamiltonian.Quanty` constructs:

- 4 hopping terms (`t * (cdag(0)c(2) + cdag(1)c(3) + h.c.)`) at length 2
- 1 on-site Coulomb term (`U * n_0 n_1` on a 4-mode system) at length 4

The matching `test_moad.jl` builds the same Hamiltonian via `c`, `cdag`,
`n` calls and the `*`, `+` algebra. After Tier-2 canonicalization both
codes report:

| Term | Quanty (descending, 0-indexed) | MOADyna (ascending, 1-indexed) |
|---|---|---|
| Hopping (×4) | `C 0 A 2 \| 1.5`, `C 1 A 3 \| 1.5`, `C 2 A 0 \| 1.5`, `C 3 A 1 \| 1.5` | `cdag(s,1) c(s,3)`, `cdag(s,2) c(s,4)`, `cdag(s,3) c(s,1)`, `cdag(s,4) c(s,2)` — all coef 1.5 |
| Coulomb | `C 1 C 0 A 1 A 0 \| -4.0` | `cdag(s,1) cdag(s,2) c(s,1) c(s,2)` — coef -4.0 |

Coefficients match exactly. Chain orderings differ only by the
ascending/descending convention.

## Running

```bash
# Quanty (need Quanty in PATH)
Quanty test_print_hamiltonian.Quanty

# MOADyna — from the repo root:
julia --project=. docs/dev/validation/test_moad.jl
```

## Full z_only model verification ✓

`z_only_print.Quanty` builds the standalone z_only Hamiltonian
(progressive_models/01_z_only/model.lua, with default parameters and
Bz=0): 5 spinless orbitals × 2 spin = 10 fermionic modes, with on-site
energies, Hubbard U on each Ni z, and apical-mediated hopping.

`z_only_compare.jl` builds the same Hamiltonian in MOADyna, parses the
Quanty dump (`z_only_quanty_output.txt`), and compares term-by-term
(after a convention bridge: 1-indexed → 0-indexed mode numbers, and
ascending → descending chain order within each statistics group).

Result: **28 of 28 terms match**, both chain content and coefficient
(within `1e-10`). Breakdown:

- 10 length-2 number diagonals (4 on Ni z's, 6 on apicals; with mixed
  ed, eL, dE_apin, dE_apout coefficients combining correctly)
- 16 length-2 hopping pairs (8 forward + 8 h.c., across z1↔a0, z1↔a1,
  z2↔a0, z2↔a2 with proper ±t signs)
- 2 length-4 Coulomb terms (one per Ni; coefficient -U=-6.0 in normal
  form, after the n_↑ n_↓ rearrangement that Tier-2 canonicalization
  produces)

This validates that MOADyna's Tier-2 algebra produces algebraically identical
output to Quanty for a real research-grade Hamiltonian. The canonical
form differs only by an ascending/descending convention; coefficients
are exact bit-for-bit matches.

## Future work

Extend the validation to:
- spin operators (Sx, Sy, Sz) — separate Quanty convention to confirm
- larger models with quartic terms beyond simple Hubbard U
- complex-valued Hamiltonians
- restrictions (`MOADyna.Bases` is in; matrix-element comparison vs a
  Quanty-side `printOperator(matrix)` dump is the next regression target)
