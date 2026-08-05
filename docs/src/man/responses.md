# Responses

```@meta
CurrentModule = MOAD
DocTestSetup  = quote
    using MOAD
end
```

`MOAD.Responses` is the foundational layer beneath all of [Spectroscopy](@ref): it
represents a matrix-valued frequency response ``C_{ab}(\omega)`` and computes it from an
assembled Hamiltonian with a block-Lanczos / continued-fraction kernel. Every spectrum in
MOAD — XAS, RIXS, ``\sigma(\omega)``, ``S(\mathbf q,\omega)`` — is a thin wrapper over the one constructor
[`correlator`](@ref).

## The response abstraction

A response is an [`AbstractResponse`](@ref)`{T}` carrying the matrix-of-ω data in one of
three representations, related by exact (or controlled) conversions:

- [`LanczosResponse`](@ref) — the raw block-tridiagonal Lanczos coefficients
  ``(\alpha_k, \beta_k)``. The native output of [`correlator`](@ref); the most compact and the source
  for cheap re-broadening, since ``\Gamma`` is applied only at evaluation.
- [`PoleResponse`](@ref) — the diagonalized `(pole, residue)` representation, from
  [`to_pole`](@ref). Useful for pole arithmetic and exact δ-resolved spectra.
- [`GridResponse`](@ref) — the response evaluated on an explicit ω grid with a chosen
  broadening, from [`to_grid`](@ref). This is what spectroscopy wrappers ultimately
  return inside a `SpectraTensor`.

[`GreensFunction`](@ref)`{T, R}` wraps any of these as a single-particle Green's function
with the extra book-keeping (orbital labels, channel) needed for inv / Dyson / periodise
operations (planned for later lattice releases).

## The `correlator` constructor

[`correlator`](@ref)`(H, basis, As, Bs; Γ, …)` computes
``C_{ab}(\omega) = \langle\psi| A_a^\dagger (\omega + E_0 - H + i\Gamma/2)^{-1} B_b |\psi\rangle`` for vectors `As`, `Bs` of
[`OperatorSum`](@ref). The current public contract is **autocorrelator-only**: pass the
*same* operator list as both arguments (`As === Bs`, identity not just equality); the
diagonal `a = a` entries are the per-operator spectra and the off-diagonal `a ≠ b` entries
are their cross-terms. (General cross-correlators with `As ≠ Bs` are deferred to a future release.)
Channels select the physical process:

| `channel` | meaning |
|---|---|
| `:neutral` (default) | the one-sided neutral autocorrelator building block in a single source block — ``S(\mathbf q,\omega)`` and the regular ``\sigma`` as the simple same-block cases; XAS/RIXS are sector-spanning workflows built on it via wrapper-managed embeddings. The full Kubo retarded ``\chi``/``\sigma`` (commutator weights) is deferred. |
| `:addition` / `:removal` | the ``N\pm1`` single-particle Green's function (electron addition / removal). |
| `:both` | the full ``G(\omega)`` assembled from both `N±1` blocks. |

The ground state is computed internally (`state = :ground_state`) or supplied as
`state = ψ₀` together with `Eg`. Broadening `Γ` is the Lorentzian FWHM. With `ω = nothing`
(the default) the constructor returns a [`LanczosResponse`](@ref); pass an ω grid (or
call [`to_grid`](@ref)) to evaluate.

```@example resp
using MOAD

# Hubbard dimer, half-filled.
s = [FermionSite{2}(:a), FermionSite{2}(:b)]
h = Hilbert(x.name => x for x in s)
H = -1.0 * sum(c'(s[1], σ) * c(s[2], σ) + c'(s[2], σ) * c(s[1], σ) for σ in 1:2) +
     4.0 * sum(n(s[i], 1) * n(s[i], 2) for i in 1:2)
b   = EagerBasis(h, n_fermion(h) == 2)
Hsp = assemble(compile(H, b), b)

# Neutral (same-sector) charge response of the staggered density.
# (`correlator` is autocorrelator-only for now — pass the same operator list
#  as both `As` and `Bs`; cross-correlators are deferred to a future release.)
O   = (n(s[1], 1) + n(s[1], 2)) - (n(s[2], 1) + n(s[2], 2))
ops = [O]
G   = correlator(Hsp, b, ops, ops; Γ = 0.3)

(G isa AbstractResponse, G isa LanczosResponse)
```

## Representations and conversions

A `LanczosResponse` converts losslessly to poles, or to a grid at any broadening — without
recomputing the Lanczos recursion. This is why re-broadening (e.g.
[`re_broaden_table`](@ref) in spectroscopy) is cheap.

```@example resp
Gpole = to_pole(G)                                 # (pole, residue) form
Ggrid = to_grid(G, range(0.0, 8.0; length = 100))  # evaluate on an ω grid
(Gpole isa PoleResponse, Ggrid isa GridResponse)
```

A response round-trips to HDF5 with [`save_response`](@ref) / [`load_response`](@ref),
preserving the representation and metadata.

## Single-particle Green's function (charged channels)

Beyond the neutral same-sector response, `correlator` computes the ``N\pm 1`` single-particle
Green's function — the photoemission (`:removal`) and inverse-photoemission (`:addition`)
spectral functions ``A(\omega) = -\tfrac1\pi\operatorname{Im}G(\omega)``. Two requirements:
the basis must **span the ``N\pm1`` sectors** reached by the source operators, and the
reference ground state must sit at the target filling (here a particle–hole-symmetric
``\mu = U/2`` keeps the half-filled sector lowest). `Eg` is computed automatically.

```@example resp
μ    = 2.0                                            # particle–hole symmetric: half-filling is GS
Ntot = sum(n(s[i], σ) for i in 1:2, σ in 1:2)
bc   = EagerBasis(h, n_fermion(h) in 1:3)             # spans N−1, N, N+1
Hμsp = assemble(compile(H - μ * Ntot, bc), bc)
cs   = [c(s[1], 1)]                                   # remove / add a ↑ electron on site a
ωs   = range(-8, 8; length = 400)

A_removal  = -imag.(to_grid(correlator(Hμsp, bc, cs, cs; Γ = 0.3, channel = :removal),  ωs).data) ./ π
A_addition = -imag.(to_grid(correlator(Hμsp, bc, cs, cs; Γ = 0.3, channel = :addition), ωs).data) ./ π
(removal_peak  = round(ωs[argmax(vec(A_removal))];  digits = 2),   # lower Hubbard band (occupied)
 addition_peak = round(ωs[argmax(vec(A_addition))]; digits = 2))   # upper Hubbard band (empty)
```

The two peaks straddle the Fermi level (here at ``\mp 1.82``) — the lower and upper Hubbard
bands separated by the Mott gap. `channel = :both` (with `addition_indices`/`removal_indices`)
assembles the full ``G(\omega)`` as a [`GreensFunction`](@ref).

## Finite temperature

Passing `T` (with `k_B = 1`) replaces the single ground-state expectation by a
Boltzmann-weighted average over the low-lying eigenstates. The truncation is
degeneracy-complete (it never splits a degenerate multiplet), and a warning is emitted if
the lowest excluded state still carries non-negligible weight — increase `N_states` to
converge. This same-sector finite-`T` neutral average is what ``\sigma(\omega)`` and ``S(\mathbf q,\omega)`` use at finite
temperature; the full Kubo retarded ``\chi``/``\sigma`` and finite-`T` charged-channel GF are deferred to a
follow-on sub-phase.
