# Atomic Parameters

```@meta
CurrentModule = MOADyna
DocTestSetup  = quote
    using MOADyna
end
```

`MOADyna.AtomicParameters` supplies the numerical inputs a multiplet Hamiltonian needs —
Slater–Condon ``F^k``/``G^k``, spin–orbit ``\zeta``, and the radial moments ``\langle r^k\rangle`` — for
transition-metal and lanthanide configurations, via [`atomic_parameters`](@ref). The values
come from a static Haverkort-derived dictionary (Cowan RCN36K Hartree–Fock), with an
optional Cowan runner for configurations outside the static set.

## Looking up a configuration

```@example atom
using MOADyna

p = atomic_parameters(:Ni, "3d8")
(p.Fdd, p.zeta, p.provenance)
```

The return is a `NamedTuple`. For a d-block element it carries `Fdd` (fields `F2`, `F4` — the
d–d direct integrals ``F^2``, ``F^4``), `Fpd`/`Gpd` (the 2p–d direct/exchange, `NaN` when the
configuration has no 2p hole), `zeta` (fields `d`, `p` — the spin–orbit constants
``\zeta_d``, ``\zeta_p``), `r` (fields `r2`, `r4` — ``\langle r^2\rangle``,
``\langle r^4\rangle`` in `Å^k`), plus `configuration`, `provenance`, `scaling`, and an
`unreliable` flag (set for thesis values marked uncertain). f-block elements return the
analogous `Fff`/`Fdf`/`Gdf` set.

These plug straight into the multiplet builders — e.g.
`coulomb(m, :Ni_3d; U, F = p.Fdd)` and `p.zeta.d * LS(m, :Ni_3d)` (see
[Multiplets & Standard Operators](@ref)). The `coulomb` keywords `F` and `G` take
either a plain `Tuple`, in which the ranks are positional, or a `NamedTuple` such as
`Fdd`/`Gpd`, whose fields are checked to be named `F2`, `F4`, … / `G1`, `G3`, … in
ascending rank — an out-of-order `NamedTuple` is rejected rather than silently
assigned to the wrong ranks.

## Coverage and provenance

[`covered_elements`](@ref) and [`covered_configurations`](@ref) report what the static
dictionary contains. Every result records its `provenance` (`:Haverkort_thesis_2005` for the
static set) so the source of every number is traceable — important for reproducibility.

## Configurations outside the static set

For a configuration not in the dictionary, `atomic_parameters(element, configuration;
cowan = path)` shells out to a local Cowan `RCN` binary to compute the integrals. The
Cowan path is opt-in (the static dictionary is the default and needs no external tools); see
the function docstring for the runner options.

## Units and conventions

``F^k``, ``G^k``, and ``\zeta`` are in eV; ``\langle r^k\rangle`` is in ``\text{Å}^k``. These are the *atomic* (unscaled)
Hartree–Fock values — the customary ~80% reduction of the ``F^k``/``G^k`` to mimic
configuration-interaction screening is a modeling choice left to the caller, not baked into
the table. Note the radial moments here are the tabulated ``\langle r^k\rangle``; the momentum-dependent
nIXS Bessel moments are a separate quantity computed with [`radial_integral`](@ref).
