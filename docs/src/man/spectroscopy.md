# Spectroscopy

```@meta
CurrentModule = MOADyna
DocTestSetup  = quote
    using MOADyna
end
```

`MOADyna.Spectroscopy` turns an assembled Hamiltonian and a ground (or thermal) state into a
frequency-resolved spectrum. It covers two families:

- **Core-level spectroscopy** — [`xas`](@ref) (X-ray absorption), [`rixs`](@ref)
  (resonant inelastic X-ray scattering), [`fluorescence_yield`](@ref). These use the
  transition operators from the [Multiplets & Standard Operators](@ref) chapter.
- **Response functions** — [`optical_conductivity`](@ref) (regular ``\operatorname{Re}\sigma_{\alpha\beta}(\omega>0)``) and
  [`dynamical_structure_factor`](@ref) (``S(\mathbf{q},\omega)``, including the nIXS spectrum). These are
  thin one-sided wrappers over [`correlator`](@ref) with bring-your-own operators.

Every entry point returns a [`SpectraTensor`](@ref): a tensor of spectra over an ω grid,
with HDF5 round-trip and a shared broadening convention (Lorentzian FWHM `Γ`) and unified
scalar frequency ``z = \omega + E_g + i\Gamma/2`` and resolvent ``(zI - H)^{-1}`` (see the [Conventions](@ref) appendix). The Lanczos-backed
core-level spectra (XAS/RIXS/FY) additionally carry the block-Lanczos chunks, so they can be
re-broadened and polarization-sliced without recomputing ([`re_broaden_table`](@ref)); the
response wrappers ([`optical_conductivity`](@ref), [`dynamical_structure_factor`](@ref))
return an already-evaluated algebraic tensor (`chunks = nothing`), so `re_broaden_table` does
not apply to them — re-run with a new `Γ` instead.

## X-ray absorption (XAS)

XAS evaluates ``I(\omega) = -\tfrac{1}{\pi}\operatorname{Im}\langle g|T^\dagger (\omega + E_g - H + i\Gamma/2)^{-1} T|g\rangle`` for a transition
operator `T` (a Cartesian [`dipole`](@ref) component) acting on the ground state ``|g\rangle``.
The Hamiltonian `H` is the *final-state* Hamiltonian (with the core hole); ``|g\rangle`` is the
initial ground state embedded into the final-state basis.

```julia
# (Illustrative — the full runnable workflow is examples/05_nio_xas.jl and the
#  NiO tutorial. T = dipole(m, :Ni_2p => :Ni_3d) is built in the Multiplets chapter.)
gs   = eigen(H_GS, basis_gs; n = 1)
psi0 = embed(gs.vectors[:, 1], basis_gs => basis_xas)
H_xas = assemble(compile(H_XAS, basis_xas), basis_xas)

T = dipole(m, :Ni_2p => :Ni_3d)          # the full Cartesian vector [T_x, T_y, T_z]
spec = xas(H_xas, basis_xas, T, psi0;    # ⇒ polarisation tensor χ_ab(ω), shape (3, 3, n_ω)
           omega_grid = range(-15, 25; length = 801),
           Gamma      = 0.6,
           Eg         = gs.values[1])
```

Pass a **single** operator (e.g. `Tz`) and `spec.tensor` is the scalar complex response
(`-imag` of it is the absorption). Pass the **whole vector** `T` and `xas` returns the full
**polarisation tensor** ``\chi_{ab}(\omega)``. Any polarisation then follows from one
contraction with [`polarise`](@ref): for a unit Jones vector ``\boldsymbol\varepsilon``,
``I(\omega) = -\operatorname{Im}\sum_{ab}\varepsilon_a^{*}\chi_{ab}(\omega)\varepsilon_b``.

```julia
polarise(spec, [0, 0, 1])               # linear ∥ z
polarise(spec, [1, im, 0] / sqrt(2))    # circular (right) → XMCD against its conjugate
polarise(spec)                          # for a scalar-T result: just −Im(tensor)
```

Linear dichroism is two real ``\boldsymbol\varepsilon``; XMCD is the difference of the two
circular helicities; the isotropic spectrum is the average over any three orthogonal
``\boldsymbol\varepsilon`` (equivalently ``\tfrac13\operatorname{tr}\chi``). The NiO tutorial
works a full magnetic-linear-dichroism (XMLD) example end to end.

## Resonant inelastic X-ray scattering (RIXS)

RIXS is the two-step coherent process: excite with `T_in`, propagate in the intermediate
(core-hole) Hamiltonian, de-excite with `T_out`, and resolve in both incident and
energy-loss axes. [`rixs`](@ref) takes the final- and intermediate-state Hamiltonians
separately.

```julia
# (Illustrative — full runnable workflow: examples/06_nio_rixs.jl.)
result = rixs(H_final, H_intermediate, basis, T_in, T_out, psi0;
              omega_in_grid      = incident_grid,
              omega_out_grid     = loss_grid,
              Gamma_intermediate = 0.6,
              Gamma_final        = 0.1,
              Eg                 = Eg)
```

Scalar `T_in`/`T_out` give a single ``(\omega_\mathrm{in}, \omega_\mathrm{out})`` map. Pass
Cartesian **vectors** for both and `rixs` returns the four-index Kramers–Heisenberg tensor
``\chi_{ijkl}``; [`polarise`](@ref)`(result, ε_in, ε_out)` contracts it for any incoming /
outgoing polarisation pair,
``\sigma = -\operatorname{Im}\sum_{ijkl}\varepsilon^{*}_{\mathrm{in},i}\varepsilon_{\mathrm{out},j}\chi_{ijkl}\varepsilon^{*}_{\mathrm{out},k}\varepsilon_{\mathrm{in},l}`` —
so parallel vs crossed linear channels, or any circular combination, come from one
calculation. The NiO tutorial works a polarisation-resolved RIXS example.

[`fluorescence_yield`](@ref) (`fluorescence_yield(H_i, basis, T_excite, T_decay, ψ; …)`)
gives the integrated decay intensity used for partial-fluorescence-yield detection.

## Response functions: optical conductivity and dynamical structure factor

The response wrappers are statistics-agnostic and operator-agnostic — you supply the
operator. [`dynamical_structure_factor`](@ref) returns ``S(\mathbf{q},\omega) = -\operatorname{Im} C(\mathbf{q},\omega)/\pi`` over the
full ω grid for a momentum-resolved operator `O_q`; the **nIXS** spectrum is exactly this
with `O_q = ` [`nixs`](@ref)`(…)`. The example below is small enough to run at build time.

```@example spec
using MOADyna

# 2-site, 2-orbital Hubbard dimer.
s = [FermionSite{2}(:a), FermionSite{2}(:b)]
h = Hilbert(x.name => x for x in s)
t, U = 1.0, 4.0
H = -t * sum(c'(s[1], σ) * c(s[2], σ) + c'(s[2], σ) * c(s[1], σ) for σ in 1:2) +
     U * sum(n(s[i], 1) * n(s[i], 2) for i in 1:2)

b   = EagerBasis(h, n_fermion(h) == 2)
Hsp = assemble(compile(H, b), b)
gs  = eigen(Hsp, b; n = 1)

# Staggered (q = π) charge operator as the probe.
O_q = (n(s[1], 1) + n(s[1], 2)) - (n(s[2], 1) + n(s[2], 2))

S = dynamical_structure_factor(Hsp, b, O_q;
                               omega_grid = range(0.0, 8.0; length = 200),
                               Gamma      = 0.3,
                               ψ₀         = gs.vectors[:, 1],
                               Eg         = gs.values[1])
(ndims(S.tensor), round(maximum(S.tensor); digits = 4))
```

Results are `SpectraTensor`s, and the post-processing toolkit combines them: [`average`](@ref)
and [`weighted_sum`](@ref) build ensembles (orientational, thermal, configurational),
[`restrict_to_window`](@ref) crops the ω axis, and [`save_spectra`](@ref)/[`load_spectra`](@ref)
round-trip to HDF5:

```@example spec
O_spin = (n(s[1], 1) - n(s[1], 2)) - (n(s[2], 1) - n(s[2], 2))   # a second probe
S_spin = dynamical_structure_factor(Hsp, b, O_spin;
                                    omega_grid = range(0.0, 8.0; length = 200),
                                    Gamma = 0.3, ψ₀ = gs.vectors[:, 1], Eg = gs.values[1])
S_avg = average([S, S_spin])                    # equal-weight ensemble
S_w   = weighted_sum([S, S_spin], [0.7, 0.3])   # arbitrary weights
(round(maximum(S_avg.tensor); digits = 4), round(maximum(S_w.tensor); digits = 4))
```

[`optical_conductivity`](@ref) returns the regular part ``\operatorname{Re}\sigma_{\alpha\beta}(\omega>0) = (1 - e^{-\omega/T})\,\dfrac{-\operatorname{Im}\Lambda_{\alpha\beta}}{\omega V}`` from a bring-your-own current operator `j` (the Drude /
diamagnetic weight is not included). Both wrappers take the finite-`T` path via `T=…`
(thermal average over the low-lying states); see [Responses](@ref) for the underlying
[`correlator`](@ref).

## Restriction-resolved partial excitations

To decompose a spectrum into final-state channels (e.g. the ``d^9`` vs ``d^{10}\underline{L}`` contributions of a
charge-transfer system, or ``t_{2g}``/``e_g`` channels), pass a `restrictions` predicate to
[`xas`](@ref)/[`rixs`](@ref): the Lanczos recursion applies it after every matvec, projecting
the propagated state into the chosen sector — an *in-recursion* projection of the final-state
dynamics, not a post-hoc filter of the result. The partials sum to the total only when the
restriction is a conserved sector of `H` (otherwise restricting cuts the inter-sector
coupling — a physical, documented caveat). The NiO tutorial works this end to end (the
final-state `nshells(m, :Ni_3d) == 9` channel of the L-edge).

## Finite temperature

Finite temperature in v0.2 applies to the **neutral same-sector response wrappers** —
[`optical_conductivity`](@ref) and [`dynamical_structure_factor`](@ref) — via a `T` keyword
(with `k_B = 1`; not to be confused with the transition-operator argument `T` of the
core-level functions). It replaces the single ground-state expectation by a
Boltzmann-weighted sum over the low-lying eigenstates (degeneracy-complete truncation). The
thermal machinery lives in [`correlator`](@ref) and is described in the [Responses](@ref)
chapter. Finite-temperature XAS/RIXS/fluorescence yield are **not** exposed in v0.2
(deferred to a follow-on sub-phase).
