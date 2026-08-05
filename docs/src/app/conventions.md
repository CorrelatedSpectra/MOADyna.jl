# Conventions

This appendix collects the cross-cutting conventions used throughout MOAD.

## Resolvent and broadening

All spectra use the same resolvent convention. With the scalar frequency
``z = \omega + E_g + i\Gamma/2``, the response is
``C(\omega) = \langle\psi_0|\, A^\dagger (zI - H)^{-1} B\, |\psi_0\rangle``, evaluated
relative to the ground-state energy ``E_g``, so the ``\omega`` axis is the **excitation
energy** (energy loss / photon energy relative to threshold), not the absolute eigenvalue.
The spectral function is ``-\operatorname{Im} C(\omega)/\pi``.

`Γ` is the broadening as a **Lorentzian FWHM** (not a half-width): a pole at ``\omega_0`` contributes
``\dfrac{\Gamma/2\pi}{(\omega-\omega_0)^2 + (\Gamma/2)^2}``. The same `Γ` convention is used by every entry point.

## Energy reference

Energies are measured from the ground state (`E_g` subtracted), so `ω = 0` is the elastic /
threshold point. Absolute atomic edge energies (e.g. the ≈853 eV Ni L₃ position) are *not*
added — the calculation gives the multiplet structure relative to threshold; line up to
experiment by shifting the axis.

## Greek and ASCII keyword aliases

Keyword arguments accept both a Greek-letter name and an ASCII alias, so scripts work
without Unicode entry:

| Greek | ASCII |
|---|---|
| `ω_grid` | `omega_grid` |
| `Γ` | `Gamma` |

Either spelling is accepted; passing both conflicting values is an error.

## Spherical-tensor / operator conventions

Multipole and crystal-field operators use **Racah's normalized ``C^k_q``** spherical tensors
(``C^k_0(0,0) = 1``). The orbital basis is the complex
spherical-harmonic basis ``|\ell, m\rangle``, ``m = -\ell,\ldots,\ell``; real (tesseral) combinations follow the
Condon–Shortley convention (the ``(-1)^m`` that makes the dipole Cartesian assembly
``T_x = (T_{-1} - T_{+1})/\sqrt{2}``). Cross-shell transition operators are built in the `shellA => shellB`
(absorption) direction ``c^\dagger(b)\,c(a)``; emission is the adjoint.

## Units

There are no hard-wired physical units — MOAD works in whatever energy unit the inputs use
(eV throughout the examples and the atomic-parameter tables). The atomic radial moments
``\langle r^k\rangle`` are in `Å^k`. Temperatures in the finite-`T` paths are in energy units (`k_B = 1`).
