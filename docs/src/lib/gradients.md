# Gradients

Differentiable ``T = 0`` XAS forward model: the map from physical
parameters ``\theta`` (Slater reductions, charge-transfer ``\Delta``,
``10Dq`` / ``A_{km}`` scale, hybridization and ``\zeta`` scales) to a
spectrum, made differentiable in ``\theta``.

Derivatives are computed **forward-mode with an analytic resolvent** —
not by reverse-mode AD through the eigensolver. The differentiated path
uses frozen Krylov and ``\omega``-grid settings, and the exposed
contract is a VJP/pullback (and JVP) over a *degeneracy-clustered
spectral measure*: ``\partial(\text{cluster energy},\ \text{summed
residues})/\partial\theta``.

!!! note "Accessing these names"
    `MOADyna.Gradients` is not re-exported by `using MOADyna`, because several
    of its names (`spectrum`, `jacobian`, `hamiltonian`, `groundstate`,
    …) are deliberately generic. Reach them module-qualified, or bind
    the module once:

    ```julia
    using MOADyna
    const G = MOADyna.Gradients

    model = G.XASGradientModel(...)
    S     = G.spectrum(model, θ)
    J     = G.jacobian(model, θ)
    ```

Parameter *inference* — the observation model (broadening, background,
energy calibration, normalization, noise) and the Bayesian layer — lives
in a separate sibling package that depends on MOADyna; MOADyna itself never
hard-depends on an ML stack. The deterministic point-estimate fit
[`MOADyna.Gradients.fit_spectrum`](@ref) is available when `Optim` is
loaded (weak-dependency extension).

`AffineModel` accepts sparse matrices or matrix-free terms on a fixed basis.
Matrix-free combinations retain compiled operator plans, and the ground-state
gap is checked with an iterative two-state solve. Ground-state and resolvent
derivatives use operator products throughout. The selected basis must stay fixed
while differentiating; derivatives of determinant selection are not defined.
Full-spectrum analysis, including automatic spectral-window construction,
requires explicitly stored matrices.

```@autodocs
Modules = [MOADyna.Gradients]
Private = false
```
