# NiO L_{2,3} XAS — comparison plots

Apples-to-apples plots comparing MOADyna's `xas` output with PyQuanty's
Lanczos-cont-frac reference and Quanty's `CreateSpectra` output, on
the same NiO multiplet calculation.

## One-time install

The plot scripts use a separate Julia environment so `Plots.jl` and
`GR_jll` don't become hard dependencies of MOADyna itself.

```bash
cd docs/dev/validation/spectroscopy/nio_xas/plots
julia --project=. -e 'import Pkg; Pkg.develop(path="../../../../../.."); Pkg.instantiate()'
```

(The path has six `..` segments: `plots/ → nio_xas/ → spectroscopy/ →
validation/ → dev/ → docs/ → MOADyna/`.)

## Run

```bash
julia --project=docs/dev/validation/spectroscopy/nio_xas/plots \
      docs/dev/validation/spectroscopy/nio_xas/plots/make_xas_comparison.jl
```

(from the package root). The script regenerates four PNGs in this
directory:

| File | Content |
|---|---|
| `XAS_x.png` | x-polarisation spectrum (MOADyna / PyQuanty / Quanty) + residuals |
| `XAS_y.png` | y-polarisation spectrum + residuals |
| `XAS_z.png` | z-polarisation spectrum + residuals |
| `XAS_isotropic.png` | Sum over polarisations + residuals |

Each figure has a top panel with all three codes overlaid (–Im of the
complex spectrum tensor → physical absorption intensity) and a bottom
panel showing residuals normalised to the PyQuanty peak.
