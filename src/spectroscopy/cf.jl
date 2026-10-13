# =====================================================================
# evaluate_on_grid — Spectroscopy-side continued-fraction grid evaluation
# =====================================================================
#
# The continued-fraction primitive and shared grid workspaces live in Responses.
# This wrapper accepts the Spectroscopy-domain LanczosChunk, avoiding a circular
# dependency between the two modules.

"""
    evaluate_on_grid(chunk::LanczosChunk, ω_grid; Γ::Real, Eg::Real)
        -> Array{Complex,3}

Evaluate the user-basis spectral tensor for one Lanczos chunk over the
frequency grid. Returns an array of shape `(B_raw, B_raw, length(ω_grid))`
with the cf-evaluated complex tensor sandwiched by the chunk's `R`
factor:

```
χ[a, b, ω_idx] = (R† · G₁(z) · R)_{ab}    z = ω + Eg + iΓ/2
```

Real intensity `-Im` is taken at contraction time by `polarise`, NOT
here — the complex tensor preserves phase information for arbitrary
polarisations (e.g. circular).

`Γ` is the FWHM of the resulting Lorentzian; the resolvent shift in
the denominator is `iΓ/2`. `Eg` is the ground-state energy used as
the zero-frequency reference (`ω = 0` corresponds to the GS pole).

Small Hermitian block chains may use an eigendecomposition when the grid
and numerical range permit it. Other block grids reuse call-local solve
workspaces; scalar and generic-precision chains retain single-point CF.
`ω_grid` is any iterable supporting `length`.
"""
function evaluate_on_grid(chunk::LanczosChunk, ω_grid;
                          Γ::Real, Eg::Real)
    R       = chunk.R
    B_raw   = chunk.raw_block_size
    α       = chunk.α
    β       = chunk.β
    Tc      = eltype(R)
    Tc <: Complex ||
        throw(ArgumentError("evaluate_on_grid: chunk eltype must be Complex; got $Tc"))

    z_at = ω -> Tc(ω + Eg + im * (Γ / 2))
    return Responses._cf_grid(α, β, R, ω_grid, z_at, Tc; raw_block_size=B_raw)
end
