# =====================================================================
# evaluate_on_grid — Spectroscopy-side wrapper over MOAD.Responses.cf_block
# =====================================================================
#
# cf_block itself lives in src/responses/cf.jl (MOAD.Responses.cf_block).
# evaluate_on_grid wraps a LanczosChunk (a Spectroscopy-domain type), so it
# lives here to avoid a circular dependency.

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

`ω_grid` is any iterable supporting `length`.
"""
function evaluate_on_grid(chunk::LanczosChunk, ω_grid;
                          Γ::Real, Eg::Real)
    R       = chunk.R
    B_raw   = chunk.raw_block_size
    n_ω     = length(ω_grid)
    α       = chunk.α
    β       = chunk.β
    Tc      = eltype(R)
    Tc <: Complex ||
        throw(ArgumentError("evaluate_on_grid: chunk eltype must be Complex; got $Tc"))

    out = Array{Tc, 3}(undef, B_raw, B_raw, n_ω)
    Rdag = adjoint(R)

    @inbounds for (i, ω) in enumerate(ω_grid)
        z   = Tc(ω + Eg + im * (Γ / 2))
        G1  = Responses.cf_block(α, β, z)  # B_active × B_active
        χ   = Rdag * G1 * R              # B_raw × B_raw
        out[:, :, i] = χ
    end

    return out
end
