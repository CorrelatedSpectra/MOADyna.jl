# =====================================================================
# DEFAULTS — module-level mutable defaults
# =====================================================================
#
# A single mutable `Defaults` instance (`DEFAULTS`) holds the default
# values for the keyword arguments shared by every spectroscopy entry
# point. Edit a field once to change behaviour for the rest of the
# session; per-call kwargs always override the live default.
#
# Greek field names (`Γ_xas`, `Γ_intermediate`, `Γ_final`) are the
# canonical ones; assignment also accepts the ASCII Latin alias
# (`Gamma_xas`, …) via overloaded `Base.setproperty!`. Reading is
# Greek-only — code that reads from `DEFAULTS` stays self-documenting
# in physics notation, while users on keyboards without convenient
# Greek input can still write to it.

"""
    Defaults

Module-level mutable defaults shared by every spectroscopy entry point
(`xas`, `rixs`, `fluorescence_yield`, `optical_conductivity`,
`dynamical_structure_factor`). The single live instance is `DEFAULTS`. Edit a
field in place to change behaviour for the entire session; any
per-call keyword argument overrides the live default for that call
only.

**Lanczos controls**

| Field | Default | Meaning |
|:------|--------:|:--------|
| `krylovdim` | 200 | Maximum Krylov-space dimension per block-Lanczos run |
| `min_iter` | 5 | Minimum number of Lanczos iterations |
| `max_iter` | 500 | Hard iteration cap |
| `tol` | 1e-10 | Convergence tolerance (relative residual) |
| `deflate_tol` | 1e-12 | Deflation threshold for near-zero singular values |
| `reorth` | `:none` | Re-orthogonalisation strategy (`:none`, `:partial`, `:full`) |

**Broadenings (FWHM, in eV)**

| Field | ASCII alias | Default | Used by |
|:------|:------------|--------:|:--------|
| `Γ_xas` | `Gamma_xas` | 1.0 eV | XAS outer Lorentzian (core-hole lifetime) |
| `Γ_intermediate` | `Gamma_intermediate` | 1.0 eV | RIXS intermediate-state lifetime |
| `Γ_final` | `Gamma_final` | 0.05 eV | RIXS final-state / `fluorescence_yield` resolution |
| `Γ_response` | `Gamma_response` | 0.1 eV | `optical_conductivity` and `dynamical_structure_factor` |

Reading always uses the Greek names (`DEFAULTS.Γ_xas`). Assignment
accepts either form (`DEFAULTS.Gamma_xas = 0.8` is equivalent to
`DEFAULTS.Γ_xas = 0.8`).

**Auto-range grid (active when `ω_grid = :auto`)**

| Field | Default | Meaning |
|:------|--------:|:--------|
| `auto_range_padding` | 5.0 | Pad each spectrum edge by `padding · Γ` on each side |
| `auto_range_density` | 3.0 | Minimum number of grid points per FWHM |
"""
@kwdef mutable struct Defaults
    # ---- Lanczos --------------------------------------------------------
    krylovdim::Int           = 200
    reorth::Symbol           = :none
    tol::Float64             = 1e-10
    min_iter::Int            = 5
    max_iter::Int            = 500
    deflate_tol::Float64     = 1e-12

    # ---- Broadenings (FWHM, eV) -----------------------------------------
    Γ_xas::Float64           = 1.0
    Γ_intermediate::Float64  = 1.0
    Γ_final::Float64         = 0.05
    Γ_response::Float64      = 0.1   # σ(ω) / S(q,ω) wrappers (2e)

    # ---- Auto-range (used when ω_grid = :auto) --------------------------
    auto_range_padding::Float64   = 5.0
    auto_range_density::Float64   = 3.0
end

"""
    DEFAULTS

The module-level `Defaults` instance. Reading uses canonical Greek
field names (`DEFAULTS.Γ_intermediate`); writing accepts either Greek
or ASCII Latin (`DEFAULTS.Gamma_intermediate = 0.6` is equivalent).
"""
const DEFAULTS = Defaults()

# --- ASCII Latin → Greek field aliases --------------------------------------
#
# Only the broadenings are Greek-named. Other fields (krylovdim, tol, ...)
# are already ASCII and pass through unchanged.

const _DEFAULTS_ASCII_TO_GREEK = Dict{Symbol,Symbol}(
    :Gamma_xas          => :Γ_xas,
    :Gamma_intermediate => :Γ_intermediate,
    :Gamma_final        => :Γ_final,
    :Gamma_response     => :Γ_response,
)

function Base.setproperty!(d::Defaults, name::Symbol, val)
    canonical = get(_DEFAULTS_ASCII_TO_GREEK, name, name)
    isdefined(d, canonical) || hasfield(Defaults, canonical) ||
        throw(ArgumentError("Defaults has no field `$name`"))
    return setfield!(d, canonical,
                     convert(fieldtype(Defaults, canonical), val))
end

# Reading: always canonical. Looking up `Gamma_xas` should error to keep
# the asymmetry honest. (Default `getproperty` already errors via
# `getfield` on unknown names, so we don't need a custom getter — but
# we make the error message explicit.)

function Base.getproperty(d::Defaults, name::Symbol)
    if !hasfield(Defaults, name)
        if haskey(_DEFAULTS_ASCII_TO_GREEK, name)
            greek = _DEFAULTS_ASCII_TO_GREEK[name]
            throw(ErrorException(
                "Defaults: `$name` is an ASCII alias accepted only on " *
                "*assignment*; read via the canonical Greek name `$greek`"))
        else
            throw(ErrorException("Defaults has no field `$name`"))
        end
    end
    return getfield(d, name)
end
