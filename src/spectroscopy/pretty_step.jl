# =====================================================================
# Pretty grid snapping for `ω_grid = :auto`
# =====================================================================
#
# The auto-range
# machinery samples each Lorentzian by ≥ `density` points, but snaps
# the chosen step to a clean `{1, 2, 5} · 10^n` value so beamline
# plots have honest tick labels (e.g. 0.1 / 0.2 / 0.5 eV, never
# 0.1224423).

"""
    pretty_step(target::Real) -> Float64

Return the largest value of the form `{1, 2, 5} · 10^n` that is `≤
target`. Used by [`auto_grid`](@ref) when `ω_grid = :auto` to snap the
sampling step to a clean resolution.

```julia
pretty_step(0.333) == 0.2
pretty_step(0.20)  == 0.2
pretty_step(0.05)  == 0.05
pretty_step(0.01)  == 0.01
pretty_step(7.0)   == 5.0
pretty_step(15.0)  == 10.0
```

`target` must be strictly positive.
"""
function pretty_step(target::Real)
    target > 0 ||
        throw(ArgumentError("pretty_step: target must be > 0; got $target"))
    e    = floor(log10(target))
    mant = target / 10.0^e        # ∈ [1, 10), modulo FP drift at boundaries
    # Boundary tolerance: 0.05 / 10^-2 may evaluate to 4.999...8 due to
    # 10^-2 being inexact; treat any mant within 1 ulp of {1, 2, 5} as
    # the boundary value when picking pmant.
    ε = 1e-12
    pmant = mant < 2.0 - ε ? 1.0 :
            mant < 5.0 - ε ? 2.0 : 5.0
    # Round to one significant digit so e.g. `5.0 * 10^-2` collapses to the
    # literal `0.05` rather than `0.05000000000000001`. With pmant ∈
    # {1, 2, 5} this is exact for the sigdigits = 1 representation.
    return round(pmant * 10.0^e; sigdigits = 1)
end

"""
    auto_grid(e_min::Real, e_max::Real, Γ_axis::Real;
              padding::Real = DEFAULTS.auto_range_padding,
              density::Real = DEFAULTS.auto_range_density) -> AbstractRange

Build the auto `ω_grid` covering bandwidth `[e_min, e_max]` (already
shifted so `0` corresponds to the ground-state pole), broadened by
`Γ_axis`. The endpoints are padded by `padding · Γ_axis` and snapped
outward to integer multiples of the chosen step; the step is
`pretty_step(Γ_axis / density)`, guaranteeing each Lorentzian gets
≥ `density` samples.

```julia
auto_grid(-1.0, 4.0, 1.0)              # XAS-style, Γ = 1.0 eV
auto_grid(-2.0, 6.0, 0.05; density=4)  # high-resolution Γ_final
```

Used by `xas`, `rixs`, and `fluorescence_yield` whenever their
respective `ω*_grid` kwargs are `:auto`.
"""
function auto_grid(e_min::Real, e_max::Real, Γ_axis::Real;
                   padding::Real = DEFAULTS.auto_range_padding,
                   density::Real = DEFAULTS.auto_range_density)
    Γ_axis > 0 ||
        throw(ArgumentError("auto_grid: Γ_axis must be > 0; got $Γ_axis"))
    density > 0 ||
        throw(ArgumentError("auto_grid: density must be > 0; got $density"))
    padding ≥ 0 ||
        throw(ArgumentError("auto_grid: padding must be ≥ 0; got $padding"))
    e_min ≤ e_max ||
        throw(ArgumentError("auto_grid: e_min ($e_min) must be ≤ e_max ($e_max)"))

    target_step = Γ_axis / density
    step        = pretty_step(target_step)

    ω_min = floor((e_min - padding * Γ_axis) / step) * step
    ω_max = ceil((e_max + padding * Γ_axis) / step) * step
    n_points = round(Int, (ω_max - ω_min) / step) + 1
    return range(ω_min, ω_max; length = n_points)
end
