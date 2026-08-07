"""
    MOADyna.Units

Energy-unit conversions for spectroscopy work, plus the one physical constant
the finite-temperature workflow needs (the Boltzmann constant in eV/K).

MOADyna's Hamiltonians carry no intrinsic unit — energies are in whatever units the
caller built `H` in, and the finite-temperature kwarg `temperature` is in those
same units (`k_B = 1`; see [`xas`](@ref MOADyna.Spectroscopy.xas), [`kubo_response`](@ref MOADyna.Spectroscopy.kubo_response), …). This
module is a small convenience for users who *do* work in physical units (most
commonly eV) and want to convert a temperature in Kelvin, a broadening in meV, or
a RIXS shift in cm⁻¹ into the working unit.

It is deliberately **dependency-free** and minimal: it does not attempt to be a
general physical-constants database. For arbitrary CODATA constants (masses,
moments, …), use [`PhysicalConstants.jl`](https://github.com/JuliaPhysics/PhysicalConstants.jl)
(`using PhysicalConstants.CODATA2018`), which is the idiomatic, `Unitful`-typed
Julia source.

# Provenance

The conversion factors are **exact by the post-2019 SI redefinition** (the values
of `h`, `c`, `e`, `k_B` are defined constants, so the energy-unit ratios below are
exact, not measured):

- `k_B = 1.380649e-23 J/K`  (exact)
- `h   = 6.62607015e-34 J s` (exact)
- `c   = 299792458 m/s`      (exact)
- `e   = 1.602176634e-19 C`  (exact, so 1 eV = 1.602176634e-19 J exactly)

The Rydberg energy `Ry = 13.605693122990(15) eV` is a *measured* value (CODATA
2018) — the only non-exact factor here.

# Examples

```julia
using MOADyna.Units

convert_energy(300, :K => :eV)     # ≈ 0.02585  (k_B·300 K, in eV)
convert_energy(1.0, :eV => :invcm) # ≈ 8065.54  (1 eV in cm⁻¹)
Units.kB_eV                        # 8.617333262e-5  (Boltzmann constant, eV/K)
```
"""
module Units

export convert_energy

# --- exact post-2019-SI defined constants (SI) ---
const _kB_SI = 1.380649e-23      # J / K        (exact)
const _h_SI  = 6.62607015e-34    # J s          (exact)
const _c_SI  = 299792458.0       # m / s        (exact)
const _e_SI  = 1.602176634e-19   # C  ⇒ 1 eV in J (exact)

"""
    Units.kB_eV

Boltzmann constant in eV/K, `8.617333262...e-5` (exact: `_kB_SI / _e_SI`). Use it
to turn a Kelvin temperature into the eV-based `temperature` kwarg, e.g.
`temperature = Units.kB_eV * 300`.
"""
const kB_eV = _kB_SI / _e_SI

# Each supported energy unit expressed in JOULES (1 unit = value J).
# Conversion between any two units a, b is x_b = x_a * (J[a] / J[b]).
const _RY_EV = 13.605693122990  # Rydberg in eV (CODATA 2018; measured, not exact)

const _UNIT_IN_JOULE = Dict{Symbol,Float64}(
    :J     => 1.0,                       # Joule
    :eV    => _e_SI,                     # electron volt
    :meV   => _e_SI * 1e-3,              # milli-eV
    :K     => _kB_SI,                    # Kelvin (via k_B)
    :invcm => _h_SI * _c_SI * 100.0,     # cm⁻¹  (E = h c · (1/λ), 1 cm⁻¹ = 100 m⁻¹)
    :THz   => _h_SI * 1e12,              # terahertz (E = h ν)
    :Ry    => _RY_EV * _e_SI,            # Rydberg
    :Ha    => 2 * _RY_EV * _e_SI,        # Hartree (= 2 Ry)
)

# Human-readable aliases → canonical symbol.
const _ALIASES = Dict{Symbol,Symbol}(
    :Joule => :J, :joule => :J,
    :electronvolt => :eV, :ev => :eV,
    :kelvin => :K,
    :cm_inv => :invcm, Symbol("cm^-1") => :invcm, :wavenumber => :invcm,
    :Rydberg => :Ry, :rydberg => :Ry, :Hartree => :Ha, :hartree => :Ha,
    :terahertz => :THz, :thz => :THz,
)

_canon(u::Symbol) = get(_ALIASES, u, u)

const _SUPPORTED = sort!(collect(keys(_UNIT_IN_JOULE)))

function _factor(u::Symbol)
    c = _canon(u)
    haskey(_UNIT_IN_JOULE, c) || throw(ArgumentError(
        "convert_energy: unknown energy unit :$u. Supported units (and aliases): " *
        join(string.(_SUPPORTED), ", ") * "."))
    return _UNIT_IN_JOULE[c]
end

"""
    convert_energy(x, from => to) -> Float64
    convert_energy(x, from, to)   -> Float64

Convert the energy-like quantity `x` from unit `from` to unit `to`. A Kelvin
temperature is treated as the energy `k_B·T` (so `:K` participates like any other
energy unit). Returns a `Float64`.

Supported units (case-sensitive symbols; common aliases accepted):
`:J`, `:eV`, `:meV`, `:K`, `:invcm` (cm⁻¹), `:THz`, `:Ry` (Rydberg), `:Ha`
(Hartree). All factors are exact by the post-2019 SI definition except `:Ry`/`:Ha`
(CODATA-2018 measured).

# Examples
```julia
convert_energy(300, :K => :eV)     # ≈ 0.02585
convert_energy(2.5, :eV, :K)       # ≈ 29011.4
convert_energy(1.0, :eV => :invcm) # ≈ 8065.54
```
"""
convert_energy(x::Real, from::Symbol, to::Symbol) =
    Float64(x) * (_factor(from) / _factor(to))

convert_energy(x::Real, conv::Pair{Symbol,Symbol}) =
    convert_energy(x, conv.first, conv.second)

end # module
