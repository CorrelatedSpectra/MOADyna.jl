# =====================================================================
# Finite-temperature core-level spectra (XAS / RIXS / fluorescence yield)
# =====================================================================
#
# Boltzmann ensemble average over thermally populated INITIAL states:
#
#     A_T(ω) = Σ_m ρ_m A_m(ω),   ρ_m = e^{-βE_m} / Z,
#
# where each A_m(ω) is the spectrum computed from initial state |m⟩ referenced
# to its own E_m. The initial ensemble is supplied as a `LinearAlgebra.Eigen`
# whose eigenvectors live in the propagation `basis` (in MOADyna's single-`H`
# core-spectroscopy formulation the initial multiplet ARE the lowest eigenstates
# of `H` on `basis`; the dipole-like operator maps to the core-excited sector,
# which sits hundreds of eV higher and carries negligible Boltzmann weight). The
# weights / degeneracy-complete truncation reuse `Responses._boltzmann_ensemble`.
#
# Design notes:
#  - Thermal wrappers REQUIRE an explicit ω grid: per-state `:auto` windows differ
#    (each state's Eg = E_m shifts the pole window) and `weighted_sum` needs
#    identical grids.
#  - Output is grid-only: `weighted_sum` drops `chunks`, so `re_broaden` is
#    unavailable on a thermal spectrum (same as `average`).
#  - The combined result is relabelled with `Eg = E₀` and carries
#    `:temperature`, `:ensemble_energies`, `:ensemble_weights`, `:N_kept`.
#  - Users with a *separate* initial basis must build an `Eigen` whose vectors
#    live in `basis` (embed) or run the per-state loop themselves.

# Relabel a `weighted_sum` result: stamp Eg → E₀ and the thermal metadata.
# (`weighted_sum` inherits the FIRST per-state Eg, which is wrong for the
# ensemble; the physical reference is the ground energy E₀.)
function _relabel_thermal(s::SpectraTensor, E0::Real;
                          temperature, energies, weights)
    md = copy(s.metadata)
    md[:thermal]           = true
    md[:temperature]       = temperature
    md[:ensemble_energies] = collect(Float64, energies)
    md[:ensemble_weights]  = collect(Float64, weights)
    md[:N_kept]            = length(weights)
    F = typeof(s.ω_grid)
    N = ndims(s.tensor)
    return SpectraTensor{eltype(s.tensor), Float64, N, F}(
        s.tensor, nothing, s.ω_grid, Float64(E0), s.block_size, md)
end

# Shared guard: temperature must be a real ≥ 0.
function _check_temperature(temperature, fname::AbstractString)
    (temperature isa Real && temperature ≥ 0) || throw(ArgumentError(
        "$fname: temperature must be a real number ≥ 0; got $(repr(temperature))"))
end

# --- finite-T XAS ----------------------------------------------------------
function _xas_thermal(H, basis::AbstractBasis, T_op, E::LinearAlgebra.Eigen;
                      temperature, N_states = nothing, degen_tol = 1e-8,
                      ω_grid = nothing, omega_grid = nothing, kwargs...)
    _check_temperature(temperature, "xas")
    ω_use = _resolve_pair(ω_grid, omega_grid; default = nothing, name = "ω_grid")
    ω_use === nothing && throw(ArgumentError(
        "xas: finite-T (temperature=...) requires an explicit ω_grid — per-state " *
        ":auto grids differ (each state's Eg shifts the window) and cannot be " *
        "combined; pass ω_grid = ..."))
    states, energies, weights, E0 = Responses._boltzmann_ensemble(
        E.values, E.vectors, Float64(temperature), N_states, degen_tol)
    χs = [xas(H, basis, T_op, states[m]; Eg = energies[m], ω_grid = ω_use, kwargs...)
          for m in eachindex(states)]
    return _relabel_thermal(weighted_sum(χs, weights), E0;
                            temperature = temperature, energies = energies,
                            weights = weights)
end

# --- finite-T RIXS ---------------------------------------------------------
function _rixs_thermal(H_f, H_i, basis::AbstractBasis, T_in, T_out,
                       E::LinearAlgebra.Eigen;
                       temperature, N_states = nothing, degen_tol = 1e-8,
                       ω_in_grid = nothing, omega_in_grid = nothing,
                       ω_out_grid = nothing, omega_out_grid = nothing, kwargs...)
    _check_temperature(temperature, "rixs")
    ω_in_use = _resolve_pair(ω_in_grid, omega_in_grid; default = nothing, name = "ω_in_grid")
    ω_out_use = _resolve_pair(ω_out_grid, omega_out_grid; default = nothing, name = "ω_out_grid")
    (ω_in_use === nothing || ω_out_use === nothing) && throw(ArgumentError(
        "rixs: finite-T (temperature=...) requires explicit ω_in_grid AND " *
        "ω_out_grid — per-state :auto grids differ and cannot be combined"))
    states, energies, weights, E0 = Responses._boltzmann_ensemble(
        E.values, E.vectors, Float64(temperature), N_states, degen_tol)
    χs = [rixs(H_f, H_i, basis, T_in, T_out, states[m];
               Eg = energies[m], ω_in_grid = ω_in_use, ω_out_grid = ω_out_use, kwargs...)
          for m in eachindex(states)]
    return _relabel_thermal(weighted_sum(χs, weights), E0;
                            temperature = temperature, energies = energies,
                            weights = weights)
end

# --- finite-T fluorescence yield ------------------------------------------
function _fy_thermal(H_i, basis::AbstractBasis, T_excite, T_decay,
                     E::LinearAlgebra.Eigen;
                     temperature, N_states = nothing, degen_tol = 1e-8,
                     ω_in_grid = nothing, omega_in_grid = nothing, kwargs...)
    _check_temperature(temperature, "fluorescence_yield")
    ω_in_use = _resolve_pair(ω_in_grid, omega_in_grid; default = nothing, name = "ω_in_grid")
    ω_in_use === nothing && throw(ArgumentError(
        "fluorescence_yield: finite-T (temperature=...) requires an explicit " *
        "ω_in_grid — per-state :auto grids differ and cannot be combined"))
    states, energies, weights, E0 = Responses._boltzmann_ensemble(
        E.values, E.vectors, Float64(temperature), N_states, degen_tol)
    χs = [fluorescence_yield(H_i, basis, T_excite, T_decay, states[m];
                             Eg = energies[m], ω_in_grid = ω_in_use, kwargs...)
          for m in eachindex(states)]
    return _relabel_thermal(weighted_sum(χs, weights), E0;
                            temperature = temperature, energies = energies,
                            weights = weights)
end
