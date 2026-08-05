module MOADPlotsExt

# Package extension activated automatically when both `MOAD` and a
# plot package built on RecipesBase (e.g. `Plots.jl`) are loaded in the
# user's session. Provides a `plot(spec::SpectraTensor; …)` recipe that
# converts a spectroscopy result into either a 1-D line plot
# (XAS / FY / scalar RIXS along a single grid) or a 2-D heatmap (RIXS
# with two ω axes).
#
# Only `RecipesBase` is required at compile time; the actual plotting
# backend is supplied by the user's `Plots.jl` install. This keeps the
# extension lightweight and backend-agnostic.

using MOAD: SpectraTensor
using MOAD.Spectroscopy: polarise
using RecipesBase

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# Convert a SpectraTensor into the (x, y) pair expected by a line plot,
# or the (x_in, x_out, Z) triple expected by a heatmap. The
# `polarisation` kwarg has three forms:
#
#   :isotropic              — XAS / FY: trace over operator axes
#                              when the result was computed with vector
#                              T (sum over diagonal a==b); otherwise
#                              `polarise(spec)`. RIXS: not supported
#                              in the heatmap branch — falls back to
#                              the scalar tensor (call `polarise` first
#                              for a polarised slice).
#   AbstractVector          — XAS only: contract with ε.
#   (ε_in, ε_out)           — RIXS only: full Kramers-Heisenberg
#                              contraction.
#
# ψ-list inputs (extra leading axis) are not auto-reduced; the caller
# should index/slice the tensor first.

function _xas_intensity(spec::SpectraTensor, polarisation)
    md = spec.metadata
    md[:ψ_is_list_input]::Bool && throw(ArgumentError(
        "MOADPlotsExt: SpectraTensor was built from a list of ψ; " *
        "select a single ψ slice (e.g. average / weighted_sum / " *
        "explicit indexing) before plotting."))
    T_is_vector = md[:T_is_vector_input]::Bool
    if !T_is_vector
        # scalar T: tensor is already (n_ω,)
        return polarise(spec)
    end
    # vector T branch
    if polarisation isa AbstractVector
        return polarise(spec, polarisation)
    elseif polarisation === :isotropic
        # trace over operator axes: sum diagonal a==b of -Im χ_{ab}
        N_T = md[:n_T]::Int
        tensor = spec.tensor       # shape (N_T, N_T, n_ω)
        n_ω = size(tensor, 3)
        out = Vector{Float64}(undef, n_ω)
        @inbounds for i in 1:n_ω
            s = 0.0
            for a in 1:N_T
                s += -imag(tensor[a, a, i])
            end
            out[i] = s
        end
        return out
    else
        throw(ArgumentError(
            "MOADPlotsExt: unsupported `polarisation` for XAS — " *
            "expected an AbstractVector or :isotropic, got $(polarisation)."))
    end
end

function _fy_intensity(spec::SpectraTensor, polarisation)
    md = spec.metadata
    md[:ψ_is_list_input]::Bool && throw(ArgumentError(
        "MOADPlotsExt: fluorescence_yield SpectraTensor with a ψ " *
        "list axis; select a single ψ before plotting."))
    polarisation === :isotropic || throw(ArgumentError(
        "MOADPlotsExt: `polarisation` is not meaningful for " *
        "fluorescence_yield (already polarisation-summed); " *
        "leave it at :isotropic."))
    return polarise(spec)              # identity for FY
end

function _rixs_intensity(spec::SpectraTensor, polarisation)
    md = spec.metadata
    md[:ψ_is_list_input]::Bool && throw(ArgumentError(
        "MOADPlotsExt: RIXS SpectraTensor with a ψ list axis; " *
        "select a single ψ before plotting."))
    T_in_vec  = md[:T_in_is_vector_input]::Bool
    T_out_vec = md[:T_out_is_vector_input]::Bool
    if polarisation isa Tuple && length(polarisation) == 2
        ε_in, ε_out = polarisation
        return polarise(spec, ε_in, ε_out)
    elseif polarisation === :isotropic
        if !T_in_vec && !T_out_vec
            return polarise(spec)              # already (n_in, n_out)
        else
            throw(ArgumentError(
                "MOADPlotsExt: RIXS result has polarisation axes " *
                "(T_in/T_out vector); supply " *
                "`polarisation = (ε_in, ε_out)` or call " *
                "`polarise` upstream."))
        end
    else
        throw(ArgumentError(
            "MOADPlotsExt: unsupported `polarisation` for RIXS — " *
            "expected (ε_in, ε_out) tuple or :isotropic, got $(polarisation)."))
    end
end

# ---------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------

@recipe function f(spec::SpectraTensor; polarisation = :isotropic)
    fn = Symbol(spec.metadata[:function])
    if fn === :xas
        intensity = _xas_intensity(spec, polarisation)
        ω = collect(spec.ω_grid)
        xlabel --> "ω (eV)"
        ylabel --> "Intensity (arb. u.)"
        legend --> false
        return ω, intensity
    elseif fn === :fluorescence_yield
        intensity = _fy_intensity(spec, polarisation)
        ω = collect(spec.ω_grid)
        xlabel --> "ω_in (eV)"
        ylabel --> "Fluorescence yield (arb. u.)"
        legend --> false
        return ω, intensity
    elseif fn === :rixs
        Z = _rixs_intensity(spec, polarisation)
        ω_in_grid, ω_out_grid = spec.ω_grid
        seriestype := :heatmap
        xlabel --> "ω_in (eV)"
        ylabel --> "ω_out (eV)"
        colorbar_title --> "Intensity (arb. u.)"
        # Plots heatmap convention: heatmap(x, y, Z) with size(Z) == (length(y), length(x)).
        # `Z` from polarise is (n_in, n_out); transpose so rows index ω_out (y) and
        # cols index ω_in (x).
        return collect(ω_in_grid), collect(ω_out_grid), permutedims(Z)
    elseif fn === :algebraic_combination
        # Best-effort: treat as a plain 1-D / 2-D array along ω_grid.
        ω_grid = spec.ω_grid
        if ω_grid isa AbstractRange
            xlabel --> "ω (eV)"
            ylabel --> "Intensity (arb. u.)"
            legend --> false
            tensor = spec.tensor
            y = eltype(tensor) <: Complex ? -imag.(tensor) : tensor
            return collect(ω_grid), vec(y)
        else
            ω_in_grid, ω_out_grid = ω_grid
            seriestype := :heatmap
            xlabel --> "ω_in (eV)"
            ylabel --> "ω_out (eV)"
            colorbar_title --> "Intensity (arb. u.)"
            tensor = spec.tensor
            y = eltype(tensor) <: Complex ? -imag.(tensor) : tensor
            return collect(ω_in_grid), collect(ω_out_grid), permutedims(y)
        end
    else
        throw(ArgumentError(
            "MOADPlotsExt: unsupported SpectraTensor metadata[:function] = $(fn)"))
    end
end

end # module MOADPlotsExt
