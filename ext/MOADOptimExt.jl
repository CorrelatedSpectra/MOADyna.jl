module MOADOptimExt

# Package extension activated automatically when both `MOAD` and `Optim` are loaded.
# Provides `MOAD.Gradients.fit_spectrum`, a deterministic gradient-driven least-squares
# direct fit of the affine-model parameters `θ` to a target XAS spectrum. The gradient
# is the analytic step-6 VJP (pullback) — no finite differences, no AD-through-eigensolver.
# This is a baseline / diagnostic; calibrated inference lives in the sibling package.
#
# Loss:     L(θ) = ½ ‖ w ∘ (S(θ) − target) ‖²,   S = spectrum(model, θ) = −Im C/π
# Gradient: ∂L/∂θ = pullback( w² ∘ (S(θ) − target) )      (the real-cotangent VJP)

using MOAD
using Optim
const G = MOAD.Gradients

function G.fit_spectrum(model::G.XASGradientModel, target::AbstractArray,
                        θ0::AbstractVector; optimizer = Optim.LBFGS(),
                        options::Optim.Options = Optim.Options(),
                        loss_weights = nothing)
    S0 = G.spectrum(model, θ0)
    # The VJP cotangent must be REAL (S = −Im C/π is real); validate the inputs so a
    # complex target or a bad weight array cannot silently violate that contract.
    size(target) == size(S0) || throw(DimensionMismatch(
        "fit_spectrum: target shape $(size(target)) ≠ spectrum shape $(size(S0))."))
    (eltype(target) <: Real && all(isfinite, target)) || throw(ArgumentError(
        "fit_spectrum: target must be real and finite (the spectrum S = −Im C/π is real)."))
    if loss_weights !== nothing
        (size(loss_weights) == size(S0) && eltype(loss_weights) <: Real &&
         all(isfinite, loss_weights) && all(≥(0), loss_weights)) || throw(ArgumentError(
            "fit_spectrum: loss_weights must be real, finite, ≥ 0, and the same shape as " *
            "the spectrum $(size(S0))."))
    end
    θ = collect(float.(θ0))            # own a fresh Float vector; never mutate the caller's

    function fg!(_F, grad, θ)
        S, pull = G.spectrum_with_pullback(model, θ)
        r = S .- target
        if grad !== nothing
            cotangent = loss_weights === nothing ? r : (loss_weights .^ 2) .* r
            grad .= pull(cotangent)
        end
        wr = loss_weights === nothing ? r : loss_weights .* r
        return 0.5 * sum(abs2, wr)
    end

    res = Optim.optimize(Optim.only_fg!(fg!), θ, optimizer, options)
    return (θ = Optim.minimizer(res), loss = Optim.minimum(res), result = res)
end

end # module MOADOptimExt
