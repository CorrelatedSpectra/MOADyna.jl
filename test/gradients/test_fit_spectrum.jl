# test/gradients/test_fit_spectrum.jl
#
# Build-step 7 of the v0.3 differentiable forward model (MOADyna.Gradients): the
# deterministic direct-fit baseline `fit_spectrum` (Optim weakdep extension). Loads
# Optim to activate MOADynaOptimExt, then checks that the gradient-driven fit recovers a
# synthetic-truth θ, validates inputs (the VJP cotangent must be real), and that the
# fallback errors helpfully when Optim is NOT loaded (checked in a separate process).
# Reproducible (no RNG).

using MOADyna
using LinearAlgebra
using Test
using Optim                                   # activates MOADynaOptimExt → fit_spectrum

const Gr = MOADyna.Gradients

# Step-6 synthetic coupled fixture (H_g 4-dim, H_f 6-dim complex, shared θ=[a,b]).
function _fit_fixture()
    names = [:a, :b]
    Dg  = ComplexF64[0 0 0 0; 0 1 0 0; 0 0 2.5 0; 0 0 0 4]
    C1g = ComplexF64[0 1 0.5 0; 1 0 0 0.2; 0.5 0 0 0; 0 0.2 0 0]
    C2g = ComplexF64[0 0 0 0.3im; 0 0 0.1im 0; 0 -0.1im 0 0; -0.3im 0 0 0]
    g_model = Gr.AffineModel([Dg, C1g, C2g],
                             Gr.AffineMap([1.0, 0, 0], [0.0 0; 1 0; 0 1]), names)
    Df  = Matrix{ComplexF64}(Diagonal([0.0, 1, 2, 3, 4, 5]))
    C1f = zeros(ComplexF64, 6, 6)
    C1f[1, 2] = 0.7; C1f[2, 1] = 0.7; C1f[3, 5] = 0.4; C1f[5, 3] = 0.4
    C1f[1, 4] = 0.2; C1f[4, 1] = 0.2
    C2f = zeros(ComplexF64, 6, 6)
    C2f[1, 3] = 0.5im; C2f[3, 1] = -0.5im; C2f[2, 4] = 0.3im; C2f[4, 2] = -0.3im
    f_model = Gr.AffineModel([Df, C1f, C2f],
                             Gr.AffineMap([1.0, 0, 0], [0.0 0; 1 0; 0 1]), names)
    M0  = ComplexF64[1 0 0 0; 0.3 1 0 0; 0 0.2 1 0; 0 0 0.5 1; 0.1im 0 0 0.2; 0 0.4 0 0]
    Emb = Matrix(qr(M0).Q)[:, 1:4]
    A1 = ComplexF64[0 1 0 0 0 0; 0 0 0.5im 0 0 0; 0 0 0 1 0 0; 0 0 0 0 0 0; 0 0 0 0.3 0 0; 0 0 0 0 0 0]
    B1 = ComplexF64[0 0 1 0 0 0; 0 0 0 0 1 0; 0 0 0 0 0 0; 0.4im 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0]
    ωs = [-1.0, 0.5, 2.0, 4.5]
    return Gr.XASGradientModel(g_model, f_model, (A1,), (B1,), Emb, ωs; Γ = 0.4)
end

@testset "Gradients build-step 7: fit_spectrum (Optim baseline)" begin
    model = _fit_fixture()
    θ_true = [0.30, 0.20]
    target = Gr.spectrum(model, θ_true)

    opts = Optim.Options(g_tol = 1e-12, iterations = 500)

    # --- synthetic-truth recovery (zero-residual least squares) ---
    fit = Gr.fit_spectrum(model, target, [0.35, 0.15]; options = opts)
    @test fit.θ ≈ θ_true atol = 1e-5
    @test fit.loss < 1e-12

    # --- a different start reaches the same θ (deterministic, locally well-posed) ---
    fit2 = Gr.fit_spectrum(model, target, [0.24, 0.27]; options = opts)
    @test fit2.θ ≈ θ_true atol = 1e-5

    # --- input validation (the VJP cotangent must be real) ---
    @test_throws DimensionMismatch Gr.fit_spectrum(model, target[:, :, 1:3], θ_true)
    @test_throws ArgumentError Gr.fit_spectrum(model, ComplexF64.(target) .+ im, θ_true)
    badw = fill(-1.0, size(target))
    @test_throws ArgumentError Gr.fit_spectrum(model, target, θ_true; loss_weights = badw)

    # weighted fit still recovers θ_true (positive weights)
    w = fill(2.0, size(target))
    fitw = Gr.fit_spectrum(model, target, [0.33, 0.18]; loss_weights = w, options = opts)
    @test fitw.θ ≈ θ_true atol = 1e-5

    # --- fallback: Optim NOT loaded ⇒ helpful error (separate process, so it does not
    #     depend on this suite having already loaded Optim) ---
    proj = Base.active_project()
    script = raw"""
    using MOADyna
    using LinearAlgebra
    const G = MOADyna.Gradients
    gm = G.AffineModel([ComplexF64[0 0; 0 1]], G.AffineMap([1.0], zeros(1, 1)), [:x])
    II = Matrix{ComplexF64}(I, 2, 2)
    m = G.XASGradientModel(gm, gm, (II,), (II,), II, [0.0, 1.0]; Γ = 0.5)
    try
        G.fit_spectrum(m, zeros(1, 1, 2), [0.0])
        print("NOERROR")
    catch e
        print(occursin("Optim", sprint(showerror, e)) ? "OPTIM_FALLBACK_OK" :
              "WRONG:" * sprint(showerror, e))
    end
    """
    out = read(`$(Base.julia_cmd()) --project=$(proj) --startup-file=no -e $(script)`, String)
    @test occursin("OPTIM_FALLBACK_OK", out)
end
