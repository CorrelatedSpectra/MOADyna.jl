# test/gradients/test_spectrum_pullback.jl
#
# Build-step 6a of the v0.3 differentiable forward model (MOADyna.Gradients): the
# grid-spectrum pullback API. The spectrum is S = −Im C_AB(ω)/π; this file checks
# spectrum / spectrum_and_jvp / jacobian and — the key test — the VJP pullback
# adjoint identity g[k] = Σ λ·dS_k = Σ Re(conj(W)·dC_k) with W = −iλ/π. Reuses the
# step-4 synthetic coupled fixture (H_g 4-dim, H_f 6-dim complex, shared θ=[a,b]).
# Reproducible (no RNG).

using MOADyna
using LinearAlgebra
using Test

const G = MOADyna.Gradients

# Step-4 coupled fixture (verbatim from test_xas_jvp.jl) wrapped as an XASGradientModel.
function _fixture()
    names = [:a, :b]
    Dg  = ComplexF64[0 0 0 0; 0 1 0 0; 0 0 2.5 0; 0 0 0 4]
    C1g = ComplexF64[0 1 0.5 0; 1 0 0 0.2; 0.5 0 0 0; 0 0.2 0 0]
    C2g = ComplexF64[0 0 0 0.3im; 0 0 0.1im 0; 0 -0.1im 0 0; -0.3im 0 0 0]
    g_model = G.AffineModel([Dg, C1g, C2g],
                            G.AffineMap([1.0, 0, 0], [0.0 0; 1 0; 0 1]), names)
    Df  = Matrix{ComplexF64}(Diagonal([0.0, 1, 2, 3, 4, 5]))
    C1f = zeros(ComplexF64, 6, 6)
    C1f[1, 2] = 0.7; C1f[2, 1] = 0.7; C1f[3, 5] = 0.4; C1f[5, 3] = 0.4
    C1f[1, 4] = 0.2; C1f[4, 1] = 0.2
    C2f = zeros(ComplexF64, 6, 6)
    C2f[1, 3] = 0.5im; C2f[3, 1] = -0.5im; C2f[2, 4] = 0.3im; C2f[4, 2] = -0.3im
    f_model = G.AffineModel([Df, C1f, C2f],
                            G.AffineMap([1.0, 0, 0], [0.0 0; 1 0; 0 1]), names)
    M0  = ComplexF64[1 0 0 0; 0.3 1 0 0; 0 0.2 1 0; 0 0 0.5 1; 0.1im 0 0 0.2; 0 0.4 0 0]
    Emb = Matrix(qr(M0).Q)[:, 1:4]
    A1 = ComplexF64[0 1 0 0 0 0; 0 0 0.5im 0 0 0; 0 0 0 1 0 0; 0 0 0 0 0 0; 0 0 0 0.3 0 0; 0 0 0 0 0 0]
    A2 = ComplexF64[0 0 0 0 0 0; 1 0 0 0 0 0; 0 0.2im 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0.5 0]
    B1 = ComplexF64[0 0 1 0 0 0; 0 0 0 0 1 0; 0 0 0 0 0 0; 0.4im 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0]
    ω_grid = [-1.0, 0.5, 2.0, 4.5]
    model = G.XASGradientModel(g_model, f_model, (A1, A2), (B1,), Emb, ω_grid; Γ = 0.4)
    return model, [0.3, 0.2]
end

@testset "Gradients build-step 6a: spectrum + VJP pullback" begin
    model, θ = _fixture()
    δ = 1e-5

    S = G.spectrum(model, θ)
    @test size(S) == (2, 1, 4)
    @test eltype(S) <: Real
    # definitional consistency: S = −Im C/π
    C = G.xas_response_C(model.g_model, model.f_model, model.A, model.B, model.embedding,
                         θ, model.ω_grid; Γ = model.Γ)
    @test maximum(abs, S - (-imag(C) ./ π)) < 1e-14

    J = G.jacobian(model, θ)
    @test size(J) == (2, 1, 4, 2)

    for k in 1:2
        ek = zeros(2); ek[k] = 1.0
        S2, dS = G.spectrum_and_jvp(model, θ, ek)
        @test S2 ≈ S
        @test J[:, :, :, k] ≈ dS                                   # jacobian column = JVP
        # central FD of the spectrum
        Sp = G.spectrum(model, θ .+ δ .* ek)
        Sm = G.spectrum(model, θ .- δ .* ek)
        @test maximum(abs, dS - (Sp - Sm) ./ (2δ)) < 1e-6
    end

    # --- the pullback adjoint identity (key test) ---
    Sb, pull = G.spectrum_with_pullback(model, θ)
    @test Sb ≈ S
    # a deterministic, non-trivial real cotangent (no RNG)
    λ = Float64[sin(i) + a - b for a in 1:2, b in 1:1, i in 1:4]
    g = pull(λ)
    @test length(g) == 2
    for k in 1:2
        ek = zeros(2); ek[k] = 1.0
        _, dS = G.spectrum_and_jvp(model, θ, ek)
        dC = G.xas_response_jvp(model.g_model, model.f_model, model.A, model.B,
                                model.embedding, θ, ek, model.ω_grid; Γ = model.Γ)
        W = (-im / π) .* λ
        @test isapprox(g[k], sum(λ .* dS); atol = 1e-10)           # g[k] = ⟨λ, dS_k⟩
        @test isapprox(g[k], sum(real(conj(W) .* dC)); atol = 1e-10)  # = Σ Re(conj(W)·dC_k)
    end

    # pullback guards: wrong shape, and a non-real cotangent (violates the VJP contract)
    @test_throws DimensionMismatch pull(zeros(2, 1, 5))
    @test_throws ArgumentError pull(fill(1.0 + 0.0im, 2, 1, 4))

    # --- ground-state degeneracy gate propagates through spectrum ---
    g_deg = G.AffineModel([Matrix{ComplexF64}(Diagonal([0.0, 0.0, 1.0, 2.0]))],
                          G.AffineMap([1.0], zeros(1, 2)), [:a, :b])
    model_deg = G.XASGradientModel(g_deg, model.f_model, model.A, model.B,
                                   model.embedding, model.ω_grid; Γ = model.Γ)
    @test_throws ArgumentError G.spectrum(model_deg, θ)
end
