# test/gradients/test_xas_jvp.jl
#
# Build-step 4 of the v0.3 differentiable forward model (MOADyna.Gradients): the full
# coupled XAS JVP. C_AB(ω)=X_A† G X_B with θ entering BOTH H_g (initial state) and
# H_f (resolvent), so the derivative threads dE0 (Hellmann–Feynman), dψ0
# (Sternheimer) → dX, and dH_f. Validated on a synthetic coupled fixture against a
# dense resolvent, the dense explicit oracle, and the end-to-end finite difference
# of the fully coupled response. Reproducible (no RNG).

using MOADyna
using LinearAlgebra
using SparseArrays
using Test

const G = MOADyna.Gradients

@testset "Gradients build-step 4: coupled XAS JVP" begin
    names = [:a, :b]

    # H_g (4-dim complex), affine in θ=[a,b]
    Dg  = ComplexF64[0 0 0 0; 0 1 0 0; 0 0 2.5 0; 0 0 0 4]
    C1g = ComplexF64[0 1 0.5 0; 1 0 0 0.2; 0.5 0 0 0; 0 0.2 0 0]
    C2g = ComplexF64[0 0 0 0.3im; 0 0 0.1im 0; 0 -0.1im 0 0; -0.3im 0 0 0]
    g_model = G.AffineModel([Dg, C1g, C2g],
                            G.AffineMap([1.0, 0, 0], [0.0 0; 1 0; 0 1]), names)

    # H_f (6-dim complex), affine in the SAME θ
    Df  = Matrix{ComplexF64}(Diagonal([0.0, 1, 2, 3, 4, 5]))
    C1f = zeros(ComplexF64, 6, 6)
    C1f[1, 2] = 0.7; C1f[2, 1] = 0.7; C1f[3, 5] = 0.4; C1f[5, 3] = 0.4
    C1f[1, 4] = 0.2; C1f[4, 1] = 0.2
    C2f = zeros(ComplexF64, 6, 6)
    C2f[1, 3] = 0.5im; C2f[3, 1] = -0.5im; C2f[2, 4] = 0.3im; C2f[4, 2] = -0.3im
    @test ishermitian(C1f) && ishermitian(C2f)
    f_model = G.AffineModel([Df, C1f, C2f],
                            G.AffineMap([1.0, 0, 0], [0.0 0; 1 0; 0 1]), names)

    # fixed embedding isometry E (6×4), E'E = I
    M0 = ComplexF64[1 0 0 0; 0.3 1 0 0; 0 0.2 1 0; 0 0 0.5 1; 0.1im 0 0 0.2; 0 0.4 0 0]
    Emb = Matrix(qr(M0).Q)[:, 1:4]
    @test norm(Emb' * Emb - I) < 1e-10

    # transition operators: A≠B, nA=2, nB=1, complex NON-Hermitian (catches
    # transpose/conjugation bugs)
    A1 = ComplexF64[0 1 0 0 0 0; 0 0 0.5im 0 0 0; 0 0 0 1 0 0; 0 0 0 0 0 0; 0 0 0 0.3 0 0; 0 0 0 0 0 0]
    A2 = ComplexF64[0 0 0 0 0 0; 1 0 0 0 0 0; 0 0.2im 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0.5 0]
    B1 = ComplexF64[0 0 1 0 0 0; 0 0 0 0 1 0; 0 0 0 0 0 0; 0.4im 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0]
    @test !ishermitian(A1)
    A = (A1, A2); B = (B1,)

    θ = [0.3, 0.2];  Γ = 0.4;  ω_grid = [-1.0, 0.5, 2.0, 4.5];  δ = 1e-5

    C = G.xas_response_C(g_model, f_model, A, B, Emb, θ, ω_grid; Γ = Γ)
    @test size(C) == (2, 1, 4)

    # dense references
    E0, ψ0, _ = G.groundstate(g_model, θ)
    Hg = G.hamiltonian(g_model, θ);  Hf = G.hamiltonian(f_model, θ)
    Eψ0 = Emb * ψ0
    XA = hcat(A1 * Eψ0, A2 * Eψ0);  XB = reshape(B1 * Eψ0, :, 1)

    # (a) C vs dense resolvent
    for (i, ω) in enumerate(ω_grid)
        z = ω + E0 + im * Γ / 2
        @test norm(C[:, :, i] - XA' * inv(z * I - Hf) * XB) < 1e-9
    end

    # exact dense dψ0 via sum-over-states (for the explicit oracle)
    F = eigen(Hermitian(Matrix(Hg)))
    function dψ0_sos(θ̇)
        dHg = G.dhamiltonian(g_model, θ, θ̇)
        acc = zeros(ComplexF64, length(ψ0))
        for n in 2:length(F.values)
            ψn = F.vectors[:, n]
            acc .+= ψn .* (-(ψn' * (dHg * ψ0)) / (F.values[n] - E0))
        end
        return acc
    end

    for k in 1:2
        ek = zeros(2); ek[k] = 1.0
        dC = G.xas_response_jvp(g_model, f_model, A, B, Emb, θ, ek, ω_grid; Γ = Γ)
        @test size(dC) == (2, 1, 4)
        @test norm(dC) > 1e-3                                   # (d) nonzero response

        # (c) dense explicit oracle
        dE0 = real(ψ0' * (G.dhamiltonian(g_model, θ, ek) * ψ0))
        Edψ = Emb * dψ0_sos(ek)
        dXA = hcat(A1 * Edψ, A2 * Edψ);  dXB = reshape(B1 * Edψ, :, 1)
        dHf = G.dhamiltonian(f_model, θ, ek)
        for (i, ω) in enumerate(ω_grid)
            z = ω + E0 + im * Γ / 2;  Gz = inv(z * I - Hf)
            dCden = dXA' * Gz * XB + XA' * Gz * dXB + XA' * Gz * (dHf - dE0 * I) * Gz * XB
            @test norm(dC[:, :, i] - dCden) < 1e-8
        end

        # (b) end-to-end FD of the FULL coupled C (threads dE0, dψ0→dX, dH_f)
        Cp = G.xas_response_C(g_model, f_model, A, B, Emb, θ .+ δ .* ek, ω_grid; Γ = Γ)
        Cm = G.xas_response_C(g_model, f_model, A, B, Emb, θ .- δ .* ek, ω_grid; Γ = Γ)
        @test maximum(abs, dC - (Cp - Cm) / (2δ)) < 1e-6
    end

    # (f) final-state degeneracy in H_f does NOT error (GS-only gate)
    Ddeg = Matrix{ComplexF64}(Diagonal([0.0, 1, 1, 2, 3, 4]))   # eigenvalue 1 is 2-fold
    f_deg = G.AffineModel([Ddeg], G.AffineMap([1.0], zeros(1, 2)), names)
    Cdeg = G.xas_response_C(g_model, f_deg, A, B, Emb, θ, ω_grid; Γ = Γ)
    @test all(isfinite, Cdeg)

    # (g) mismatched parameter schema (swapped names) → error
    f_swap = G.AffineModel([Df, C1f, C2f],
                           G.AffineMap([1.0, 0, 0], [0.0 0; 1 0; 0 1]), [:b, :a])
    @test_throws ArgumentError G.xas_response_C(g_model, f_swap, A, B, Emb, θ, ω_grid; Γ = Γ)

    # (h) non-isometric embedding → error
    @test_throws ArgumentError G.xas_response_C(g_model, f_model, A, B, 2 .* Emb, θ,
                                                ω_grid; Γ = Γ)
end
