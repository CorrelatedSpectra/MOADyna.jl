# test/gradients/test_groundstate_grad.jl
#
# Build-step 2 of the v0.3 differentiable forward model (MOADyna.Gradients):
# ground-state derivatives — Hellmann–Feynman dE0 and the projected Sternheimer
# dψ0 — validated against finite differences and the exact sum-over-states
# response.
#
# The PRIMARY fixture is a small synthetic *complex* AffineModel whose ground
# state genuinely rotates with the parameters (norm(dψ0) > 0), so the Sternheimer
# path is actually exercised — not the trivial zero-response case. A real d⁸
# multiplet then provides the energy-gradient and degeneracy-guard checks.
# Reproducible (no RNG).

using MOADyna
using LinearAlgebra
using SparseArrays
using Test

const G = MOADyna.Gradients

@testset "Gradients build-step 2: ground-state derivatives" begin

    # === Synthetic complex fixture (nonzero dψ0, complex ψ0) ================
    # H(θ) = D + a·C1 + b·C2 on four levels. C1, C2 couple the ground level to
    # excited levels, so the ground state genuinely rotates with (a, b).
    D  = ComplexF64[0 0 0 0; 0 1 0 0; 0 0 2.5 0; 0 0 0 4.0]
    C1 = ComplexF64[0 1 0.5 0; 1 0 0 0.2; 0.5 0 0 0; 0 0.2 0 0]            # real-symmetric
    C2 = ComplexF64[0 0 0 0.3im; 0 0 0.1im 0; 0 -0.1im 0 0; -0.3im 0 0 0]  # complex Hermitian
    @test ishermitian(C1) && ishermitian(C2)
    model = G.AffineModel([D, C1, C2],
                          G.AffineMap([1.0, 0.0, 0.0], [0.0 0.0; 1.0 0.0; 0.0 1.0]),
                          [:a, :b])              # coeffs: D→1 (const), C1→a, C2→b
    θ = [0.3, 0.2]
    δ = 1e-6

    E0, ψ0, gap = G.groundstate(model, θ)
    @test gap > 1e-3
    H0 = G.hamiltonian(model, θ)
    @test norm(H0 * ψ0 - E0 * ψ0) < 1e-10
    @test maximum(abs, imag.(ψ0)) > 1e-6          # genuinely complex ground state

    # (a) Hellmann–Feynman dE0 vs central FD of E0
    gE = G.grad_E0(model, θ, ψ0)
    for k in 1:2
        ek = zeros(2); ek[k] = 1.0
        Ep, _, _ = G.groundstate(model, θ .+ δ .* ek)
        Em, _, _ = G.groundstate(model, θ .- δ .* ek)
        @test (Ep - Em) / (2δ) ≈ gE[k] atol = 1e-6
    end

    # exact sum-over-states response (dense)
    F = eigen(Hermitian(Matrix(H0)))
    function dψ0_sos(θ̇)
        dH = G.dhamiltonian(model, θ, θ̇)
        acc = zeros(ComplexF64, length(ψ0))
        for nn in 2:length(F.values)              # n > 0 (skip the GS at index 1)
            ψn = F.vectors[:, nn]
            acc .+= ψn .* (-(ψn' * (dH * ψ0)) / (F.values[nn] - E0))
        end
        return acc
    end
    Pproj(v) = v .- ψ0 .* (ψ0' * v)

    for k in 1:2
        ek = zeros(2); ek[k] = 1.0
        dψ = G.sternheimer_dψ0(model, θ, E0, ψ0, ek)
        @test norm(dψ) > 1e-3                                   # GS actually rotates
        @test norm(dψ - dψ0_sos(ek)) < 1e-8                     # (b) == sum-over-states
        @test abs(ψ0' * dψ) < 1e-9                              # (c) ⟨ψ0|dψ0⟩ ≈ 0
        dH = G.dhamiltonian(model, θ, ek)
        @test norm((H0 - E0 * I) * dψ + Pproj(dH * ψ0)) < 1e-8  # (c) residual ≈ 0
    end

    # (d) observable FD (gauge-invariant end-to-end), independent observable
    Oop = ComplexF64[0 0 0 0; 0 1 0 0; 0 0 2 0; 0 0 0 3]
    expectO(φ) = (E = G.groundstate(model, φ); real(E[2]' * (Oop * E[2])))
    for k in 1:2
        ek = zeros(2); ek[k] = 1.0
        dψ = G.sternheimer_dψ0(model, θ, E0, ψ0, ek)
        dO_an = 2 * real(dψ' * (Oop * ψ0))
        dO_fd = (expectO(θ .+ δ .* ek) - expectO(θ .- δ .* ek)) / (2δ)
        @test dO_an ≈ dO_fd atol = 1e-5
    end

    # input guards
    @test_throws ArgumentError G.groundstate(model, θ; gap_tol = -1.0)
    @test_throws ArgumentError G.groundstate(model, θ; dense_max = 1)
    @test_throws ArgumentError G.sternheimer_dψ0(model, θ, E0, ψ0, [1.0, 0.0]; rtol = 0.0)

    # === Real d⁸ multiplet: energy gradient + degeneracy guard ==============
    m    = ShellModel([:d])
    site = site_of(m, :d)
    w    = repeat([1, -1], 5)
    bsz  = basis(m, total(m) == 8, WeightedParticleCount([site], w) == 2)  # max-Sz, non-degenerate
    asm(op) = assemble(compile(op, bsz), bsz)
    U_dd, F2_0, F4_0, TenDq0, r0 = 5.0, 10.0, 6.2, 1.0, 0.8
    rc0 = [U_dd, 0.0, 0.0, 0.0]
    rJ  = zeros(4, 2); rJ[2, 1] = F2_0; rJ[3, 1] = F4_0; rJ[4, 2] = 1.0
    rmodel = G.AffineModel([asm(coulomb(m, :d; U = 1.0, F = (0.0, 0.0))),
                            asm(coulomb(m, :d; U = 0.0, F = (1.0, 0.0))),
                            asm(coulomb(m, :d; U = 0.0, F = (0.0, 1.0))),
                            asm(Akm(m, :d, :Oh, [0.6, -0.4]))],
                           G.AffineMap(rc0, rJ), [:r, :TenDq])
    θr = [r0, TenDq0]
    Er, ψr, gapr = G.groundstate(rmodel, θr)
    @test gapr > 1e-3
    gEr = G.grad_E0(rmodel, θr, ψr)
    for k in 1:2
        ek = zeros(2); ek[k] = 1.0
        Ep, _, _ = G.groundstate(rmodel, θr .+ 1e-5 .* ek)
        Em, _, _ = G.groundstate(rmodel, θr .- 1e-5 .* ek)
        @test (Ep - Em) / (2e-5) ≈ gEr[k] atol = 1e-6
    end

    # degeneracy guard: full d⁸ basis (no Sz sector) → 3-fold spin-triplet GS
    bdeg = basis(m, total(m) == 8)
    asmd(op) = assemble(compile(op, bdeg), bdeg)
    dmodel = G.AffineModel([asmd(coulomb(m, :d; U = 1.0, F = (0.0, 0.0))),
                            asmd(coulomb(m, :d; U = 0.0, F = (1.0, 0.0))),
                            asmd(coulomb(m, :d; U = 0.0, F = (0.0, 1.0))),
                            asmd(Akm(m, :d, :Oh, [0.6, -0.4]))],
                           G.AffineMap(rc0, rJ), [:r, :TenDq])
    @test_throws ArgumentError G.groundstate(dmodel, θr)
end
