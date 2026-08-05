# =====================================================================
# MOAD.Spectroscopy — kubo_response: full Kubo retarded χ_AB(ω,T) (Item C)
# =====================================================================
#
# Validated against the closed-form two-level susceptibility. A single fermion
# in two modes (N=1 sector → exactly two states |10⟩, |01⟩ at energies ε₁, ε₂)
# with the "σ_x" operator A = c†₁c₂ + c†₂c₁ gives
#
#   χ(ω,T) = (w₁ − w₂) [ 1/(ω + iΓ/2 − Δ) − 1/(ω + iΓ/2 + Δ) ],  Δ = ε₂ − ε₁,
#
# with wₖ = e^{−βEₖ}/Z. This is the exact two-level Kubo response.

using Test
using LinearAlgebra
using SparseArrays
using MOAD
using MOAD: kubo_response
using MOAD.Spectroscopy: SpectraTensor

@testset "kubo_response (Item C)" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    ε1, ε2 = -1.0, 2.0
    H_op  = ε1 * n(s, 1) + ε2 * n(s, 2)
    basis = EagerBasis(h, n_fermion(h) == 1)          # 2 states: |10⟩, |01⟩
    H_sp  = assemble(compile(H_op, basis), basis)
    A     = cdag(s, 1) * c(s, 2) + cdag(s, 2) * c(s, 1)   # σ_x

    Γ  = 0.2
    ωs = collect(range(-6.0, 6.0; length = 200))
    τ  = 0.9
    Δ  = ε2 - ε1
    E0 = ε1

    # analytic reference
    Zp = 1 + exp(-Δ / τ)
    w1, w2 = 1 / Zp, exp(-Δ / τ) / Zp
    ref = [(w1 - w2) * (1 / (ω + im * Γ / 2 - Δ) - 1 / (ω + im * Γ / 2 + Δ)) for ω in ωs]

    χ = kubo_response(H_sp, basis, A, A; temperature = τ, ω_grid = ωs, Γ = Γ)
    @test χ isa SpectraTensor
    @test eltype(χ.tensor) == ComplexF64
    @test ndims(χ.tensor) == 1
    @test maximum(abs, χ.tensor .- ref) < 1e-10
    @test χ.metadata[:drude_excluded] === true
    @test χ.metadata[:observable] === :kubo_chi
    @test χ.Eg ≈ E0 atol = 1e-12

    # operator-vector form → (N_A, N_B, n_ω) tensor
    χv = kubo_response(H_sp, basis, [A], [A]; temperature = τ, ω_grid = ωs, Γ = Γ)
    @test size(χv.tensor) == (1, 1, length(ωs))
    @test maximum(abs, χv.tensor[1, 1, :] .- ref) < 1e-10

    # cross-correlator A ≠ B: with B = i·(c†₁c₂ − c†₂c₁) ("σ_y"), the off-diagonal
    # matrix elements pick up ±i, giving a distinct (purely reactive at ω=0) χ_AB.
    By = im * (cdag(s, 1) * c(s, 2) - cdag(s, 2) * c(s, 1))
    χxy = kubo_response(H_sp, basis, A, By; temperature = τ, ω_grid = ωs, Γ = Γ)
    # A₁₂=A₂₁=1; By₁₂=+i, By₂₁=−i (By Hermitian). Numerator is A_{mn}·B_{nm}:
    #   (m,n)=(1,2): (w1−w2)·A₁₂·By₂₁ /(ω+iΓ/2−Δ) = (w1−w2)(−i)/(ω+iΓ/2−Δ)
    #   (m,n)=(2,1): (w2−w1)·A₂₁·By₁₂ /(ω+iΓ/2+Δ) = (w1−w2)(−i)/(ω+iΓ/2+Δ)
    refxy = [(w1 - w2) * (-im) * (1 / (ω + im * Γ / 2 - Δ) + 1 / (ω + im * Γ / 2 + Δ)) for ω in ωs]
    @test maximum(abs, χxy.tensor .- refxy) < 1e-10

    # required kwargs
    @test_throws UndefKeywordError kubo_response(H_sp, basis, A, A; ω_grid = ωs, Γ = Γ)
    @test_throws ArgumentError kubo_response(H_sp, basis, A, A; temperature = τ, Γ = Γ)

    # N_eigs truncation: project onto the single ground state → no pairs → χ = 0,
    # with a warning that both initial and final states are truncated.
    χp = @test_logs (:warn,) match_mode = :any kubo_response(
        H_sp, basis, A, A; temperature = τ, ω_grid = ωs, Γ = Γ, N_eigs = 1)
    @test maximum(abs, χp.tensor) < 1e-12
    @test χp.metadata[:N_eigs] == 1

    # temperature = 0 → only the ground state is occupied (w₁=1, w₂=0).
    χ0 = kubo_response(H_sp, basis, A, A; temperature = 0.0, ω_grid = ωs, Γ = Γ)
    ref0 = [1.0 * (1 / (ω + im * Γ / 2 - Δ) - 1 / (ω + im * Γ / 2 + Δ)) for ω in ωs]
    @test maximum(abs, χ0.tensor .- ref0) < 1e-10

    # --- N_eigs must not split a degenerate manifold ----------------------
    # One fermion in 3 modes: |100⟩,|010⟩ degenerate (E=-1), |001⟩ at E=+3.
    s3 = FermionSite{3}(:t)
    h3 = Hilbert(:t => s3)
    Hdeg = -1.0 * (n(s3, 1) + n(s3, 2)) + 3.0 * n(s3, 3)
    b3   = EagerBasis(h3, n_fermion(h3) == 1)                 # 3 states
    Hdeg_sp = assemble(compile(Hdeg, b3), b3)
    Adeg = cdag(s3, 1) * c(s3, 3) + cdag(s3, 3) * c(s3, 1) +
           cdag(s3, 2) * c(s3, 3) + cdag(s3, 3) * c(s3, 2)
    # N_eigs = 1 cuts the 2-fold degenerate ground manifold → raise.
    @test_throws ArgumentError kubo_response(Hdeg_sp, b3, Adeg, Adeg;
                                temperature = 0.5, ω_grid = ωs, Γ = Γ, N_eigs = 1)
    # N_eigs = 2 keeps the full degenerate manifold → no raise.
    χ2 = @test_logs (:warn,) match_mode = :any kubo_response(
        Hdeg_sp, b3, Adeg, Adeg; temperature = 0.5, ω_grid = ωs, Γ = Γ, N_eigs = 2)
    @test χ2 isa SpectraTensor
    @test χ2.metadata[:N_eigs] == 2
end
