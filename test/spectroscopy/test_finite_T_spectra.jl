# =====================================================================
# MOADyna.Spectroscopy — finite-T core-level spectra (Item A, v0.2 closeout)
# =====================================================================
#
# The finite-T xas/rixs/fy wrappers Boltzmann-average the per-state spectra:
#   A_T(ω) = Σ_m ρ_m A_m(ω),  ρ_m = e^{-βE_m}/Z,  each A_m referenced to E_m.
# These tests check the wrapper reproduces an INDEPENDENT manual ensemble sum,
# stamps the right metadata (Eg = E₀, weights), and enforces its contracts
# (explicit grid; thermal-only kwargs). A half-filled Hubbard dimer in the
# (N=2, Sz=0) sector supplies a clean fixed-sector spectrum; a neutral bond
# operator stands in as the "transition" operator purely to exercise the
# thermal machinery.

using Test
using LinearAlgebra
using SparseArrays
using MOADyna
using MOADyna: xas
using MOADyna.Spectroscopy: SpectraTensor

@testset "Finite-T core-level spectra (Item A)" begin

    function _dimer()
        s = FermionSite{4}(:s)                       # 1↑,1↓,2↑,2↓
        h = Hilbert(:s => s)
        t, U = 1.0, 4.0
        H_hop = -t * (cdag(s, 1) * c(s, 3) + cdag(s, 2) * c(s, 4))
        H_op  = (H_hop + H_hop') + U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))
        basis = EagerBasis(h,
                           n_fermion(h) == 2,
                           WeightedParticleCount([s], [1, -1, 1, -1]) == 0)
        H_sp = assemble(compile(H_op, basis), basis)
        # Neutral, in-sector bond operator (Hermitian); active on the low states.
        Top  = cdag(s, 1) * c(s, 3) + cdag(s, 3) * c(s, 1)
        F    = eigen(Hermitian(Matrix{Float64}(H_sp)))
        return (basis = basis, H = H_sp, T = Top, E = F)
    end

    d  = _dimer()
    Γ  = 0.1
    ωs = range(-6.0, 6.0, length = 200)
    τ  = 0.8
    Nst = 3

    # --- wrapper vs independent manual Boltzmann sum -----------------------
    r = xas(d.H, d.basis, d.T, d.E; temperature = τ, N_states = Nst, ω_grid = ωs, Γ = Γ)
    @test r isa SpectraTensor
    @test r.metadata[:thermal] === true
    @test r.metadata[:temperature] == τ
    @test r.chunks === nothing                       # grid-only (no re_broaden)

    ev = real.(d.E.values); vc = d.E.vectors
    p  = sortperm(ev); ev = ev[p]; vc = vc[:, p]
    E0 = ev[1]
    Es = ev[1:Nst]
    ws = exp.(-(Es .- E0) ./ τ); ws ./= sum(ws)
    manual = zeros(ComplexF64, length(ωs))
    for m in 1:Nst
        χm = xas(d.H, d.basis, d.T, ComplexF64.(vc[:, m]); Eg = ev[m], ω_grid = ωs, Γ = Γ)
        manual .+= ws[m] .* χm.tensor
    end
    @test maximum(abs, r.tensor .- manual) < 1e-10
    @test r.Eg ≈ E0 atol = 1e-12
    @test r.metadata[:ensemble_weights] ≈ ws atol = 1e-12
    @test r.metadata[:N_kept] == Nst
    @test r.metadata[:ensemble_energies] ≈ Es atol = 1e-12

    # --- explicit grid required on the thermal path ------------------------
    @test_throws ArgumentError xas(d.H, d.basis, d.T, d.E;
                                   temperature = τ, N_states = Nst, Γ = Γ)

    # --- thermal-only kwargs require temperature ---------------------------
    @test_throws ArgumentError xas(d.H, d.basis, d.T, d.E;
                                   N_states = 2, ω_grid = ωs, Γ = Γ)

    # --- temperature = 0 → (degenerate) GS average; unique GS ⇒ equals GS path
    r0  = xas(d.H, d.basis, d.T, d.E; temperature = 0.0, N_states = Nst, ω_grid = ωs, Γ = Γ)
    χgs = xas(d.H, d.basis, d.T, d.E; ω_grid = ωs, Γ = Γ)
    @test maximum(abs, r0.tensor .- χgs.tensor) < 1e-10
    @test r0.metadata[:N_kept] == 1                  # zero-weight excited states dropped

    # --- σ/S temperature alias: temperature= matches T= --------------------
    s = FermionSite{4}(:s2)
    h = Hilbert(:s2 => s)
    H_hop = -1.0 * (cdag(s, 1) * c(s, 3) + cdag(s, 2) * c(s, 4))
    H_op  = (H_hop + H_hop') + 4.0 * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))
    basis = EagerBasis(h, n_fermion(h) == 2, WeightedParticleCount([s], [1, -1, 1, -1]) == 0)
    H_sp  = assemble(compile(H_op, basis), basis)
    j_op  = im * (cdag(s, 1) * c(s, 3) - cdag(s, 3) * c(s, 1))
    ωσ    = range(0.1, 8.0, length = 60)
    σ_T   = optical_conductivity(H_sp, basis, j_op; ω_grid = ωσ, Γ = 0.1, volume = 2.0, T = 0.7)
    σ_tmp = optical_conductivity(H_sp, basis, j_op; ω_grid = ωσ, Γ = 0.1, volume = 2.0, temperature = 0.7)
    @test σ_T.tensor ≈ σ_tmp.tensor atol = 1e-12
    @test_throws ArgumentError optical_conductivity(H_sp, basis, j_op;
                                ω_grid = ωσ, Γ = 0.1, volume = 2.0, T = 0.7, temperature = 0.7)
end
