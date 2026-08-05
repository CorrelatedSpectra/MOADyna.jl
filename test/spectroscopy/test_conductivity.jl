# =====================================================================
# MOAD.Spectroscopy — optical_conductivity (2e Task 1) API + unit tests
# =====================================================================
#
# Thin-wrapper checks for the regular optical conductivity
#   Re σ_αβ(ω,T) = factor(ω,T) · (−Im Λ_αβ(ω)) / (ω·V),   ω > 0,
#   factor(ω,T) = (T===nothing || iszero(T)) ? 1 : -expm1(-ω/T).
# Built on a small half-filled Hubbard dimer with a bond current
#   j = i·(c†₁↑ c₂↑ − c†₂↑ c₁↑)   (current on the ↑ bond).

using Test
using LinearAlgebra
using SparseArrays
using MOAD
using MOAD: GridResponse
using MOAD.Spectroscopy: optical_conductivity

# Dense one-sided correlator Λ(ω) = ⟨ψ₀|A†(ω+Eg−H+iΓ/2)⁻¹A|ψ₀⟩.
function _dense_corr(H::Matrix, A::Matrix, ψ₀::Vector, Eg::Float64,
                     ωs::AbstractVector{<:Real}, Γ::Float64)
    x   = A * ψ₀
    out = Vector{ComplexF64}(undef, length(ωs))
    for (k, ω) in enumerate(ωs)
        z = ω + Eg + im * Γ / 2
        out[k] = dot(x, (z * I - H) \ x)
    end
    return out
end

# Half-filled Hubbard dimer in the Sz=0 sector; current on the ↑ bond.
function _hubbard_dimer_current()
    s = FermionSite{4}(:s)                   # 1↑, 1↓, 2↑, 2↓
    h = Hilbert(:s => s)
    t, U = 1.0, 4.0
    H_hop = -t * (cdag(s, 1) * c(s, 3) + cdag(s, 2) * c(s, 4))
    H_op  = (H_hop + H_hop') + U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))
    basis = EagerBasis(h,
                       n_fermion(h) == 2,
                       WeightedParticleCount([s], [1, -1, 1, -1]) == 0)
    H_sp = assemble(compile(H_op, basis), basis)
    # Paramagnetic bond current on the ↑ bond (j† = j, t = 1, e = ℏ = 1):
    j_op = im * (cdag(s, 1) * c(s, 3) - cdag(s, 3) * c(s, 1))
    F  = eigen(Hermitian(Matrix{Float64}(H_sp)))
    k  = argmin(F.values)
    return (s = s, basis = basis, H_sp = H_sp,
            H_dense = Matrix{Float64}(H_sp), j_op = j_op,
            j_mat = Matrix{ComplexF64}(assemble(compile(j_op, basis), basis)),
            ψ₀ = ComplexF64.(F.vectors[:, k]), Eg = F.values[k])
end

@testset "optical_conductivity (2e)" begin
    d = _hubbard_dimer_current()
    Γ  = 0.1
    ωs = range(-1.0, 8.0, length = 120)     # straddles 0 to exercise the drop
    V  = 2.0

    # --- validation ---------------------------------------------------------
    @test_throws ArgumentError optical_conductivity(d.H_sp, d.basis, OperatorSum[];
                                                     ω_grid = ωs, Γ = Γ)
    @test_throws ArgumentError optical_conductivity(d.H_sp, d.basis, d.j_op;
                                                     ω_grid = ωs, Γ = Γ, volume = 0.0)
    @test_throws ArgumentError optical_conductivity(d.H_sp, d.basis, d.j_op;
                                                     ω_grid = ωs, Γ = Γ, volume = -1.0)
    @test_throws ArgumentError optical_conductivity(d.H_sp, d.basis, d.j_op;
                                                     ω_grid = ωs, Γ = Γ, volume = Inf)
    # missing ω_grid (neither kwarg)
    @test_throws ArgumentError optical_conductivity(d.H_sp, d.basis, d.j_op; Γ = Γ)
    # ψ₀ together with T forbidden
    @test_throws ArgumentError optical_conductivity(d.H_sp, d.basis, d.j_op;
                                                     ω_grid = ωs, Γ = Γ,
                                                     ψ₀ = d.ψ₀, Eg = d.Eg, T = 0.5)

    # --- omega_grid ASCII alias ---------------------------------------------
    σ_ascii = optical_conductivity(d.H_sp, d.basis, d.j_op; omega_grid = ωs, Γ = Γ)
    @test σ_ascii isa SpectraTensor

    # --- single op → 1-D, vector → n×n --------------------------------------
    σ1 = optical_conductivity(d.H_sp, d.basis, d.j_op; ω_grid = ωs, Γ = Γ, volume = V)
    @test ndims(σ1.tensor) == 1
    σv = optical_conductivity(d.H_sp, d.basis, [d.j_op, d.j_op]; ω_grid = ωs, Γ = Γ)
    @test ndims(σv.tensor) == 3
    @test size(σv.tensor, 1) == 2 && size(σv.tensor, 2) == 2
    @test eltype(σ1.tensor) == Float64

    # --- ω ≤ 0 dropped ------------------------------------------------------
    n_pos = count(>(0), ωs)
    @test length(σ1.ω_grid) == n_pos
    @test all(>(0), σ1.ω_grid)
    @test σ1.metadata[:dropped_nonpositive_ω] == length(ωs) - n_pos
    @test σ1.metadata[:observable] == :sigma_regular
    @test σ1.metadata[:volume] == V

    # --- T = nothing vs explicit T = 0.0 match (factor = 1) -----------------
    σ_T0 = optical_conductivity(d.H_sp, d.basis, d.j_op; ω_grid = ωs, Γ = Γ,
                                volume = V, T = 0.0)
    @test all(isfinite, σ_T0.tensor)
    @test maximum(abs, σ_T0.tensor .- σ1.tensor) < 1e-12

    # --- T > 0 runs and returns a SpectraTensor -----------------------------
    σ_T = optical_conductivity(d.H_sp, d.basis, d.j_op; ω_grid = ωs, Γ = Γ,
                               volume = V, T = 0.5)
    @test σ_T isa SpectraTensor
    @test all(isfinite, σ_T.tensor)

    # --- numeric spot-check at an ω > 0 vs hand dense (T = 0) ---------------
    ω_pos = collect(σ1.ω_grid)
    Λ = _dense_corr(d.H_dense, d.j_mat, d.ψ₀, d.Eg, ω_pos, Γ)
    σ_ref = (-imag.(Λ)) ./ (ω_pos .* V)
    @test maximum(abs, σ1.tensor .- σ_ref) < 1e-10

    # --- finite-T detailed-balance factor sanity (full match is in Task 3) --
    # The wrapper's finite-T path uses the thermal ensemble, not the T=0 ψ₀,
    # so we only check the factor here; the element-wise finite-T σ match lives
    # in the Task 3 validation fixture.
    Tval = 0.5
    factor = -expm1.(-ω_pos ./ Tval)
    @test all(0 .<= factor .<= 1)

    # --- Eg without ψ₀ rejected; empty ω>0 grid rejected; kwargs forwarded --
    @test_throws ArgumentError optical_conductivity(d.H_sp, d.basis, d.j_op;
                                                     ω_grid = ωs, Γ = Γ, Eg = d.Eg)
    @test_throws ArgumentError optical_conductivity(d.H_sp, d.basis, d.j_op;
                                                     ω_grid = range(-2.0, -0.1, length = 5), Γ = Γ)
    @test optical_conductivity(d.H_sp, d.basis, d.j_op; ω_grid = ωs, Γ = Γ,
                               reorth = :full) isa SpectraTensor   # tuning kwarg forwarded
end
