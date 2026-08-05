# =====================================================================
# MOAD.Spectroscopy — dynamical_structure_factor (2e Task 2) tests
# =====================================================================
#
# Thin-wrapper checks for the one-sided dynamical structure factor
#   S(q,ω,T) = −Im C(q,ω,T) / π
# over the FULL ω grid (no ω-division, no ω≤0 drop, no detailed-balance
# factor). Built on a small Heisenberg dimer with O_q = S^z_1.

using Test
using LinearAlgebra
using SparseArrays
using MOAD
using MOAD: GridResponse
using MOAD.Spectroscopy: dynamical_structure_factor

# Dense one-sided correlator C(ω) = ⟨ψ₀|O†(ω+Eg−H+iΓ/2)⁻¹O|ψ₀⟩.
function _dense_corr_sf(H::Matrix, O::Matrix, ψ₀::Vector, Eg::Float64,
                        ωs::AbstractVector{<:Real}, Γ::Float64)
    x   = O * ψ₀
    out = Vector{ComplexF64}(undef, length(ωs))
    for (k, ω) in enumerate(ωs)
        z = ω + Eg + im * Γ / 2
        out[k] = dot(x, (z * I - H) \ x)
    end
    return out
end

function _heisenberg_dimer_sf()
    s1 = SpinSite{1//2}(:s1)
    s2 = SpinSite{1//2}(:s2)
    h  = Hilbert(:s1 => s1, :s2 => s2)
    H_op  = S(s1)[1] * S(s2)[1] + S(s1)[2] * S(s2)[2] + S(s1)[3] * S(s2)[3]
    basis = EagerBasis(h)
    H_sp  = assemble(compile(H_op, basis), basis)
    Oq_op = Sz(s1, -1//2) + Sz(s1, 1//2)     # S^z on site 1
    F  = eigen(Hermitian(Matrix{Float64}(H_sp)))
    k  = argmin(F.values)
    return (basis = basis, H_sp = H_sp, H_dense = Matrix{Float64}(H_sp),
            Oq_op = Oq_op,
            Oq_mat = Matrix{ComplexF64}(assemble(compile(Oq_op, basis), basis)),
            ψ₀ = ComplexF64.(F.vectors[:, k]), Eg = F.values[k])
end

@testset "dynamical_structure_factor (2e)" begin
    d  = _heisenberg_dimer_sf()
    Γ  = 0.1
    ωs = range(-1.0, 3.0, length = 120)      # full grid, includes ω ≤ 0

    # --- validation ---------------------------------------------------------
    @test_throws ArgumentError dynamical_structure_factor(d.H_sp, d.basis,
                                                           OperatorSum[]; ω_grid = ωs, Γ = Γ)
    @test_throws ArgumentError dynamical_structure_factor(d.H_sp, d.basis,
                                                           d.Oq_op; Γ = Γ)   # missing ω_grid
    @test_throws ArgumentError dynamical_structure_factor(d.H_sp, d.basis, d.Oq_op;
                                                           ω_grid = ωs, Γ = Γ,
                                                           ψ₀ = d.ψ₀, Eg = d.Eg, T = 0.5)

    # --- omega_grid ASCII alias ---------------------------------------------
    S_ascii = dynamical_structure_factor(d.H_sp, d.basis, d.Oq_op; omega_grid = ωs, Γ = Γ)
    @test S_ascii isa SpectraTensor

    # --- single op → 1-D, vector → n×n; FULL grid retained ------------------
    S1 = dynamical_structure_factor(d.H_sp, d.basis, d.Oq_op; ω_grid = ωs, Γ = Γ)
    @test ndims(S1.tensor) == 1
    @test length(S1.ω_grid) == length(ωs)          # no ω≤0 drop
    @test S1.metadata[:observable] == :S_q_omega
    @test eltype(S1.tensor) == Float64
    Sv = dynamical_structure_factor(d.H_sp, d.basis, [d.Oq_op, d.Oq_op]; ω_grid = ωs, Γ = Γ)
    @test ndims(Sv.tensor) == 3
    @test size(Sv.tensor, 1) == 2 && size(Sv.tensor, 2) == 2

    # --- T = nothing + T > 0 run --------------------------------------------
    @test all(isfinite, S1.tensor)
    S_T = dynamical_structure_factor(d.H_sp, d.basis, d.Oq_op; ω_grid = ωs, Γ = Γ, T = 0.5)
    @test S_T isa SpectraTensor
    @test all(isfinite, S_T.tensor)

    # --- numeric spot-check vs hand dense (T = 0) ---------------------------
    C   = _dense_corr_sf(d.H_dense, d.Oq_mat, d.ψ₀, d.Eg, collect(ωs), Γ)
    Sref = (-imag.(C)) ./ π
    @test maximum(abs, S1.tensor .- Sref) < 1e-10

    # --- Eg without ψ₀ rejected; kwargs forwarded ---------------------------
    @test_throws ArgumentError dynamical_structure_factor(d.H_sp, d.basis, d.Oq_op;
                                                          ω_grid = ωs, Γ = Γ, Eg = d.Eg)
    @test dynamical_structure_factor(d.H_sp, d.basis, d.Oq_op; ω_grid = ωs, Γ = Γ,
                                     reorth = :full) isa SpectraTensor   # tuning kwarg forwarded
end
