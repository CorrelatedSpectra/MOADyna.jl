# =====================================================================
# MOAD.Responses — conversion tests: Lanczos ↔ Pole ↔ Grid
# =====================================================================
#
# Covers:
#   1. to_pole(LanczosResponse) correctness on the Heisenberg dimer
#      (S=1/2, J=1, 4-dim Hilbert space).
#   2. Three-way agreement: L(ω) ≈ to_pole(L)(ω) ≈ to_grid(L, ω).
#   3. sign == -1 conversion: to_pole and to_grid succeed and are correct.
#   4. Error-path contract:
#      a. to_grid(G, ω′) with ω′ ≠ G.ω raises ArgumentError.
#      b. to_grid(G; Γ=Γ′) with Γ′ ≠ G.Γ raises ArgumentError.
#      c. to_grid(G, G.ω) (no Γ kwarg) is the no-op pass: returns G.
#      d. to_grid(L, ω; Γ=Γ′) and to_grid(P, ω; Γ=Γ′) with Γ′ ≠ R.Γ
#         SUCCEED and the returned GridResponse carries Γ′.
#
# Physical setup: Heisenberg dimer, H = J·S₁·S₂, J = 1, four-dim Hilbert
# space {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩}.  H is constructed as a plain
# Matrix{Float64} so the test has no dependency on MOAD.Algebra / Bases —
# the conversion path is the only thing under test.

using Test
using LinearAlgebra
using MOAD: LanczosResponse, PoleResponse, GridResponse
using MOAD.Responses: block_lanczos, to_pole, to_grid

# =====================================================================
# Heisenberg dimer fixture
# =====================================================================
#
# Basis: {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩} — standard 2-spin-½ product basis,
# indexed 1 … 4.
#
# S₁·S₂ = S₁ᶻS₂ᶻ + ½(S₁⁺S₂⁻ + S₁⁻S₂⁺)
#
# H matrix (J=1):
#   diagonal: [1/4, -1/4, -1/4, 1/4]
#   off-diagonal: H[2,3] = H[3,2] = 1/2
#
# Spectrum:
#   singlet: E = -3/4, ψ_s = (|↑↓⟩ - |↓↑⟩)/√2  (ground state)
#   triplet: E = +1/4 (3-fold degenerate)

function _heisenberg_H()
    H = diagm([1.0/4, -1.0/4, -1.0/4, 1.0/4])
    H[2, 3] = 0.5
    H[3, 2] = 0.5
    return H
end

# Ground state and energy from exact diagonalisation (J=1 analytic: Eg = -3/4).
function _heisenberg_gs(H)
    F    = eigen(Hermitian(H))
    k    = argmin(F.values)
    ψ₀   = F.vectors[:, k]
    Eg   = F.values[k]
    return ψ₀, Eg
end

# Build a valid LanczosResponse by running block_lanczos on the Heisenberg
# dimer with a single-column initial block X = A|ψ₀⟩.
#
# Operator chosen: Sz₁ = diag(+1/2, +1/2, -1/2, -1/2).
# Sz₁ |ψ_s⟩ is non-zero (it has non-trivial overlap with the triplet states),
# so the initial block is non-trivial.
function _dimer_lanczos(; Γ::Float64 = 0.05, krylovdim::Int = 20)
    H    = _heisenberg_H()
    ψ₀, Eg = _heisenberg_gs(H)

    # Sz₁ acting on the ground state
    Sz1  = diagm([0.5, 0.5, -0.5, -0.5])
    x    = Sz1 * ψ₀             # initial vector; must be non-zero

    # block_lanczos expects a Matrix (N × B_raw)
    X_raw = reshape(complex(x), length(x), 1)    # ComplexF64 for cf path

    result = block_lanczos(H * one(ComplexF64), X_raw; krylovdim = krylovdim)

    # Drop trailing β if block_lanczos emitted one beyond the α stack
    # (same clipping applied in correlator.jl step 7).
    α = Vector{Matrix{ComplexF64}}(result.α)
    β_raw = Vector{Matrix{ComplexF64}}(result.β)
    β = length(β_raw) == length(α) ? β_raw[1:end-1] : β_raw
    R = Matrix{ComplexF64}(result.R)

    L = LanczosResponse{ComplexF64}(α, β, R,
                                     Float64(Eg), Γ,
                                     +1, one(ComplexF64), Bool(result.converged))
    return L, H, ψ₀, Eg
end

# =====================================================================
# 1. to_pole correctness
# =====================================================================

@testset "to_pole — Heisenberg dimer" begin

    @testset "to_pole returns PoleResponse with finite poles" begin
        L, _, _, Eg = _dimer_lanczos()
        P = to_pole(L)
        @test P isa PoleResponse
        # Poles are excitation energies (eigenvalues of T_K minus Eg):
        # they should be real (PoleResponse stores Float64 poles) and finite.
        @test all(isfinite, P.poles)
        # For a 4-dim Hilbert space starting from a non-degenerate singlet
        # ground state with Sz₁ as excitation operator, the Krylov subspace
        # has dimension ≤ 3 (singlet → triplet sector spans 3 states).
        # The number of non-trivial poles is ≤ dim(Hilbert space).
        @test length(P.poles) ≥ 1
        @test length(P.poles) ≤ 4
    end

    @testset "all poles ≥ 0 (excitation energies from singlet GS)" begin
        L, H, _, Eg = _dimer_lanczos()
        P = to_pole(L)
        # Poles are λ_n - Eg; since Eg is the minimal eigenvalue of H,
        # all excitation energies must be ≥ 0.
        @test minimum(P.poles) ≥ -1e-12   # allow round-off
    end

    @testset "poles match the projected Krylov eigenvalue set exactly" begin
        L, H, _, Eg = _dimer_lanczos()
        P = to_pole(L)
        # Sz₁ |singlet⟩ = (1/2)|T₀⟩ is itself an exact H eigenvector,
        # so the Krylov subspace is 1-dimensional and to_pole must
        # return exactly one pole at the triplet excitation energy
        # 1.0 = (+1/4) − (−3/4). Anything else (extra spurious poles,
        # missing pole, wrong energy) is a regression.
        @test length(P.poles) == 1
        @test P.poles[1] ≈ 1.0 atol = 1e-10

        # General contract check: pole set equals (eigvals(T_K) − Eg).
        # Rebuild the K = 1 block-tridiagonal T from L's α stack to
        # confirm `to_pole` projects via T, not through some other path.
        T_K = L.α[1]
        expected = sort(real.(eigvals(Hermitian(T_K))) .- Eg)
        @test sort(P.poles) ≈ expected atol = 1e-10
    end

    # ------------------------------------------------------------------
    # 2. Point-wise agreement: L(ω) ≈ P(ω) on a 50-point ω grid
    # ------------------------------------------------------------------

    @testset "L(ω) ≈ to_pole(L)(ω) to 1e-10 max-abs on 50-point grid" begin
        L, _, _, Eg = _dimer_lanczos(Γ = 0.05)
        P = to_pole(L)

        # Grid spanning the positive excitation energy region, evaluated
        # in the LanczosResponse's frame (ω is the excitation energy).
        # L(ω) internally uses z = ω + Eg + iΓ/2, so evaluating at
        # ω ∈ [0.5, 1.5] probes the triplet peak at ω ≈ 1.0.
        ωs = range(0.5, 1.5, length = 50) |> collect

        max_err = maximum(
            maximum(abs, L(ω) - P(ω)) for ω in ωs
        )
        @test max_err < 1e-10
    end

end

# =====================================================================
# 2. Three-way agreement: LanczosResponse / PoleResponse / GridResponse
# =====================================================================

@testset "Three-way view agreement: Lanczos / Pole / Grid" begin

    @testset "to_grid(L) ≈ to_grid(to_pole(L)) on 50-point grid" begin
        L, _, _, _ = _dimer_lanczos(Γ = 0.05)
        ωs = range(0.5, 1.5, length = 50) |> collect

        G_lanczos = to_grid(L, ωs)
        G_pole    = to_grid(to_pole(L), ωs)

        @test G_lanczos isa GridResponse
        @test G_pole    isa GridResponse
        # data arrays should agree point-wise
        max_err = maximum(abs, G_lanczos.data - G_pole.data)
        @test max_err < 1e-10
    end

    @testset "GridResponse.data matches direct evaluation L(ω)" begin
        L, _, _, _ = _dimer_lanczos(Γ = 0.05)
        ωs = range(0.5, 1.5, length = 50) |> collect

        G = to_grid(L, ωs)
        # G.data has shape (B_raw, B_raw, nω); G.data[:, :, iω] == L(ωs[iω])
        max_err = maximum(
            maximum(abs, G.data[:, :, iω] - L(ωs[iω])) for iω in 1:length(ωs)
        )
        @test max_err < 1e-10
    end

    @testset "PoleResponse.data matches direct evaluation P(ω)" begin
        L, _, _, _ = _dimer_lanczos(Γ = 0.05)
        ωs = range(0.5, 1.5, length = 50) |> collect

        P = to_pole(L)
        G = to_grid(P, ωs)

        max_err = maximum(
            maximum(abs, G.data[:, :, iω] - P(ωs[iω])) for iω in 1:length(ωs)
        )
        @test max_err < 1e-10
    end

    @testset "Eg and Γ fields are propagated faithfully" begin
        L, _, _, Eg = _dimer_lanczos(Γ = 0.05)
        ωs = range(0.5, 1.5, length = 10) |> collect

        P = to_pole(L)
        G = to_grid(L, ωs)

        @test P.Eg ≈ Eg  atol=1e-14
        @test P.Γ  ≈ 0.05 atol=1e-14
        @test G.Eg ≈ Eg  atol=1e-14
        @test G.Γ  ≈ 0.05 atol=1e-14
    end

end

# =====================================================================
# 3. Error-path contract tests
# =====================================================================

@testset "Error paths — conversion contract" begin

    # Build a valid LanczosResponse with sign == -1 for testing.
    function _dimer_lanczos_removal()
        H    = _heisenberg_H()
        ψ₀, Eg = _heisenberg_gs(H)
        Sz1  = diagm([0.5, 0.5, -0.5, -0.5])
        x    = Sz1 * ψ₀
        X_raw = reshape(complex(x), length(x), 1)
        result = block_lanczos(H * one(ComplexF64), X_raw; krylovdim = 10)
        α = Vector{Matrix{ComplexF64}}(result.α)
        β_raw = Vector{Matrix{ComplexF64}}(result.β)
        β = length(β_raw) == length(α) ? β_raw[1:end-1] : β_raw
        R = Matrix{ComplexF64}(result.R)
        # Patch sign to -1 — deferred removal channel.
        return LanczosResponse{ComplexF64}(α, β, R, Float64(Eg), 0.05, -1,
                                            one(ComplexF64), true)
    end

    @testset "to_pole(L) with sign == -1 — pole positions and residues" begin
        # For sign = -1, poles[n] = -(λ_n - Eg) and residues are unchanged.
        # Verify both against independently-computed values from L's α/β/R.
        L_rem = _dimer_lanczos_removal()

        # Independent computation: assemble T_K and diagonalise
        K = length(L_rem.α)
        block_sizes = [size(L_rem.α[k], 1) for k in 1:K]
        M = sum(block_sizes)
        offsets = [sum(block_sizes[1:k-1]) for k in 1:K]
        Tk = zeros(ComplexF64, M, M)
        for k in 1:K
            r = offsets[k]+1 : offsets[k]+block_sizes[k]
            Tk[r, r] = L_rem.α[k]
        end
        for k in 1:K-1
            r_k1 = offsets[k+1]+1 : offsets[k+1]+block_sizes[k+1]
            c_k  = offsets[k]+1   : offsets[k]+block_sizes[k]
            Tk[r_k1, c_k] = L_rem.β[k]
            Tk[c_k, r_k1] = L_rem.β[k]'
        end
        F = eigen(Hermitian(Tk))
        λ_ref = F.values
        V_ref = F.vectors
        B_active = size(L_rem.R, 1)
        expected_poles    = -(λ_ref .- L_rem.Eg)   # sign = -1
        expected_residues = [L_rem.prefactor * (L_rem.R' * V_ref[1:B_active, n]) *
                             (L_rem.R' * V_ref[1:B_active, n])' for n in 1:M]

        P_rem = to_pole(L_rem)
        @test P_rem isa PoleResponse
        @test sort(P_rem.poles) ≈ sort(Float64.(expected_poles)) atol=1e-10
        @test all(isapprox(P_rem.residues[n], expected_residues[n]; atol=1e-10)
                  for n in 1:length(P_rem.poles))
    end

    @testset "to_grid(L, ω) with sign == -1 matches direct L(ω) evaluation" begin
        # to_grid delegates to L(ω_pt) pointwise; verifying the grid agrees
        # with direct evaluation confirms sign=-1 is no longer blocked.
        L_rem = _dimer_lanczos_removal()
        ωs = collect(range(-2.0, 0.0, length = 10))

        G_rem = to_grid(L_rem, ωs)
        max_err = maximum(
            maximum(abs, G_rem.data[:, :, iω] - L_rem(ωs[iω])) for iω in 1:length(ωs)
        )
        @test max_err < 1e-12
    end

    @testset "to_grid(GridResponse, ω′≠G.ω) raises ArgumentError" begin
        L, _, _, _ = _dimer_lanczos()
        ωs = range(0.5, 1.5, length = 20) |> collect
        G  = to_grid(L, ωs)

        # Different first element
        ωs_shifted = copy(ωs)
        ωs_shifted[1] += 0.001
        @test_throws ArgumentError to_grid(G, ωs_shifted)

        # Completely different grid (same length)
        ωs_other = range(2.0, 3.0, length = 20) |> collect
        @test_throws ArgumentError to_grid(G, ωs_other)

        # Different length
        ωs_short = range(0.5, 1.5, length = 10) |> collect
        @test_throws ArgumentError to_grid(G, ωs_short)
    end

    @testset "to_grid(GridResponse; Γ≠G.Γ) raises ArgumentError" begin
        L, _, _, _ = _dimer_lanczos(Γ = 0.05)
        ωs = range(0.5, 1.5, length = 20) |> collect
        G  = to_grid(L, ωs)     # G.Γ == 0.05

        @test_throws ArgumentError to_grid(G, ωs; Γ = 0.10)
        @test_throws ArgumentError to_grid(G, ωs; Γ = 0.01)
    end

    @testset "to_grid(G, G.ω) with no Γ override is the no-op identity" begin
        L, _, _, _ = _dimer_lanczos(Γ = 0.05)
        ωs = range(0.5, 1.5, length = 20) |> collect
        G  = to_grid(L, ωs)

        G2 = to_grid(G, G.ω)    # no Γ kwarg — should succeed and return G
        @test G2 === G           # identity: same object
    end

    @testset "to_grid(GridResponse, G.ω; Γ = G.Γ) also passes (explicit match)" begin
        L, _, _, _ = _dimer_lanczos(Γ = 0.05)
        ωs = range(0.5, 1.5, length = 20) |> collect
        G  = to_grid(L, ωs)

        G2 = to_grid(G, G.ω; Γ = G.Γ)   # explicit Γ that matches
        @test G2 === G
    end

    @testset "Γ override is allowed for LanczosResponse and PoleResponse" begin
        L, _, _, _ = _dimer_lanczos(Γ = 0.05)
        ωs = range(0.5, 1.5, length = 20) |> collect

        Γ_new = 0.15

        # LanczosResponse override
        G_L = to_grid(L, ωs; Γ = Γ_new)
        @test G_L isa GridResponse
        @test G_L.Γ ≈ Γ_new atol=1e-14   # override is stored

        # PoleResponse override
        P = to_pole(L)
        G_P = to_grid(P, ωs; Γ = Γ_new)
        @test G_P isa GridResponse
        @test G_P.Γ ≈ Γ_new atol=1e-14

        # The overridden grids agree with each other (same Γ, same data source)
        max_err = maximum(abs, G_L.data - G_P.data)
        @test max_err < 1e-10
    end

end
