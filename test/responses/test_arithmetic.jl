# =====================================================================
# MOAD.Responses — arithmetic surface tests
# =====================================================================
#
# Covers per type, where applicable:
#   1. Evaluation R(ω)
#   2. Scalar α * R / R * α with type promotion
#   3. adjoint
#   4. + (closed on Pole, lazy-promote elsewhere, mismatch errors)
#
# Source authority: src/responses/arithmetic.jl, src/responses/conversions.jl.

using Test
using LinearAlgebra
using MOAD: LanczosResponse, PoleResponse, GridResponse, GreensFunction
using MOAD.Responses: to_pole, cf_block

# =====================================================================
# Shared fixture builders
# =====================================================================

function _make_lanczos_K1(; T::Type = Float64, Eg::Float64 = 0.5, Γ::Float64 = 0.1)
    B = 2
    α = [T[1.0  0.3; 0.3  2.0]]
    β = Matrix{T}[]
    R = Matrix{T}(I, B, B)
    return LanczosResponse(α, β, R, Eg, Γ, 1, one(T))
end

function _make_lanczos_K2(; T::Type = Float64, Eg::Float64 = 0.5, Γ::Float64 = 0.1)
    B = 2
    α = [T[1.0  0.3; 0.3  2.0], T[3.0  0.1; 0.1  4.0]]
    β = [T[0.5  0.0; 0.0  0.5]]
    R = Matrix{T}(I, B, B)
    return LanczosResponse(α, β, R, Eg, Γ, 1, one(T))
end

function _make_pole_1x1(; Eg::Float64 = 0.5, Γ::Float64 = 0.1)
    return PoleResponse{ComplexF64}(zeros(ComplexF64, 1, 1),
        [1.0, 3.0],
        [fill(ComplexF64(2.0), 1, 1), fill(ComplexF64(0.5), 1, 1)],
        Eg, Γ)
end

function _make_pole_2x2(; Eg::Float64 = 0.5, Γ::Float64 = 0.1)
    return PoleResponse{ComplexF64}(zeros(ComplexF64, 2, 2),
        [1.0, 3.0],
        [Matrix{ComplexF64}([1.0  0.2im; -0.2im  1.0]),
         Matrix{ComplexF64}([0.5  0.0;    0.0   0.5])],
        Eg, Γ)
end

function _make_grid(; Eg::Float64 = 0.0, Γ::Float64 = 0.1)
    ωvec = collect(range(0.0, 4.0, length = 5))
    data = [complex(Float64(i + j), Float64(i - j + k))
            for i in 1:2, j in 1:2, k in 1:5]
    return GridResponse(Array{ComplexF64, 3}(data), ωvec; Eg = Eg, Γ = Γ)
end

# =====================================================================
# 1. Evaluation R(ω)
# =====================================================================

@testset "Evaluation R(ω)" begin

    @testset "LanczosResponse" begin
        # K=1: direct (z·I − α)⁻¹ formula at ω = 1.5
        L1 = _make_lanczos_K1()
        ω  = 1.5
        @test norm(L1(ω) - inv((ω + L1.Eg + im * L1.Γ / 2) * I - L1.α[1])) < 1e-12

        # K=1 + K=2: self-consistency vs to_pole on an ω-grid
        for L in (L1, _make_lanczos_K2())
            P  = to_pole(L)
            ωs = [-2.0, 0.0, 1.0, 4.0, 8.0]
            @test maximum(norm(L(ω) - P(ω)) for ω in ωs) < 1e-10
        end

    end

    @testset "sign = -1 evaluation" begin
        # Construct a K=1 LanczosResponse with sign = -1 using a small
        # 4×4 toy Hermitian H, starting from a single-column unit vector.
        #
        # For sign = -1 (removal channel) the unified formula gives:
        #   z    = -1 * (ω + iΓ/2) + Eg  =  Eg - ω - iΓ/2
        #   L(ω) = -1 * prefactor * R' * cf_block(α, β, z) * R
        #
        # We verify at ω = -2, 0, 2 that L(ω) matches the reference
        # computed explicitly from the formula above.

        # Toy Hermitian H (4×4, K=1 block-Lanczos run with B=1 starting block)
        H_toy = Float64[
             2.0  1.0  0.5  0.0;
             1.0  3.0  0.0  0.5;
             0.5  0.0  4.0  1.0;
             0.0  0.5  1.0  5.0
        ]
        # Starting block: single-column unit vector (B_raw = 1)
        v0 = reshape(Float64[1.0, 0.0, 0.0, 0.0], 4, 1)

        # K=1 block-Lanczos: α[1] = v0' * H * v0, β = [], R = I (1×1)
        α1 = v0' * H_toy * v0   # 1×1 matrix
        α_blocks = [α1]
        β_blocks = Matrix{Float64}[]
        R_mat    = Matrix{Float64}(I, 1, 1)

        Eg = 0.5; Γ = 0.1; pf = one(Float64)
        L_rem = LanczosResponse(α_blocks, β_blocks, R_mat, Eg, Γ, -1, pf)

        for ω in (-2.0, 0.0, 2.0)
            # Evaluated result
            val = L_rem(ω)

            # Independent reference: sign=-1 formula
            z_ref  = Eg - ω - im * Γ / 2            # = -1*(ω + iΓ/2) + Eg
            G_ref  = cf_block(α_blocks, β_blocks, z_ref)
            ref    = -pf * R_mat' * G_ref * R_mat   # sign * prefactor * R' * cf * R

            @test norm(val - ref) < 1e-12
        end
    end

    @testset "PoleResponse direct formula" begin
        P = _make_pole_1x1()
        ω, Γ = 2.0, P.Γ
        expected = P.a0 .+ P.residues[1] ./ (ω - P.poles[1] + im * Γ / 2) .+
                          P.residues[2] ./ (ω - P.poles[2] + im * Γ / 2)
        @test norm(P(ω) - expected) < 1e-15
        @test size(_make_pole_2x2()(1.5)) == (2, 2)
    end

    @testset "GridResponse on-grid + off-grid raise" begin
        G = _make_grid()
        @test G(G.ω[3]) == G.data[:, :, 3]
        @test_throws ArgumentError G(G.ω[1] + 1e-6)
    end

    @testset "GreensFunction channel delegation" begin
        P = _make_pole_1x1()
        ω = 2.0
        @test GreensFunction(P, nothing)(ω) ≈ P(ω)
        @test GreensFunction(nothing, P)(ω) ≈ P(ω)
        @test GreensFunction(P, P)(ω) ≈ 2 * P(ω)
    end
end

# =====================================================================
# 2. Scalar multiplication α * R
# =====================================================================

@testset "Scalar multiplication" begin

    @testset "LanczosResponse" begin
        L = _make_lanczos_K1()
        α = 3.0
        L2 = α * L
        @test L2 isa LanczosResponse{Float64}
        @test L2.prefactor ≈ α * L.prefactor
        @test L2.α == L.α && L2.β == L.β && L2.R == L.R

        # Commutativity
        @test (2.0 * L).prefactor ≈ (L * 2.0).prefactor

        # Type promotion: Float32 stays Float64; complex scalar promotes to ComplexF64
        @test eltype(Float32(2.0) * L) == promote_type(Float64, Float32)
        @test eltype((2.0 + 0.0im) * L) == ComplexF64

        # Functional consistency
        @test maximum(norm(L2(ω) - α * L(ω)) for ω in (0.0, 1.0, 3.0)) < 1e-12
    end

    @testset "PoleResponse" begin
        P = _make_pole_2x2()
        α = 1.5
        P2 = α * P
        @test P2 isa PoleResponse
        @test P2.a0 ≈ α * P.a0
        @test all(P2.residues[n] ≈ α * P.residues[n] for n in eachindex(P.residues))

        @test (2.0 * P).a0 ≈ (P * 2.0).a0
        @test maximum(norm(P2(ω) - α * P(ω)) for ω in (0.0, 2.0, 5.0)) < 1e-14

        # Complex scalar on real Pole promotes to ComplexF64
        P_real = PoleResponse([1.0], [Matrix{Float64}(I, 2, 2)]; Γ = 0.1)
        @test eltype((1.0 + 0.5im) * P_real) == ComplexF64
    end

    @testset "GridResponse" begin
        G = _make_grid()
        α = 2.0
        G2 = α * G
        @test G2.data ≈ α .* G.data
        @test (G2.ω, G2.Eg, G2.Γ) == (G.ω, G.Eg, G.Γ)
        @test (3.0 * G).data ≈ (G * 3.0).data
        @test G2(G.ω[2]) ≈ α * G(G.ω[2])
    end

    @testset "GreensFunction" begin
        P  = _make_pole_2x2()
        GF = GreensFunction(P, P)
        α  = 2.0
        @test (α * GF)(1.5) ≈ α * GF(1.5)
        # Commutativity on a single-channel GF
        GF1 = GreensFunction(P, nothing)
        @test (α * GF1)(1.5) ≈ (GF1 * α)(1.5)
    end
end

# =====================================================================
# 3. adjoint
# =====================================================================

@testset "adjoint" begin

    @testset "PoleResponse" begin
        P    = _make_pole_2x2()
        Padj = adjoint(P)
        @test Padj isa PoleResponse
        @test Padj.a0 ≈ adjoint(P.a0)
        @test all(Padj.residues[n] ≈ adjoint(P.residues[n]) for n in eachindex(P.residues))
        @test Padj.poles == P.poles
        @test (Padj.Eg, Padj.Γ) == (P.Eg, P.Γ)

        # Functional check on a non-Hermitian residue:
        # adjoint(P)(ω) = a₀† + Σ r_n† / (ω − p_n + iΓ/2). The +iΓ/2 sign is
        # preserved by design — adjoint(P)(ω) ≠ adjoint(P(ω)) in general.
        P_nh = PoleResponse([1.0],
                            [ComplexF64[1.0  2.0+1.0im; 3.0-1.0im  4.0]]; Γ = 0.1)
        ω, Γ = 0.5, P_nh.Γ
        expected = adjoint(P_nh.residues[1]) ./ (ω - P_nh.poles[1] + im * Γ / 2)
        @test norm(adjoint(P_nh)(ω) - expected) < 1e-13
        @test P_nh.residues[1][1, 2] != P_nh.residues[1][2, 1]   # non-trivial
    end

    @testset "LanczosResponse promotes to Pole" begin
        # adjoint(L) === adjoint(to_pole(L)). Test K=1 and K=2.
        for L in (_make_lanczos_K1(), _make_lanczos_K2())
            Ladj = adjoint(L)
            Padj = adjoint(to_pole(L))
            @test Ladj isa PoleResponse
            @test maximum(norm(Ladj(ω) - Padj(ω)) for ω in (-1.0, 0.0, 2.5, 7.0)) < 1e-10
        end
    end

    @testset "LanczosResponse adjoint with sign=-1" begin
        # Verify adjoint(L) works for removal channel (sign = -1).
        # Build a K=1 removal-channel response from the evaluation testset.
        H_toy = Float64[
             2.0  1.0  0.5  0.0;
             1.0  3.0  0.0  0.5;
             0.5  0.0  4.0  1.0;
             0.0  0.5  1.0  5.0
        ]
        v0 = reshape(Float64[1.0, 0.0, 0.0, 0.0], 4, 1)
        α1 = v0' * H_toy * v0
        L_rem = LanczosResponse([α1], Matrix{Float64}[], Matrix{Float64}(I, 1, 1), 0.5, 0.1, -1, one(Float64))

        # adjoint(L_rem) should return a PoleResponse
        Ladj = adjoint(L_rem)
        @test Ladj isa PoleResponse

        # Verify it's equivalent to adjoint(to_pole(L_rem))
        Padj = adjoint(to_pole(L_rem))
        @test maximum(norm(Ladj(ω) - Padj(ω)) for ω in (-2.0, 0.0, 2.0)) < 1e-10
    end

    @testset "GridResponse element-wise" begin
        G    = _make_grid()
        Gadj = adjoint(G)
        @test Gadj.data ≈ permutedims(conj.(G.data), (2, 1, 3))
        @test (Gadj.ω, Gadj.Eg, Gadj.Γ) == (G.ω, G.Eg, G.Γ)
        @test all(Gadj(ω) ≈ adjoint(G(ω)) for ω in G.ω)
    end

    @testset "GreensFunction per-channel" begin
        P     = _make_pole_2x2()
        Padj  = adjoint(P)
        # both-populated: GFadj(ω) = Padj(ω) + Padj(ω)
        @test adjoint(GreensFunction(P, P))(1.5) ≈ 2 * Padj(1.5)
        # addition-only: removal stays nothing
        GFadj = adjoint(GreensFunction(P, nothing))
        @test !isnothing(GFadj.addition) && isnothing(GFadj.removal)
    end
end

# =====================================================================
# 4. Addition (+)
# =====================================================================

@testset "Addition (+)" begin

    @testset "PoleResponse — closed" begin
        # Concatenation when poles are distinct
        P1 = PoleResponse([1.0, 2.0],
                          [Matrix{Float64}(I, 2, 2), Matrix{Float64}(2I, 2, 2)];
                          Eg = 0.5, Γ = 0.1)
        P2 = PoleResponse([3.0], [Matrix{Float64}(3I, 2, 2)]; Eg = 0.5, Γ = 0.1)
        Psum = P1 + P2
        @test Psum isa PoleResponse
        @test length(Psum.poles) == 3
        @test (Psum.Eg, Psum.Γ) == (P1.Eg, P1.Γ)
        # Functional check on a smaller pair
        P1a = PoleResponse([1.0], [Matrix{Float64}(I, 2, 2)]; Eg = 0.5, Γ = 0.1)
        P2a = PoleResponse([3.0], [Matrix{Float64}(2I, 2, 2)]; Eg = 0.5, Γ = 0.1)
        @test maximum(norm((P1a + P2a)(ω) - (P1a(ω) + P2a(ω))) for ω in (0.0, 2.0, 5.0)) < 1e-13

        # Coincident poles are merged with residues summed
        shared = Matrix{Float64}(I, 2, 2)
        Pmerge = PoleResponse([1.0], [shared]; Γ = 0.1) +
                 PoleResponse([1.0], [2shared]; Γ = 0.1)
        @test length(Pmerge.poles) == 1
        @test Pmerge.residues[1] ≈ 3shared

        # Eg / Γ mismatch raises
        @test_throws ArgumentError PoleResponse([1.0], [shared]; Eg = 0.0, Γ = 0.1) +
                                   PoleResponse([2.0], [shared]; Eg = 1.0, Γ = 0.1)
        @test_throws ArgumentError PoleResponse([1.0], [shared]; Eg = 0.5, Γ = 0.1) +
                                   PoleResponse([2.0], [shared]; Eg = 0.5, Γ = 0.2)
    end

    @testset "LanczosResponse + PoleResponse — lazy-promote" begin
        L = _make_lanczos_K1()
        P_other = PoleResponse([5.0], [Matrix{Float64}(I, 2, 2)]; Eg = 0.5, Γ = 0.1)
        @test L + P_other isa PoleResponse
        @test P_other + L isa PoleResponse

        # L + to_pole(L) == 2 * to_pole(L)
        P = to_pole(L)
        @test maximum(norm((L + P)(ω) - 2 * P(ω)) for ω in (-1.0, 0.5, 3.0)) < 1e-10
    end

    @testset "GridResponse — closed on matching grid" begin
        G1, G2 = _make_grid(), _make_grid()
        Gsum = G1 + G2
        @test Gsum isa GridResponse
        @test Gsum.data ≈ G1.data .+ G2.data
        @test all(Gsum(ω) ≈ G1(ω) + G2(ω) for ω in G1.ω)

        # ω mismatch raises
        ω1, ω2 = collect(range(0.0, 4.0, length = 5)), collect(range(0.0, 5.0, length = 5))
        data = rand(ComplexF64, 2, 2, 5)
        @test_throws ArgumentError GridResponse(data, ω1; Γ = 0.1) +
                                   GridResponse(data, ω2; Γ = 0.1)
    end

    @testset "GreensFunction — per-channel" begin
        P = _make_pole_2x2()
        # both-populated
        GFsum = GreensFunction(P, P) + GreensFunction(P, P)
        @test !isnothing(GFsum.addition) && !isnothing(GFsum.removal)
        @test GFsum(1.5) ≈ 4 * P(1.5)

        # addition-only + addition-only — exercises
        # `_channel_add(::Nothing, ::Nothing) = nothing`
        GFadd = GreensFunction(P, nothing) + GreensFunction(P, nothing)
        @test !isnothing(GFadd.addition) && isnothing(GFadd.removal)
        @test GFadd(1.5) ≈ 2 * P(1.5)

        # Grid + Pole on the same channel raises (Grid → Pole fitting deferred)
        G = _make_grid(Eg = 0.5, Γ = 0.1)
        @test_throws ArgumentError GreensFunction(G, nothing) +
                                   GreensFunction(P, nothing)
    end

    @testset "LanczosResponse + LanczosResponse" begin
        # Regression: the two mixed Lanczos/AbstractResponse methods are
        # equally specific for two LanczosResponse operands, so without a
        # dedicated method this raised a dispatch ambiguity error.
        L1 = _make_lanczos_K1(Eg = 0.5, Γ = 0.1)
        L2 = _make_lanczos_K2(Eg = 0.5, Γ = 0.1)

        S = L1 + L2
        @test S isa PoleResponse

        # Semantics: adding promotes both operands via to_pole and sums them.
        ωs = [-1.0, 0.0, 0.75, 2.5]
        @test maximum(maximum(abs, S(ω) - (to_pole(L1)(ω) + to_pole(L2)(ω)))
                      for ω in ωs) < 1e-10

        # Commutative, and consistent with the mixed-argument methods.
        @test maximum(maximum(abs, S(ω) - (L2 + L1)(ω)) for ω in ωs) < 1e-10
        @test maximum(maximum(abs, S(ω) - (to_pole(L1) + L2)(ω)) for ω in ωs) < 1e-10

        # Eg / Γ mismatches are rejected, as for any PoleResponse sum.
        L_badEg = _make_lanczos_K1(Eg = 0.9, Γ = 0.1)
        L_badΓ  = _make_lanczos_K1(Eg = 0.5, Γ = 0.2)
        @test_throws ArgumentError L1 + L_badEg
        @test_throws ArgumentError L1 + L_badΓ
    end
end

# =====================================================================
# 5. Lanczos ↔ Pole round-trip: L(ω) ≈ to_pole(L)(ω) for sign ∈ {±1}
# =====================================================================

@testset "Lanczos↔Pole round-trip" begin
    # Use the K=1 and K=2 fixtures; loop over sign ∈ {+1, -1}.
    # For sign = -1 we patch an existing LanczosResponse rather than
    # constructing one from scratch (the block-Lanczos data is identical;
    # only the evaluation/pole formula changes).
    for sign in (+1, -1)
        for L_base in (_make_lanczos_K1(), _make_lanczos_K2())
            L = LanczosResponse(L_base.α, L_base.β, L_base.R,
                                L_base.Eg, L_base.Γ, sign, L_base.prefactor)
            P = to_pole(L)
            # ω-grid around the positive pole region; sign=-1 poles are negative
            # but evaluation is compared pointwise regardless.
            ωs = collect(range(-2.0, 4.0, length = 10))
            max_err = maximum(maximum(abs, L(ω) - P(ω)) for ω in ωs)
            @test max_err < 1e-10
        end
    end
end
