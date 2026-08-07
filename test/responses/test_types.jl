# =====================================================================
# MOADyna.Responses — construction tests for the four concrete types
# =====================================================================
#
# Covers, per type:
#   1. Well-formed inputs construct and round-trip field values.
#   2. Malformed inputs raise ArgumentError (verified against the actual
#      inner-constructor checks in src/responses/types.jl).
#   3. Base.show produces non-empty sensible output.

using Test
using LinearAlgebra
using MOADyna: LanczosResponse, PoleResponse, GridResponse, GreensFunction

function _lanczos_parts(; T::Type = Float64, K::Int = 2, B::Int = 2)
    α = [Matrix{T}(I, B, B) * k for k in 1:K]
    β = [Matrix{T}(0.5 * I, B, B) for _ in 1:K-1]
    R = Matrix{T}(I, B, B)
    return α, β, R, 0.0, 0.1, 1, one(T)
end

# ---------------------------------------------------------------------------
# LanczosResponse
# ---------------------------------------------------------------------------

@testset "LanczosResponse — construction" begin

    @testset "valid constructions" begin
        # K=2, Float64
        α, β, R, Eg, Γ, sign, prefactor = _lanczos_parts(T = Float64)
        L = LanczosResponse(α, β, R, Eg, Γ, sign, prefactor)
        @test L isa LanczosResponse{Float64}
        @test L.α == α && length(L.β) == 1
        @test (L.Eg, L.Γ, L.sign, L.prefactor) == (Eg, Γ, 1, prefactor)
        @test L.converged == true   # default

        # ComplexF64
        @test LanczosResponse(_lanczos_parts(T = ComplexF64)...) isa LanczosResponse{ComplexF64}

        # converged flag both true and false
        L_t = LanczosResponse(α, β, R, Eg, Γ, sign, prefactor, true)
        L_f = LanczosResponse(α, β, R, Eg, Γ, sign, prefactor, false)
        @test L_t.converged && !L_f.converged

        # sign = -1 accepted
        @test LanczosResponse(α, β, R, Eg, Γ, -1, prefactor).sign == -1

        # K=1: empty β is valid
        α1 = [Matrix{Float64}(I, 2, 2)]
        β1 = Matrix{Float64}[]
        L1 = LanczosResponse(α1, β1, R, 0.0, 0.1, 1, 1.0)
        @test length(L1.α) == 1 && isempty(L1.β)
    end

    @testset "validation raises" begin
        α, β, R, Eg, Γ, sign, prefactor = _lanczos_parts()

        # R rows ≠ B_active. Use the matching β3 from _lanczos_parts(B=3)
        # so length(β) == length(α) − 1 holds and the R-row check is the
        # invariant that actually fires.
        α3, β3, _, _, _, _, _ = _lanczos_parts(B = 3)
        @test_throws ArgumentError LanczosResponse(
            α3, β3, Matrix{Float64}(I, 2, 3), Eg, Γ, sign, prefactor
        )

        # empty α
        @test_throws ArgumentError LanczosResponse(
            Matrix{Float64}[], Matrix{Float64}[], R, Eg, Γ, sign, prefactor
        )

        # non-square α block
        @test_throws ArgumentError LanczosResponse(
            [Matrix{Float64}(ones(2, 3))], Matrix{Float64}[], R, Eg, Γ, sign, prefactor
        )

        # length(β) ≠ length(α) − 1
        @test_throws ArgumentError LanczosResponse(
            α, [Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2)], R, Eg, Γ, sign, prefactor
        )

        # β shape mismatch
        @test_throws ArgumentError LanczosResponse(
            α, [Matrix{Float64}(ones(3, 2))], R, Eg, Γ, sign, prefactor
        )

        # sign not ±1
        @test_throws ArgumentError LanczosResponse(α, β, R, Eg, Γ, 0, prefactor)
        @test_throws ArgumentError LanczosResponse(α, β, R, Eg, Γ, 2, prefactor)
    end

    @testset "Base.show" begin
        L = LanczosResponse(_lanczos_parts()...)
        s = sprint(show, L)
        @test !isempty(s) && occursin("LanczosResponse", s)
    end
end

# ---------------------------------------------------------------------------
# PoleResponse
# ---------------------------------------------------------------------------

@testset "PoleResponse — construction" begin

    @testset "valid constructions" begin
        # 3 poles, Float64
        poles = [1.0, 2.0, 3.0]
        residues = [Matrix{Float64}(I, 2, 2) * k for k in 1:3]
        P = PoleResponse(poles, residues; Γ = 0.1)
        @test P isa PoleResponse{Float64}
        @test P.poles == poles && length(P.residues) == 3
        @test P.Eg == 0.0 && P.Γ == 0.1
        @test size(P.a0) == (2, 2) && all(iszero, P.a0)

        # ComplexF64
        @test PoleResponse([0.5, 1.5],
                           [Matrix{ComplexF64}(I, 2, 2), Matrix{ComplexF64}(2I, 2, 2)];
                           Γ = 0.05) isa PoleResponse{ComplexF64}

        # non-zero a0
        a0 = Matrix{Float64}(2I, 2, 2)
        @test PoleResponse([1.0], [Matrix{Float64}(I, 2, 2)]; a0 = a0, Γ = 0.1).a0 == a0

        # empty poles (constant-only response)
        P0 = PoleResponse(Float64[], Matrix{ComplexF64}[];
                          a0 = Matrix{ComplexF64}(I, 2, 2), Γ = 0.1)
        @test P0 isa PoleResponse{ComplexF64}
        @test isempty(P0.poles) && isempty(P0.residues)

        # positional (HDF5-loader) constructor
        P_pos = PoleResponse(zeros(Float64, 2, 2), [1.0, 2.0],
                             [Matrix{Float64}(I, 2, 2), Matrix{Float64}(2I, 2, 2)],
                             0.5, 0.1)
        @test (P_pos.Eg, P_pos.Γ) == (0.5, 0.1)
    end

    @testset "validation raises" begin
        # poles / residues length mismatch
        @test_throws ArgumentError PoleResponse([1.0, 2.0], [Matrix{Float64}(I, 2, 2)]; Γ = 0.1)
        # residue size mismatch across the list
        @test_throws ArgumentError PoleResponse([1.0, 2.0],
            [Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 3, 3)]; Γ = 0.1)
        # a0 / residue shape mismatch
        @test_throws ArgumentError PoleResponse([1.0], [Matrix{Float64}(I, 2, 2)];
            a0 = Matrix{Float64}(I, 3, 3), Γ = 0.1)
        # empty residues with no explicit a0
        @test_throws ArgumentError PoleResponse(Float64[], Matrix{Float64}[]; Γ = 0.1)
    end

    @testset "Base.show" begin
        s = sprint(show, PoleResponse([1.0, 2.0],
            [Matrix{Float64}(I, 2, 2), Matrix{Float64}(2I, 2, 2)]; Γ = 0.1))
        @test !isempty(s) && occursin("PoleResponse", s)
    end
end

# ---------------------------------------------------------------------------
# GridResponse
# ---------------------------------------------------------------------------

@testset "GridResponse — construction" begin

    @testset "valid constructions" begin
        # N=2 (rows × ω), Float64
        ω2 = collect(range(-5.0, 5.0, length = 10))
        G2 = GridResponse(rand(Float64, 3, 10), ω2; Γ = 0.1)
        @test G2 isa GridResponse{Float64, 2}
        @test G2.ω == ω2 && G2.Eg == 0.0 && G2.Γ == 0.1

        # N=3 (rows × cols × ω), ComplexF64, non-zero Eg
        G3 = GridResponse(rand(ComplexF64, 2, 2, 8),
                          collect(range(0.0, 4.0, length = 8)); Eg = 1.5, Γ = 0.05)
        @test G3 isa GridResponse{ComplexF64, 3}
        @test G3.Eg == 1.5 && size(G3.data) == (2, 2, 8)

        # positional (HDF5-loader) constructor
        G_pos = GridResponse(rand(Float64, 2, 2, 5), collect(1.0:5.0), 0.3, 0.2)
        @test (G_pos.Eg, G_pos.Γ) == (0.3, 0.2)
    end

    @testset "validation raises" begin
        # last-axis length ≠ length(ω)
        @test_throws ArgumentError GridResponse(rand(Float64, 2, 2, 6), collect(1.0:5.0); Γ = 0.1)
    end

    @testset "Base.show" begin
        s = sprint(show, GridResponse(rand(ComplexF64, 2, 2, 4), collect(1.0:4.0); Γ = 0.1))
        @test !isempty(s) && occursin("GridResponse", s)
    end
end

# ---------------------------------------------------------------------------
# GreensFunction
# ---------------------------------------------------------------------------

@testset "GreensFunction — construction" begin

    _lr(; T = Float64) = LanczosResponse(
        [Matrix{T}(I, 2, 2)], Matrix{T}[], Matrix{T}(I, 2, 2), 0.0, 0.1, 1, one(T))
    _pr(; T = Float64) = PoleResponse([1.0, 2.0],
        [Matrix{T}(I, 2, 2), Matrix{T}(2I, 2, 2)]; Γ = 0.1)
    _gr(; T = Float64) = GridResponse(rand(T, 2, 2, 5), collect(1.0:5.0); Γ = 0.1)

    @testset "slot configurations × inner reps" begin
        # LanczosResponse — all three slot patterns
        L = _lr()
        GF_add  = GreensFunction(L, nothing)
        GF_rem  = GreensFunction(nothing, L)
        GF_both = GreensFunction(L, L)
        @test GF_add  isa GreensFunction{Float64, LanczosResponse{Float64}}
        @test GF_add.addition === L && isnothing(GF_add.removal)
        @test isnothing(GF_rem.addition) && GF_rem.removal === L
        @test !isnothing(GF_both.addition) && !isnothing(GF_both.removal)

        # PoleResponse — both-populated + addition-only
        P = _pr()
        @test GreensFunction(P, P) isa GreensFunction{Float64, PoleResponse{Float64}}
        @test GreensFunction(P, nothing) isa GreensFunction{Float64, PoleResponse{Float64}}

        # GridResponse — both-populated + removal-only
        G_cplx = _gr(T = ComplexF64)
        @test GreensFunction(G_cplx, G_cplx) isa GreensFunction{ComplexF64, GridResponse{ComplexF64, 3}}
        G_real = _gr()
        @test isnothing(GreensFunction(nothing, G_real).addition)

        # eltype reflects inner T
        @test eltype(GreensFunction(_lr(T = ComplexF64), nothing)) == ComplexF64
    end

    @testset "both-nothing raises" begin
        @test_throws ArgumentError GreensFunction(nothing, nothing)
        @test_throws ArgumentError GreensFunction{Float64, LanczosResponse{Float64}}(nothing, nothing)
    end

    @testset "Base.show" begin
        L = _lr()
        for GF in (GreensFunction(L, L), GreensFunction(L, nothing), GreensFunction(nothing, L))
            s = sprint(show, GF)
            @test !isempty(s) && occursin("GreensFunction", s)
        end
    end
end
