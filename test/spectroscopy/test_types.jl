@testset "Spectroscopy types" begin
    using LinearAlgebra: I
    using MOADyna.Spectroscopy: SpectraTensor, LanczosChunk

    @testset "LanczosChunk{ComplexF64} basic construction + show" begin
        K = 3
        Bact, Braw = 2, 2
        α = [zeros(ComplexF64, Bact, Bact) for _ in 1:K]
        β = [zeros(ComplexF64, Bact, Bact) for _ in 1:K]
        R = Matrix{ComplexF64}(I, Bact, Braw)
        c = LanczosChunk{ComplexF64}(α, β, R, Braw, 1, 0, K, true)
        @test c.n_iter == K
        @test c.raw_block_size == Braw
        @test c.ψ_index == 1
        @test c.ω_in_index == 0
        @test c.converged
        @test c.R == R
        s = sprint(show, c)
        @test occursin("LanczosChunk", s)
        @test occursin("n_iter=$K",   s)
    end

    @testset "Rectangular R for symmetry-deflated channel" begin
        # 1 active block dimension out of 3 user channels (e.g. only one
        # symmetry-allowed channel in a 3-component dipole).
        Bact, Braw = 1, 3
        R = ComplexF64[1.0 0.0 0.0]                    # 1 × 3
        α = [zeros(ComplexF64, Bact, Bact)]
        β = [zeros(ComplexF64, Bact, Bact)]
        c = LanczosChunk{ComplexF64}(α, β, R, Braw, 1, 0, 1, true)
        @test size(c.R) == (Bact, Braw)
        @test c.raw_block_size == Braw                 # output keeps user dim
    end

    @testset "SpectraTensor: XAS scalar (N=1, S=Complex)" begin
        ω = -1.0:0.1:1.0
        n = length(ω)
        tensor = zeros(ComplexF64, n)
        Bact = 1
        chunks = [LanczosChunk{ComplexF64}(
            [zeros(ComplexF64, Bact, Bact)],
            [zeros(ComplexF64, Bact, Bact)],
            ones(ComplexF64, Bact, Bact),
            1, 1, 0, 1, true)]
        st = SpectraTensor{ComplexF64,Float64,1,typeof(ω)}(
            tensor, chunks, ω, 0.0, 1, Dict{Symbol,Any}(:edge => :L23))
        @test eltype(st.tensor) === ComplexF64
        @test ndims(st.tensor) == 1
        @test size(st.tensor) == (n,)
        @test st.Eg == 0.0
        @test st.block_size == 1
        @test st.metadata[:edge] === :L23
        s = sprint(show, st)
        @test occursin("SpectraTensor", s)
        @test occursin("chunks=1", s)
    end

    @testset "SpectraTensor: FY scalar (N=1, S=T real)" begin
        ω = -1.0:0.1:1.0
        n = length(ω)
        tensor = zeros(Float64, n)             # FY: real intensity
        st = SpectraTensor{Float64,Float64,1,typeof(ω)}(
            tensor, nothing, ω, 0.0, 1, Dict{Symbol,Any}())
        @test eltype(st.tensor) === Float64
        @test st.chunks === nothing            # algebra-derived form
        s = sprint(show, st)
        @test occursin("chunks=nothing", s)
    end

    @testset "SpectraTensor: RIXS tensor-form (N=6, S=Complex)" begin
        # 3-component T_in and T_out, n_in × n_out grid.
        N_in, N_out = 3, 3
        n_in, n_out = 11, 21
        ω_in_grid  = range(-1.0, 1.0; length = n_in)
        ω_out_grid = range(-2.0, 2.0; length = n_out)
        ω_grid = (ω_in_grid, ω_out_grid)
        tensor = zeros(ComplexF64, N_in, N_out, N_out, N_in, n_in, n_out)
        st = SpectraTensor{ComplexF64,Float64,6,typeof(ω_grid)}(
            tensor, nothing, ω_grid, 0.0, N_in * N_out, Dict{Symbol,Any}())
        @test ndims(st.tensor) == 6
        @test size(st.tensor) == (N_in, N_out, N_out, N_in, n_in, n_out)
        @test st.ω_grid === ω_grid
        @test st.block_size == N_in * N_out
    end
end
