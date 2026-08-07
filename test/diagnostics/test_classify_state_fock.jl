# test/diagnostics/test_classify_state_fock.jl
#
# Fock-space many-body classify_state(ψ, basis, m, G): the LiftedRep apply.
using Test
using MOADyna
using LinearAlgebra: norm, tr

@testset "classify_state (many-body Fock apply)" begin

    @testset "closed s-shell → totally symmetric (A1g)" begin
        m = ShellModel([:H_1s])                  # ℓ=0, 2 modes
        b = basis(m, nshells(m, :H_1s) == 2)     # one closed-shell determinant
        @test length(b) == 1
        G = pointgroup(:Oh)
        res = MOADyna.classify_state([1.0 + 0im], b, m, G)
        @test res.dominant_IR == :A1g
        @test isapprox(res.dominant_weight, 1.0; atol = 1e-8)
    end

    @testset "cubic d⁸ ground term → ³A₂g (spatial A2g)" begin
        # Spin-independent cubic d-shell: no SOC, no field ⇒ Û(g) is a real
        # symmetry. Ni²⁺ d⁸ in Oh has the ³A₂g ground term.
        m = ShellModel([:Ni_3d])
        H = coulomb(m, :Ni_3d; U = 0.0, F = (11.14, 6.87)) +
            1.0 * Akm(m, :Ni_3d, :Oh, [0.6, -0.4])
        b   = basis(m, nshells(m, :Ni_3d) == 8)
        Hsp = assemble(compile(H, b), b)
        gs  = eigen(Hsp, b; n = 3)
        G   = pointgroup(:Oh)
        # 3-fold degenerate (spin triplet); every partner is spatial A2g.
        @test isapprox(gs.values[1], gs.values[2]; atol = 1e-6)
        @test isapprox(gs.values[2], gs.values[3]; atol = 1e-6)
        for i in 1:3
            res = MOADyna.classify_state(gs.vectors[:, i], b, m, G)
            @test res.dominant_IR == :A2g
            @test isapprox(res.dominant_weight, 1.0; atol = 1e-6)
        end
    end

    @testset "1-electron d sector: lifted character → subduction (Eg ⊕ T2g)" begin
        # Exercise the Fock-space apply end-to-end. Build the character of the
        # full single-electron d sector as χ_S(g) = Σ_i ⟨e_i|Û(g)|e_i⟩, where
        # each basis vector e_i is one Slater determinant and the matrix
        # elements come from the SAME lifted apply that classify_state uses
        # (via _lift_matrix_elements). Classifying χ_S must reproduce the
        # subduction D² ↓ Oh doubled by spin: 2·(Eg ⊕ T2g) over the 10 states.
        m = ShellModel([:Ni_3d])
        b = basis(m, nshells(m, :Ni_3d) == 1)     # 10 one-electron states
        @test length(b) == 10
        G = pointgroup(:Oh)
        N = length(b)
        nG = length(G.elements)
        χ = zeros(ComplexF64, nG)
        for i in 1:N
            ei = zeros(ComplexF64, N); ei[i] = 1
            χ .+= MOADyna.Diagnostics._lift_matrix_elements(ei, b, m, G)
        end
        sub = MOADyna.classify_subspace(χ, G)
        d = Dict(sub)
        @test get(d, :Eg, 0) == 2
        @test get(d, :T2g, 0) == 2
    end

    @testset "filled d¹⁰ shell → A1g (10×10 determinant, all fermion signs)" begin
        # A closed shell is a single Slater determinant; its rotation is one
        # full (per-spin 5×5, combined 10×10) determinant per group element.
        # The result is exactly A1g, weight 1, only if every fermion sign and
        # the improper-rotation parity (-1)^ℓ are correct.
        m = ShellModel([:Ni_3d])
        b = basis(m, nshells(m, :Ni_3d) == 10)   # the single d¹⁰ state
        @test length(b) == 1
        G = pointgroup(:Oh)
        res = MOADyna.classify_state([1.0 + 0im], b, m, G)
        @test res.dominant_IR == :A1g
        @test isapprox(res.dominant_weight, 1.0; atol = 1e-8)
    end

    @testset "dimension + zero-norm guards" begin
        m = ShellModel([:H_1s])
        b = basis(m, nshells(m, :H_1s) == 2)
        G = pointgroup(:Oh)
        @test_throws DimensionMismatch MOADyna.classify_state([1.0+0im, 2.0+0im], b, m, G)
        @test_throws ArgumentError MOADyna.classify_state([0.0+0im], b, m, G)
    end

end
