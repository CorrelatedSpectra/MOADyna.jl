using Test
using MOADyna
using MOADyna.Bases: EagerBasis, get_state, get_index

@testset "embed — d⁸L¹⁰ into superset basis" begin
    m = ShellModel([:Ni_3d, :L_3d])

    # Smaller basis: d⁸L¹⁰
    basis_a = EagerBasis(m, nshells(m, :Ni_3d) == 8, nshells(m, :L_3d) == 10)
    # Bigger basis: d⁸L¹⁰ + d⁹L⁹ + d¹⁰L⁸ (sum restriction = 18)
    basis_b = EagerBasis(m, nshells(m, :Ni_3d) + nshells(m, :L_3d) == 18)

    psi_a = randn(ComplexF64, length(basis_a))
    psi_b = embed(psi_a, basis_a => basis_b)

    @test length(psi_b) == length(basis_b)
    @test sum(abs2, psi_b) ≈ sum(abs2, psi_a)   # norm preserved
end

@testset "embed — every basis_a entry maps correctly" begin
    m = ShellModel([:Ni_3d, :L_3d])
    basis_a = EagerBasis(m, nshells(m, :Ni_3d) == 8, nshells(m, :L_3d) == 10)
    basis_b = EagerBasis(m, nshells(m, :Ni_3d) + nshells(m, :L_3d) == 18)

    psi_a = randn(ComplexF64, length(basis_a))
    psi_b = embed(psi_a, basis_a => basis_b)

    # Spot-check: pick a few i, look up the basis_a state in basis_b, verify amplitude matches
    @test all(begin
        state = get_state(basis_a, i)
        j = get_index(basis_b, state)
        psi_b[j] == psi_a[i]
    end for i in 1:length(basis_a))
end

@testset "embed — entries outside basis_a are zero in basis_b" begin
    m = ShellModel([:Ni_3d, :L_3d])
    basis_a = EagerBasis(m, nshells(m, :Ni_3d) == 8, nshells(m, :L_3d) == 10)
    basis_b = EagerBasis(m, nshells(m, :Ni_3d) + nshells(m, :L_3d) == 18)

    psi_a = randn(ComplexF64, length(basis_a))
    psi_b = embed(psi_a, basis_a => basis_b)

    # All basis_b entries that are NOT in basis_a should be zero
    a_states = Set(get_state(basis_a, i) for i in 1:length(basis_a))
    @test all(begin
        state = get_state(basis_b, j)
        in(state, a_states) || psi_b[j] == 0
    end for j in 1:length(basis_b))
end

@testset "embed — error if basis_a is not a subspace of basis_b" begin
    m = ShellModel([:Ni_3d])
    basis_a = EagerBasis(m, nshells(m, :Ni_3d) == 8)
    basis_b = EagerBasis(m, nshells(m, :Ni_3d) == 7)   # disjoint

    psi_a = randn(ComplexF64, length(basis_a))
    @test_throws ArgumentError embed(psi_a, basis_a => basis_b)
end

@testset "embed — error on dimension mismatch" begin
    m = ShellModel([:Ni_3d])
    basis_a = EagerBasis(m, nshells(m, :Ni_3d) == 8)
    basis_b = EagerBasis(m, nshells(m, :Ni_3d) + nshells(m, :Ni_3d) == 0)  # tiny

    @test_throws DimensionMismatch embed(zeros(ComplexF64, 999), basis_a => basis_b)
end

@testset "embed — preserves real-typed input" begin
    m = ShellModel([:Ni_3d])
    basis_a = EagerBasis(m, nshells(m, :Ni_3d) == 5)
    basis_b = EagerBasis(m, nshells(m, :Ni_3d) in 4:6)

    psi_a = rand(Float64, length(basis_a))
    psi_b = embed(psi_a, basis_a => basis_b)
    @test eltype(psi_b) == Float64
    @test sum(abs2, psi_b) ≈ sum(abs2, psi_a)
end
