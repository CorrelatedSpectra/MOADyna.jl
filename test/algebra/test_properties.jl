@testset "algebraic properties" begin
    # Property-based sanity checks. These exercise algebraic invariants the
    # canonicalization and arithmetic should preserve.

    s = FermionSite{2}(:s)
    t = FermionSite{2}(:t)
    bs = BosonSite{4}(:bs)

    # Build a few small operators to combine
    A = c(s, 1) * cdag(t, 1) - 0.5 * n(s, 2)
    B = cdag(s, 1) * c(t, 1) + 0.3 * n(s, 1)
    C = 0.7 * cdag(s, 2) * c(s, 1)

    @testset "addition is commutative" begin
        @test A + B == B + A
        @test (A + B) + C == A + (B + C)
    end

    @testset "scalar associates" begin
        @test 2.0 * (3.0 * A) == 6.0 * A
        @test (2.0 + 3.0) * A == 2.0 * A + 3.0 * A
    end

    @testset "distributivity" begin
        @test A * (B + C) == A * B + A * C
        @test (A + B) * C == A * C + B * C
    end

    @testset "adjoint properties" begin
        @test (A')' == A
        @test (A + B)' == A' + B'
        @test (A * B)' == B' * A'
        @test (2.0 * A)' == 2.0 * A'                # real scalar — conj is identity
        @test (1.0im * A)' == -1.0im * A'           # imaginary scalar conjugates
    end

    @testset "zero is the additive identity" begin
        z = zero(A)
        @test A + z == A
        @test z + A == A
    end

    @testset "one is the multiplicative identity" begin
        i = one(OperatorSum{Float64})
        @test A * i == A
        @test i * A == A
    end

    @testset "Pauli zeros emerge from products" begin
        @test isempty(c(s, 1) * c(s, 1))
        @test isempty(cdag(s, 1) * cdag(s, 1))
        # In a longer product
        @test isempty(c(s, 1) * c(t, 1) * c(s, 1))
    end

    @testset "fermionic anticommutation: {c_i, cdag_j} = δ_ij" begin
        # Same site, same mode
        anticomm = c(s, 1) * cdag(s, 1) + cdag(s, 1) * c(s, 1)
        @test anticomm == one(OperatorSum{Float64})

        # Same site, different modes — should be zero
        anticomm2 = c(s, 1) * cdag(s, 2) + cdag(s, 2) * c(s, 1)
        @test isempty(anticomm2)

        # Different sites — also zero
        anticomm3 = c(s, 1) * cdag(t, 1) + cdag(t, 1) * c(s, 1)
        @test isempty(anticomm3)
    end

    @testset "bosonic commutation: [b, bdag] = 1" begin
        comm = b(bs) * bdag(bs) - bdag(bs) * b(bs)
        @test comm == one(OperatorSum{Float64})
    end

    @testset "fermion-boson commute: [c, b] = 0" begin
        comm = c(s, 1) * b(bs) - b(bs) * c(s, 1)
        @test isempty(comm)
    end

    @testset "fermion number commutes with itself" begin
        n1 = n(s, 1)
        @test n1 * n1 == n1   # idempotent: n^2 = n for fermionic occupation
    end
end
