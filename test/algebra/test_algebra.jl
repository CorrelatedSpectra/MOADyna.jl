# Helper: extract the type parameter T of an OperatorSum{T}
eltype_T(::OperatorSum{T}) where {T} = T

@testset "algebra" begin

    s = FermionSite{2}(:s)
    t = FermionSite{2}(:t)

    @testset "addition merges terms" begin
        a = c(s, 1)
        b = c(s, 1)
        op = a + b
        @test length(op) == 1
        @test collect(op)[1].coefficient ≈ 2.0

        # Different terms add separately
        op2 = c(s, 1) + cdag(s, 2)
        @test length(op2) == 2
    end

    @testset "addition deletes zero" begin
        op = c(s, 1) - c(s, 1)
        @test isempty(op)
    end

    @testset "scalar multiplication" begin
        op = 2.5 * c(s, 1)
        @test collect(op)[1].coefficient ≈ 2.5

        op = c(s, 1) * 3.0
        @test collect(op)[1].coefficient ≈ 3.0

        op = c(s, 1) / 2
        @test collect(op)[1].coefficient ≈ 0.5

        # Scalar 0 → empty sum
        op = 0 * c(s, 1)
        @test isempty(op)
    end

    @testset "negation and subtraction" begin
        a = c(s, 1)
        @test (-a) == (-1.0) * a

        a = c(s, 1)
        b = cdag(s, 2)
        @test (a - b) == a + (-b)
    end

    @testset "type promotion (Float → Complex)" begin
        H = -1.0 * c(s, 1) * cdag(s, 2)
        @test eltype_T(H) == Float64

        Hc = H + 0.5im * c(s, 1) * cdag(s, 2)
        @test eltype_T(Hc) == ComplexF64
    end

    @testset "constant lift: H + scalar" begin
        H = c(s, 1) * cdag(s, 2)
        H_shifted = H + 5.0
        # Should have one more term: the constant
        @test length(H_shifted) == length(H) + 1
        # The constant term has empty chain
        const_term = nothing
        for term in H_shifted
            if isempty(term.chain)
                const_term = term
            end
        end
        @test const_term !== nothing
        @test const_term.coefficient ≈ 5.0

        # Symmetric: scalar + H
        H_shifted2 = 5.0 + H
        @test H_shifted == H_shifted2

        # H - scalar
        @test (H - 5.0) == H + (-5.0)
        @test (5.0 - H) == (-H) + 5.0
    end

    @testset "adjoint" begin
        # n is hermitian
        op = n(s, 1)
        @test op' == op

        # H = c(s,1) cdag(s,2) → H' should reverse and swap kinds
        # H = -cdag(s,2) c(s,1) (after canonicalization, with -1 sign)
        # H' = -c(s,1) cdag(s,2) (reverse + swap), then canonicalize
        # → -1 * (1 - cdag(s,1) c(s,2)) = -1 + cdag(s,1) c(s,2)?
        # Adjoint involution: (H')' == H
        H = -1.0 * c(s, 1) * cdag(s, 2)
        @test (H')' == H

        # add_hc(H) returns H + H', which must itself be Hermitian
        Hh = add_hc(H)
        @test Hh' == Hh
    end

    @testset "chop" begin
        H = c(s, 1) + 1e-15 * cdag(s, 2)
        @test length(H) == 2
        Hc = chop(H)
        @test length(Hc) == 1
    end

    @testset "structural equality" begin
        @test c(s, 1) == c(s, 1)
        @test c(s, 1) != c(s, 2)
        @test (c(s, 1) + c(s, 2)) == (c(s, 2) + c(s, 1))   # order-independent
    end

end
