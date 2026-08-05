@testset "OperatorSum (basics)" begin

    s = FermionSite{2}(:s)

    @testset "construction & one/zero" begin
        z = zero(OperatorSum{Float64})
        @test isempty(z)
        @test length(z) == 0

        i = one(OperatorSum{Float64})
        @test length(i) == 1
        # the only term is the constant ()
        terms = collect(i)
        @test terms[1].chain == ()
        @test terms[1].coefficient == 1.0
    end

    @testset "single-operator sum" begin
        op = c(s, 1)
        @test length(op) == 1
        terms = collect(op)
        @test length(terms[1].chain) == 1
        @test terms[1].coefficient == 1.0
    end

    @testset "iteration" begin
        op = c(s, 1) + cdag(s, 2)
        @test length(op) == 2
        terms = collect(op)
        @test length(terms) == 2
    end

end
