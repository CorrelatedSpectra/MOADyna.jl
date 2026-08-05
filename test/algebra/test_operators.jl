@testset "operators (constructors)" begin

    @testset "fermionic" begin
        s = FermionSite{2}(:s)
        @test c(s, 1) isa OperatorSum
        @test cdag(s, 2) isa OperatorSum
        @test length(c(s, 1)) == 1
        @test length(cdag(s, 2)) == 1

        # n returns a Sum
        @test n(s, 1) isa OperatorSum
        @test n(s) isa OperatorSum  # total
    end

    @testset "bosonic" begin
        site = BosonSite{5}(:b)
        @test b(site) isa OperatorSum
        @test bdag(site) isa OperatorSum
        @test n_b(site) isa OperatorSum
    end

    @testset "spin" begin
        site = SpinSite{1//2}(:sp)
        @test Sx(site, 1//2) isa OperatorSum
        @test Sy(site, -1//2) isa OperatorSum
        @test Sz(site, 1//2) isa OperatorSum
        @test Splus(site, 1//2) isa OperatorSum
        @test Sminus(site, 1//2) isa OperatorSum

        # Bundle returns a tuple of three OperatorSums
        bundle = S(site)
        @test bundle isa Tuple{OperatorSum, OperatorSum, OperatorSum}
        @test length(bundle) == 3
    end

    @testset "label normalization in constructors" begin
        s = FermionSite{2}(:s)
        # Both equivalent label forms produce equal operators
        @test c(s, 1) == c(s, (1,))
    end
end
