@testset "hilbert" begin

    s1 = FermionSite{2}(:s1)
    s2 = FermionSite{2}(:s2)
    s3 = FermionSite{2}(:s3)

    @testset "construction" begin
        h = Hilbert(:s1 => s1, :s2 => s2)
        @test length(h) == 2
        @test h[:s1] === s1
        @test h[:s2] === s2
        @test haskey(h, :s1)
        @test !haskey(h, :s3)
        @test collect(keys(h)) == [:s1, :s2]
    end

    @testset "key must match site name" begin
        # Hilbert key must equal name(site) — chain ids use site identity,
        # so key/name mismatch would create ambiguity.
        @test_throws ArgumentError Hilbert(:wrongname => FermionSite{2}(:s1))
    end

    @testset "comprehension construction" begin
        sites = [FermionSite{2}(Symbol("s$i")) for i = 1:4]
        h = Hilbert(s.name => s for s in sites)
        @test length(h) == 4
        @test collect(keys(h)) == [:s1, :s2, :s3, :s4]
    end

    @testset "tensor composition" begin
        h₁ = Hilbert(:s1 => s1)
        h₂ = Hilbert(:s2 => s2, :s3 => s3)
        h = h₁ ⊗ h₂
        @test length(h) == 3
        @test collect(keys(h)) == [:s1, :s2, :s3]

        # overlapping name → error
        h_bad = Hilbert(:s1 => FermionSite{4}(:s1))
        @test_throws ArgumentError h₁ ⊗ h_bad
    end

end
