@testset "observables" begin

    s1 = FermionSite{2}(:s1)
    s2 = FermionSite{2}(:s2)
    bs = BosonSite{5}(:b1)
    sp = SpinSite{1//2}(:sp1)

    h = Hilbert(:s1 => s1, :s2 => s2, :b1 => bs, :sp1 => sp)

    @testset "n_fermion" begin
        nf = n_fermion(h)
        @test nf isa ParticleCount{Fermionic}
        # Should pick up only FermionSites
        @test length(nf.sites) == 2
        @test all(s isa FermionSite for s in nf.sites)
    end

    @testset "n_boson" begin
        nb = n_boson(h)
        @test nb isa ParticleCount{Bosonic}
        @test length(nb.sites) == 1
        @test nb.sites[1] === bs
    end

    @testset "Sz_total: SpinSite only, FermionSite throws" begin
        # h has s1, s2 (FermionSites), bs (BosonSite), sp (SpinSite).
        # Per layer-2 v3.x design, Sz_total picks up SpinSites only.
        # FermionSite spin requires explicit weights via WeightedParticleCount.
        sz = Sz_total(h)
        @test sz isa TotalSz
        @test length(sz.sites) == 1
        @test sz.sites[1] === sp

        # Direct misuse on a FermionSite throws with a hint.
        @test_throws ArgumentError Sz_total(s1)

        # Vector-of-FermionSites also throws (the inner constructor catches it).
        @test_throws ArgumentError Sz_total(AbstractSite[s1, s2])

        # SpinSite-only Hilberts work.
        h_spin = Hilbert(:sp1 => SpinSite{1//2}(:sp1), :sp2 => SpinSite{1}(:sp2))
        @test length(Sz_total(h_spin).sites) == 2
    end

    @testset "WeightedParticleCount" begin
        # Two-orbital site, modes 1=up 2=down each → length 4 weights.
        wpc = WeightedParticleCount([s1, s2], [1//2, -1//2, 1//2, -1//2])
        @test wpc isa WeightedParticleCount{Rational{Int}}
        @test length(wpc.sites) == 2
        @test wpc.weights == [1//2, -1//2, 1//2, -1//2]

        # Int weights: type W = Int.
        wpc_int = WeightedParticleCount([s1], [1, -1])
        @test wpc_int isa WeightedParticleCount{Int}

        # Mode-count mismatch: throw.
        @test_throws ArgumentError WeightedParticleCount([s1, s2], [1, -1])

        # Float weights: rejected with clear message.
        @test_throws ArgumentError WeightedParticleCount([s1], [0.5, -0.5])

        # Non-FermionSite: rejected.
        @test_throws ArgumentError WeightedParticleCount([bs], [1])
        @test_throws ArgumentError WeightedParticleCount([sp], [1])

        # Restriction construction with rational target.
        r_sz0 = wpc == 0
        @test r_sz0 isa Restriction
        @test r_sz0.bounds == 0:0

        r_sz_half = wpc == 1//2
        @test r_sz_half isa Restriction
        @test r_sz_half.bounds == (1//2):(1//2)

        # Float bound: rejected.
        @test_throws ArgumentError (wpc == 0.5)

        # Composition: disjoint sites.
        s3 = FermionSite{1}(:s3)
        wpc_a = WeightedParticleCount([s1], [1, -1])
        wpc_b = WeightedParticleCount([s3], [1])
        merged = wpc_a + wpc_b
        @test merged isa WeightedParticleCount{Int}
        @test length(merged.sites) == 2
        @test merged.weights == [1, -1, 1]

        # Composition: overlapping site names → throw.
        wpc_c = WeightedParticleCount([s1], [1, 1])
        @test_throws ArgumentError (wpc_a + wpc_c)

        # Cross-type composition (Int + Rational) → promotes.
        merged2 = wpc_int + WeightedParticleCount([s3], [1//2])
        @test merged2 isa WeightedParticleCount{Rational{Int}}
        @test merged2.weights == [1//1, -1//1, 1//2]
    end

    @testset "ParticleCount composition" begin
        nf1 = ParticleCount{Fermionic}([s1])
        nf2 = ParticleCount{Fermionic}([s2])
        merged = nf1 + nf2
        @test merged isa ParticleCount{Fermionic}
        @test length(merged.sites) == 2
    end

    @testset "Restriction construction" begin
        r1 = n_fermion(h) == 8
        @test r1 isa Restriction
        @test r1.bounds == 8:8

        r2 = n_fermion(h) ∈ 6:8
        @test r2 isa Restriction
        @test r2.bounds == 6:8

        r3 = n_fermion(h) >= 5
        @test r3 isa Restriction
        @test r3.bounds.start == 5

        r4 = n_fermion(h) <= 10
        @test r4 isa Restriction
        @test r4.bounds == 0:10

        # Symmetric ==
        r5 = 8 == n_fermion(h)
        @test r5.bounds == 8:8
    end

end
