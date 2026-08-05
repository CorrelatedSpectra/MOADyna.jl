@testset "sites" begin

    @testset "FermionSite" begin
        s = FermionSite{2}(:s1)
        @test local_dim(s) == 4
        @test statistics(s) === Fermionic()
        @test encoding_bits(s) == 2
        @test collect(mode_labels(s)) == [1, 2]
        @test collect(mode_labels_canonical(s)) == [(1,), (2,)]

        # normalize_label accepts both Int and 1-tuple
        @test normalize_label(s, 1) == (1,)
        @test normalize_label(s, (2,)) == (2,)

        # Out-of-range labels are rejected
        @test_throws ArgumentError normalize_label(s, 0)        # below 1
        @test_throws ArgumentError normalize_label(s, 3)        # above N=2
        @test_throws ArgumentError normalize_label(s, (3,))
        @test_throws ArgumentError c(s, 3)                      # error propagates through constructor
        @test_throws ArgumentError cdag(s, -1)

        # Equality by name
        @test s == FermionSite{2}(:s1)
        @test s != FermionSite{2}(:s2)
        @test s != FermionSite{3}(:s1)   # different N
        @test hash(s) == hash(FermionSite{2}(:s1))
    end

    @testset "BosonSite" begin
        b = BosonSite{5}(:phon)
        @test local_dim(b) == 6
        @test statistics(b) === Bosonic()
        @test encoding_bits(b) == 3   # ceil(log2(6)) = 3
        @test collect(mode_labels(b)) == [()]
        @test collect(mode_labels_canonical(b)) == [()]
        @test normalize_label(b) == ()
        @test normalize_label(b, ()) == ()
    end

    @testset "SpinSite" begin
        s12 = SpinSite{1//2}(:s)
        @test local_dim(s12) == 2
        @test statistics(s12) === Bosonic()
        @test encoding_bits(s12) == 1
        @test collect(mode_labels(s12)) == [-1//2, 1//2]

        s1 = SpinSite{1}(:t)
        @test local_dim(s1) == 3
        @test collect(mode_labels(s1)) == [-1, 0, 1]

        s32 = SpinSite{3//2}(:u)
        @test local_dim(s32) == 4
        @test collect(mode_labels(s32)) == [-3//2, -1//2, 1//2, 3//2]

        # normalize_label range/step checking
        @test normalize_label(s12, 1//2) == (1//2,)
        @test normalize_label(s12, -1//2) == (-1//2,)
        @test_throws ArgumentError normalize_label(s12, 1)        # outside -1/2..1/2
        @test_throws ArgumentError normalize_label(s12, 0)        # half-integer step required
    end

end
