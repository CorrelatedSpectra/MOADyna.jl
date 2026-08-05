@testset "EagerBasis" begin

    @testset "fermion full Fock" begin
        s = FermionSite{3}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)
        @test length(b) == 8
        # All states must be distinct.
        states = Set([Tuple(get_state(b, i)) for i in 1:length(b)])
        @test length(states) == 8
        # And sorted lexicographically.
        @test issorted(get_state(b, i)[1] for i in 1:length(b))
    end

    @testset "fermion with particle-count restriction" begin
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)
        @test all(length(EagerBasis(h, n_fermion(h) == k)) == binomial(4, k)
                  for k in 0:4)
    end

    @testset "fermion with range restriction" begin
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h, n_fermion(h) ∈ 1:3)
        @test length(b) == binomial(4,1) + binomial(4,2) + binomial(4,3)
    end

    @testset "two fermion sites with disjoint particle restrictions" begin
        # T3-style: independent up-channel and down-channel particle counts.
        s_up = FermionSite{4}(:up)
        s_dn = FermionSite{4}(:dn)
        h = Hilbert(:up => s_up, :dn => s_dn)
        b = EagerBasis(h, n_fermion([s_up]) == 2, n_fermion([s_dn]) == 1)
        @test length(b) == binomial(4,2) * binomial(4,1)
    end

    @testset "boson full" begin
        b_site = BosonSite{3}(:b)
        h = Hilbert(:b => b_site)
        bb = EagerBasis(h)
        @test length(bb) == 4
    end

    @testset "spin full" begin
        sp = SpinSite{1}(:sp)
        h = Hilbert(:sp => sp)
        b = EagerBasis(h)
        @test length(b) == 3
    end

    @testset "WeightedParticleCount Sz=0" begin
        # 4-mode site with weights [1,-1,1,-1]: Σ wᵢnᵢ = 0 sector
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)
        wpc = WeightedParticleCount([s], [1, -1, 1, -1])
        b = EagerBasis(h, wpc == 0)
        # Hand-computed: 6 states (00,11,0011,1001,0110,1111 in mode order)
        @test length(b) == 6
    end

    @testset "WeightedParticleCount Sz=1//2 (rational target)" begin
        # 2-mode site, weights [+1//2, -1//2]: target Sz=+1//2 means up-down=1, so
        # bit pattern (1, 0) only.
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        wpc = WeightedParticleCount([s], [1//2, -1//2])
        b = EagerBasis(h, wpc == 1//2)
        @test length(b) == 1
        @test get_state(b, 1)[1] == 0x01
    end

    @testset "get_index roundtrip" begin
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)
        @test all(get_index(b, collect(get_state(b, i))) == i for i in 1:length(b))
        # Absent state.
        @test get_index(b, [UInt64(0)]) == 1   # 0 occupation present
        # Construct an absent query via wrong-length state — DimensionMismatch.
        @test_throws DimensionMismatch get_index(b, [UInt64(0), UInt64(0)])
    end

    @testset "basis_id is unique" begin
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        b1 = EagerBasis(h)
        b2 = EagerBasis(h)
        @test basis_id(b1) != basis_id(b2)
    end

    @testset "overlapping restrictions on same modes are unioned" begin
        # Hubbard-style fixed-N + Sz-resolved sector on a single FermionSite{4}:
        # n_fermion == 2 AND Σ wᵢ nᵢ == 0 with weights [+1, -1, +1, -1].
        # The two restrictions touch the same modes; the planner unions them
        # into one partition and filters candidates against both.
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)
        w = WeightedParticleCount([s], [1, -1, 1, -1])
        b = EagerBasis(h, n_fermion(h) == 2, w == 0)
        # Hand-counted Sz=0, n=2 states (modes 1=up,2=dn,3=up,4=dn):
        # (1,1,0,0), (0,1,1,0), (1,0,0,1), (0,0,1,1).
        @test length(b) == 4
        bits_per_state = [digits(Int(get_state(b, i)[1]), base=2, pad=4)
                          for i in 1:length(b)]
        @test all(sum(bits) == 2 for bits in bits_per_state)
        @test all(bits[1] - bits[2] + bits[3] - bits[4] == 0
                  for bits in bits_per_state)
    end

    @testset "popcount-with-filter is order-independent for multi-POPCOUNT" begin
        # Two FermionSite{2} sites, two POPCOUNT restrictions:
        #   (a) total n_fermion == 2 (covers all 4 modes)
        #   (b) n_fermion(s1) == 1   (covers only the 2 modes of s1)
        # The merged partition has both. The walker must pick the COVERING
        # POPCOUNT (a) as the enumeration driver, not whichever appears first
        # in `p.restrictions`. Subset POPCOUNT (b) is just another filter.
        s1 = FermionSite{2}(:s1)
        s2 = FermionSite{2}(:s2)
        h = Hilbert(:s1 => s1, :s2 => s2)
        # Expected count: s1 has 1 of 2 fermions, s2 has 1 of 2 → C(2,1)·C(2,1)=4.
        a = EagerBasis(h, n_fermion(h) == 2, n_fermion([s1]) == 1)
        b = EagerBasis(h, n_fermion([s1]) == 1, n_fermion(h) == 2)
        @test length(a) == 4
        @test length(b) == 4
        # Same set of states regardless of restriction order.
        states_a = sort([Tuple(get_state(a, i)) for i in 1:length(a)])
        states_b = sort([Tuple(get_state(b, i)) for i in 1:length(b)])
        @test states_a == states_b
    end

    @testset "popcount-with-filter fast path (Hubbard sectors)" begin
        # POPCOUNT covers all partition modes plus a WEIGHTED_SUM filter.
        # 8 modes interpreted as 4 up/down pairs, n=4 ∧ Sz=0 → C(4,2)·C(4,2).
        s = FermionSite{8}(:s)
        h = Hilbert(:s => s)
        w = WeightedParticleCount([s], [1, -1, 1, -1, 1, -1, 1, -1])

        b = EagerBasis(h, n_fermion(h) == 4, w == 0)
        @test length(b) == binomial(4, 2)^2          # = 36

        # n=2 ∧ Sz=0 → C(4,1)·C(4,1) = 16
        b2 = EagerBasis(h, n_fermion(h) == 2, w == 0)
        @test length(b2) == binomial(4, 1)^2

        # n=6 ∧ Sz=0 → C(4,3)·C(4,3) = 16
        b6 = EagerBasis(h, n_fermion(h) == 6, w == 0)
        @test length(b6) == binomial(4, 3)^2

        # Range-form POPCOUNT: n ∈ 2:6 ∧ Sz=0
        br = EagerBasis(h, n_fermion(h) ∈ 2:6, w == 0)
        @test length(br) == sum(binomial(4, k)^2 for k in 1:3)   # n_up=1..3 valid

        # 12-mode scale: n=6 ∧ Sz=0 → C(6,3)·C(6,3) = 400
        s12 = FermionSite{12}(:s)
        h12 = Hilbert(:s => s12)
        w12 = WeightedParticleCount([s12], [(-1)^(i+1) for i in 1:12])
        b12 = EagerBasis(h12, n_fermion(h12) == 6, w12 == 0)
        @test length(b12) == binomial(6, 3)^2
    end

    @testset "suffix-pruned weighted-sum walk" begin
        # Pure-weighted partition: dispatcher routes to _walk_weighted_pruned!.
        # 8-mode FermionSite, weights [+1,-1,+1,-1,+1,-1,+1,-1], target 0
        # — Sz=0 over all n_fermion. Σ_{k=0..4} C(4,k)^2 = 70.
        s = FermionSite{8}(:s)
        h = Hilbert(:s => s)
        w = WeightedParticleCount([s], [1, -1, 1, -1, 1, -1, 1, -1])
        b = EagerBasis(h, w == 0)
        @test length(b) == sum(binomial(4, k)^2 for k in 0:4)

        # TotalSz on multiple SpinSite{1//2}: also routes to weighted-pruned
        # (BYTESUM kernel, no POPCOUNT). 6 spins, Sz=0 → C(6, 3) = 20.
        sites = [SpinSite{1//2}(Symbol("s$i")) for i in 1:6]
        hsp = Hilbert(s.name => s for s in sites)
        b_sp = EagerBasis(hsp, Sz_total(hsp) == 0)
        @test length(b_sp) == binomial(6, 3)

        # SpinSite{1} chain Sz=2: encoded values 0/1/2; sum-check correctness.
        sp1_sites = [SpinSite{1}(Symbol("p$i")) for i in 1:3]
        hsp1 = Hilbert(s.name => s for s in sp1_sites)
        b_sp1 = EagerBasis(hsp1, Sz_total(hsp1) == 2)
        # mz_total = 2 across 3 sites with mz_i ∈ {-1,0,1}: count compositions.
        ok = 0
        for a in -1:1, b in -1:1, c in -1:1
            (a + b + c == 2) && (ok += 1)
        end
        @test length(b_sp1) == ok
    end

    @testset "popcount enumerator handles 64-mode boundary" begin
        # Previous Gosper-trick implementation computed `last = UInt64(1) << n`
        # which wraps to 0 at n == 64, silently emitting an empty basis.
        # Verify each k-sector at the boundary produces the correct count.
        s = FermionSite{64}(:s)
        h = Hilbert(:s => s)
        @test length(EagerBasis(h, n_fermion(h) == 0))  == 1
        @test length(EagerBasis(h, n_fermion(h) == 1))  == 64
        @test length(EagerBasis(h, n_fermion(h) == 2))  == binomial(64, 2)
        @test length(EagerBasis(h, n_fermion(h) == 63)) == 64
        @test length(EagerBasis(h, n_fermion(h) == 64)) == 1
    end

    @testset "basis() factory function" begin
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)

        # Default: eager
        b_default = basis(h, n_fermion(h) == 2)
        @test b_default isa EagerBasis
        @test length(b_default) == 6   # C(4,2)

        # Equivalence with direct EagerBasis
        b_explicit = EagerBasis(h, n_fermion(h) == 2)
        @test length(b_default) == length(b_explicit)
        @test get_state(b_default, 1) == get_state(b_explicit, 1)

        # ShellModel overload
        m = ShellModel([:Ni_3d])
        b_shell = basis(m, nshells(m, :Ni_3d) == 5)
        @test b_shell isa EagerBasis
        @test length(b_shell) == binomial(10, 5)

        # lazy=true currently rejects
        @test_throws ArgumentError basis(h, n_fermion(h) == 2; lazy = true)
    end
end
