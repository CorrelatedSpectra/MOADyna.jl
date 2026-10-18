@testset "Restricted basis sampling" begin
    using Random: MersenneTwister

    f = FermionSite{4}(:f)
    b, s = BosonSite{2}(:b), SpinSite{1//2}(:s)
    h = Hilbert(:f => f, :b => b, :s => s)
    allowed = (n_fermion([f]) == 2,
               WeightedParticleCount([f], [1, -1, 2, -2]) ∈ -1:1,
               n_boson([b]) ∈ 1:2, Sz_total([s]) == 1//2)
    full = basis(h, allowed...)
    sampled, dimension = MOADyna.Bases._sample_basis(h, allowed, 100; rng=MersenneTwister(4))
    @test dimension == length(full)
    @test sampled.states == full.states
    small, _ = MOADyna.Bases._sample_basis(h, allowed, 3; rng=MersenneTwister(4))
    repeat, _ = MOADyna.Bases._sample_basis(h, allowed, 3; rng=MersenneTwister(4))
    @test small.states == repeat.states
    @test length(small) == 3
    @test all(state -> state in full.states, small.states)

    sites = [FermionSite{1}(Symbol("f$i")) for i in 1:70]
    large = Hilbert(s.name => s for s in sites)
    seed, total = MOADyna.Bases._sample_basis(large, n_fermion(large) == 35, 4;
                                              rng=MersenneTwister(1))
    @test total == binomial(big(70), 35) > typemax(Int)
    @test seed.nwords == 2 && length(seed) == 4
    @test all(i -> sum(count_ones, get_state(seed, i)) == 35, 1:4)
    empty, total = MOADyna.Bases._sample_basis(h, n_fermion([f]) == 5, 3;
                                               rng=MersenneTwister(1))
    @test total == 0 && length(empty) == 0
    @test_throws ArgumentError MOADyna.Bases._sample_basis(h, allowed, 3;
        rng=MersenneTwister(1), max_work=1)
end
