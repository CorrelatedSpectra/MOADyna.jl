@testset "Residual-controlled eigenstate selection" begin
    using LinearAlgebra
    f = FermionSite{2}(:f)
    h = Hilbert(:f => f)
    restriction = n_fermion(h) == 1
    full = basis(h, restriction)
    seed = MOADyna.Bases._selected_basis(full, UInt64[1]; restrictions=restriction)
    H = n(f, 2) + 0.01add_hc(cdag(f, 2) * c(f, 1))
    exact = eigen(H, full)
    two_states = selected_eigen(H, seed; restrictions=restriction, max_states=2,
        n=2, residual_tol=0.1)
    @test length(two_states.eigen.values) == 2
    for backend in (:matrixfree, :sparse)
        result = selected_eigen(H, seed; restrictions=restriction, max_states=2,
            residual_tol=1e-10, backend)
        @test length(result.basis) == 2
        @test result.eigen.values ≈ exact.values atol=1e-12
        @test result.residual_norm <= 1e-10
        ψ = embed(result.eigen.vectors[:, 1], result.basis => full)
        @test norm(matrixfree(H, full) * ψ - result.eigen.values[1] * ψ) <= 1e-10
    end
    failure = try
        selected_eigen(H, seed; restrictions=restriction, max_states=1, residual_tol=1e-10)
    catch e
        e
    end
    @test failure isa SelectionConvergenceError
    @test failure.reason == :max_states
    @test failure.result.residual_norm ≈ 0.01
    @test_throws ArgumentError selected_eigen(H, seed;
        restrictions=n_fermion(h) == 0, max_states=2)

    # A complete degenerate target block must survive selection.
    s1, s2 = SpinSite{1//2}(:s1), SpinSite{1//2}(:s2)
    spin_basis = basis(Hilbert(:s1 => s1, :s2 => s2))
    x1, y1, z1 = S(s1)
    x2, y2, z2 = S(s2)
    ferro = -(x1 * x2 + y1 * y2 + z1 * z2)
    result = selected_eigen(ferro, spin_basis; restrictions=nothing,
        max_states=4, residual_tol=1e-10)
    @test length(result.eigen.values) == 3
    @test result.eigen.values ≈ fill(-0.25, 3)
    @test result.eigen.vectors' * result.eigen.vectors ≈ I atol=1e-12

    # Virtual hopping resolves a flat seed before its full manifold is needed.
    flat_site = FermionSite{8}(:flat)
    flat_hilbert = Hilbert(:flat => flat_site)
    flat_allowed = n_fermion(flat_hilbert) == 1
    flat_full = basis(flat_hilbert, flat_allowed)
    flat_seed = MOADyna.Bases._selected_basis(flat_full, UInt64[1, 2, 4, 8]; restrictions=flat_allowed)
    virtual = sum(4n(flat_site, i + 4) + 0.1i * add_hc(cdag(flat_site, i + 4) * c(flat_site, i))
                  for i in 1:4)
    failure = try
        selected_eigen(virtual, flat_seed; restrictions=flat_allowed,
            max_states=8, max_candidates=1, residual_tol=1e-9)
    catch e
        e
    end
    @test failure isa SelectionConvergenceError && length(failure.result.eigen.values) == 1
    resolved = selected_eigen(virtual, flat_seed; restrictions=flat_allowed,
        max_states=8, residual_tol=1e-9)
    @test resolved.eigen.values[1] ≈ (4 - sqrt(16 + 4 * 0.4^2)) / 2 atol=1e-11

    sites = [FermionSite{2}(Symbol("s$i")) for i in 1:4]
    h = Hilbert(s.name => s for s in sites)
    allowed = [n_fermion(h) == 4, WeightedParticleCount(sites, repeat([1, -1], 4)) == 0]
    full = basis(h, allowed...)
    seed = MOADyna.Bases._selected_basis(full, full.states[1:1]; restrictions=allowed)
    H = -sum(add_hc(cdag(sites[i], s) * c(sites[i + 1], s)) for i in 1:3, s in 1:2) +
        4sum(n(s, 1) * n(s, 2) for s in sites)
    exact = eigen(H, full)
    result = selected_eigen(H, seed; restrictions=allowed, max_states=length(full),
        residual_tol=1e-9, threaded=true)
    @test result.iterations > 1
    @test result.eigen.values ≈ exact.values atol=1e-10
    ψ = embed(result.eigen.vectors[:, 1], result.basis => full)
    @test norm(matrixfree(H, full) * ψ - result.eigen.values[1] * ψ) <= 1e-9
    failure = try
        selected_eigen(H, seed; restrictions=allowed, max_states=length(full), max_candidates=1)
    catch e
        e
    end
    @test failure isa SelectionConvergenceError
    @test failure.reason == :max_candidates
    @test isinf(failure.result.residual_norm)
    @test failure.result.candidate_count === nothing
end
