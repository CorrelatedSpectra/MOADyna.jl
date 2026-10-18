@testset "Selected basis and external action" begin
    using LinearAlgebra
    B = MOADyna.Bases
    f = FermionSite{4}(:f)
    h = Hilbert(:f => f)
    allowed = n_fermion(h) == 1
    full = basis(h, allowed)
    seed = B._selected_basis(full, UInt64[2, 1, 2]; restrictions=allowed)
    @test seed.states == UInt64[1, 2]
    @test basis_id(seed) != basis_id(full)
    @test_throws ArgumentError B._selected_basis(full, UInt64[16]; restrictions=allowed)
    @test_throws ArgumentError B._selected_basis(full, UInt64[3]; restrictions=allowed)
    H = add_hc((0.2 + 0.1im) * cdag(f, 3) * c(f, 1) +
               0.3cdag(f, 3) * c(f, 2) + 0.4cdag(f, 4) * c(f, 2))
    A = matrixfree(H, seed)
    X = ComplexF64[1 0.2im; -0.4 1]
    reference = matrixfree(H, full) * reduce(hcat, [embed(X[:, j], seed => full) for j in 1:2])
    for threaded in (false, true)
        external = B._external_action(A, X; restrictions=allowed, max_candidates=2, threaded)
        @test external.states == UInt64[4, 8]
        @test external.values ≈ reference[3:4, :]
        Q = ComplexF64[1 im; im 1] / sqrt(2)
        rotated = B._external_action(A, X * Q; restrictions=allowed, max_candidates=2, threaded)
        @test [norm(row) for row in eachrow(rotated.values)] ≈ [norm(row) for row in eachrow(external.values)]
        @test_throws B._CandidateLimit B._external_action(A, X;
            restrictions=allowed, max_candidates=1, threaded)
    end
    # Distinct chains can cancel on the selected source state.
    cancel = cdag(f, 3) * c(f, 1) - cdag(f, 3) * c(f, 1) * n(f, 1)
    external = B._external_action(matrixfree(cancel, seed), X;
        restrictions=allowed, max_candidates=2, threaded=false)
    @test isempty(external.states)

    pad, bos = FermionSite{64}(:pad), BosonSite{2}(:bos)
    multi = basis(Hilbert(:pad => pad, :bos => bos), n_fermion([pad]) == 0)
    @test multi.nwords == 2
    @test_throws ArgumentError B._selected_basis(multi, UInt64[0, 3]; restrictions=nothing)
    @test_throws ArgumentError B._selected_basis(multi, UInt64[0, 4]; restrictions=nothing)
    small = B._selected_basis(multi, UInt64[0, 0]; restrictions=nothing)
    external = B._external_action(matrixfree(bdag(bos), small), ones(1, 1);
        restrictions=nothing, max_candidates=1)
    @test external.states == UInt64[0, 1]
    @test external.values == ones(1, 1)
end
