@testset "Residual-controlled response selection" begin
    using LinearAlgebra
    B = MOADyna.Bases
    f = FermionSite{4}(:f)
    h = Hilbert(:f => f)
    allowed = n_fermion(h) == 1
    full = basis(h, allowed)
    seed = B._selected_basis(full, UInt64[1, 2]; restrictions=allowed)
    H = 0.5n(f, 2) + 1.4n(f, 3) + 2n(f, 4) +
        add_hc(0.3cdag(f, 3) * c(f, 1) + (0.2 + 0.1im) * cdag(f, 4) * c(f, 2) +
               0.07cdag(f, 4) * c(f, 3))
    source = ComplexF64[1 0.2im; 0.1im 0.5]
    shifts = [0.2 + 0.3im, 1.1 + 0.07im]
    Hfull = Matrix(matrixfree(H, full))
    source_full = B._embed_block(source, seed, full)
    for backend in (:matrixfree, :sparse)
        result = selected_response(H, seed, source; restrictions=allowed,
            shifts, max_states=4, error_tol=1e-9, backend)
        @test result.iterations > 1 && all(result.error_bounds .<= 1e-9)
        @test B._embed_block(result.sources, result.basis, full) == source_full
        @test all(norm((z * I - Hfull) \ source_full -
            B._embed_block(resolvent(result, z), result.basis, full)) <= bound + 1e-12
            for (z, bound) in zip(shifts, result.error_bounds))
        grid = to_grid(result, real.(shifts); Γ=0.6)
        @test all(grid.data[:, :, k] ≈ source_full' *
            (((real(shifts[k]) + 0.3im) * I - Hfull) \ source_full) for k in 1:2)
    end
    coarse = selected_response(H, seed, source; restrictions=allowed,
        shifts, max_states=2, error_tol=100.0)
    @test all(norm((z * I - Hfull) \ source_full -
        B._embed_block(resolvent(coarse, z), coarse.basis, full)) <= bound
        for (z, bound) in zip(shifts, coarse.error_bounds))
    U = ComplexF64[1 im; im 1] / sqrt(2)
    rotated = selected_response(H, seed, source * U; restrictions=allowed,
        shifts, max_states=2, error_tol=100.0)
    @test rotated.error_bounds ≈ coarse.error_bounds

    # A QR-discarded source must remain visible to the residual certificate.
    weak = ComplexF64[1 0; 0 1e-5]
    failure = try
        selected_response(n(f, 2), seed, weak; restrictions=allowed,
            shifts=[1 + 1e-6im], max_states=2, error_tol=1e-5, deflate_tol=1e-4)
    catch e
        e
    end
    @test failure isa SelectionConvergenceError && failure.reason == :inner_residual
    @test failure.result.error_bounds[1] ≈ 10.0

    # A capped recurrence can retain a trailing block outside its tridiagonal.
    failure = try
        selected_response(H, full, source_full[:, 1]; restrictions=allowed,
            shifts, max_states=4, error_tol=1e-10, krylovdim=1)
    catch e
        e
    end
    @test failure isa SelectionConvergenceError && failure.reason == :inner_residual
    @test size(resolvent(failure.result, shifts[1])) == (4, 1)
    @test_throws SelectionConvergenceError selected_response(H, seed, source;
        restrictions=allowed, shifts, max_states=4, max_candidates=1)

    # Transition support is discovered in the destination particle sector.
    initial = B._selected_basis(full, UInt64[1]; restrictions=allowed)
    target = n_fermion(h) == 2
    target_full = basis(h, target)
    transitions = [cdag(f, 3), cdag(f, 4)]
    result = selected_response(H, initial, transitions, [1.0]; restrictions=target,
        shifts, max_states=6, error_tol=1e-9)
    expected = hcat((matrixfree(T, target_full, initial) * [1.0] for T in transitions)...)
    @test B._embed_block(result.sources, result.basis, target_full) == expected
    cancel = cdag(f, 3) - cdag(f, 3) * n(f, 1)
    cancelled_support = selected_response(n(f, 2), initial, cdag(f, 2) + cancel, [1.0];
        restrictions=target, shifts, max_states=1, max_candidates=2)
    @test cancelled_support.basis.states == UInt64[3]
    dark = selected_response(H, initial, cancel, [1.0]; restrictions=target,
        shifts, max_states=6)
    @test length(dark.basis) == 0 && all(iszero, dark.error_bounds)
    @test size(resolvent(dark, shifts[1])) == (0, 1)
    @test all(iszero, to_grid(dark, [0.0, 1.0]; Γ=0.1).data)
end
