@testset "Matrix-free spectra" begin
    using LinearAlgebra, SparseArrays

    site = SpinSite{1}(:spin)
    b = basis(Hilbert(:spin => site))
    sx, sy, sz = S(site)
    H = 0.7sz + 0.2sx + 0.1sy + 0.4sz * sz
    A = matrixfree(H, b)
    M = assemble(compile(H, b), b)
    E = eigen(Hermitian(Matrix(M)))
    ψ, Eg = E.vectors[:, 1], E.values[1]
    transitions = [sx + 0.3sz, sy + 0.2sx]
    actions = [matrixfree(t, b) for t in transitions]
    other_b = basis(Hilbert(:spin => site))
    mixed = [first(transitions), matrixfree(last(transitions), b, other_b)]
    grid = range(-0.5, 2.5; length=9)
    tuning = (; krylovdim=5, reorth=:full, tol=1e-12, threaded=false)
    @test_throws ArgumentError xas(M, b, mixed, ψ; Eg, Γ=0.3, ω_grid=grid, tuning...)
    @test MOADyna.Bases._transition_action(M, b, M; threaded=true) === M
    @test MOADyna.Responses._supports_scalar_lanczos_action(A)
    for T in (first(transitions), transitions)
        exact = xas(M, b, T, ψ; Eg, Γ=0.3, ω_grid=grid, tuning...)
        result = xas(A, b, T, ψ; Eg, Γ=0.3, ω_grid=grid, tuning...)
        @test result.tensor ≈ exact.tensor atol=1e-10
    end
    exact = xas(M, b, transitions, ψ; Eg, Γ=0.3, ω_grid=grid, tuning...)
    result = xas(A, b, actions, ψ; Eg, Γ=0.3, ω_grid=grid, tuning...)
    @test result.tensor ≈ exact.tensor atol=1e-10
    response = correlator(A, b, transitions, transitions; state=ψ, Eg,
                          Γ=0.3, ω=grid, form=:grid, tuning...)
    @test response.data ≈ exact.tensor atol=1e-10
    thermal = correlator(A, b, transitions, transitions; T=0.3,
                         Γ=0.3, ω=grid, form=:grid, tuning...)
    thermal_reference = correlator(M, b, transitions, transitions; T=0.3,
                                   Γ=0.3, ω=grid, form=:grid, tuning...)
    @test thermal.data ≈ thermal_reference.data atol=1e-10
    settings = (; Eg, Γ_intermediate=0.4, Γ_final=0.3,
                 ω_in_grid=range(0.2, 1.1; length=2), ω_out_grid=grid, tuning...)
    exact_rixs = rixs(M, M, b, transitions, transitions, ψ; settings...)
    result_rixs = rixs(A, A, b, actions, actions, ψ; settings...)
    @test result_rixs.tensor ≈ exact_rixs.tensor atol=1e-9
    fy_settings = (; Eg, Γ_intermediate=0.4, ω_in_grid=settings.ω_in_grid, tuning...)
    exact_fy = fluorescence_yield(M, b, transitions, transitions, ψ; fy_settings...)
    result_fy = fluorescence_yield(A, b, actions, actions, ψ; fy_settings...)
    @test result_fy.tensor ≈ exact_fy.tensor atol=1e-9

    g, i, f = FermionSite{1}(:g), FermionSite{2}(:i), FermionSite{3}(:f)
    h = Hilbert(:g => g, :i => i, :f => f)
    all_states = basis(h, n_fermion(h) == 1)
    bg = basis(h, n_fermion(h) == 1, n_fermion([g]) == 1)
    bi = basis(h, n_fermion(h) == 1, n_fermion([i]) == 1)
    bf = basis(h, n_fermion(h) == 1, n_fermion([f]) == 1)
    Hi = 4n(i, 1) + 5.2n(i, 2) +
         (0.2 + 0.1im) * cdag(i, 1) * c(i, 2) +
         (0.2 - 0.1im) * cdag(i, 2) * c(i, 1)
    Hf = 1.1n(f, 1) + 2n(f, 2) + 2.8n(f, 3)
    Tin = [cdag(i, 1) * c(g, 1), (0.4 + 0.2im) * cdag(i, 2) * c(g, 1)]
    Tout = [cdag(i, 1) * c(f, 1) + 0.3cdag(i, 2) * c(f, 2),
            cdag(i, 2) * c(f, 3) + 0.2im * cdag(i, 1) * c(f, 2)]
    Ai, Af = matrixfree(Hi, bi), matrixfree(Hf, bf)
    incoming = [matrixfree(t, bi, bg) for t in Tin]
    outgoing = [matrixfree(t, bi, bf) for t in Tout]
    Mi, Mf = assemble(compile(Hi, all_states), all_states),
             assemble(compile(Hf, all_states), all_states)
    ψg = ComplexF64[1]
    ψfull = embed(ψg, bg => all_states)
    settings = (; settings..., Eg=0.0, ω_in_grid=range(3.9, 5.1; length=2),
                 restrictions_intermediate=n_fermion([i]) == 1,
                 restrictions_final=n_fermion([f]) == 1)
    reference = rixs(Mf, Mi, all_states, Tin, Tout, ψfull; settings...)
    for (F, I, Ti, To) in ((Af, Ai, incoming, outgoing),
                           (sparse(Af), sparse(Ai), sparse.(incoming), sparse.(outgoing)))
        result = rixs(F, I, bf, Ti, To, ψg; intermediate_basis=bi, settings...)
        @test result.tensor ≈ reference.tensor atol=1e-10
    end
    xr = xas(Mi, all_states, Tin, ψfull; Eg=0, Γ=0.4, ω_grid=grid, tuning...)
    xm = xas(Ai, bi, incoming, ψg; Eg=0, Γ=0.4, ω_grid=grid, tuning...)
    @test xm.tensor ≈ xr.tensor atol=1e-10
    fy_settings = (; fy_settings..., Eg=0.0, ω_in_grid=settings.ω_in_grid)
    fr = fluorescence_yield(Mi, all_states, Tin, Tout, ψfull; fy_settings...)
    fm = fluorescence_yield(Ai, bi, incoming, outgoing, ψg; fy_settings...)
    @test fm.tensor ≈ fr.tensor atol=1e-10
    different_bi = basis(h, n_fermion(h) == 1, n_fermion([i]) == 1)
    @test_throws ArgumentError xas(Ai, different_bi, incoming, ψg;
                                   Eg=0, Γ=0.4, ω_grid=grid, tuning...)
    @test_throws ArgumentError rixs(Af, Ai, bf, incoming, outgoing, ψg;
                                   intermediate_basis=different_bi, settings...)
end
