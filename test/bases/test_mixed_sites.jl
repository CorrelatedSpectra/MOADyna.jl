using LinearAlgebra: norm, I

@testset "mixed-site Jordan-Wigner parity" begin
    f1 = FermionSite{1}(:f1)
    f2 = FermionSite{1}(:f2)
    phonon = BosonSite{3}(:p)
    large_phonon = BosonSite{31}(:p)
    spin = SpinSite{1}(:s)
    padding63 = FermionSite{63}(:padding)
    padding64 = FermionSite{64}(:padding)

    function mixed_site_errors(sites, independent, restrictions...)
        h = Hilbert(site.name => site for site in sites)
        bb = EagerBasis(h, restrictions...)
        A(op) = Matrix(assemble(compile(op, bb), bb))
        fermion_ops = (c(f2, 1), cdag(f2, 1))
        independent_ops = independent isa BosonSite ?
            (b(independent), bdag(independent)) : S(independent)[1:2]

        commutator_error = 0.0
        product_error = 0.0
        for F in fermion_ops, B in independent_ops
            FM, BM = A(F), A(B)
            commutator_error = max(commutator_error, norm(FM * BM - BM * FM))
            product_error = max(product_error,
                                norm(A(F * B) - FM * BM),
                                norm(A(B * F) - BM * FM))
        end

        # The preceding f1 mode is absent from compile(c(f2), bb), but its
        # occupied states must still contribute to f2's fermionic sign.
        C1, C2 = A(c(f1, 1)), A(c(f2, 1))
        car_error = max(norm(C1 * C2' + C2' * C1),
                        norm(C2 * C2' + C2' * C2 - I))
        return (; commutator_error, product_error, car_error)
    end

    errors = [
        mixed_site_errors((f1, f2, phonon), phonon),
        mixed_site_errors((f1, phonon, f2), phonon),
        mixed_site_errors((phonon, f1, f2), phonon),
        # 128 states exercise multiple assembly chunks with several threads.
        mixed_site_errors((f1, large_phonon, f2), large_phonon),
        mixed_site_errors((f1, spin, f2), spin),
        # Restrict only the inactive spectator site. f1 remains free in an
        # earlier word; f2 lands at bit offset 63 or 0 in a later word.
        mixed_site_errors((f1, phonon, padding63, f2), phonon,
                          n_fermion([padding63]) == 0),
        mixed_site_errors((f1, phonon, padding64, f2), phonon,
                          n_fermion([padding64]) == 0),
    ]

    @test maximum(e.commutator_error for e in errors) < 1e-12
    @test maximum(e.product_error for e in errors) < 1e-12
    @test maximum(e.car_error for e in errors) < 1e-12
end
