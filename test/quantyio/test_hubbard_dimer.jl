# =====================================================================
# Hubbard-dimer integration test for the QuantyIO bridge
# =====================================================================
#
# A 2-site, 2-orbital Hubbard dimer (4 fermion modes) is a standard
# many-body test case: small enough to inline a checked-in Quanty operator
# dump, large enough to exercise the bridge end-to-end (operator parse +
# build + algebraic equivalence + eigenvalue check via MOADyna's eigen).
#
# This replaced the project-specific z_only round-trip in the official
# test suite. The z_only validation script is preserved in
# `docs/dev/validation/` for the original research model.

@testset "QuantyIO: 2-site Hubbard dimer round-trip" begin

    dump_path = joinpath(@__DIR__, "data", "hubbard_dimer.txt")
    @assert isfile(dump_path) "missing test fixture $dump_path"

    # Parameters used to generate the dump (see /tmp/hubbard_dimer.Quanty)
    t = 1.0
    U = 4.0

    # ----- Bridge: read Quanty's operator dump into MOADyna -----
    s = FermionSite{4}(:s)            # 4 modes: site0↑, site0↓, site1↑, site1↓
    h = Hilbert(:s => s)
    map_fn = i -> (:s, i + 1)         # Quanty 0-indexed → MOADyna 1-indexed
    H_bridge = read_quanty_operator(dump_path, h, map_fn)

    # ----- Algebra: same H built natively -----
    # Mode mapping: 0=site0↑, 1=site0↓, 2=site1↑, 3=site1↓
    H_hop = -t * (cdag(s, 1) * c(s, 3) + cdag(s, 2) * c(s, 4) +
                  cdag(s, 3) * c(s, 1) + cdag(s, 4) * c(s, 2))
    H_U   = U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))
    H_native = H_hop + H_U

    # Term count matches.
    @test length(H_bridge) == length(H_native)

    # Coefficients agree per-chain (within a small float tolerance — both
    # sides go through Tier-2 canonicalization, so chain ordering is the
    # same; we still allow ε for accumulation differences).
    for term in H_native
        other = get(H_bridge.terms, term.chain, nothing)
        @test other !== nothing
        if other !== nothing
            @test other.coefficient ≈ term.coefficient atol = 1e-12
        end
    end

    # ----- End-to-end ED: 2-electron Sz=0 sector via Quanty bridge -----
    # Sz weights: +1 on up modes (1, 3), −1 on down modes (2, 4).
    sz_weights = [1, -1, 1, -1]
    basis = EagerBasis(h,
                       n_fermion(h) == 2,
                       WeightedParticleCount([s], sz_weights) == 0)
    @test length(basis) == 4   # |↑↓,⋅⟩, |⋅,↑↓⟩, |↑,↓⟩, |↓,↑⟩

    E_bridge = eigen(H_bridge, basis; n = 4)
    E_native = eigen(H_native, basis; n = 4)
    @test E_bridge.values ≈ E_native.values atol = 1e-12

    # Analytical GS for the 2-site Hubbard dimer at half filling, Sz=0:
    #   E_0 = (U - sqrt(U^2 + 16 t^2)) / 2
    E0_analytic = (U - sqrt(U^2 + 16t^2)) / 2
    @test E_bridge.values[1] ≈ E0_analytic atol = 1e-10
end
