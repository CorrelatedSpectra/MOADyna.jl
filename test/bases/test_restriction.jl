@testset "CompiledRestriction" begin
    using MOAD.Bases: EncodingMap, compile_restriction, _check, POPCOUNT, BYTESUM, WEIGHTED_SUM

    @testset "POPCOUNT (n_fermion == 2 on FermionSite{4})" begin
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)
        enc = EncodingMap(h)
        c = compile_restriction(n_fermion(h) == 2, enc)
        @test c.op === POPCOUNT
        @test c.scale == 1
        @test c.min == 2
        @test c.max == 2
        # State 0b0011 has popcount 2, passes.
        buf = UInt64[0x03]
        @test _check(c, buf, 1)
        # State 0b0001 has popcount 1, fails.
        buf2 = UInt64[0x01]
        @test !_check(c, buf2, 1)
    end

    @testset "BYTESUM (boson n_b == 1)" begin
        b_site = BosonSite{3}(:b)
        h = Hilbert(:b => b_site)
        enc = EncodingMap(h)
        c = compile_restriction(n_boson(h) == 1, enc)
        @test c.op === BYTESUM
        # encoded value = 1 → satisfies.
        buf = UInt64[0x01]
        @test _check(c, buf, 1)
        buf3 = UInt64[0x03]
        @test !_check(c, buf3, 1)
    end

    @testset "WEIGHTED_SUM (fermionic Sz)" begin
        # 2-mode site with weights [+1//2, -1//2]; target Sz = 0 means
        # n_up = n_down. After scaling (LCM of 2): scale=2, weights=[+1,-1],
        # bounds = [0, 0]·scale = [0, 0].
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        enc = EncodingMap(h)
        wpc = WeightedParticleCount([s], [1//2, -1//2])
        c = compile_restriction(wpc == 0, enc)
        @test c.op === WEIGHTED_SUM
        @test c.scale == 2
        @test c.weights == [1, -1]
        @test c.min == 0 && c.max == 0

        # Sz=0 states: (0,0) → 0, (1,1) → 1-1=0; (1,0) → 1, (0,1) → -1
        @test _check(c, UInt64[0b00], 1)
        @test _check(c, UInt64[0b11], 1)
        @test !_check(c, UInt64[0b01], 1)
        @test !_check(c, UInt64[0b10], 1)
    end

    @testset "apply_restriction! — public projection API" begin
        # Building blocks reused across cases.
        s_up = FermionSite{4}(:up)
        s_dn = FermionSite{4}(:dn)
        h    = Hilbert(:up => s_up, :dn => s_dn)

        @testset "(a) Restriction = nothing → no-op" begin
            b = EagerBasis(h)            # full Fock, 2^8 = 256 states
            v = collect(1.0:Float64(length(b)))
            v_ref = copy(v)
            r = apply_restriction!(v, nothing, b)
            @test r === v
            @test v == v_ref
        end

        @testset "(b) single ParticleCount restriction" begin
            # Full Fock basis; project to N_up = 2.
            b = EagerBasis(h)
            v = ones(Float64, length(b))
            apply_restriction!(v, n_fermion([s_up]) == 2, b)
            # The set of indices i with v[i] != 0 must equal exactly the
            # indices whose state encoding has popcount(up_modes) == 2.
            kept = findall(!iszero, v)
            n_up_per_state = [count_ones(get_state(b, i)[1] & UInt64(0x0F))
                              for i in 1:length(b)]
            expected_kept = findall(==(2), n_up_per_state)
            @test kept == expected_kept
            # And the kept count matches a parallel basis built with the
            # restriction baked in.
            b_restr = EagerBasis(h, n_fermion([s_up]) == 2)
            @test length(kept) == length(b_restr)
        end

        @testset "(c) coupled multi-restriction" begin
            # Full basis; project to (N_up=2, N_dn=1) → 6 × 4 = 24 states.
            b = EagerBasis(h)
            v = ones(ComplexF64, length(b))
            apply_restriction!(v,
                               [n_fermion([s_up]) == 2, n_fermion([s_dn]) == 1],
                               b)
            kept = findall(!iszero, v)
            b_restr = EagerBasis(h, n_fermion([s_up]) == 2, n_fermion([s_dn]) == 1)
            @test length(kept) == length(b_restr)
            # Every kept index must encode the correct popcounts.
            words = [get_state(b, i)[1] for i in kept]
            @test all(count_ones(w & UInt64(0x0F)) == 2 for w in words)  # up modes
            @test all(count_ones(w & UInt64(0xF0)) == 1 for w in words)  # dn modes
        end

        @testset "(d) empty restriction vector → no-op" begin
            b = EagerBasis(h)
            v = collect(1.0:Float64(length(b)))
            v_ref = copy(v)
            apply_restriction!(v, Restriction[], b)
            @test v == v_ref
        end

        @testset "(e) length mismatch throws DimensionMismatch" begin
            b = EagerBasis(h)
            @test_throws DimensionMismatch apply_restriction!(
                zeros(length(b) + 1), n_fermion([s_up]) == 2, b)
            @test_throws DimensionMismatch apply_restriction!(
                zeros(length(b) - 1),
                [n_fermion([s_up]) == 2, n_fermion([s_dn]) == 1], b)
        end

        @testset "(f) projector idempotence and AND ordering" begin
            # Applying the same restriction twice yields the same result.
            b = EagerBasis(h)
            v1 = randn(Float64, length(b))
            v2 = copy(v1)
            apply_restriction!(v1, n_fermion([s_up]) == 2, b)
            apply_restriction!(v2, n_fermion([s_up]) == 2, b)
            apply_restriction!(v2, n_fermion([s_up]) == 2, b)
            @test v1 == v2
            # And A∧B applied as a vector matches A then B applied sequentially.
            vA = randn(Float64, length(b))
            vB = copy(vA)
            apply_restriction!(vA,
                               [n_fermion([s_up]) == 2, n_fermion([s_dn]) == 1], b)
            apply_restriction!(vB, n_fermion([s_up]) == 2, b)
            apply_restriction!(vB, n_fermion([s_dn]) == 1, b)
            @test vA == vB
        end

        @testset "(g) AbstractVector compatibility (view)" begin
            b = EagerBasis(h)
            buf = ones(Float64, length(b) + 4)
            v = @view buf[3:2 + length(b)]
            apply_restriction!(v, n_fermion([s_up]) == 2, b)
            # Padding outside the view is untouched.
            @test buf[1:2]      == [1.0, 1.0]
            @test buf[end-1:end] == [1.0, 1.0]
            # Inside the view, the right number of entries are zeroed.
            b_restr = EagerBasis(h, n_fermion([s_up]) == 2)
            @test count(!iszero, v) == length(b_restr)
        end
    end

    @testset "TotalSz on SpinSites (BYTESUM, scale=2)" begin
        # Two SpinSite{1//2}: encoded ∈ {0,1} per site (mz ∈ {-1/2, +1/2}).
        # TotalSz = (encoded1 - 1/2) + (encoded2 - 1/2). target = 0 means
        # encoded1 + encoded2 == 1.
        s1 = SpinSite{1//2}(:s1)
        s2 = SpinSite{1//2}(:s2)
        h = Hilbert(:s1 => s1, :s2 => s2)
        enc = EncodingMap(h)
        c = compile_restriction(Sz_total(h) == 0, enc)
        @test c.op === BYTESUM
        @test c.scale == 2
        # Sz=0 states: |↑↓⟩ encoded(0,1) = bits ?, |↓↑⟩ encoded(1,0)
        # Encoding: site1 at bit 0, site2 at bit 1.
        # encoded1=1, encoded2=0 → byte = 0b01
        # encoded1=0, encoded2=1 → byte = 0b10
        @test _check(c, UInt64[0b01], 1)
        @test _check(c, UInt64[0b10], 1)
        # Sz=±1 states fail.
        @test !_check(c, UInt64[0b00], 1)   # ↓↓
        @test !_check(c, UInt64[0b11], 1)   # ↑↑
    end
end
