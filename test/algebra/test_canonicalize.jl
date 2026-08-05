@testset "canonicalization (Tier 2)" begin

    s = FermionSite{2}(:s)
    t = FermionSite{2}(:t)

    @testset "Pauli zeros" begin
        # c c on same fermion mode is zero
        op = c(s, 1) * c(s, 1)
        @test isempty(op)

        # c' c' on same fermion mode is zero
        op = cdag(s, 1) * cdag(s, 1)
        @test isempty(op)

        # c(s,1) c(s,2) c(s,1) — non-adjacent same mode → still zero after reordering
        op = c(s, 1) * c(s, 2) * c(s, 1)
        @test isempty(op)
    end

    @testset "fermionic anticommutation" begin
        # c(s,1) c(s,2) should reorder to -c(s,2) c(s,1) — but canonical order
        # is by (site_name, label), so c(s,1) c(s,2) IS already canonical (1<2).
        op1 = c(s, 1) * c(s, 2)
        op2 = -1 * c(s, 2) * c(s, 1)
        @test op1 == op2
        @test length(op1) == 1

        # Two different sites: c(s,1) c(t,1)
        op = c(s, 1) * c(t, 1)
        # Canonical: site name :s < :t, so c(s,1) c(t,1) is canonical.
        @test length(op) == 1
        # Reverse order should give -1 * canonical
        op_rev = c(t, 1) * c(s, 1)
        @test op_rev == -op
    end

    @testset "anticommutator constants" begin
        # c(s,1) cdag(s,1) = δ - cdag(s,1) c(s,1) = 1 - n(s,1)
        op = c(s, 1) * cdag(s, 1)
        expected = one(OperatorSum{Float64}) - cdag(s, 1) * c(s, 1)
        @test op == expected

        # c(s,1) cdag(s,2): different modes, no δ, just -cdag(s,2) c(s,1)
        op = c(s, 1) * cdag(s, 2)
        expected = -1 * cdag(s, 2) * c(s, 1)
        @test op == expected
    end

    @testset "normal ordering: cdag-left, c-right" begin
        # c(s,1) cdag(s,2) → canonical form
        op = c(s, 1) * cdag(s, 2)
        # Result: -1 * cdag(s,2) * c(s,1)
        terms = collect(op)
        @test length(terms) == 1
        chain = terms[1].chain
        @test chain[1].kind === :cdag
        @test chain[2].kind === :c
        @test terms[1].coefficient ≈ -1.0
    end

    @testset "merging equivalent terms" begin
        # Two different ways of writing the same thing should merge in +
        # c(s,1) cdag(s,2) and -cdag(s,2) c(s,1) are algebraically equal
        a = c(s, 1) * cdag(s, 2)
        b = -1 * cdag(s, 2) * c(s, 1)
        s_op = a + b
        # 2 × the same term
        @test length(s_op) == 1
        @test collect(s_op)[1].coefficient ≈ -2.0
    end

    @testset "bosonic commutation" begin
        bp = BosonSite{5}(:p)
        bq = BosonSite{5}(:q)

        # b(p) bdag(p) = δ + bdag(p) b(p) = 1 + n_b(p)
        op = b(bp) * bdag(bp)
        expected = one(OperatorSum{Float64}) + bdag(bp) * b(bp)
        @test op == expected

        # b(p) bdag(q) = bdag(q) b(p) — commute, no sign, no δ
        op = b(bp) * bdag(bq)
        expected = bdag(bq) * b(bp)
        @test op == expected
    end

    @testset "cross-statistics commutation" begin
        # Fermion and boson commute
        f = FermionSite{2}(:f)
        bs = BosonSite{5}(:b)

        op = b(bs) * c(f, 1)
        # Canonical: fermion before boson (statistics group order: 0 < 1)
        # so b * c → c * b with no sign change
        expected = c(f, 1) * b(bs)
        @test op == expected

        # Same for spin × fermion
        sp = SpinSite{1//2}(:sp)
        op = Sz(sp, 1//2) * c(f, 1)
        expected = c(f, 1) * Sz(sp, 1//2)
        @test op == expected
    end

    @testset "spin operators (Tier 2 punt on same-site)" begin
        sp1 = SpinSite{1//2}(:sp1)
        sp2 = SpinSite{1//2}(:sp2)

        # Different spin sites commute
        op = Sx(sp2, 1//2) * Sx(sp1, 1//2)
        expected = Sx(sp1, 1//2) * Sx(sp2, 1//2)
        @test op == expected

        # Same site: leave in user order
        op1 = Splus(sp1, 1//2) * Sminus(sp1, 1//2)
        op2 = Sminus(sp1, 1//2) * Splus(sp1, 1//2)
        @test op1 != op2  # not auto-simplified
    end

end
