@testset "builder" begin

    s = FermionSite{4}(:s)
    h = Hilbert(:s => s)
    # mode_map: Quanty 0-indexed → MOAD 1-indexed on the single site
    map_fn = i -> (:s, i + 1)

    @testset "single hopping term" begin
        text = """
        Operator of Length 2
        C  0 A  2 |  1.50000000000000E+00
        """
        op = build_operator(parse_quanty_operator(text), h, map_fn)
        expected = 1.5 * cdag(s, 1) * c(s, 3)
        @test op == expected
    end

    @testset "Coulomb (length-4) reproduces n*n" begin
        text = """
        Operator of Length 4
        C  1 C  0 A  1 A  0 | -4.00000000000000E+00
        """
        op = build_operator(parse_quanty_operator(text), h, map_fn)
        # In Quanty's normal form: cdag(1) cdag(0) c(1) c(0) | -4
        # That is algebraically equal to U n_0 n_1 with U=4.
        expected = 4.0 * n(s, 1) * n(s, 2)
        @test op == expected
    end

    @testset "multiple terms accumulate via canonicalization" begin
        text = """
        Operator of Length 2
        C  0 A  2 |  1.50000000000000E+00
        C  2 A  0 |  1.50000000000000E+00
        """
        op = build_operator(parse_quanty_operator(text), h, map_fn)
        # 1.5 * cdag(1) c(3)  +  1.5 * cdag(3) c(1)  (h.c. pair)
        expected = 1.5 * cdag(s, 1) * c(s, 3) + 1.5 * cdag(s, 3) * c(s, 1)
        @test op == expected
    end

    @testset "Dict mode_map works too" begin
        text = "C  0 A  1 |  2.0\n"
        # Force a different convention via Dict
        d = Dict(0 => (:s, 4), 1 => (:s, 2))
        op = build_operator(parse_quanty_operator("Operator of Length 2\n" * text), h, d)
        expected = 2.0 * cdag(s, 4) * c(s, 2)
        @test op == expected
    end

    @testset "empty parsed → zero operator" begin
        op = build_operator(parse_quanty_operator(""), h, map_fn)
        @test isempty(op)
    end

    @testset "missing site name throws" begin
        text = "Operator of Length 2\nC  0 A  1 | 1.0\n"
        bad = i -> (:nope, i + 1)
        @test_throws ArgumentError build_operator(parse_quanty_operator(text), h, bad)
    end

end
