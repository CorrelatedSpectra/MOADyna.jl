@testset "parser" begin

    @testset "length-0 constant block" begin
        # Quanty emits length-0 blocks for constant offsets (e.g. the energy
        # zero of a number-operator combination). Format:
        #   Operator of Length   0
        #   QComplex      =          0
        #   N             =          1
        #   |  <coef>
        dump = """
        Operator: Test
        QComplex         =          0
        MaxLength        =          0
        NFermionic modes =          2
        NBosonic modes   =          0

        Operator of Length   0
        QComplex      =          0
        N             =          1
        |  3.14000000000000E+00
        """
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        path = tempname()
        write(path, dump)
        H = read_quanty_operator(path, h, i -> (:s, i + 1))
        # Single identity term with coef 3.14
        terms = collect(H)
        @test length(terms) == 1
        @test terms[1].coefficient ≈ 3.14
        @test terms[1].chain == ()       # empty chain = identity
    end

    # A minimal real-coefficient Quanty dump
    sample = """
    Operator: CrAn
    QComplex         =          0 (Real==0 or Complex==1 or Mixed==2)
    MaxLength        =          2 (largest number of product of lader operators)
    NFermionic modes =          4 (Number of fermionic modes ...)
    NBosonic modes   =          0 (Number of bosonic modes ...)

    Operator of Length   2
    QComplex      =          0 (Real==0 or Complex==1)
    N             =          4 (number of operators of length   2)
    C  0 A  2 |  1.50000000000000E+00
    C  1 A  3 |  1.50000000000000E+00
    C  2 A  0 |  1.50000000000000E+00
    C  3 A  1 |  1.50000000000000E+00

    Operator of Length   4
    QComplex      =          0 (Real==0 or Complex==1)
    N             =          1 (number of operators of length   4)
    C  1 C  0 A  1 A  0 | -4.00000000000000E+00
    """

    parsed = parse_quanty_operator(sample)

    @testset "header bookkeeping" begin
        @test parsed.nfermion == 4
        @test parsed.nboson == 0
        # Quanty only emits one MaxLength header at the top of the dump
        # (its value reflects the operator's longest chain). Per-block
        # "Operator of Length N" headers are walked but not aggregated
        # into max_length.
        @test parsed.max_length == 2
    end

    @testset "term count" begin
        @test length(parsed.terms) == 5
    end

    @testset "length-2 terms" begin
        first = parsed.terms[1]
        @test first.kinds == [:cdag, :c]
        @test first.indices == [0, 2]
        @test first.coefficient ≈ 1.5
    end

    @testset "length-4 term" begin
        last = parsed.terms[end]
        @test last.kinds == [:cdag, :cdag, :c, :c]
        @test last.indices == [1, 0, 1, 0]
        @test last.coefficient ≈ -4.0
    end

    @testset "empty input" begin
        empty_parsed = parse_quanty_operator("")
        @test isempty(empty_parsed.terms)
        @test empty_parsed.nfermion == 0
    end

    @testset "negative and scientific coefficients" begin
        text = """
        Operator of Length 2
        C  4 A  4 |  1.70000000000000E+00
        C  6 A  6 |  2.64000000000000E+00
        C  0 A  0 | -4.50000000000000E+00
        C  2 A  8 | -9.90000000000000E-01
        """
        p = parse_quanty_operator(text)
        @test length(p.terms) == 4
        @test p.terms[3].coefficient ≈ -4.5
        @test p.terms[4].coefficient ≈ -0.99
    end

    # ---- Fail-loud tests: silent drops would corrupt the operator sum ----

    @testset "complex blocks (QComplex=1) round-trip into ComplexF64" begin
        text = """
        Operator of Length 2
        QComplex      =          1
        N             =          1
        C  0 A  1 |  1.0  2.0
        """
        p = parse_quanty_operator(text)
        @test length(p.terms) == 1
        @test p.terms[1].coefficient == complex(1.0, 2.0)
    end

    @testset "Mixed (QComplex=2) blocks raise (not yet supported)" begin
        text = """
        Operator of Length 2
        QComplex      =          2
        N             =          1
        C  0 A  1 |  1.0  2.0
        """
        @test_throws ArgumentError parse_quanty_operator(text)
    end

    @testset "malformed chain line raises (no silent skip)" begin
        # Coefficient missing → cannot parse Float64
        text = """
        Operator of Length 2
        C  0 A  1 |
        """
        @test_throws ArgumentError parse_quanty_operator(text)

        # Two-token coef field (looks like complex but no QComplex header)
        text2 = """
        Operator of Length 2
        C  0 A  1 |  1.0  2.0
        """
        @test_throws ArgumentError parse_quanty_operator(text2)

        # Garbled chain tokens
        text3 = """
        Operator of Length 2
        X  0 Y  1 | 1.0
        """
        @test_throws ArgumentError parse_quanty_operator(text3)
    end

    @testset "per-block N count mismatch raises" begin
        text = """
        Operator of Length 2
        QComplex      =          0
        N             =          3
        C  0 A  1 |  1.0
        C  1 A  2 |  1.0
        """
        # Declares 3 terms but only 2 chain lines
        @test_throws ArgumentError parse_quanty_operator(text)

        text2 = """
        Operator of Length 2
        QComplex      =          0
        N             =          1
        C  0 A  1 |  1.0
        C  1 A  2 |  1.0
        """
        # Declares 1 but has 2
        @test_throws ArgumentError parse_quanty_operator(text2)
    end

end
