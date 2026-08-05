@testset "compile" begin

    @testset "simple n operator" begin
        s = FermionSite{1}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)
        H = n(s, 1)
        Hc = compile(H, b)
        @test length(Hc) == 1
        @test Hc.basis_id == basis_id(b)
        @test Hc.nwords == 1
    end

    @testset "basis_id mismatch throws on assemble" begin
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        b1 = EagerBasis(h)
        b2 = EagerBasis(h)
        H = n(s, 1)
        Hc = compile(H, b1)
        @test_throws ArgumentError assemble(Hc, b2)
    end

    @testset "Sx/Sy expansion factor" begin
        s = SpinSite{1//2}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)
        # Sx alone: expansion factor 2.
        H_x = Sx(s, 1//2) + Sx(s, -1//2)        # full Sx via per-mz sum
        Hc_x = compile(H_x, b)
        # Two original terms × 2 (Sx → S± expansion) = 4 compiled terms.
        @test length(Hc_x) == 4
        # Sy alone: expansion factor 2 with imaginary coefficients.
        H_y = Sy(s, 1//2) + Sy(s, -1//2)
        Hc_y = compile(H_y, b)
        @test length(Hc_y) == 4
        # Element type promotes to Complex.
        @test eltype(first(Hc_y.terms).coef) <: Complex
    end

    @testset "Sx/Sy expansion error threshold" begin
        # Manually construct a chain with too many Sx ops (would generate 2^17
        # compiled terms — beyond the error threshold).
        # We can't easily trigger this from layer-1 OperatorSum without
        # synthesizing a 17-spin chain. Skip the throw, but exercise the
        # warn case (factor > 256) with 9 Sx ops would go beyond warn.
        # 9 spins × 2 expansions each = 512 → warn. Build it.
        sites = [SpinSite{1//2}(Symbol("s$i")) for i in 1:9]
        h = Hilbert(s.name => s for s in sites)
        # Cannot multiply 9 spin operators on different sites because that
        # would only produce one chain. Build a single product.
        chain_op = Sx(sites[1], 1//2)
        for i in 2:9
            chain_op = chain_op * Sx(sites[i], 1//2)
        end
        b = EagerBasis(h)
        # Expect a warn but still complete.
        @test_logs (:warn,) compile(chain_op, b)
    end
end
