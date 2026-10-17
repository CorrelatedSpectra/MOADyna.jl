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
        # Nine distinct spins give 512 expansions: warn, but still compile.
        sites = [SpinSite{1//2}(Symbol("s$i")) for i in 1:9]
        h = Hilbert(s.name => s for s in sites)
        # Build one chain whose nine Sx entries each contribute two choices.
        chain_op = Sx(sites[1], 1//2)
        for i in 2:9
            chain_op = chain_op * Sx(sites[i], 1//2)
        end
        b = EagerBasis(h)
        # Expect a warn but still complete.
        @test_logs (:warn,) compile(chain_op, b)
    end

    @testset "Sx/Sy limit precedes Cartesian expansion" begin
        s = SpinSite{1//2}(:s)
        b = EagerBasis(Hilbert(:s => s))
        spin_ops = (Sx(s, 1//2), Sy(s, 1//2))
        # Repeated actions on one spin keep the Hilbert space tiny. The
        # expansion itself must reject 2^17 terms before constructing them.
        errors = map(spin_ops) do op
            chain = ntuple(_ -> only(op).chain[1], 17)
            try
                MOADyna.Algebra._expand_spin_xy(chain)
                nothing
            catch err
                err
            end
        end
        @test all(err -> err isa ArgumentError, errors)

        # Exercise counts that cannot fit their expansion factor in Int,
        # only after early rejection is established; a regression must not
        # attempt to allocate 2^64 entries in the test process.
        if all(err -> err isa ArgumentError, errors)
            long_chain_errors = map(Iterators.product(spin_ops, (64, 128))) do (op, count)
                # Build the fixture directly rather than specializing
                # symbolic multiplication at every intermediate length.
                chain = ntuple(_ -> only(op).chain[1], count)
                long_op = zero(op)
                long_op.terms[chain] = OperatorTerm(1.0, chain)
                try
                    compile(long_op, b)
                    nothing
                catch err
                    err
                end
            end
            @test all(err -> err isa ArgumentError, long_chain_errors)
        end

        # The existing limits are inclusive: 256 is quiet, 65536 warns.
        @test_logs MOADyna.Algebra._check_spin_expansion(256)
        @test_logs (:warn,) MOADyna.Algebra._check_spin_expansion(65_536)
    end
end
