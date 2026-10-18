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

    @testset "Fermion chains across packed words" begin
        f, padding, g = FermionSite{2}(:f), FermionSite{64}(:padding), FermionSite{2}(:g)
        phonon = BosonSite{2}(:phonon)
        h = Hilbert(:f => f, :phonon => phonon, :padding => padding, :g => g)
        bb = basis(h, n_fermion(h) == 0)
        B = MOADyna.Bases
        bit_position(e) = 64(e.word_idx - 1) + e.bit_offset
        fermion_bits = sort([bit_position(e) for e in values(bb.encoding.entries) if e.kind === :fermion])
        occupied(state, bit) = !iszero(state[bit ÷ 64 + 1] & (UInt64(1) << (bit % 64)))
        function primitive_action(state, entries)
            output, phase = copy(state), 1
            for entry in reverse(entries)
                bit = bit_position(entry)
                occupied(output, bit) == (entry.kind === B.FERM_C) || return output, 0
                isodd(count(p -> p < bit && occupied(output, p), fermion_bits)) && (phase = -phase)
                output[entry.word_idx] ⊻= UInt64(1) << entry.bit_offset
            end
            return output, phase
        end
        ladders = map(op -> only(op).chain[1], (c(f, 1), cdag(f, 1), c(g, 2), cdag(g, 2)))
        active = map(op -> B._lower_entry(only(op).chain[1], bb.encoding),
                     (c(f, 1), c(f, 2), c(padding, 64), c(g, 2)))
        phonon_mode = B.mode_entry(bb.encoding, :phonon, ())
        errors = zeros(Int, 3)
        for chain_length in 1:3, chain in Iterators.product(ntuple(_ -> ladders, chain_length)...)
            raw = OperatorSum{ComplexF64}()
            raw.terms[chain] = OperatorTerm(1.0 + 2im, chain)
            compiled = compile(raw, bb)
            for plan in (compiled, B._adjoint_plan(compiled, basis_id(bb)))
                term = only(plan.terms)
                for occupations in 0:15, phonons in (0, 2)
                    state = zeros(UInt64, bb.nwords)
                    for (k, mode) in enumerate(active)
                        isodd(occupations >> (k - 1)) && B.set_bit!(state, 1, mode.word_idx, mode.bit_offset)
                    end
                    B.set_span!(state, 1, phonon_mode, phonons)
                    expected, sign = primitive_action(state, plan.entries)
                    output = copy(state)
                    phase = B._apply_chain_rtl!(output, plan, term)
                    errors[1] += phase != sign
                    errors[2] += !iszero(sign) && output != expected
                    errors[3] += B._term_precheck_fails(plan, term, state, 1) != iszero(sign)
                end
            end
        end
        @test errors == [0, 0, 0]
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
