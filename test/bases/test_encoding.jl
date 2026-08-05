@testset "encoding" begin
    using MOAD.Bases: EncodingMap, mode_entry, get_span, set_span!,
                      get_bit, set_bit!, clear_bit!,
                      _state_less_flat

    @testset "fermion encoding" begin
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)
        enc = EncodingMap(h)
        @test enc.nwords == 1
        @test enc.total_bits == 4
        entries = [mode_entry(enc, :s, (i,)) for i in 1:4]
        @test all(e.kind === :fermion && e.nbits == 1 && e.max_val == 1 &&
                  e.word_idx == 1 for e in entries)
        @test [e.bit_offset for e in entries] == [0, 1, 2, 3]
    end

    @testset "boson encoding (multi-bit span)" begin
        b = BosonSite{3}(:b)            # ⌈log₂(4)⌉ = 2 bits
        h = Hilbert(:b => b)
        enc = EncodingMap(h)
        @test enc.total_bits == 2
        e = mode_entry(enc, :b, ())
        @test e.kind === :boson
        @test e.nbits == 2
        @test e.max_val == 3
    end

    @testset "spin encoding" begin
        sp = SpinSite{1}(:sp)            # 2S+1 = 3, ⌈log₂(3)⌉ = 2
        h = Hilbert(:sp => sp)
        enc = EncodingMap(h)
        @test enc.total_bits == 2
        e = mode_entry(enc, :sp, ())
        @test e.kind === :spin
        @test e.nbits == 2
        @test e.max_val == 2          # 2S
    end

    @testset "bit get/set roundtrip" begin
        buf = zeros(UInt64, 1)
        set_span!(buf, 1, 1, 3, 4, 11)     # write 11 (= 0b1011) at bit 3, width 4
        @test get_span(buf, 1, 1, 3, 4) == 11
        # Adjacent bits must remain zero.
        @test get_span(buf, 1, 1, 0, 3) == 0
        @test get_span(buf, 1, 1, 7, 4) == 0
    end

    @testset "_state_less_flat lex order" begin
        nwords = 2
        # Two states; second is lexicographically larger in the high word.
        buf = UInt64[0x0000_0000_0000_0001, 0x0000_0000_0000_0000,   # state 1: word0=1, word1=0
                     0x0000_0000_0000_0000, 0x0000_0000_0000_0001]   # state 2: word0=0, word1=1
        # Lex: compare word0 first; 1 < 0? No, 1 > 0. So state 1 > state 2.
        @test !_state_less_flat(buf, nwords, 1, 2)
        @test _state_less_flat(buf, nwords, 2, 1)
        @test !_state_less_flat(buf, nwords, 1, 1)
    end

    @testset "padding when site span would straddle a word" begin
        # Two FermionSite{40} sites: each is 40 bits, total 80 bits, but the
        # second straddles word 1, so it must move to word 2.
        s1 = FermionSite{40}(:s1)
        s2 = FermionSite{40}(:s2)
        h = Hilbert(:s1 => s1, :s2 => s2)
        enc = EncodingMap(h)
        @test enc.nwords == 2
        e_s2 = mode_entry(enc, :s2, (1,))
        @test e_s2.word_idx == 2
        @test e_s2.bit_offset == 0
    end
end
