using Test
using MOADyna
using MOADyna.Shells: ShellModel, onsite_energies

@testset "onsite_energies" begin

    @testset "NiO 2-shell GS solve" begin
        # ZSA anchors: d⁸L¹⁰ at 0, d⁹L⁹ at Δ. Udd = 7.3 eV, Δ = 4.7 eV.
        # Hand derivation:
        #   8·ed + 10·eL = 0   − 8·7/2·7.3 = −204.4
        #   9·ed +  9·eL = 4.7 − 9·8/2·7.3 = −258.1
        # ⇒ ed = −41.18888…, eL = 12.51111…
        m = ShellModel([:Ni_3d, :L_3d])
        Udd   = 7.3   # eV
        Delta = 4.7   # eV
        es = onsite_energies(m;
            U = (Ni_3d = Udd,),
            anchors = [(Ni_3d = 8, L_3d = 10) => 0.0,
                       (Ni_3d = 9, L_3d = 9)  => Delta])
        @test es.Ni_3d ≈ -41.18888888888889 atol = 1e-12
        @test es.L_3d  ≈  12.51111111111111 atol = 1e-12
    end

    @testset "single-shell trivial" begin
        m  = ShellModel([:Ni_3d])
        es = onsite_energies(m; anchors = [(Ni_3d = 1,) => 5.0])
        @test es.Ni_3d ≈ 5.0 atol = 1e-12
    end

    @testset "NiO 3-anchor XAS solve with Upd cross-pair" begin
        # ZSA-style 3-shell XAS anchors: ground state (d⁸L¹⁰), CT (d⁹L⁹),
        # and core-hole (p⁵d⁹L¹⁰). Hand derivation:
        #   6·ep + 8·ed + 10·eL = 0   − 8·7/2·7.3 − 6·8·8.5  = −612.4
        #   6·ep + 9·ed +  9·eL = 4.7 − 9·8/2·7.3 − 6·9·8.5  = −717.1
        #   5·ep + 9·ed + 10·eL = 0   − 9·8/2·7.3 − 5·9·8.5  = −645.3
        # ⇒ ep = −44.46666…, ed = −77.36666…, eL = 27.33333…
        m   = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
        Udd = 7.3   # eV
        Upd = 8.5   # eV
        Δ   = 4.7   # eV
        es  = onsite_energies(m;
            U     = (Ni_3d = Udd,),
            pairs = ((:Ni_2p, :Ni_3d) => Upd,),
            anchors = [
                (Ni_2p = 6, Ni_3d = 8, L_3d = 10) => 0.0,
                (Ni_2p = 6, Ni_3d = 9, L_3d = 9)  => Δ,
                (Ni_2p = 5, Ni_3d = 9, L_3d = 10) => 0.0,
            ])
        @test es.Ni_2p ≈ -44.46666666666667 atol = 1e-12
        @test es.Ni_3d ≈ -77.36666666666666 atol = 1e-12
        @test es.L_3d  ≈  27.33333333333333 atol = 1e-12
    end

    @testset "empty anchors → ArgumentError" begin
        m = ShellModel([:Ni_3d])
        @test_throws ArgumentError onsite_energies(m;
            anchors = Pair{NamedTuple, Float64}[])
    end

    @testset "under-determined → ArgumentError" begin
        # 1 anchor, 2 shells → under-determined.
        m = ShellModel([:Ni_3d, :L_3d])
        @test_throws ArgumentError onsite_energies(m;
            anchors = [(Ni_3d = 8, L_3d = 10) => 0.0])
    end

    @testset "anchor mentions shell not in m → ArgumentError" begin
        m = ShellModel([:Ni_3d, :L_3d])
        err = try
            onsite_energies(m; anchors = [
                (Ni_3d = 8, Nonexistent = 10) => 0.0,
                (Ni_3d = 9, L_3d = 9)         => 4.7,
            ])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Nonexistent", err.msg) || occursin("not in", err.msg)
    end

    @testset "shells kwarg restricts the unknowns" begin
        # Solve only for Ni_3d and L_3d, ignoring Ni_2p in the model.
        m  = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
        es = onsite_energies(m;
            shells  = (:Ni_3d, :L_3d),
            U       = (Ni_3d = 7.3,),
            anchors = [(Ni_3d = 8, L_3d = 10) => 0.0,
                       (Ni_3d = 9, L_3d = 9)  => 4.7])
        @test es.Ni_3d ≈ -41.18888888888889 atol = 1e-12
        @test es.L_3d  ≈  12.51111111111111 atol = 1e-12
        @test !haskey(es, :Ni_2p)
    end

end
