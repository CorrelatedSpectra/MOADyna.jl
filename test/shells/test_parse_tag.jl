# test/shells/test_parse_tag.jl
using Test
using MOADyna.Shells: _parse_shell_tag

@testset "shell-tag parser" begin
    @testset "standard atomic shell tags" begin
        @test _parse_shell_tag(:Ni_3d)   == (atom = "Ni",   n = 3, orbital = :d, ell = 2)
        @test _parse_shell_tag(:O_2p)    == (atom = "O",    n = 2, orbital = :p, ell = 1)
        @test _parse_shell_tag(:Sm_4f)   == (atom = "Sm",   n = 4, orbital = :f, ell = 3)
        @test _parse_shell_tag(:H_1s)    == (atom = "H",    n = 1, orbital = :s, ell = 0)
    end

    @testset "ligand and indexed atom labels" begin
        @test _parse_shell_tag(:L_3d).atom    == "L"
        @test _parse_shell_tag(:Ni1_3d).atom  == "Ni1"
        @test _parse_shell_tag(:Ni2_3d).atom  == "Ni2"
        @test _parse_shell_tag(:Cu_planar_3d).atom == "Cu_planar"
    end

    @testset "atom-label-free form" begin
        @test _parse_shell_tag(:_3d) == (atom = "", n = 3, orbital = :d, ell = 2)
        @test _parse_shell_tag(:_p)  == (atom = "", n = 0, orbital = :p, ell = 1)  # n=0 ≡ unspecified
    end

    @testset "rejection — no orbital character" begin
        @test_throws ArgumentError _parse_shell_tag(:Ni)
        @test_throws ArgumentError _parse_shell_tag(:foo_bar)
        @test_throws ArgumentError _parse_shell_tag(:_)
    end

    @testset "rejection — unknown orbital character" begin
        @test_throws ArgumentError _parse_shell_tag(:Ni_3z)   # z is not s/p/d/f/g/h
        @test_throws ArgumentError _parse_shell_tag(:Ni_3X)
    end

    @testset "case sensitivity — orbital must be lowercase" begin
        @test_throws ArgumentError _parse_shell_tag(:Ni_3D)
    end
end

using MOADyna.Shells: _shell_mode_labels, _orbital_mode_pair, _m_values

@testset "m-major mode layout" begin
    @testset "m_values for ℓ" begin
        @test _m_values(0) == [0]
        @test _m_values(1) == [-1, 0, 1]
        @test _m_values(2) == [-2, -1, 0, 1, 2]
        @test _m_values(3) == [-3, -2, -1, 0, 1, 2, 3]
    end

    @testset "_orbital_mode_pair returns (dn_idx, up_idx) within shell" begin
        # ℓ=2 (d-shell), 10 modes: dn,up,dn,up,... with m=-2..+2
        @test _orbital_mode_pair(2, -2) == (1, 2)
        @test _orbital_mode_pair(2, -1) == (3, 4)
        @test _orbital_mode_pair(2,  0) == (5, 6)
        @test _orbital_mode_pair(2,  1) == (7, 8)
        @test _orbital_mode_pair(2,  2) == (9, 10)
        # ℓ=1 (p-shell), 6 modes
        @test _orbital_mode_pair(1, -1) == (1, 2)
        @test _orbital_mode_pair(1,  0) == (3, 4)
        @test _orbital_mode_pair(1,  1) == (5, 6)
    end

    @testset "_orbital_mode_pair rejects m out of range" begin
        @test_throws ArgumentError _orbital_mode_pair(2, 3)
        @test_throws ArgumentError _orbital_mode_pair(2, -3)
        @test_throws ArgumentError _orbital_mode_pair(0, 1)
    end

    @testset "_shell_mode_labels enumerates shell in m-major dn-then-up order" begin
        labels = _shell_mode_labels(2)  # d-shell
        @test length(labels) == 10
        @test labels[1] == (m = -2, sigma = :dn)
        @test labels[2] == (m = -2, sigma = :up)
        @test labels[3] == (m = -1, sigma = :dn)
        @test labels[10] == (m = 2, sigma = :up)
    end

    @testset "_shell_mode_labels handles s and p" begin
        s_labels = _shell_mode_labels(0)
        @test length(s_labels) == 2
        @test s_labels[1] == (m = 0, sigma = :dn)
        @test s_labels[2] == (m = 0, sigma = :up)

        p_labels = _shell_mode_labels(1)
        @test length(p_labels) == 6
        @test p_labels[1] == (m = -1, sigma = :dn)
        @test p_labels[6] == (m = 1, sigma = :up)
    end

    @testset "negative ell rejected by all three helpers" begin
        @test_throws ArgumentError _m_values(-1)
        @test_throws ArgumentError _orbital_mode_pair(-1, 0)
        @test_throws ArgumentError _shell_mode_labels(-1)
    end
end
