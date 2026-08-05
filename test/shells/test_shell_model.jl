using Test
using MOAD
using MOAD.Algebra: FermionSite
using MOAD.Shells: ShellModel, ell_of, range_of, site_of

@testset "ShellModel — single shell, d-shell" begin
    m = ShellModel([:Ni_3d])

    @test m.shells == [:Ni_3d]
    @test ell_of(m, :Ni_3d) == 2
    @test range_of(m, :Ni_3d) == 1:10
    @test site_of(m, :Ni_3d) isa FermionSite{10}

    # Underlying Hilbert has one site of size 10
    @test length(m.hilbert.sites) == 1
end

@testset "ShellModel — single shell, p-shell" begin
    m = ShellModel([:Ni_2p])
    @test ell_of(m, :Ni_2p) == 1
    @test range_of(m, :Ni_2p) == 1:6
    @test site_of(m, :Ni_2p) isa FermionSite{6}
end

@testset "ShellModel — single shell, f-shell" begin
    m = ShellModel([:Sm_4f])
    @test ell_of(m, :Sm_4f) == 3
    @test range_of(m, :Sm_4f) == 1:14
    @test site_of(m, :Sm_4f) isa FermionSite{14}
end

@testset "ShellModel — single shell, s-shell" begin
    m = ShellModel([:H_1s])
    @test ell_of(m, :H_1s) == 0
    @test range_of(m, :H_1s) == 1:2
    @test site_of(m, :H_1s) isa FermionSite{2}
end

@testset "ShellModel — show formats nicely" begin
    m = ShellModel([:Ni_3d])
    str = sprint(show, m)
    @test str == "ShellModel([:Ni_3d])"

    # MIME text/plain includes ℓ and mode-range detail
    str_plain = sprint(show, MIME"text/plain"(), m)
    @test occursin("ℓ=2", str_plain)
    @test occursin("10 modes", str_plain)
end

@testset "ShellModel — three-shell NiO layout" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])

    @test m.shells == [:Ni_2p, :Ni_3d, :L_3d]
    @test ell_of(m, :Ni_2p) == 1
    @test ell_of(m, :Ni_3d) == 2
    @test ell_of(m, :L_3d)  == 2
    @test range_of(m, :Ni_2p) == 1:6
    @test range_of(m, :Ni_3d) == 7:16
    @test range_of(m, :L_3d)  == 17:26
    # Three FermionSites in the underlying Hilbert
    @test length(m.hilbert.sites) == 3
end

@testset "ShellModel — concrete FermionSite{N} per shell in multi-shell" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    @test site_of(m, :Ni_2p) isa FermionSite{6}
    @test site_of(m, :Ni_3d) isa FermionSite{10}
    @test site_of(m, :L_3d)  isa FermionSite{10}
end

@testset "ShellModel — error: duplicate tags" begin
    @test_throws ArgumentError ShellModel([:Ni_3d, :Ni_3d])
    @test_throws ArgumentError ShellModel([:Ni_2p, :Ni_3d, :Ni_3d])
end

@testset "ShellModel — error: empty list" begin
    @test_throws ArgumentError ShellModel(Symbol[])
end

@testset "ShellModel — error: malformed tag (no orbital)" begin
    @test_throws ArgumentError ShellModel([:Ni])
    @test_throws ArgumentError ShellModel([:Ni_3d, :Cu])
end

@testset "ShellModel — error: unknown orbital character" begin
    @test_throws ArgumentError ShellModel([:Ni_3z])
    @test_throws ArgumentError ShellModel([:Ni_3X])
end

@testset "ShellModel — atom-label uniqueness independent" begin
    # :Ni1_3d and :Ni2_3d are different shells even though both are nominally Ni d
    m = ShellModel([:Ni1_3d, :Ni2_3d])
    @test m.shells == [:Ni1_3d, :Ni2_3d]
    @test ell_of(m, :Ni1_3d) == 2
    @test ell_of(m, :Ni2_3d) == 2
    @test range_of(m, :Ni1_3d) == 1:10
    @test range_of(m, :Ni2_3d) == 11:20
end
