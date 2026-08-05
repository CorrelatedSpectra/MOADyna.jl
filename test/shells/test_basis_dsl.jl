using Test
using MOAD
using MOAD.Algebra: ParticleCount, Fermionic, Restriction
using MOAD.Bases: EagerBasis

@testset "basis-restriction DSL — nshells single-shell" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    pc = nshells(m, :Ni_2p)
    @test pc isa ParticleCount{Fermionic}
    r = pc == 6
    @test r isa Restriction
end

@testset "basis-restriction DSL — nshells multi-shell vararg" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    pc = nshells(m, :Ni_3d, :L_3d)
    @test pc isa ParticleCount{Fermionic}
end

@testset "basis-restriction DSL — nshells(m) zero-arg rejected" begin
    # Regression: nshells(m) with no shells previously built an empty
    # ParticleCount and silently left the basis unrestricted. Must throw.
    m = ShellModel([:Ni_3d])
    @test_throws ArgumentError nshells(m)
end

@testset "basis-restriction DSL — nshells via + composition" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    pc_combined = nshells(m, :Ni_3d) + nshells(m, :L_3d)
    @test pc_combined isa ParticleCount{Fermionic}
    r = pc_combined == 16
    @test r isa Restriction
end

@testset "basis-restriction DSL — total(m) sugar" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    pc = total(m)
    @test pc isa ParticleCount{Fermionic}
    r = pc == 24
    @test r isa Restriction
end

@testset "basis-restriction DSL — EagerBasis(m, ...) overload, single shell" begin
    m = ShellModel([:Ni_3d])
    basis = EagerBasis(m, nshells(m, :Ni_3d) == 8)
    # 5 orbitals × 2 spins = 10 modes; choose 8 → C(10, 8) = 45
    @test length(basis) == binomial(10, 8)
end

@testset "basis-restriction DSL — EagerBasis(m, ...) overload, multi-shell" begin
    m = ShellModel([:Ni_3d, :L_3d])
    basis = EagerBasis(m, nshells(m, :Ni_3d) == 8, nshells(m, :L_3d) == 10)
    # d⁸ × L¹⁰ = C(10,8) × C(10,10) = 45 × 1 = 45
    @test length(basis) == 45
end

@testset "basis-restriction DSL — EagerBasis(m, ...) with combined sum restriction" begin
    m = ShellModel([:Ni_3d, :L_3d])
    basis = EagerBasis(m, nshells(m, :Ni_3d) + nshells(m, :L_3d) == 18)
    # d⁸L¹⁰ + d⁹L⁹ + d¹⁰L⁸ → C(10,8)×C(10,10) + C(10,9)×C(10,9) + C(10,10)×C(10,8)
    # = 45 + 100 + 45 = 190
    @test length(basis) == 190
end

@testset "basis-restriction DSL — EagerBasis(m, ...) with `in` range" begin
    m = ShellModel([:Ni_2p])
    basis = EagerBasis(m, nshells(m, :Ni_2p) in 5:6)
    # C(6,5) + C(6,6) = 6 + 1 = 7
    @test length(basis) == 7
end

@testset "basis-restriction DSL — total(m) with `==` matches sum of nshells" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    basis_total = EagerBasis(m, total(m) == 24)
    basis_sum   = EagerBasis(m,
        nshells(m, :Ni_2p) + nshells(m, :Ni_3d) + nshells(m, :L_3d) == 24)
    @test length(basis_total) == length(basis_sum)
end

@testset "basis-restriction DSL — EagerBasis(m, ...) with NO restrictions" begin
    m = ShellModel([:Ni_3d])
    basis = EagerBasis(m)   # no restrictions; full Hilbert
    @test length(basis) == 2^10
end
