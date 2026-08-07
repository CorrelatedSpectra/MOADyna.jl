using Test
using MOADyna
using MOADyna.Algebra: OperatorSum
using MOADyna.Shells: ShellModel

@testset "shell-keyed n — single shell" begin
    m = ShellModel([:Ni_3d])
    op = n(m, :Ni_3d)
    @test op isa OperatorSum
    # 10 modes for a d-shell → 10 terms in the total-n operator
    @test length(op.terms) == 10
    # Hermitian (n is self-adjoint)
    @test op' == op
end

@testset "shell-keyed n — multi-shell, all" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    op_total = n(m)
    op_p = n(m, :Ni_2p)
    op_d = n(m, :Ni_3d)
    op_L = n(m, :L_3d)
    @test op_total == op_p + op_d + op_L
end

@testset "shell-keyed n — vararg subset" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    op_pd = n(m, :Ni_2p, :Ni_3d)
    @test op_pd == n(m, :Ni_2p) + n(m, :Ni_3d)
end

@testset "shell-keyed n — single orbital by m value" begin
    m = ShellModel([:Ni_3d])
    op = n(m, :Ni_3d => 0)
    # Single orbital → 2 terms (dn + up)
    @test length(op.terms) == 2
    @test op' == op
end

@testset "shell-keyed n — orbital subset by vector of m values" begin
    m = ShellModel([:Ni_3d])
    op = n(m, :Ni_3d => [-1, 1])
    # Two orbitals × 2 spins → 4 terms
    @test length(op.terms) == 4
end

@testset "shell-keyed n — orbital subset matches sum of singletons" begin
    m = ShellModel([:Ni_3d])
    @test n(m, :Ni_3d => [-2, 0, 2]) == n(m, :Ni_3d => -2) + n(m, :Ni_3d => 0) + n(m, :Ni_3d => 2)
end

@testset "shell-keyed n — error on unknown shell" begin
    m = ShellModel([:Ni_3d])
    @test_throws KeyError n(m, :Cu_3d)
    @test_throws KeyError n(m, :Cu_3d => 0)
end

@testset "shell-keyed Sz — single shell" begin
    m = ShellModel([:Ni_3d])
    op = Sz(m, :Ni_3d)
    @test op isa OperatorSum
    # 10 terms: for each of 5 orbitals, n_up minus n_dn → 2 terms each, 10 total
    @test length(op.terms) == 10
    # Hermitian
    @test op' == op
end

@testset "shell-keyed Sz — multi-shell sum" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    @test Sz(m) == Sz(m, :Ni_2p) + Sz(m, :Ni_3d) + Sz(m, :L_3d)
end

@testset "shell-keyed Sz — vararg subset" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    @test Sz(m, :Ni_2p, :Ni_3d) == Sz(m, :Ni_2p) + Sz(m, :Ni_3d)
end

@testset "shell-keyed Sz — single orbital by m value" begin
    m = ShellModel([:Ni_3d])
    op = Sz(m, :Ni_3d => 0)
    # 1 orbital, Sz = (1/2)(n_up − n_dn) → 2 terms
    @test length(op.terms) == 2
    @test op' == op
end

@testset "shell-keyed Sz — orbital subset matches sum of singletons" begin
    m = ShellModel([:Ni_3d])
    @test Sz(m, :Ni_3d => [-1, 0, 1]) == Sz(m, :Ni_3d => -1) + Sz(m, :Ni_3d => 0) + Sz(m, :Ni_3d => 1)
end

@testset "shell-keyed Sz — physically meaningful: vacuum gives Sz=0" begin
    # Sanity check: in any basis the vacuum-state Sz expectation is 0.
    # This is purely structural: Sz must contain only number-operator terms,
    # so on a vacuum (no electrons) it gives zero.
    m = ShellModel([:Ni_3d])
    op = Sz(m, :Ni_3d)
    # For each term, the coefficient is ±1/2 and the operator is a single n.
    # Total of 10 such number-operator terms, sum of coefficients = 0 (5 +1/2, 5 -1/2)
    @test sum(t.coefficient for t in op) == 0
end

@testset "shell-keyed Sz — error on unknown shell" begin
    m = ShellModel([:Ni_3d])
    @test_throws KeyError Sz(m, :Cu_3d)
    @test_throws KeyError Sz(m, :Cu_3d => 0)
end

@testset "shell-keyed Sz — error on out-of-range m value" begin
    m = ShellModel([:Ni_3d])
    @test_throws ArgumentError Sz(m, :Ni_3d => 3)    # m=3 invalid for ℓ=2
    @test_throws ArgumentError Sz(m, :Ni_3d => -3)   # m=-3 invalid for ℓ=2
end

@testset "shell-keyed Splus — single shell, structural" begin
    m = ShellModel([:Ni_3d])
    op = Splus(m, :Ni_3d)
    @test op isa OperatorSum
    # 5 orbitals × one cdag c term per orbital → 5 terms
    @test length(op.terms) == 5
end

@testset "shell-keyed Sminus = adjoint(Splus)" begin
    m = ShellModel([:Ni_3d])
    @test Sminus(m, :Ni_3d) == adjoint(Splus(m, :Ni_3d))
    @test Sminus(m) == adjoint(Splus(m))
    @test Sminus(m, :Ni_3d => 0) == adjoint(Splus(m, :Ni_3d => 0))
end

@testset "shell-keyed Sx = (Splus + Sminus)/2" begin
    m = ShellModel([:Ni_3d])
    Sp = Splus(m, :Ni_3d)
    Sm = Sminus(m, :Ni_3d)
    @test Sx(m, :Ni_3d) == (1//2) * (Sp + Sm)
end

@testset "shell-keyed Sy = (-i/2) (Splus - Sminus)" begin
    m = ShellModel([:Ni_3d])
    Sp = Splus(m, :Ni_3d)
    Sm = Sminus(m, :Ni_3d)
    @test Sy(m, :Ni_3d) == (-im/2) * (Sp - Sm)
end

@testset "shell-keyed Sx, Sy Hermiticity" begin
    m = ShellModel([:Ni_3d])
    @test Sx(m, :Ni_3d)' == Sx(m, :Ni_3d)
    @test Sy(m, :Ni_3d)' == Sy(m, :Ni_3d)
end

@testset "shell-keyed Splus' == Sminus" begin
    m = ShellModel([:Ni_3d])
    @test Splus(m, :Ni_3d)' == Sminus(m, :Ni_3d)
end

@testset "shell-keyed Splus — single orbital" begin
    m = ShellModel([:Ni_3d])
    op = Splus(m, :Ni_3d => 0)
    # cdag_up * c_dn at m=0 → 1 term
    @test length(op.terms) == 1
end

@testset "shell-keyed Splus — orbital subset matches sum of singletons" begin
    m = ShellModel([:Ni_3d])
    @test Splus(m, :Ni_3d => [-1, 0, 1]) == Splus(m, :Ni_3d => -1) + Splus(m, :Ni_3d => 0) + Splus(m, :Ni_3d => 1)
end

@testset "shell-keyed Splus — vararg multi-shell" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    @test Splus(m, :Ni_2p, :Ni_3d) == Splus(m, :Ni_2p) + Splus(m, :Ni_3d)
end

@testset "shell-keyed Splus — total over all shells" begin
    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    @test Splus(m) == Splus(m, :Ni_2p) + Splus(m, :Ni_3d) + Splus(m, :L_3d)
end

@testset "shell-keyed S± — error on unknown shell" begin
    m = ShellModel([:Ni_3d])
    @test_throws KeyError Splus(m, :Cu_3d)
    @test_throws KeyError Sminus(m, :Cu_3d)
    @test_throws KeyError Sx(m, :Cu_3d)
    @test_throws KeyError Sy(m, :Cu_3d)
end

@testset "shell-keyed S± — error on out-of-range m value" begin
    m = ShellModel([:Ni_3d])
    @test_throws ArgumentError Splus(m, :Ni_3d => 3)
    @test_throws ArgumentError Sminus(m, :Ni_3d => 3)
    @test_throws ArgumentError Sx(m, :Ni_3d => 3)
    @test_throws ArgumentError Sy(m, :Ni_3d => 3)
end

@testset "shell-keyed Ssqr — single shell, structural" begin
    m = ShellModel([:Ni_3d])
    op = Ssqr(m, :Ni_3d)
    @test op isa OperatorSum
    # Hermitian
    @test op' == op
end

@testset "shell-keyed Ssqr — equals Sx² + Sy² + Sz²" begin
    m = ShellModel([:Ni_3d])
    op_explicit = Sx(m, :Ni_3d)*Sx(m, :Ni_3d) + Sy(m, :Ni_3d)*Sy(m, :Ni_3d) + Sz(m, :Ni_3d)*Sz(m, :Ni_3d)
    @test Ssqr(m, :Ni_3d) == op_explicit
end

@testset "shell-keyed Ssqr — total over all shells" begin
    m = ShellModel([:Ni_2p, :Ni_3d])
    sx = Sx(m); sy = Sy(m); sz = Sz(m)
    @test Ssqr(m) == sx*sx + sy*sy + sz*sz
end

@testset "shell-keyed Ssqr — single orbital" begin
    m = ShellModel([:Ni_3d])
    op = Ssqr(m, :Ni_3d => 0)
    sx = Sx(m, :Ni_3d => 0); sy = Sy(m, :Ni_3d => 0); sz = Sz(m, :Ni_3d => 0)
    @test op == sx*sx + sy*sy + sz*sz
end

@testset "shell-keyed Ssqr — error on unknown shell" begin
    m = ShellModel([:Ni_3d])
    @test_throws KeyError Ssqr(m, :Cu_3d)
end
