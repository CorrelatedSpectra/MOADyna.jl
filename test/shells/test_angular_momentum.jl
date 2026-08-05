using Test
using MOAD
using MOAD.Algebra: OperatorSum
using MOAD.Shells: ShellModel, Lz, Lplus, Lminus, Lx, Ly, Lsqr, site_of, ell_of
using MOAD.Shells: Jx, Jy, Jz, Jplus, Jminus, Jsqr
using MOAD.Shells: LS
using LinearAlgebra: eigvals
using LinearAlgebra: Hermitian

@testset "Lz — Hermitian on Ni_3d" begin
    m = ShellModel([:Ni_3d])
    op = Lz(m, :Ni_3d)
    @test op isa OperatorSum
    @test op' == op
end

@testset "Lz — 1-particle spectrum on d-shell matches m_ℓ × spin degeneracy" begin
    # Lz |m_ℓ, σ⟩ = m_ℓ |m_ℓ, σ⟩, independent of σ.
    # 1-particle sector of a d-shell has 10 single-particle states with
    # eigenvalues {-2, -2, -1, -1, 0, 0, 1, 1, 2, 2}.
    m = ShellModel([:Ni_3d])
    op = Lz(m, :Ni_3d)
    bas = EagerBasis(m, nshells(m, :Ni_3d) == 1)
    M = Matrix(assemble(compile(op, bas), bas))
    evals = sort(real.(eigvals(M)))
    expected = sort([-2.0, -2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    @test maximum(abs, evals .- expected) < 1e-12
end

@testset "Lz — vacuum carries zero" begin
    # Lz is a sum of m_ℓ · n terms, so it annihilates the empty shell.
    m = ShellModel([:Ni_3d])
    op = Lz(m, :Ni_3d)
    bas = EagerBasis(m, nshells(m, :Ni_3d) == 0)
    M = Matrix(assemble(compile(op, bas), bas))
    @test size(M) == (1, 1)
    @test abs(M[1, 1]) < 1e-14
end

@testset "Lz — s-shell (ℓ=0) is identically zero" begin
    m = ShellModel([:A_1s])
    op = Lz(m, :A_1s)
    bas = EagerBasis(m)
    M = Matrix(assemble(compile(op, bas), bas))
    @test maximum(abs, M) < 1e-14
end

@testset "Lz — error on unknown shell" begin
    m = ShellModel([:Ni_3d])
    @test_throws KeyError Lz(m, :Cu_3d)
end

@testset "Lplus/Lminus — adjoint relation on Ni_3d" begin
    m = ShellModel([:Ni_3d])
    Lp = Lplus(m, :Ni_3d)
    Lm = Lminus(m, :Ni_3d)
    @test Lp isa OperatorSum
    @test Lm isa OperatorSum
    @test Lp' == Lm
    @test Lm' == Lp
end

@testset "Lplus/Lminus — [L+, L-] = 2 Lz on 1-electron d-shell" begin
    m = ShellModel([:Ni_3d])
    Lp = Lplus(m, :Ni_3d)
    Lm = Lminus(m, :Ni_3d)
    Lz_op = Lz(m, :Ni_3d)
    bas = EagerBasis(m, nshells(m, :Ni_3d) == 1)
    LpM = Matrix(assemble(compile(Lp, bas), bas))
    LmM = Matrix(assemble(compile(Lm, bas), bas))
    LzM = Matrix(assemble(compile(Lz_op, bas), bas))
    commutator = LpM * LmM - LmM * LpM - 2 * LzM
    @test maximum(abs, commutator) < 1e-12
end

@testset "Lplus — raises m_ℓ on 1-electron sector with right amplitudes" begin
    # On the 1-particle d-shell, L⁺ acts as L⁺|m,σ⟩ = sqrt(ℓ(ℓ+1) − m(m+1)) |m+1,σ⟩.
    # Aggregate spectral check: L⁺ L⁻ = L² − Lz² + Lz, so on each |m, σ⟩
    # the diagonal of L⁺ L⁻ is ℓ(ℓ+1) − m(m+1) + 2m? No — explicitly:
    # L⁻ |m,σ⟩ = sqrt(ℓ(ℓ+1) − m(m−1)) |m−1, σ⟩, then L⁺ raises it back, so
    # ⟨m,σ| L⁺ L⁻ |m,σ⟩ = ℓ(ℓ+1) − m(m−1).
    # For ℓ=2 the 5 m-values (-2,-1,0,1,2) give diagonals (0, 4, 6, 6, 4) — each twice for spin.
    m = ShellModel([:Ni_3d])
    Lp = Lplus(m, :Ni_3d)
    Lm = Lminus(m, :Ni_3d)
    Lz_op = Lz(m, :Ni_3d)
    bas = EagerBasis(m, nshells(m, :Ni_3d) == 1)
    LpM = Matrix(assemble(compile(Lp, bas), bas))
    LmM = Matrix(assemble(compile(Lm, bas), bas))
    LzM = Matrix(assemble(compile(Lz_op, bas), bas))
    # L² = (1/2)(L⁺ L⁻ + L⁻ L⁺) + Lz² should be ℓ(ℓ+1) = 6 on every 1-particle d-state.
    Lsqr = (LpM * LmM + LmM * LpM) / 2 + LzM * LzM
    @test maximum(abs, Lsqr - 6 * one(Lsqr)) < 1e-12
end

@testset "Lplus — s-shell (ℓ=0) is identically zero" begin
    m = ShellModel([:A_1s])
    Lp = Lplus(m, :A_1s)
    Lm = Lminus(m, :A_1s)
    bas = EagerBasis(m)
    @test maximum(abs, Matrix(assemble(compile(Lp, bas), bas))) < 1e-14
    @test maximum(abs, Matrix(assemble(compile(Lm, bas), bas))) < 1e-14
end

@testset "Lplus — error on unknown shell" begin
    m = ShellModel([:Ni_3d])
    @test_throws KeyError Lplus(m, :Cu_3d)
    @test_throws KeyError Lminus(m, :Cu_3d)
end

@testset "Lx, Ly — Hermiticity + [Lx,Ly] = i Lz on 1-electron d-shell" begin
    m = ShellModel([:Ni_3d])
    Lx_op = Lx(m, :Ni_3d)
    Ly_op = Ly(m, :Ni_3d)
    Lz_op = Lz(m, :Ni_3d)
    @test Lx_op isa OperatorSum
    @test Ly_op isa OperatorSum
    @test Lx_op' == Lx_op
    @test Ly_op' == Ly_op
    bas = EagerBasis(m, nshells(m, :Ni_3d) == 1)
    LxM = Matrix(assemble(compile(Lx_op, bas), bas))
    LyM = Matrix(assemble(compile(Ly_op, bas), bas))
    LzM = Matrix(assemble(compile(Lz_op, bas), bas))
    commutator_xy = LxM * LyM - LyM * LxM - im * LzM
    @test maximum(abs, commutator_xy) < 1e-12
end

@testset "Lsqr — eigenvalue ℓ(ℓ+1) = 6 on 1-electron d-shell" begin
    m = ShellModel([:Ni_3d])
    L2 = Lsqr(m, :Ni_3d)
    @test L2 isa OperatorSum
    bas = EagerBasis(m, nshells(m, :Ni_3d) == 1)
    L2M = Matrix(assemble(compile(L2, bas), bas))
    evals = sort(real.(eigvals(L2M)))
    # ℓ(ℓ+1) = 2·3 = 6 for d-shell, 10-fold (5 m-values × 2 spins).
    @test maximum(abs, evals .- 6.0) < 1e-12
end

@testset "Lsqr — eigenvalue ℓ(ℓ+1) = 2 on 1-electron p-shell" begin
    m = ShellModel([:Ni_2p])
    L2 = Lsqr(m, :Ni_2p)
    bas = EagerBasis(m, nshells(m, :Ni_2p) == 1)
    L2M = Matrix(assemble(compile(L2, bas), bas))
    evals = sort(real.(eigvals(L2M)))
    # ℓ(ℓ+1) = 1·2 = 2 for p-shell, 6-fold (3 m-values × 2 spins).
    @test maximum(abs, evals .- 2.0) < 1e-12
end

@testset "Jx, Jy, Jz — Hermiticity + [Jx, Jy] = i Jz on 1-electron d-shell" begin
    m = ShellModel([:Ni_3d])
    Jx_op = Jx(m, :Ni_3d)
    Jy_op = Jy(m, :Ni_3d)
    Jz_op = Jz(m, :Ni_3d)
    @test Jx_op isa OperatorSum
    @test Jy_op isa OperatorSum
    @test Jz_op isa OperatorSum
    @test Jx_op' == Jx_op
    @test Jy_op' == Jy_op
    @test Jz_op' == Jz_op
    bas = EagerBasis(m, nshells(m, :Ni_3d) == 1)
    JxM = Matrix(assemble(compile(Jx_op, bas), bas))
    JyM = Matrix(assemble(compile(Jy_op, bas), bas))
    JzM = Matrix(assemble(compile(Jz_op, bas), bas))
    # [Jx, Jy] = i Jz
    @test maximum(abs, JxM * JyM - JyM * JxM - im * JzM) < 1e-12
end

@testset "Jplus, Jminus — adjoint relation on Ni_3d" begin
    m = ShellModel([:Ni_3d])
    Jp = Jplus(m, :Ni_3d)
    Jm = Jminus(m, :Ni_3d)
    @test Jp isa OperatorSum
    @test Jm isa OperatorSum
    @test Jp' == Jm
    @test Jm' == Jp
end

@testset "Jsqr — j(j+1) eigenvalues on 1-electron d-shell" begin
    # For ℓ=2, s=1/2: j ∈ {3/2, 5/2}. Eigenvalues of J² are j(j+1):
    # j=3/2 → 15/4 (4-fold), j=5/2 → 35/4 (6-fold). Total 10 = 5 m × 2 σ.
    m = ShellModel([:Ni_3d])
    J2 = Jsqr(m, :Ni_3d)
    @test J2 isa OperatorSum
    bas = EagerBasis(m, nshells(m, :Ni_3d) == 1)
    J2M = Matrix(assemble(compile(J2, bas), bas))
    # Hermitian on the 1-electron sector.
    @test maximum(abs, J2M - J2M') < 1e-12
    evals = sort(real.(eigvals((J2M + J2M') / 2)))
    expected = sort(vcat(fill(15/4, 4), fill(35/4, 6)))
    @test maximum(abs, evals .- expected) < 1e-12
end

@testset "LS — Hermiticity on Ni_3d" begin
    m = ShellModel([:Ni_3d])
    op = LS(m, :Ni_3d)
    @test op isa OperatorSum
    @test op' == op
end

@testset "LS — STRUCTURAL: every term has at most 2 ladder entries (one-body sentinel)" begin
    # Defends against the "L·S as product of total operators" mistake.
    # H_SO must be a one-body operator: every chain has length ≤ 2
    # (n = c†c is length 2; c†c hopping is length 2). A length-4 chain
    # (c†c†cc) would mean a two-body cross term Σ_{i≠j} l_i · s_j sneaked in.
    m = ShellModel([:Ni_3d])
    op = LS(m, :Ni_3d)
    @test all(length(t.chain) ≤ 2 for t in op)
end

@testset "LS — j=1/2 / j=3/2 spectrum on 1-electron 2p" begin
    # Eigenvalues of l·s on j-coupled basis: (j(j+1) − ℓ(ℓ+1) − s(s+1)) / 2.
    # For ℓ=1, s=1/2:
    #   j=1/2: (3/4 − 2 − 3/4)/2 = −1, multiplicity 2
    #   j=3/2: (15/4 − 2 − 3/4)/2 = +1/2, multiplicity 4
    m = ShellModel([:Ni_2p])
    op = LS(m, :Ni_2p)
    bas = EagerBasis(m, nshells(m, :Ni_2p) == 1)
    H = Matrix(assemble(compile(op, bas), bas))
    evals = sort(real.(eigvals(Hermitian((H + H') / 2))))
    expected = sort([-1.0, -1.0, 0.5, 0.5, 0.5, 0.5])
    @test maximum(abs, evals .- expected) < 1e-12
end

@testset "LS — j=3/2 / j=5/2 spectrum on 1-electron 3d" begin
    # For ℓ=2, s=1/2:
    #   j=3/2: (15/4 − 6 − 3/4)/2 = −3/2, multiplicity 4
    #   j=5/2: (35/4 − 6 − 3/4)/2 = +1, multiplicity 6
    m = ShellModel([:Ni_3d])
    op = LS(m, :Ni_3d)
    bas = EagerBasis(m, nshells(m, :Ni_3d) == 1)
    H = Matrix(assemble(compile(op, bas), bas))
    evals = sort(real.(eigvals(Hermitian((H + H') / 2))))
    expected = sort(vcat(fill(-1.5, 4), fill(1.0, 6)))
    @test maximum(abs, evals .- expected) < 1e-12
end

@testset "LS — s-shell (ℓ=0) is identically zero" begin
    m = ShellModel([:A_1s])
    op = LS(m, :A_1s)
    bas = EagerBasis(m)
    M = Matrix(assemble(compile(op, bas), bas))
    @test maximum(abs, M) < 1e-14
end

@testset "LS — error on unknown shell" begin
    m = ShellModel([:Ni_3d])
    @test_throws KeyError LS(m, :Cu_3d)
end
