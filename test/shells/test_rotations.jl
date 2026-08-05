using Test
using MOAD
using MOAD.Algebra: FermionSite, Hilbert, c, cdag, n, rotate, OperatorSum
using MOAD.Shells: ShellModel, to_real, to_jlmj, ell_of, site_of
using LinearAlgebra: I as LinAlgI, Diagonal, diag

# A handful of fixed real-K transforms are tested at the matrix level
# (unitarity, hand-checked entries, deferred-shell error path) and one
# many-body round-trip via `rotate` exercises the spin-fast layout
# (kron(U_orb, I_2)) against the live operator engine.

@testset "to_real — s shell is identity (spin block only)" begin
    m = ShellModel([:H_1s])
    U = to_real(m, :H_1s)
    @test size(U) == (2, 2)
    @test U == ComplexF64[1 0; 0 1]
end

@testset "to_real — p shell shape, unitarity, hand-checked entries" begin
    m = ShellModel([:Ni_2p])
    U = to_real(m, :Ni_2p)
    @test size(U) == (6, 6)
    @test maximum(abs, U' * U - LinAlgI) < 1e-13
    @test maximum(abs, U * U' - LinAlgI) < 1e-13

    inv_sqrt2 = 1 / sqrt(2)
    # Spin block kron(U_orb, I_2): U[2(i-1)+s, 2(j-1)+t] = U_orb[i,j] * δ_{s,t}.
    # p_y (m_K=-1, col 1): U_orb[1,1] = i/√2, U_orb[3,1] = i/√2.
    @test U[1, 1] ≈ im * inv_sqrt2 atol = 1e-13   # (m_old=-1,↓) → (p_y,↓)
    @test U[5, 1] ≈ im * inv_sqrt2 atol = 1e-13   # (m_old=+1,↓) → (p_y,↓)
    # p_z (m_K=0, col 3): U_orb[2,2] = 1.
    @test U[3, 3] ≈ 1 atol = 1e-13                 # (m_old=0,↓) → (p_z,↓)
    @test U[4, 4] ≈ 1 atol = 1e-13                 # (m_old=0,↑) → (p_z,↑)
    # p_x (m_K=+1, col 5): U_orb[1,3] = 1/√2, U_orb[3,3] = -1/√2.
    @test U[1, 5] ≈ inv_sqrt2 atol = 1e-13
    @test U[5, 5] ≈ -inv_sqrt2 atol = 1e-13
end

@testset "to_real — d shell shape, unitarity, hand-checked entries" begin
    m = ShellModel([:Ni_3d])
    U = to_real(m, :Ni_3d)
    @test size(U) == (10, 10)
    @test maximum(abs, U' * U - LinAlgI) < 1e-13
    @test maximum(abs, U * U' - LinAlgI) < 1e-13

    inv_sqrt2 = 1 / sqrt(2)
    # d_xy (m_K=-2, col 1): U_orb[1,1] = i/√2, U_orb[5,1] = -i/√2.
    @test U[1, 1] ≈ im * inv_sqrt2 atol = 1e-13
    @test U[9, 1] ≈ -im * inv_sqrt2 atol = 1e-13
    # d_{3z²-r²} (m_K=0, col 5): U_orb[3,3] = 1.
    @test U[5, 5] ≈ 1 atol = 1e-13
    @test U[6, 6] ≈ 1 atol = 1e-13   # spin-up partner
    # d_{x²-y²} (m_K=+2, col 9): U_orb[1,5] = 1/√2, U_orb[5,5] = 1/√2.
    @test U[1, 9] ≈ inv_sqrt2 atol = 1e-13
    @test U[9, 9] ≈ inv_sqrt2 atol = 1e-13
end

@testset "to_real — kron block diagonal in spin (no spin-flip)" begin
    # kron(U_orb, I_2) means rows/cols of opposite spin are decoupled:
    # U[odd, even] and U[even, odd] entries are exactly zero.
    m = ShellModel([:Ni_3d])
    U = to_real(m, :Ni_3d)
    @test all(U[i, j] == 0 for i in 1:2:10 for j in 2:2:10)
    @test all(U[i, j] == 0 for i in 2:2:10 for j in 1:2:10)
end

@testset "to_real — round-trip through rotate (square unitary)" begin
    # rotate(rotate(op, h, U), h, U') should recover op to FP tolerance.
    # This locks the spin layout: if `kron(U_orb, I_2)` mismatches the
    # actual mode order (m-major, dn-then-up), this test fails first.
    m = ShellModel([:Ni_3d])
    s = site_of(m, :Ni_3d)
    h = m.hilbert
    U = to_real(m, :Ni_3d)
    op = n(s, 1) + n(s, 5)            # arbitrary diagonal in (ℓm) basis
    op_real = rotate(op, h, U)
    op_back = rotate(op_real, h, U')
    @test op_back ≈ op atol = 1e-12
end

@testset "to_real — total number invariant under basis change" begin
    # Tr(c†c) = total number is independent of orbital basis.
    m = ShellModel([:Ni_2p])
    s = site_of(m, :Ni_2p)
    h = m.hilbert
    U = to_real(m, :Ni_2p)
    N_total = n(s)
    @test rotate(N_total, h, U) ≈ N_total atol = 1e-12
end

@testset "to_real — f shell deferred" begin
    m = ShellModel([:Ce_4f])
    err = try
        to_real(m, :Ce_4f)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("f-shell", err.msg)
end

@testset "to_jlmj — s shell trivial (j = 1/2 only)" begin
    m = ShellModel([:H_1s])
    U = to_jlmj(m, :H_1s)
    @test size(U) == (2, 2)
    @test U == ComplexF64[1 0; 0 1]
end

@testset "to_jlmj — unitarity for p and d shells" begin
    for tag in (:Ni_2p, :Ni_3d)
        m = ShellModel([tag])
        U = to_jlmj(m, tag)
        ell = ell_of(m, tag)
        n_dim = 2 * (2 * ell + 1)
        @test size(U) == (n_dim, n_dim)
        @test maximum(abs, U' * U - LinAlgI) < 1e-13
        @test maximum(abs, U * U' - LinAlgI) < 1e-13
    end
end

@testset "to_jlmj — column conservation (m_ℓ + σ = m_j)" begin
    # Each column of U lives in a single m_j eigenspace, so it should
    # have at most two nonzero rows: (m_ℓ=m_j-1/2, σ=+1/2) and
    # (m_ℓ=m_j+1/2, σ=-1/2). Stretched-j states have only one.
    m = ShellModel([:Ni_3d])
    U = to_jlmj(m, :Ni_3d)
    nzc = count(!iszero, U; dims = 1)
    @test all(1 .<= nzc .<= 2)
    # Only j_max stretched states |j_max, ±j_max⟩ are pure (single
    # product state). For d (ℓ=2), j_max = 5/2 → cols 5, 10. The lower
    # j=3/2 block columns mix two (m_ℓ, σ) basis vectors.
    @test count(!iszero, U[:, 5]) == 1     # |5/2, -5/2⟩ = (m_ℓ=-2, ↓)
    @test count(!iszero, U[:, 10]) == 1    # |5/2, +5/2⟩ = (m_ℓ=+2, ↑)
end

@testset "to_jlmj — p shell stretched-j entries (CG by hand)" begin
    # |j=3/2, m_j=+3/2⟩ = |m_ℓ=+1, σ=+1/2⟩  (CG = 1).
    # |j=3/2, m_j=-3/2⟩ = |m_ℓ=-1, σ=-1/2⟩  (CG = 1).
    # In p-shell column ordering: j=1/2 first (cols 1-2), then
    # j=3/2 (cols 3-6 with m_j = -3/2, -1/2, +1/2, +3/2).
    # Row layout (m-major + dn-then-up): row 1 = (m_ℓ=-1, ↓), row 5 = (m_ℓ=+1, ↓),
    # row 6 = (m_ℓ=+1, ↑).
    m = ShellModel([:Ni_2p])
    U = to_jlmj(m, :Ni_2p)
    @test U[1, 3] ≈ 1.0 atol = 1e-13   # |3/2,-3/2⟩ = (m_ℓ=-1, ↓)
    @test U[6, 6] ≈ 1.0 atol = 1e-13   # |3/2,+3/2⟩ = (m_ℓ=+1, ↑)

    # Standard p_{1/2}, p_{3/2} mixing (Condon-Shortley CG):
    #   |j=1/2, m_j=+1/2⟩ = +√(2/3) |m_ℓ=+1, ↓⟩ - √(1/3) |m_ℓ=0, ↑⟩
    #   |j=3/2, m_j=+1/2⟩ = +√(1/3) |m_ℓ=+1, ↓⟩ + √(2/3) |m_ℓ=0, ↑⟩
    # Row 4 = (m_ℓ=0, ↑); row 5 = (m_ℓ=+1, ↓).
    @test U[5, 2] ≈ sqrt(2 / 3) atol = 1e-13
    @test U[4, 2] ≈ -sqrt(1 / 3) atol = 1e-13
    @test U[5, 5] ≈ sqrt(1 / 3) atol = 1e-13
    @test U[4, 5] ≈ sqrt(2 / 3) atol = 1e-13
end

@testset "to_jlmj — diagonalizes l·s for the p shell (inline 6×6)" begin
    # Build l·s on the 1-particle p basis directly (no LS operator
    # needed — Task 8 not yet shipped). Old basis order (rows/cols):
    #   1: (m_ℓ=-1, ↓), 2: (m_ℓ=-1, ↑),
    #   3: (m_ℓ= 0, ↓), 4: (m_ℓ= 0, ↑),
    #   5: (m_ℓ=+1, ↓), 6: (m_ℓ=+1, ↑).
    # l·s = l_z s_z + (1/2)(l_+ s_- + l_- s_+).
    LS = zeros(ComplexF64, 6, 6)
    # l_z s_z diagonal: m_ℓ * σ.
    for (i, (m_l, σ)) in enumerate([(-1, -0.5), (-1, 0.5), (0, -0.5),
                                    (0, 0.5), (1, -0.5), (1, 0.5)])
        LS[i, i] = m_l * σ
    end
    # (1/2) l_+ s_- : connects (m_ℓ, ↑) → (m_ℓ+1, ↓), coeff
    # (1/2) sqrt(ℓ(ℓ+1) - m_ℓ(m_ℓ+1)) with ℓ = 1.
    # m_ℓ = -1 (state 2, ↑) → m_ℓ = 0 (state 3, ↓): (1/2)·sqrt(2) = 1/√2.
    LS[3, 2] += 0.5 * sqrt(2)
    # m_ℓ = 0 (state 4, ↑) → m_ℓ = +1 (state 5, ↓): (1/2)·sqrt(2) = 1/√2.
    LS[5, 4] += 0.5 * sqrt(2)
    # Hermitian conjugate (l_- s_+ is the adjoint):
    LS[2, 3] += 0.5 * sqrt(2)
    LS[4, 5] += 0.5 * sqrt(2)

    m = ShellModel([:Ni_2p])
    U = to_jlmj(m, :Ni_2p)
    LS_j = U' * LS * U
    # Should be diagonal with eigenvalues j=1/2 → -1 (×2), j=3/2 → +1/2 (×4).
    @test maximum(abs, LS_j - Diagonal(diag(LS_j))) < 1e-12
    expected = sort([-1.0, -1.0, 0.5, 0.5, 0.5, 0.5])
    @test maximum(abs, sort(real.(diag(LS_j))) - expected) < 1e-12
end

# LS one-body cross-check (using MOAD's `LS(m, shell)`) deferred until
# Plan 2 Task 8 ships. When ready, U' * LS_M * U should be diagonal with
# eigenvalues {-1, -1, 1/2, 1/2, 1/2, 1/2} on the 1-particle 2p basis,
# and {-3/2, -3/2, -3/2, -3/2, 1, 1, 1, 1, 1, 1} on the 1-particle 2d
# basis (in units where ζ = 1).
