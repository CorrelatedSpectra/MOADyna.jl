using Test
using MOADyna
using MOADyna.Algebra: OperatorSum
using MOADyna.Shells: Akm, hop
using LinearAlgebra: norm

@testset "Akm — Oh d-shell eg/t2g splitting" begin
    # Cubic ligand field on a 3d shell. `subduce(Oh, 2) = [Eg, T2g]`,
    # both multiplicity-free. The packed irrep_coeffs `[ε_Eg, ε_T2g]`
    # therefore feed `expand_clm_central(G, 2, [ε_Eg, ε_T2g])`. Choosing
    # ε_Eg = +0.6, ε_T2g = -0.4 gives the canonical 10Dq=1 splitting.
    m = ShellModel([:Ni_3d])
    op = Akm(m, :Ni_3d, :Oh, [0.6, -0.4])

    @test op isa OperatorSum
    @test length(op) > 0

    # Hermiticity: real-symmetric per-IR blocks ⇒ Hermitian operator.
    # The spherical-tensor expansion produces ComplexF64 coefficients
    # whose terms organise into Hermitian pairs.
    h = op - op'
    # Empty (or numerical noise) means Hermitian.
    @test isempty(h)

    # 1-electron sector: 5 orbitals × 2 spins = 10 states. The Eg
    # 2-orbital block sits at +0.6, the T2g 3-orbital block at -0.4;
    # each is 2-fold spin-degenerate.
    bas = EagerBasis(m.hilbert, n_fermion(m.hilbert) == 1)
    sys = eigen(op, bas; n = length(bas))
    evals = sort(real.(sys.values))

    n_eg  = count(e -> abs(e - 0.6) < 1e-10, evals)
    n_t2g = count(e -> abs(e + 0.4) < 1e-10, evals)
    @test n_eg  == 4   # 2 orbitals × 2 spins
    @test n_t2g == 6   # 3 orbitals × 2 spins
    @test n_eg + n_t2g == length(evals)
end

@testset "Akm — Oh d-shell central scalar shift" begin
    # ε_Eg = ε_T2g = ε ⇒ degenerate identity on the shell. Spectrum is
    # ε on every single-particle state (10 modes).
    m = ShellModel([:Ni_3d])
    op = Akm(m, :Ni_3d, :Oh, [1.5, 1.5])
    bas = EagerBasis(m.hilbert, n_fermion(m.hilbert) == 1)
    sys = eigen(op, bas; n = length(bas))
    evals = real.(sys.values)
    @test all(e -> abs(e - 1.5) < 1e-10, evals)
end

@testset "Akm — D4h d-shell tetragonal splitting" begin
    # `subduce(D4h, 2) = A1g ⊕ B1g ⊕ B2g ⊕ Eg`, all multiplicity-free
    # (4 scalars). Mulliken correspondence (Quanty/Bilbao):
    #   d_{z²}   ↔ A1g (1 state)
    #   d_{x²-y²} ↔ B1g (1 state)
    #   d_{xy}    ↔ B2g (1 state)
    #   d_{xz},d_{yz} ↔ Eg (2 states)
    # Each is 2-fold spin-degenerate in the 1-electron sector.
    m = ShellModel([:Ni_3d])
    op = Akm(m, :Ni_3d, :D4h, [1.0, 2.0, 3.0, -0.5])

    bas = EagerBasis(m.hilbert, n_fermion(m.hilbert) == 1)
    sys = eigen(op, bas; n = length(bas))
    evals = sort(real.(sys.values))

    @test length(evals) == 10
    # Each scalar appears with the multiplicity (orbital × 2 spins) of
    # its IR.
    @test count(e -> abs(e - 1.0)  < 1e-10, evals) == 2  # A1g × 2 spins
    @test count(e -> abs(e - 2.0)  < 1e-10, evals) == 2  # B1g × 2 spins
    @test count(e -> abs(e - 3.0)  < 1e-10, evals) == 2  # B2g × 2 spins
    @test count(e -> abs(e + 0.5)  < 1e-10, evals) == 4  # Eg × 2 orbitals × 2 spins
end

@testset "Akm — linearity in irrep_coeffs" begin
    # Akm is R-linear in the packed coefficient vector.
    m = ShellModel([:Ni_3d])
    p1 = [0.6, -0.4]
    p2 = [0.1,  0.7]
    α, β = 0.7, -1.3
    lhs = Akm(m, :Ni_3d, :Oh, α .* p1 .+ β .* p2)
    rhs = α * Akm(m, :Ni_3d, :Oh, p1) + β * Akm(m, :Ni_3d, :Oh, p2)
    # Chop residual roundoff (each Akm call chops independently, so the
    # subtraction can leave terms at the chop floor).
    diff = chop(lhs - rhs; tol = 1e-10)
    @test isempty(diff)
end

@testset "hop — Hermiticity (Oh d-shell, eg and t2g)" begin
    # The constructed operator is c†_A P_Γ c_B + h.c., which is
    # self-adjoint by construction. After symmetrize+chop the residual
    # `op - op'` is exactly empty.
    m = ShellModel([:Ni_3d, :L_3d])
    op_eg  = hop(m, :Ni_3d, :L_3d, :Oh; irrep = :Eg)
    op_t2g = hop(m, :Ni_3d, :L_3d, :Oh; irrep = :T2g)
    @test isempty(op_eg  - op_eg')
    @test isempty(op_t2g - op_t2g')
    @test length(op_eg)  > 0
    @test length(op_t2g) > 0
end

@testset "hop — different-ℓ shells rejected" begin
    m = ShellModel([:Ni_2p, :Ni_3d])   # ℓ=1 vs ℓ=2
    @test_throws ArgumentError hop(m, :Ni_2p, :Ni_3d, :Oh; irrep = :Eg)
end

@testset "hop — eg / t2g block disjointness" begin
    # P_eg + P_t2g = 1_d (the d-shell identity). Therefore the eg and
    # t2g hopping operators are linearly independent and their sum
    # equals the full c†_A c_B + h.c. on the d-shell — a stronger
    # structural check than a simple eigenvalue match.
    m = ShellModel([:Ni_3d, :L_3d])
    op_eg  = hop(m, :Ni_3d, :L_3d, :Oh; irrep = :Eg)
    op_t2g = hop(m, :Ni_3d, :L_3d, :Oh; irrep = :T2g)

    # Single-particle basis: 5 d-orbitals × 2 shells × 2 spins = 20 states.
    # On this basis the hop matrix has a block-off-diagonal structure
    # H = [0 P; P' 0] (size 10+10) for each spin sector. Eigenvalues are
    # ± singular values of P. For an idempotent P, those are 0 and 1
    # with multiplicities (n_orb - rank) and rank.
    bas = EagerBasis(m.hilbert, n_fermion(m.hilbert) == 1)

    # eg: rank 2 ⇒ four ±1 eigenvalues per spin (eight total) and the
    # remaining 12 eigenvalues are 0.
    sys_eg = eigen(op_eg, bas; n = length(bas))
    evals_eg = sort(real.(sys_eg.values))
    n_pos_eg = count(e -> abs(e - 1.0) < 1e-10, evals_eg)
    n_neg_eg = count(e -> abs(e + 1.0) < 1e-10, evals_eg)
    n_zero_eg = count(e -> abs(e) < 1e-10, evals_eg)
    @test n_pos_eg  == 4   # 2 (rank) × 2 spins
    @test n_neg_eg  == 4
    @test n_zero_eg == 12

    # t2g: rank 3 ⇒ six ±1 eigenvalues per spin (twelve total).
    sys_t2g = eigen(op_t2g, bas; n = length(bas))
    evals_t2g = sort(real.(sys_t2g.values))
    n_pos_t2g = count(e -> abs(e - 1.0) < 1e-10, evals_t2g)
    n_neg_t2g = count(e -> abs(e + 1.0) < 1e-10, evals_t2g)
    n_zero_t2g = count(e -> abs(e) < 1e-10, evals_t2g)
    @test n_pos_t2g  == 6   # 3 (rank) × 2 spins
    @test n_neg_t2g  == 6
    @test n_zero_t2g == 8

    # P_eg + P_t2g = 1 ⇒ op_eg + op_t2g = full c†_A c_B + h.c.
    # which has rank 5 (per spin) ⇒ 10 eigenvalues at +1, 10 at -1.
    op_full = op_eg + op_t2g
    sys_full = eigen(op_full, bas; n = length(bas))
    evals_full = sort(real.(sys_full.values))
    @test count(e -> abs(e - 1.0) < 1e-10, evals_full) == 10
    @test count(e -> abs(e + 1.0) < 1e-10, evals_full) == 10
end

@testset "hop — unknown irrep label rejected" begin
    m = ShellModel([:Ni_3d, :L_3d])
    @test_throws ArgumentError hop(m, :Ni_3d, :L_3d, :Oh; irrep = :nope)
end

# Quanty OppVeg / OppVt2g bit-exact hop regression intentionally NOT wired
# here — fixtures are not in the repo (deferred sidetrack tracker, see
# STATUS.md "Open threads" section). The structural disjointness test above
# (P_Eg ⊥ P_T2g, P_Eg + P_T2g = 1_d) is convention-independent evidence the
# operator is correctly built. When fixtures are captured, replace this
# comment with the bit-exact regression.
