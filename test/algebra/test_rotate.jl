using Test, MOADyna
using MOADyna.Algebra: FermionSite, Hilbert, c, cdag, n
using LinearAlgebra: I as LinAlgI, qr
using Random: MersenneTwister, randn

@testset "rotate — single-site identity is no-op" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    op = cdag(s, 1) * c(s, 2)
    I2 = [1.0 0.0; 0.0 1.0]
    @test rotate(op, h, I2) == op
end

@testset "rotate — single-site, real 2x2 unitary, Hermiticity preserved" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    U = (1 / sqrt(2)) * [1.0 1.0; -1.0 1.0]   # columns = new basis in old, real orthogonal
    op_rotated = rotate(n(s, 1), h, U)
    @test op_rotated' == op_rotated
end

@testset "rotate — round-trip with U†" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    U = (1 / sqrt(2)) * [1.0 1.0; -1.0 1.0]
    op = cdag(s, 1) * c(s, 2)
    op_round = rotate(rotate(op, h, U), h, U')
    # FP arithmetic: (1/√2)² ≠ 0.5 exactly, so strict == fails by ~3e-16.
    # Test the structural property (round-trip recovers op) at FP tolerance.
    @test op_round ≈ op atol = 1e-12
end

@testset "rotate — non-square U rejected when project=false" begin
    s = FermionSite{4}(:s)
    h = Hilbert(:s => s)
    U_rect = randn(4, 2)
    err = try
        rotate(n(s, 1), h, U_rect)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("project=true", err.msg)
end

@testset "rotate — non-unitary square U rejected" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    U_bad = [1.0 0.5; 0.0 1.0]   # not unitary
    @test_throws ArgumentError rotate(n(s, 1), h, U_bad)
end

@testset "rotate — multi-site block-diagonal unitary preserves total number on rotated block" begin
    # Two-shell Hilbert mirroring the Ni 2p (6 modes) + Ni 3d (10 modes) layout
    # used downstream. Rotate only the d-block by a real orthogonal U_d; leave
    # the p-block as identity. Total number on the d-shell is the trace of
    # c† c on that block, which is invariant under any orthogonal rotation.
    s_p = FermionSite{6}(:Ni_2p)
    s_d = FermionSite{10}(:Ni_3d)
    h   = Hilbert(:Ni_2p => s_p, :Ni_3d => s_d)

    # Reproducible real-orthogonal 10x10 from a fixed-seed QR.
    rng = MersenneTwister(20260507)
    A = randn(rng, 10, 10)
    Q_d, _ = qr(A)
    U_d = Matrix(Q_d)                             # 10x10 real orthogonal
    U   = zeros(Float64, 16, 16)
    U[1:6,    1:6   ] = Matrix(LinAlgI, 6, 6)     # p-block identity
    U[7:16,   7:16  ] = U_d                       # d-block rotated

    N_d = sum(n(s_d, k) for k in 1:10)            # total number on d-shell
    N_d_rot = rotate(N_d, h, U)
    @test N_d_rot ≈ N_d atol = 1e-12

    # And the p-block number on a single mode is unchanged (identity sub-block).
    @test rotate(n(s_p, 3), h, U) ≈ n(s_p, 3) atol = 1e-12
end

@testset "rotate — sign/phase-sensitive substitution rule (locks Sakurai convention)" begin
    # Diagonal complex U with a phase only on mode 2. Acting on the
    # non-Hermitian operator c†_1 c_2, the substitution rule
    #     c†_old_1 → conj(U[1,1]) c†_new_1 = c†_new_1
    #     c_old_2  → U[2,2] c_new_2          = exp(iφ) c_new_2
    # gives op_rotated = exp(iφ) · c†_new_1 c_new_2.
    # If the conjugate were placed on c instead of c†, the result would be
    # exp(-iφ), so this test detects a silent inversion of the rule.
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    φ = π / 3
    U = ComplexF64[1.0  0.0;
                   0.0  exp(im * φ)]
    op = cdag(s, 1) * c(s, 2)
    op_rotated = rotate(op, h, U)
    expected   = exp(im * φ) * cdag(s, 1) * c(s, 2)
    @test op_rotated ≈ expected atol = 1e-12
end

@testset "rotate — project=true downfolding keeps modes 1,2 of a 4-mode site" begin
    # Rectangular isometry U = [I_2; 0_2] selects the first two modes.
    # Under c_old_j → Σ_i U[j,i] c_new_i, only j=1,2 contribute and they map
    # to c_new_1, c_new_2 respectively. Therefore n(s_in, 1) restricted to
    # the kept subspace equals n(s_out, 1).
    s_in  = FermionSite{4}(:s)
    s_out = FermionSite{2}(:s)
    h_in  = Hilbert(:s => s_in)
    h_out = Hilbert(:s => s_out)
    U = zeros(Float64, 4, 2)
    U[1, 1] = 1.0
    U[2, 2] = 1.0
    @test rotate(n(s_in, 1), h_in, U; h_out = h_out, project = true) ≈
          n(s_out, 1) atol = 1e-12
    @test rotate(n(s_in, 2), h_in, U; h_out = h_out, project = true) ≈
          n(s_out, 2) atol = 1e-12
    # Modes 3,4 are projected out: their number drops to zero.
    op_dropped = rotate(n(s_in, 3), h_in, U; h_out = h_out, project = true)
    @test isempty(op_dropped)
end

@testset "rotate — non-orthonormal rectangular U rejected with project=true" begin
    # Columns of U are not orthonormal: ⟨col1, col2⟩ = 1 ≠ 0, so U†U ≠ I_2.
    s_in  = FermionSite{4}(:s)
    s_out = FermionSite{2}(:s)
    h_in  = Hilbert(:s => s_in)
    h_out = Hilbert(:s => s_out)
    U_bad = [1.0 1.0;
             0.0 1.0;
             0.0 0.0;
             0.0 0.0]
    err = try
        rotate(n(s_in, 1), h_in, U_bad; h_out = h_out, project = true)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("isometric", err.msg) || occursin("U†U", err.msg)
end
