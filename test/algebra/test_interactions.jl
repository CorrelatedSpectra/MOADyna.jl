using Test
using MOAD
using MOAD.Algebra: FermionSite, Hilbert, OperatorSum

# =====================================================================
# Algebra-level interaction primitives — FermionSite + orbital-pair forms
# =====================================================================
#
# A FermionSite{4} models 2 orbitals × 2 spins in m-major dn-then-up order:
#   orbital 0: modes 1 (dn), 2 (up)
#   orbital 1: modes 3 (dn), 4 (up)

const _FS4 = FermionSite{4}(:test_site)
const _PAIRS2 = [(1, 2), (3, 4)]   # 2 orbital pairs

# --- density_density primitives ---

@testset "density_density(FermionSite) — result is OperatorSum" begin
    op = density_density(_FS4, _PAIRS2; U = 3.0, J = 0.5)
    @test op isa OperatorSum
end

@testset "density_density(FermionSite) — Hermiticity" begin
    op = density_density(_FS4, _PAIRS2; U = 3.0, J = 0.5)
    @test op' == op
end

@testset "density_density(FermionSite) — J=0 means U_eff = Up = U" begin
    # With J=0, Up=U, parallel coefficient = U-J = U, antiparallel = U.
    # All inter-orbital terms have coefficient U.
    op = density_density(_FS4, _PAIRS2; U = 4.0, J = 0.0)
    @test op' == op
    # Every term coefficient should have magnitude U (intra) or U (inter).
    coeffs = abs.(real.(t.coefficient for t in op))
    @test all(c -> isapprox(c, 4.0, atol=1e-12), coeffs)
end

@testset "density_density(FermionSite) — Up override changes operator" begin
    op_default = density_density(_FS4, _PAIRS2; U = 5.0, J = 0.5)
    op_custom  = density_density(_FS4, _PAIRS2; U = 5.0, J = 0.5, Up = 3.0)
    @test op_default != op_custom
end

@testset "density_density(FermionSite) — 2-orbital N=2 eigenvalue structure" begin
    # Density-density only (no spin-flip / pair-hop):
    # C(4, 2) = 6 states for 2 orbitals × 2 spins.
    # Intra-orbital pairs (1 state): E = U
    # Inter-orbital, parallel spin (2 states, one for ↑↑ and one for ↓↓): E = U-3J
    # Inter-orbital, opposite spin (2 states, ordering σ): E = U-2J
    # Inter-orbital, opposite spin (remaining 1 state): E = U-2J
    # Actual count for 2 orbitals:
    #   1 intra (α=0, n_dn n_up) + 1 intra (α=1) = 2 at U
    #   1 parallel ↑↑ + 1 parallel ↓↓ = 2 at U-3J
    #   1 opposite (↑_0 ↓_1) + 1 opposite (↓_0 ↑_1) = 2 at U-2J
    #   Total = 6 ✓
    site = FermionSite{4}(:dd_test)
    h = Hilbert(:dd_test => site)
    H = density_density(site, _PAIRS2; U = 5.0, J = 0.5)
    U, J = 5.0, 0.5
    basis = EagerBasis(h, n_fermion(h) == 2)
    sys = eigen(H, basis; n = length(basis))
    evals = sort(real.(sys.values))

    @test count(e -> abs(e - U)       < 1e-10, evals) == 2
    @test count(e -> abs(e - (U-3J))  < 1e-10, evals) == 2
    @test count(e -> abs(e - (U-2J))  < 1e-10, evals) == 2
    @test length(evals) == 6
end

# --- kanamori primitives ---

@testset "kanamori(FermionSite) — result is OperatorSum" begin
    H = kanamori(_FS4, _PAIRS2; U = 3.4, J = 0.6)
    @test H isa OperatorSum
end

@testset "kanamori(FermionSite) — Hermiticity" begin
    H = kanamori(_FS4, _PAIRS2; U = 3.4, J = 0.6)
    @test H' == H
end

@testset "kanamori(FermionSite) — reduces to density_density at J=0" begin
    H_kan = kanamori(_FS4, _PAIRS2; U = 4.0, J = 0.0)
    H_dd  = density_density(_FS4, _PAIRS2; U = 4.0, J = 0.0)
    @test H_kan == H_dd
end

@testset "kanamori(FermionSite) — Up override changes operator" begin
    op_default = kanamori(_FS4, _PAIRS2; U = 5.0, J = 0.5)
    op_custom  = kanamori(_FS4, _PAIRS2; U = 5.0, J = 0.5, Up = 3.0)
    @test op_default != op_custom
end

@testset "kanamori(FermionSite) — 2-orbital N=2 spectrum: 3 at U-3J, 2 at U-J, 1 at U+J" begin
    # 2 orbitals × 2 spins → C(4,2)=6 states in the N=2 sector.
    # With pair-hopping the two intra-orbital states |0↑0↓⟩, |1↑1↓⟩ mix via J:
    #   2×2 block with U on diagonal and J on off-diagonal →
    #   eigenvalues U+J (symmetric, ×1) and U-J (antisymmetric, ×1).
    # The two parallel-spin states |0↑1↑⟩, |0↓1↓⟩ have E = U-3J.
    # The two opposite-spin states |0↑1↓⟩, |0↓1↑⟩ mix under spin-flip:
    #   triplet S_z=0 at U-3J (×1), singlet at U-J (×1).
    # Net: 3 at U-3J, 2 at U-J, 1 at U+J. Total = 6 ✓.
    site = FermionSite{4}(:kan_test)
    h = Hilbert(:kan_test => site)
    U, J = 5.0, 0.5
    H = kanamori(site, _PAIRS2; U = U, J = J)
    basis = EagerBasis(h, n_fermion(h) == 2)
    sys = eigen(H, basis; n = length(basis))
    evals = sort(real.(sys.values))

    @test count(e -> abs(e - (U - 3J)) < 1e-10, evals) == 3
    @test count(e -> abs(e - (U - J))  < 1e-10, evals) == 2
    @test count(e -> abs(e - (U + J))  < 1e-10, evals) == 1
    @test length(evals) == 6
end

# --- Spin operators on FermionSite ---

@testset "Sz(FermionSite) — result is OperatorSum" begin
    op = Sz(_FS4, _PAIRS2)
    @test op isa OperatorSum
end

@testset "Sz(FermionSite) — Hermiticity" begin
    op = Sz(_FS4, _PAIRS2)
    @test op' == op
end

@testset "Sz(FermionSite) — coefficient sum is zero (equal up/dn weights)" begin
    op = Sz(_FS4, _PAIRS2)
    # For each orbital: +1/2 on up mode, -1/2 on dn mode → sum = 0
    @test sum(t.coefficient for t in op) == 0
end

@testset "Splus(FermionSite) — result is OperatorSum" begin
    op = Splus(_FS4, _PAIRS2)
    @test op isa OperatorSum
end

@testset "Sminus(FermionSite) — equals adjoint(Splus)" begin
    @test Sminus(_FS4, _PAIRS2) == adjoint(Splus(_FS4, _PAIRS2))
end

@testset "Splus(FermionSite)' == Sminus(FermionSite)" begin
    @test Splus(_FS4, _PAIRS2)' == Sminus(_FS4, _PAIRS2)
end

@testset "Sx(FermionSite) — Hermiticity" begin
    @test Sx(_FS4, _PAIRS2)' == Sx(_FS4, _PAIRS2)
end

@testset "Sy(FermionSite) — Hermiticity" begin
    @test Sy(_FS4, _PAIRS2)' == Sy(_FS4, _PAIRS2)
end

@testset "Sx(FermionSite) — equals (Splus + Sminus)/2" begin
    Sp = Splus(_FS4, _PAIRS2)
    Sm = Sminus(_FS4, _PAIRS2)
    @test Sx(_FS4, _PAIRS2) == (1//2) * (Sp + Sm)
end

@testset "Sy(FermionSite) — equals (-im/2)(Splus - Sminus)" begin
    Sp = Splus(_FS4, _PAIRS2)
    Sm = Sminus(_FS4, _PAIRS2)
    @test Sy(_FS4, _PAIRS2) == (-im/2) * (Sp - Sm)
end

@testset "Ssqr(FermionSite) — Hermiticity" begin
    op = Ssqr(_FS4, _PAIRS2)
    @test op' == op
end

@testset "Ssqr(FermionSite) — equals Sx² + Sy² + Sz² structurally" begin
    sx = Sx(_FS4, _PAIRS2)
    sy = Sy(_FS4, _PAIRS2)
    sz = Sz(_FS4, _PAIRS2)
    @test Ssqr(_FS4, _PAIRS2) == sx*sx + sy*sy + sz*sz
end

@testset "Ssqr(FermionSite) — eigenvalue S(S+1) on maximally polarized 2e state" begin
    # A 2-electron state with both spins up (|0↑1↑⟩) has total S=1 → S²=2.
    site = FermionSite{4}(:ssqr_test)
    h = Hilbert(:ssqr_test => site)
    Hsz = Sz(site, _PAIRS2)   # use Sz to select the Sz=+1 sector
    Hs2 = Ssqr(site, _PAIRS2)
    # Build basis for N=2, Sz=+1 (only state is |0↑1↑⟩ = modes 2 and 4 filled)
    basis = EagerBasis(h,
        n_fermion(h) == 2,
        WeightedParticleCount([site], [0, 1, 0, 1]) == 1,   # exactly 1 up electron per site
        WeightedParticleCount([site], [1, 0, 1, 0]) == 1)   # exactly 1 dn electron per site
    # Actually: Sz=+1 means n_up - n_dn = 2, so n_up=2, n_dn=0.
    # Use WeightedParticleCount with weights [dn, up, dn, up] = [-1, 1, -1, 1] == 2.
    basis2 = EagerBasis(h,
        n_fermion(h) == 2,
        WeightedParticleCount([site], [-1, 1, -1, 1]) == 2)
    sys = eigen(Hs2, basis2; n = length(basis2))
    # All eigenvalues should be S(S+1) = 1*(1+1) = 2.
    @test all(e -> isapprox(real(e), 2.0, atol=1e-10), sys.values)
end
