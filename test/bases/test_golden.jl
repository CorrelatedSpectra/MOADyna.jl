# =====================================================================
# Golden tests G1–G6
# =====================================================================
# Per §8.1 of docs/architecture/02_hilbert.md. Each test pins down compile +
# assemble correctness against a hand-computed reference matrix. These run
# in milliseconds and are mandatory before T1/T2/T3.

using SparseArrays
using LinearAlgebra: eigvals, Diagonal

@testset "golden G1: one-mode number operator" begin
    s = FermionSite{1}(:s)
    h = Hilbert(:s => s)
    b = EagerBasis(h)
    H = n(s, 1)
    M = Matrix(assemble(compile(H, b), b))
    @test M ≈ Diagonal([0.0, 1.0])
end

@testset "golden G2: two-mode hopping" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    H = cdag(s, 1) * c(s, 2) + cdag(s, 2) * c(s, 1)
    b = EagerBasis(h, n_fermion(h) == 1)
    M = Matrix(assemble(compile(H, b), b))
    @test size(M) == (2, 2)
    # Off-diagonal hopping entry: +1 (no Pauli sign for n=1 sector).
    @test M ≈ [0.0 1.0; 1.0 0.0]
end

@testset "golden G3: Hubbard U n_↑ n_↓" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    U = 4.0
    H = U * n(s, 1) * n(s, 2)
    b = EagerBasis(h)
    M = Matrix(assemble(compile(H, b), b))
    # Only |11⟩ has both modes occupied. State indices: 0,1,2,3.
    expected = zeros(4, 4)
    expected[4, 4] = U
    @test M ≈ expected
end

@testset "golden G4: bosonic ladder" begin
    b_site = BosonSite{3}(:b)
    h = Hilbert(:b => b_site)
    H = n_b(b_site)
    b = EagerBasis(h)
    M = Matrix(assemble(compile(H, b), b))
    @test M ≈ Diagonal([0.0, 1.0, 2.0, 3.0])
end

@testset "golden G5: Heisenberg dimer" begin
    s1 = SpinSite{1//2}(:s1)
    s2 = SpinSite{1//2}(:s2)
    h = Hilbert(:s1 => s1, :s2 => s2)
    J = 1.0
    ⋅(a::NTuple{3, OperatorSum}, b::NTuple{3, OperatorSum}) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    H = J * (S(s1) ⋅ S(s2))
    b = EagerBasis(h)
    M = Matrix(assemble(compile(H, b), b))
    eigs = sort(real.(eigvals(M)))
    # Singlet: -3J/4; triplet (×3): +J/4.
    @test eigs[1] ≈ -0.75
    @test eigs[2] ≈ 0.25
    @test eigs[3] ≈ 0.25
    @test eigs[4] ≈ 0.25
    # Hermitian.
    @test M ≈ M'
end

@testset "golden G5b: spin-1 dimer" begin
    s1 = SpinSite{1}(:s1)
    s2 = SpinSite{1}(:s2)
    h = Hilbert(:s1 => s1, :s2 => s2)
    J = 1.0
    ⋅(a::NTuple{3, OperatorSum}, b::NTuple{3, OperatorSum}) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    H = J * (S(s1) ⋅ S(s2))
    b = EagerBasis(h)
    M = Matrix(assemble(compile(H, b), b))
    # Spin-1 dimer: S_total(S_total+1)/2 - S(S+1) on each level.
    # Levels: S_total = 0 (deg 1, E=-2J), 1 (deg 3, E=-J), 2 (deg 5, E=+J).
    eigs = sort(real.(eigvals(M)))
    @test eigs[1] ≈ -2.0
    @test all(eigs[2:4] .≈ -1.0)
    @test all(eigs[5:9] .≈ 1.0)
end

@testset "golden G6: WeightedParticleCount Sz=0 sector" begin
    # 4-mode FermionSite with weights [+1,-1,+1,-1]: Σ w_i n_i == 0.
    # This is the Sz=0 sector with two up modes (bits 0,2) and two down
    # modes (bits 1,3), n_up == n_down.
    s = FermionSite{4}(:s)
    h = Hilbert(:s => s)
    wpc = WeightedParticleCount([s], [1, -1, 1, -1])
    b = EagerBasis(h, wpc == 0)
    @test length(b) == 6   # binomial(2,0)*binomial(2,0) + ... = 1+4+1 = 6

    # Verify each state satisfies the constraint.
    enc = MOADyna.Bases.encoding(b)
    @test all(begin
        bits = digits(Int(get_state(b, i)[1]), base=2, pad=4)
        bits[1] - bits[2] + bits[3] - bits[4] == 0
    end for i in 1:length(b))

    # Build a Hamiltonian that exercises hopping inside this sector:
    # H = c'(s,1)c(s,3) + c'(s,3)c(s,1)  (spin-↑ hopping between modes 1↔3)
    H = cdag(s, 1) * c(s, 3) + cdag(s, 3) * c(s, 1)
    M = Matrix(assemble(compile(H, b), b))
    @test size(M) == (6, 6)
    # Hermitian.
    @test M ≈ M'
end
