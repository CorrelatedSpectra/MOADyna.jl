using Test
using MOAD
using MOAD.Algebra: OperatorSum

@testset "density_density — single d-shell, J=0" begin
    m = ShellModel([:Ni_3d])
    op = density_density(m, :Ni_3d; U = 1.0, J = 0.0)
    @test op isa OperatorSum
    # Hermitian
    @test op' == op
    # Non-trivial
    @test length(op.terms) > 0
end

@testset "density_density — single d-shell, J ≠ 0" begin
    m = ShellModel([:Ni_3d])
    op = density_density(m, :Ni_3d; U = 5.0, J = 0.5)
    @test op' == op
end

@testset "density_density — eigenvalues real on a fixed-N sector" begin
    m = ShellModel([:Ni_3d])
    H = density_density(m, :Ni_3d; U = 5.0, J = 0.5)

    # All occupations are eigenvalues of density-density (no off-diagonal terms),
    # so eigenvalues on any fixed-N sector are all real.
    basis = EagerBasis(m.hilbert, n_fermion(m.hilbert) == 4)
    sys = eigen(H, basis; n = length(basis))
    @test all(isreal, sys.values)
end

@testset "density_density — Up override changes the operator" begin
    m = ShellModel([:Ni_3d])
    op_default = density_density(m, :Ni_3d; U = 5.0, J = 0.5)             # Up = 5 - 1 = 4
    op_custom  = density_density(m, :Ni_3d; U = 5.0, J = 0.5, Up = 3.0)   # Up = 3 (overridden)
    @test op_default != op_custom
end

@testset "density_density — convention check on a 2-electron d sector" begin
    # Two electrons on a d-shell with density-density only (no hopping):
    # All eigenvalues are direct configuration energies. Verify a few:
    #
    #   - Intra-orbital pair (one orbital doubly occupied): E = U
    #   - Inter-orbital, parallel spin (e.g., two ↑'s on different orbitals): E = U - 3J
    #   - Inter-orbital, opposite spin: E = U - 2J
    #
    # Multiplicities (5 orbitals, 10 modes, C(10,2) = 45 states):
    #   - 5 intra-orbital states (1 per orbital): E = U
    #   - C(5,2) × 2 σ = 20 parallel-spin pairs: E = U - 3J
    #   - C(5,2) × 2 σ-orderings = 20 opposite-spin pairs: E = U - 2J
    m = ShellModel([:Ni_3d])
    U, J = 5.0, 0.5
    H = density_density(m, :Ni_3d; U = U, J = J)
    basis = EagerBasis(m.hilbert, n_fermion(m.hilbert) == 2)
    sys = eigen(H, basis; n = length(basis))
    evals = sort(real.(sys.values))

    n_intra      = count(e -> abs(e - U)        < 1e-10, evals)
    n_parallel   = count(e -> abs(e - (U - 3J)) < 1e-10, evals)
    n_antiparall = count(e -> abs(e - (U - 2J)) < 1e-10, evals)

    @test n_intra      == 5
    @test n_parallel   == 20
    @test n_antiparall == 20
    @test n_intra + n_parallel + n_antiparall == 45
end

@testset "density_density — error on unknown shell" begin
    m = ShellModel([:Ni_3d])
    @test_throws KeyError density_density(m, :Cu_3d; U = 1.0, J = 0.0)
end

@testset "kanamori — Hermiticity" begin
    m = ShellModel([:Ni_3d])
    H = kanamori(m, :Ni_3d; U = 5.0, J = 0.5)
    @test H' == H
end

@testset "kanamori — equals density_density when J=0" begin
    # When J=0, spin-flip and pair-hop both vanish; kanamori = density_density.
    m = ShellModel([:Ni_3d])
    @test kanamori(m, :Ni_3d; U = 5.0, J = 0.0) == density_density(m, :Ni_3d; U = 5.0, J = 0.0)
end

@testset "kanamori — Up override" begin
    m = ShellModel([:Ni_3d])
    op_default = kanamori(m, :Ni_3d; U = 5.0, J = 0.5)              # Up = 4.0
    op_custom  = kanamori(m, :Ni_3d; U = 5.0, J = 0.5, Up = 3.0)
    @test op_default != op_custom
end

@testset "kanamori — closed-form 2-orbital N=2 spectrum check" begin
    # Two electrons in a d-shell with full Kanamori (U, J), 5 orbitals.
    # C(10,2) = 45 states total. The pair-hopping term mixes intra-orbital
    # doubly-occupied states, splitting them away from the bare U:
    #
    #   - 30 states at E = U - 3J  (parallel-spin S=1 triplets × 10 pairs)
    #   - 14 states at E = U - J   (10 inter-orbital spin singlets +
    #                                4 antisymmetric intra-orbital superpositions)
    #   -  1 state  at E = U + 4J  (symmetric intra-orbital pair-hop superposition)
    # Total: 30 + 14 + 1 = 45 ✓
    #
    # The 5 intra-orbital bare |α↑α↓⟩ states hybridise under pair-hopping
    # (off-diagonal coupling J between all pairs) into a 5×5 block with
    # U on the diagonal and J on all off-diagonals. Its eigenvalues are
    # U + 4J (×1, symmetric) and U − J (×4, antisymmetric).

    m = ShellModel([:Ni_3d])
    U, J = 5.0, 0.5

    H = kanamori(m, :Ni_3d; U = U, J = J)
    basis = EagerBasis(m.hilbert, n_fermion(m.hilbert) == 2)
    sys = eigen(H, basis; n = length(basis))
    evals = sort(real.(sys.values))

    n_at_U_minus_3J = count(e -> abs(e - (U - 3J)) < 1e-10, evals)
    n_at_U_minus_J  = count(e -> abs(e - (U - J))  < 1e-10, evals)
    n_at_U_plus_4J  = count(e -> abs(e - (U + 4J)) < 1e-10, evals)

    @test n_at_U_minus_3J == 30
    @test n_at_U_minus_J  == 14
    @test n_at_U_plus_4J  == 1
    @test n_at_U_minus_3J + n_at_U_minus_J + n_at_U_plus_4J == 45
end

@testset "kanamori — structural match against the standard decomposition" begin
    # Build the standard rotationally-invariant Kanamori Hamiltonian
    # term-by-term, independently of MOAD's builder, and require the two
    # operators to be ==-equal at the OperatorSum level.
    #
    #   H = U Σ_i n_i↑ n_i↓
    #     + (U − 2J) Σ_{i≠j} n_i↑ n_j↓
    #     + (U − 3J) Σ_{i<j, σ} n_iσ n_jσ
    #     + J Σ_{i≠j} c†_i↑ c†_i↓ c_j↓ c_j↑        (pair hopping)
    #     − J Σ_{i≠j} c†_i↑ c_i↓ c†_j↓ c_j↑        (spin flip)
    #
    # See J. Kanamori, Prog. Theor. Phys. 30, 275 (1963), and the review by
    # A. Georges, L. de' Medici and J. Mravlje, Annu. Rev. Condens. Matter
    # Phys. 4, 137 (2013), §2.
    #
    # The mode ordering below follows the m-major convention (2k = dn_k,
    # 2k+1 = up_k) used by Quanty and other multiplet codes.
    # In MOAD's _orbital_mode_pair(ell, m_value), with ms = -ell..ell, the
    # k-th orbital (0-indexed) corresponds to m_value = -ell + k. Both
    # conventions produce the same (dn, up) mode-index pair for orbital k.

    m = ShellModel([:Ni_3d])    # ℓ=2, 5 orbitals
    site = MOAD.Shells.site_of(m, :Ni_3d)
    Norb = 5
    U, J = 5.0, 0.5

    # Mode-pair lookup matching Quanty's 1-indexed loop:
    # orbinds[k] = k-1 (0-indexed orbital), so dn_k = 2*(k-1)+1, up_k = 2*(k-1)+2 in 1-based.
    # For ℓ=2 in MOAD's convention: orbital k has m_value = -2 + (k-1).
    function quanty_pair(k)
        m_value = -2 + (k - 1)   # k = 1..5
        return MOAD.Shells._orbital_mode_pair(2, m_value)
    end

    # Seed accumulator
    H_quanty = 0 * (cdag(site, 1) * c(site, 1))

    # U term
    for k in 1:Norb
        d, u = quanty_pair(k)
        H_quanty += U * cdag(site, d) * c(site, d) * cdag(site, u) * c(site, u)
    end

    # V term: (U-2J) for i ≠ j
    for k_i in 1:Norb, k_j in 1:Norb
        k_i == k_j && continue
        di, ui = quanty_pair(k_i)
        dj, uj = quanty_pair(k_j)
        H_quanty += (U - 2J) * cdag(site, ui) * c(site, ui) * cdag(site, dj) * c(site, dj)
    end

    # V-J term: (U-3J) for i < j, both σ
    for k_i in 1:Norb, k_j in (k_i + 1):Norb
        di, ui = quanty_pair(k_i)
        dj, uj = quanty_pair(k_j)
        # σ = dn (s=0): use d indices
        H_quanty += (U - 3J) * cdag(site, di) * c(site, di) * cdag(site, dj) * c(site, dj)
        # σ = up (s=1): use u indices
        H_quanty += (U - 3J) * cdag(site, ui) * c(site, ui) * cdag(site, uj) * c(site, uj)
    end

    # Pair hopping: J · c†_up_i c†_dn_i c_dn_j c_up_j  for i ≠ j
    for k_i in 1:Norb, k_j in 1:Norb
        k_i == k_j && continue
        di, ui = quanty_pair(k_i)
        dj, uj = quanty_pair(k_j)
        H_quanty += J * cdag(site, ui) * cdag(site, di) * c(site, dj) * c(site, uj)
    end

    # Spin flip: -J · c†_up_i c_dn_i c†_dn_j c_up_j  for i ≠ j
    for k_i in 1:Norb, k_j in 1:Norb
        k_i == k_j && continue
        di, ui = quanty_pair(k_i)
        dj, uj = quanty_pair(k_j)
        H_quanty += (-J) * cdag(site, ui) * c(site, di) * cdag(site, dj) * c(site, uj)
    end

    # MOAD's kanamori
    H_moad = kanamori(m, :Ni_3d; U = U, J = J)

    # Structural equality: both must canonicalize to the exact same OperatorSum.
    @test H_moad == H_quanty
end

@testset "kanamori — error on unknown shell" begin
    m = ShellModel([:Ni_3d])
    @test_throws KeyError kanamori(m, :Cu_3d; U = 5.0, J = 0.5)
end
