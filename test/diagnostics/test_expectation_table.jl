using Test
using LinearAlgebra: dot
using MOADyna
using MOADyna.Diagnostics: expectation_table

@testset "expectation_table — round-trip on a diagonal Hamiltonian" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    b = EagerBasis(h)            # 4 Fock states: |00⟩, |10⟩, |01⟩, |11⟩
    H = 1.0 * n(s, 1) + 2.0 * n(s, 2)
    sys = eigen(H, b; n = 4)

    ops = ["n_1" => n(s, 1), "n_2" => n(s, 2)]
    tbl = expectation_table(sys, b, ops; digits = 3, header = true)

    @test isa(tbl, String)
    @test occursin("n_1", tbl)
    @test occursin("n_2", tbl)
    # Spectrum is {0, 1, 2, 3} (sorted ascending under :SR).
    @test all(occursin(string(round(E; digits = 3)), tbl) for E in (0.0, 1.0, 2.0, 3.0))

    # No header path: still a String, still contains the rounded eigenvalues,
    # but should NOT contain the literal column header "n_1".
    tbl_no_header = expectation_table(sys, b, ops; digits = 3, header = false)
    @test isa(tbl_no_header, String)
    # First non-empty token of the no-header table is the eigenstate index "1".
    @test startswith(lstrip(tbl_no_header), "1")
end

@testset "expectation_table — values match direct computation" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    b = EagerBasis(h)
    # Mix the two number operators so eigenstates aren't pure |n_1, n_2⟩
    # Fock states; here the basis is product of number eigenstates, so the
    # ⟨ψ|n_α|ψ⟩ values are still 0/1, but they exercise the matvec path.
    H = 0.7 * n(s, 1) + 1.3 * n(s, 2)
    sys = eigen(H, b; n = 4)

    ops = ["n_1" => n(s, 1), "n_2" => n(s, 2), "n_tot" => n(s, 1) + n(s, 2)]
    tbl = expectation_table(sys, b, ops; digits = 6, header = true)

    # Recompute directly and confirm each value appears in the table.
    M_n1   = assemble(compile(n(s, 1), b), b)
    M_n2   = assemble(compile(n(s, 2), b), b)
    M_ntot = assemble(compile(n(s, 1) + n(s, 2), b), b)
    direct = Matrix{Float64}(undef, 4, 3)
    for i in 1:4
        ψ = sys.vectors[:, i]
        direct[i, 1] = real(dot(ψ, M_n1   * ψ))
        direct[i, 2] = real(dot(ψ, M_n2   * ψ))
        direct[i, 3] = real(dot(ψ, M_ntot * ψ))
    end

    @test all(occursin(string(round(direct[i, j]; digits = 6)), tbl)
              for i in 1:4, j in 1:3)
    # Sanity: n_tot column equals n_1 + n_2 row by row.
    @test all(direct[:, 3] .≈ direct[:, 1] .+ direct[:, 2])
end
