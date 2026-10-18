using LinearAlgebra

@testset "Clusters and supercells" begin
    hoppings = Dict((0, 0) => fill(0.2, 1, 1), (1, 0) => fill(0.3im, 1, 1),
                    (-1, 0) => fill(-0.3im, 1, 1), (0, 1) => fill(-0.4, 1, 1),
                    (0, -1) => fill(-0.4, 1, 1))
    model = TightBindingModel(hoppings; lattice=[2.0 0.7; 0.0 3.0], centres=zeros(2, 1))
    selection = [((0, 0), 1), ((-1, 0), 1), ((0, 1), 1)]
    cluster = finite_cluster(model, selection)
    @test cluster.orbitals == selection
    @test cluster.hamiltonian ≈ [0.2 -0.3im -0.4; 0.3im 0.2 0; -0.4 0 0.2]
    @test cluster.positions ≈ [0.0 -2.0 0.7; 0.0 0.0 3.0]
    site = FermionSite{3}(:cluster)
    hilbert = Hilbert(:cluster => site)
    b1 = basis(hilbert, n_fermion(hilbert) == 1)
    op = onebody(cluster, hilbert)
    @test Matrix(assemble(compile(op, b1), b1)) ≈ cluster.hamiltonian
    @test finite_cluster(model; cells=[(0, 0), (-1, 0)]).hamiltonian ≈ cluster.hamiltonian[1:2, 1:2]
    @test_throws ArgumentError finite_cluster(model, [selection[1], selection[1]])

    S = [2 1; 0 1]
    sc = supercell(model, S)
    @test size(sc) == (2, 2)
    @test sc.lattice ≈ model.lattice * S
    for q in ([0.0, 0.0], [0.23, -0.17])
        k = S' \ q
        primitive = sort([real(bloch_hamiltonian(model, k)[1, 1]),
                          real(bloch_hamiltonian(model, k+[0.5, -0.5])[1, 1])])
        @test eigvals(Hermitian(bloch_hamiltonian(sc, q))) ≈ primitive
    end
    @test finite_cluster(sc).hamiltonian == hopping(sc, (0, 0))
    @test !(hopping(sc, (0, 0)) ≈ bloch_hamiltonian(sc, [0.0, 0.0]))
    @test_throws ArgumentError supercell(model, [1 2; 2 4])
    @test_throws ArgumentError supercell(model, S; representatives=[(0, 0), (2, 0)])
    rounded = TightBindingModel(Dict((0,) => fill(1+1e-7im, 1, 1)); atol=1e-6)
    same = supercell(rounded, reshape([1], 1, 1); atol=1e-6)
    @test hopping(same, (0,)) == hopping(rounded, (0,))

    h0 = ComplexF64[0.2 0.4im; -0.4im -0.3]
    t = ComplexF64[0.1im 0.3+0.2im; -0.5+0.1im 0.2]
    two = TightBindingModel(Dict((0, 0) => h0, (1, 0) => t, (-1, 0) => t');
                            lattice=model.lattice, centres=[0.1 0.65; 0.2 -0.1])
    doubled = supercell(two, S)
    @test hopping(doubled, (0, 0)) ≈ [h0 t; t' h0]
    q = [0.23, -0.17]
    k = S' \ q
    folded = sort(vcat(eigvals(Hermitian(bloch_hamiltonian(two, k))),
                      eigvals(Hermitian(bloch_hamiltonian(two, k+[0.5, -0.5])))))
    @test eigvals(Hermitian(bloch_hamiltonian(doubled, q))) ≈ folded
    expected_centres = hcat([S \ (collect(cell) + two.centres[:, i])
                             for (cell, i) in doubled.metadata[:orbital_map]]...)
    @test doubled.centres ≈ expected_centres
end

@testset "Complete ligand rotation" begin
    h = ComplexF64[1 1 im 0; 1 2 0.2 0.4; -im 0.2 3 0.1im; 0 0.4 -0.1im 4]
    result = ligand_basis(h, [1])
    @test result.U' * result.U ≈ I
    @test result.hamiltonian ≈ result.U' * h * result.U
    @test eigvals(Hermitian(result.hamiltonian)) ≈ eigvals(Hermitian(h))
    @test result.direct == 2:2
    @test result.tail == 3:4
    @test result.coupling_residual < 1e-12
    @test result.tail_coupling > 0.1
    zero_coupling = ligand_basis(Matrix(Diagonal([1.0, 2.0, 3.0])), [1])
    @test isempty(zero_coupling.direct)
    @test eigvals(Hermitian(zero_coupling.hamiltonian)) ≈ [1.0, 2.0, 3.0]

    C = ComplexF64[1 im; 0.3 0.7; 0.2im 0.4]
    h2 = [Matrix{ComplexF64}(I, 2, 2) C'; C Matrix{ComplexF64}(I, 3, 3)]
    aligned = ligand_basis(h2, 1:2)
    q = aligned.U[3:5, aligned.direct]
    @test q' * C ≈ C' * q
    @test minimum(eigvals(Hermitian(q' * C))) > 0
    W = ComplexF64[1 im; im 1] / sqrt(2)
    changed = [Matrix{ComplexF64}(I, 2, 2) W'*C'; C*W Matrix{ComplexF64}(I, 3, 3)]
    transformed = ligand_basis(changed, 1:2)
    @test transformed.U[3:5, transformed.direct] ≈ q * W
    wide = ligand_basis(h2, 3:5)
    @test length(wide.direct) == 2 && isempty(wide.tail)
    @test wide.U' * wide.U ≈ I
end
