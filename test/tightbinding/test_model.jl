using LinearAlgebra

@testset "Tight-binding model" begin
    h0 = ComplexF64[0.2 0.4im; -0.4im -0.3]
    t = ComplexF64[0.1im 0.3+0.2im; -0.5+0.1im 0.2]
    lattice = [2.0 0.7; 0.0 3.0]
    centres = [0.1 0.65; 0.2 -0.1]
    hoppings = Dict((0, 0) => h0, (1, -1) => t, (-1, 1) => t')
    model = TightBindingModel(hoppings; lattice, centres, labels=[:a, :b])
    k = [0.17, 0.23]
    phase = cispi(2 * (k[1] - k[2]))
    expected = h0 + phase * t + conj(phase) * t'

    @test size(model) == (2, 2)
    @test bloch_hamiltonian(model, k) ≈ expected
    @test bloch_hamiltonian(model, k) ≈ bloch_hamiltonian(model, k)'
    @test hopping(model, (0, 0)) == h0
    @test iszero(hopping(model, (0, 1)))
    dest = fill(ComplexF64(NaN), 2, 2)
    @test bloch_hamiltonian!(dest, model, k) === dest
    @test dest ≈ expected

    kcart = 2π * (lattice' \ k)
    @test bloch_hamiltonian(model, kcart; coordinates=:cartesian) ≈ expected
    D = Diagonal(cispi.(2 .* (centres' * k)))
    @test bloch_hamiltonian(model, k; gauge=:orbital) ≈ D' * expected * D
    @test eigvals(Hermitian(D' * expected * D)) ≈ eigvals(Hermitian(expected))

    plain = TightBindingModel(hoppings)
    @test bloch_hamiltonian(plain, k) ≈ expected
    @test_throws ArgumentError bloch_hamiltonian(plain, k; gauge=:orbital)
    @test_throws ArgumentError bloch_hamiltonian(plain, k; coordinates=:cartesian)
    @test_throws ArgumentError TightBindingModel(Dict((0,) => h0, (1,) => t))
    @test_throws DimensionMismatch TightBindingModel(hoppings; centres=zeros(2, 3))
    h0[1, 1] = 99
    @test bloch_hamiltonian(model, k) ≈ expected
end
