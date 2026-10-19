using LinearAlgebra

@testset "One-body matrix conversion" begin
    s = FermionSite{3}(:orb)
    hilbert = Hilbert(:orb => s)
    b1 = basis(hilbert, n_fermion(hilbert) == 1)
    h = ComplexF64[1 2im 0.3; 0.7 0 -0.2im; 0 0.4 -1]
    op = onebody(h, hilbert)
    @test Matrix(assemble(compile(op, b1), b1)) ≈ h
    @test onebody(h, [(s, 1), (s, (2,)), (s, 3)]) == op
    @test_throws ArgumentError onebody(ones(2, 2), [(s, 1), (s, (1,))])
    @test_throws DimensionMismatch onebody(h, [(s, 1), (s, 2)])
    U = ComplexF64[1 0 0; 0 inv(sqrt(2)) im/sqrt(2); 0 im/sqrt(2) inv(sqrt(2))]
    @test rotate(op, hilbert, U) ≈ onebody(U' * h * U, hilbert)
end
