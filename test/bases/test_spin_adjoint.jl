@testset "Adjoints of resolved spin operators" begin
    using LinearAlgebra
    s = SpinSite{1}(:s)
    bs = basis(Hilbert(:s => s))
    operators = [Splus(s, 0), Sminus(s, 0), Sx(s, 0), Sy(s, 0),
                 Splus(s, 1), Sminus(s, -1), Sx(s, 1), Sy(s, -1),
                 (0.2 + 0.3im) * Sx(s, 0) * Sminus(s, 1)]
    errors = [norm(Matrix(matrixfree(op', bs)) - Matrix(matrixfree(op, bs))') for op in operators]
    @test maximum(errors) < 1e-13
    @test all(op -> ishermitian(matrixfree(add_hc(op), bs)), operators)
    @test all(op -> Matrix(matrixfree(op'', bs)) ≈ Matrix(matrixfree(op, bs)), operators)
end
