using LinearAlgebra

@testset "Band paths" begin
    lattice = [2.0 0.7; 0.0 3.0]
    model = TightBindingModel(Dict((0, 0) => [0.0 0.3im; -0.3im 0.5],
                                  (1, 0) => [-0.4 0.0; 0.0 0.2],
                                  (-1, 0) => [-0.4 0.0; 0.0 0.2]); lattice)
    vertices = [0.0 0.5 0.5 0.0; 0.0 0.0 0.5 0.5]
    path = kpath(model, vertices; labels=["Γ", "X", "M", "Y"],
                 intervals=4, breaks=[3])
    B = 2π * inv(lattice')
    @test size(path.points) == (2, 10)
    @test path.breaks == [6]
    @test path.distance[5] == path.distance[6]
    @test path.distance[end] ≈ norm(B * [0.5, 0.0]) * 2
    @test path.labels == [1 => "Γ", 5 => "X", 6 => "M", 10 => "Y"]
    result = bandstructure(model, path)
    @test result.path === path
    @test size(result.energies) == (2, 10)
    for i in (1, 3, 10)
        a = -0.8cospi(2path.points[1, i])
        d = 0.5 + 0.4cospi(2path.points[1, i])
        centre, split = (a + d)/2, sqrt(((a-d)/2)^2 + 0.3^2)
        @test result.energies[:, i] ≈ [centre-split, centre+split]
    end
    @test_throws ArgumentError kpath(TightBindingModel(Dict((0,) => ones(1, 1))), [0.0 0.5])
    @test_throws ArgumentError bandstructure(model, KPath(nothing, [0.0, 1.0]))
    @test_throws ArgumentError KPath(nothing, [1.0, 0.0])
    @test KPath(nothing, [0.0, 1.0]; breaks=[2]).distance == [0.0, 1.0]
end
