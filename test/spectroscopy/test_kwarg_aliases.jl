@testset "Spectroscopy kwarg-alias resolver" begin
    using MOAD.Spectroscopy: _resolve_pair

    @testset "neither specified → default" begin
        @test _resolve_pair(nothing, nothing; default = 1.0,  name = "Γ") == 1.0
        @test _resolve_pair(nothing, nothing; default = :auto, name = "ω_grid") === :auto
    end

    @testset "Greek-only specified" begin
        @test _resolve_pair(0.6, nothing; default = 1.0, name = "Γ") == 0.6
        r = -2.0:0.1:5.0
        @test _resolve_pair(r, nothing; default = :auto, name = "ω_grid") === r
    end

    @testset "ASCII-only specified" begin
        @test _resolve_pair(nothing, 0.6; default = 1.0, name = "Γ") == 0.6
        r = -2.0:0.1:5.0
        @test _resolve_pair(nothing, r; default = :auto, name = "ω_grid") === r
    end

    @testset "both specified → ArgumentError" begin
        @test_throws ArgumentError _resolve_pair(0.6, 0.7;
                                                 default = 1.0, name = "Γ")
        @test_throws ArgumentError _resolve_pair(:auto, -2.0:0.1:5.0;
                                                 default = :auto, name = "ω_grid")
    end

    @testset "default of any type works" begin
        # Sanity: the resolver doesn't care about types.
        @test _resolve_pair(nothing, nothing; default = "fallback", name = "x") == "fallback"
        @test _resolve_pair(:foo, nothing; default = :bar, name = "x") === :foo
    end
end
