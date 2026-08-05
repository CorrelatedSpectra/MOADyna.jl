@testset "Spectroscopy pretty_step + auto_grid" begin
    using MOAD.Spectroscopy: pretty_step, auto_grid, DEFAULTS

    @testset "pretty_step: snap target to {1, 2, 5}·10^n" begin
        # The spec says: largest value of {1, 2, 5} · 10^n that is ≤ target.
        @test pretty_step(0.333) == 0.2
        @test pretty_step(0.200) == 0.2
        @test pretty_step(0.05)  == 0.05
        @test pretty_step(0.0167) == 0.01
        @test pretty_step(0.01)  == 0.01

        # Larger / smaller magnitudes.
        @test pretty_step(7.0)   == 5.0
        @test pretty_step(15.0)  == 10.0
        @test pretty_step(99.0)  == 50.0
        @test pretty_step(100.0) == 100.0
        @test pretty_step(0.001) == 0.001
        @test pretty_step(2.0)   == 2.0     # boundary (mant == 2.0 → 2.0)
        @test pretty_step(5.0)   == 5.0     # boundary (mant == 5.0 → 5.0)
        @test pretty_step(1.0)   == 1.0
    end

    @testset "pretty_step rejects non-positive target" begin
        @test_throws ArgumentError pretty_step(0.0)
        @test_throws ArgumentError pretty_step(-1.0)
    end

    @testset "auto_grid: snapped to step, ≥ density samples per Γ" begin
        # XAS-like: Γ = 1.0, density = 3 → target = 0.333 → snapped 0.2
        ω = auto_grid(-1.0, 4.0, 1.0)
        @test step(ω) == 0.2
        # padding = 5·Γ = 5 → endpoints snapped outward to 0.2 multiples.
        @test first(ω) ≤ -1.0 - 5.0
        @test last(ω)  ≥  4.0 + 5.0
        # Endpoints land on integer multiples of step.
        @test first(ω) ≈ round(first(ω) / 0.2) * 0.2  atol = 1e-12
        @test last(ω)  ≈ round(last(ω)  / 0.2) * 0.2  atol = 1e-12
        # Each Γ is sampled by ≥ density points.
        @test 1.0 / step(ω) ≥ DEFAULTS.auto_range_density
    end

    @testset "auto_grid with high resolution Γ_final" begin
        ω = auto_grid(-2.0, 6.0, 0.05)
        @test step(ω) == 0.01     # 0.05 / 3 ≈ 0.0167 → snapped 0.01
        @test 0.05 / step(ω) ≥ DEFAULTS.auto_range_density
    end

    @testset "auto_grid kwargs override defaults" begin
        ω1 = auto_grid(-1.0, 1.0, 1.0; padding = 0.0, density = 3.0)
        ω2 = auto_grid(-1.0, 1.0, 1.0; padding = 5.0, density = 3.0)
        @test first(ω1) > first(ω2)            # less padding → tighter grid
        @test last(ω1)  < last(ω2)

        ω_fine = auto_grid(-1.0, 1.0, 1.0; density = 10.0)
        @test step(ω_fine) ≤ step(ω1)          # higher density → finer step
    end

    @testset "auto_grid input validation" begin
        @test_throws ArgumentError auto_grid(-1.0, 1.0, 0.0)        # Γ_axis ≤ 0
        @test_throws ArgumentError auto_grid(-1.0, 1.0, -1.0)
        @test_throws ArgumentError auto_grid(-1.0, 1.0, 1.0; density = 0.0)
        @test_throws ArgumentError auto_grid(-1.0, 1.0, 1.0; padding = -1.0)
        @test_throws ArgumentError auto_grid(2.0, 1.0, 1.0)         # e_min > e_max
    end
end
