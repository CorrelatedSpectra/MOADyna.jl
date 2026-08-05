@testset "Spectroscopy DEFAULTS" begin
    using MOAD.Spectroscopy
    using MOAD.Spectroscopy: DEFAULTS, Defaults

    # Save and restore the live DEFAULTS so this test never bleeds into
    # other tests in the suite.
    original = (
        krylovdim          = DEFAULTS.krylovdim,
        reorth             = DEFAULTS.reorth,
        tol                = DEFAULTS.tol,
        Γ_xas              = DEFAULTS.Γ_xas,
        Γ_intermediate     = DEFAULTS.Γ_intermediate,
        Γ_final            = DEFAULTS.Γ_final,
        auto_range_padding = DEFAULTS.auto_range_padding,
        auto_range_density = DEFAULTS.auto_range_density,
    )

    try
        @testset "factory defaults match locked spec" begin
            d = Defaults()
            @test d.krylovdim          == 200
            @test d.reorth             === :none
            @test d.tol                == 1e-10
            @test d.min_iter           == 5
            @test d.max_iter           == 500
            @test d.deflate_tol        == 1e-12
            @test d.Γ_xas              == 1.0
            @test d.Γ_intermediate     == 1.0
            @test d.Γ_final            == 0.05
            @test d.auto_range_padding == 5.0
            @test d.auto_range_density == 3.0
        end

        @testset "Greek field assignment" begin
            DEFAULTS.Γ_xas = 0.7
            @test DEFAULTS.Γ_xas == 0.7
            DEFAULTS.Γ_intermediate = 0.6
            @test DEFAULTS.Γ_intermediate == 0.6
            DEFAULTS.Γ_final = 0.04
            @test DEFAULTS.Γ_final == 0.04
        end

        @testset "ASCII alias assignment writes through to Greek field" begin
            DEFAULTS.Gamma_xas = 0.55
            @test DEFAULTS.Γ_xas == 0.55
            DEFAULTS.Gamma_intermediate = 0.65
            @test DEFAULTS.Γ_intermediate == 0.65
            DEFAULTS.Gamma_final = 0.075
            @test DEFAULTS.Γ_final == 0.075
        end

        @testset "ASCII read raises (canonical Greek read only)" begin
            @test_throws ErrorException DEFAULTS.Gamma_xas
            @test_throws ErrorException DEFAULTS.Gamma_intermediate
            @test_throws ErrorException DEFAULTS.Gamma_final
        end

        @testset "non-Greek fields pass through untouched" begin
            DEFAULTS.krylovdim = 250
            @test DEFAULTS.krylovdim == 250
            DEFAULTS.tol = 1e-9
            @test DEFAULTS.tol == 1e-9
        end

        @testset "unknown field assignment raises" begin
            @test_throws ArgumentError setproperty!(DEFAULTS, :nonexistent, 1.0)
        end

        @testset "unknown field read raises" begin
            @test_throws ErrorException DEFAULTS.nonexistent
        end
    finally
        # Restore live state — order doesn't matter; each is independent.
        DEFAULTS.krylovdim          = original.krylovdim
        DEFAULTS.reorth             = original.reorth
        DEFAULTS.tol                = original.tol
        DEFAULTS.Γ_xas              = original.Γ_xas
        DEFAULTS.Γ_intermediate     = original.Γ_intermediate
        DEFAULTS.Γ_final            = original.Γ_final
        DEFAULTS.auto_range_padding = original.auto_range_padding
        DEFAULTS.auto_range_density = original.auto_range_density
    end
end
