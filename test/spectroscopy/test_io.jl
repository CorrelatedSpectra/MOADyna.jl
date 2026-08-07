@testset "Spectroscopy I/O — save / load round-trip" begin
    using LinearAlgebra
    using MOADyna: xas
    using MOADyna.Spectroscopy: SpectraTensor, save_spectra, load_spectra,
                              re_broaden

    function build_2level()
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        b = EagerBasis(h)
        H = -1.0 * n(s, 1) + 4.0 * n(s, 2)
        H_sp = assemble(compile(H, b), b)
        T = cdag(s, 2) * c(s, 1)
        ψ = zeros(ComplexF64, length(b))
        for i in 1:length(b)
            w = get_state(b, i)[1]
            (w & 0x01) != 0 && (w & 0x02) == 0 && (ψ[i] = 1.0; break)
        end
        return (basis = b, H = H_sp, T = T, ψ = ψ, Eg = -1.0)
    end

    # SpectraTensor round-trip equality (numeric tensor + grid + Eg + key meta).
    # `chunks` is format-dependent (ASCII drops them), so it's checked per-case.
    function _spec_match(loaded, result; atol)
        return loaded isa SpectraTensor &&
               eltype(loaded.tensor) === eltype(result.tensor) &&
               size(loaded.tensor) == size(result.tensor) &&
               isapprox(loaded.tensor, result.tensor; atol = atol) &&
               length(loaded.ω_grid) == length(result.ω_grid) &&
               isapprox(first(loaded.ω_grid), first(result.ω_grid); atol = 1e-12) &&
               isapprox(last(loaded.ω_grid),  last(result.ω_grid);  atol = 1e-12) &&
               loaded.Eg == result.Eg &&
               loaded.metadata[:Γ] == result.metadata[:Γ] &&
               Symbol(loaded.metadata[:function]) === :xas
    end

    @testset "ASCII round-trip: scalar XAS" begin
        sys = build_2level()
        ωs  = -1.0:0.1:8.0
        result = xas(sys.H, sys.basis, sys.T, sys.ψ;
                     ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)

        path = tempname() * ".txt"
        try
            save_spectra(result, path)
            loaded = load_spectra(path)
            @test _spec_match(loaded, result; atol = 1e-7)
            @test loaded.chunks === nothing            # ASCII doesn't carry chunks
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "ASCII round-trip: tensor-form XAS" begin
        sys = build_2level()
        ωs  = -1.0:0.2:8.0
        T_vec = [sys.T, 0.5 * sys.T]
        result = xas(sys.H, sys.basis, T_vec, sys.ψ;
                     ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)
        @test ndims(result.tensor) == 3

        path = tempname() * ".txt"
        try
            save_spectra(result, path)
            loaded = load_spectra(path)
            @test ndims(loaded.tensor) == 3
            @test size(loaded.tensor) == size(result.tensor)
            @test loaded.tensor ≈ result.tensor    atol = 1e-7
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "HDF5 round-trip: scalar XAS preserves chunks" begin
        sys = build_2level()
        ωs  = -1.0:0.1:8.0
        result = xas(sys.H, sys.basis, sys.T, sys.ψ;
                     ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)

        path = tempname() * ".h5"
        try
            save_spectra(result, path)
            @test isfile(path)
            loaded = load_spectra(path)
            @test loaded.tensor ≈ result.tensor   atol = 1e-12
            @test loaded.Eg == result.Eg
            # Chunks preserved + structurally match (α/β/R round-tripped).
            chunk_eq(a, b) =
                a.raw_block_size == b.raw_block_size && a.ψ_index == b.ψ_index &&
                a.ω_in_index == b.ω_in_index && a.n_iter == b.n_iter &&
                a.converged == b.converged && isapprox(a.R, b.R; atol = 1e-12) &&
                length(a.α) == length(b.α) && length(a.β) == length(b.β) &&
                all(isapprox(a.α[k], b.α[k]; atol = 1e-12) for k in 1:length(a.α)) &&
                all(isapprox(a.β[k], b.β[k]; atol = 1e-12) for k in 1:length(a.β))
            @test loaded.chunks !== nothing &&
                  length(loaded.chunks) == length(result.chunks) &&
                  all(chunk_eq(a, b) for (a, b) in zip(loaded.chunks, result.chunks))
            # Round-tripped chunks support re_broaden after load.
            r2 = re_broaden(loaded; Γ = 0.5)
            r2_direct = xas(sys.H, sys.basis, sys.T, sys.ψ;
                            ω_grid = ωs, Γ = 0.5, Eg = sys.Eg)
            @test r2.tensor ≈ r2_direct.tensor   atol = 1e-10
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "HDF5 round-trip: tensor-form XAS" begin
        sys = build_2level()
        ωs  = -1.0:0.2:8.0
        T_vec = [sys.T, 0.5 * sys.T]
        result = xas(sys.H, sys.basis, T_vec, sys.ψ;
                     ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)

        path = tempname() * ".h5"
        try
            save_spectra(result, path)
            loaded = load_spectra(path)
            @test ndims(loaded.tensor) == 3
            @test size(loaded.tensor) == size(result.tensor)
            @test loaded.tensor ≈ result.tensor    atol = 1e-12
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "HDF5 metadata: Symbols / Bools / Nothing round-trip" begin
        using MOADyna.Spectroscopy: re_broaden, polarise

        sys = build_2level()
        ωs  = -1.0:0.5:8.0
        result = xas(sys.H, sys.basis, sys.T, sys.ψ;
                     ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)

        path = tempname() * ".h5"
        try
            save_spectra(result, path)
            loaded = load_spectra(path)

            # Critical: :function must round-trip as Symbol, not String,
            # so downstream helpers' === :xas comparisons work.
            @test loaded.metadata[:function] === :xas
            @test loaded.metadata[:reorth] === :none      # Symbol round-trip
            @test loaded.metadata[:converged] isa Bool
            @test loaded.metadata[:T_is_vector_input] isa Bool
            @test loaded.metadata[:ψ_is_list_input] isa Bool
            # Nothing-valued field round-trips.
            @test loaded.metadata[:restrictions] === nothing

            # Helpers that branch on :function work after load.
            @test polarise(loaded) ≈ -imag.(loaded.tensor)
            r_rebroad = re_broaden(loaded; Γ = 0.4)
            @test r_rebroad.metadata[:function] === :xas
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "format inference + explicit override" begin
        sys = build_2level()
        ωs  = -1.0:0.5:8.0
        result = xas(sys.H, sys.basis, sys.T, sys.ψ;
                     ω_grid = ωs, Γ = 0.2, Eg = sys.Eg)

        path_dat = tempname() * ".dat"
        path_h5  = tempname() * ".h5"
        path_misc = tempname() * ".bogus"
        try
            save_spectra(result, path_dat)
            @test load_spectra(path_dat).tensor ≈ result.tensor   atol = 1e-7

            save_spectra(result, path_h5)
            @test load_spectra(path_h5).tensor ≈ result.tensor   atol = 1e-12

            # Unknown extension → ArgumentError on auto.
            @test_throws ArgumentError save_spectra(result, path_misc)

            # Explicit format works regardless of extension.
            save_spectra(result, path_misc; format = :h5)
            loaded = load_spectra(path_misc; format = :h5)
            @test loaded.tensor ≈ result.tensor   atol = 1e-12
        finally
            for p in (path_dat, path_h5, path_misc)
                isfile(p) && rm(p)
            end
        end
    end
end
