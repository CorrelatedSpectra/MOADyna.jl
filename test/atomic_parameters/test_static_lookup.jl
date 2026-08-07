@testset "atomic_parameters — primary form (config string)" begin
    p = atomic_parameters(:Ni, "3d8")
    @test p.Fdd.F2 ≈ 12.234 atol = 1e-3   # canonical Cowan Ni²⁺
    @test p.Fdd.F4 ≈ 7.598 atol = 1e-3
    @test p.zeta.d ≈ 0.083 atol = 1e-3
    @test isnan(p.Fpd.F2)                  # ground row → no 2p-d coupling
    @test isnan(p.zeta.p)
    @test p.configuration == "3d8"
    @test p.provenance === :Haverkort_thesis_2005
    @test p.scaling === :HF
end

@testset "atomic_parameters — sugar form (charge kwarg, ground only)" begin
    # Sugar regression: charge-keyed call returns the same NamedTuple as
    # the primary config form for Ni²⁺ ground.
    p_sugar   = atomic_parameters(:Ni; charge = 2)
    p_primary = atomic_parameters(:Ni, "3d8")
    @test p_sugar.Fdd.F2 == p_primary.Fdd.F2
    @test p_sugar.Fdd.F4 == p_primary.Fdd.F4
    @test p_sugar.zeta.d == p_primary.zeta.d
    @test p_sugar.configuration == p_primary.configuration
end

@testset "atomic_parameters — primary form covers core-hole rows" begin
    # The L_{2,3} intermediate "2p5 3d9" is reachable only via the
    # primary form (sugar is ground-only by design).
    p_l23 = atomic_parameters(:Ni, "2p5 3d9")
    @test p_l23.configuration == "2p5 3d9"
    @test !isnan(p_l23.Fpd.F2)
    @test !isnan(p_l23.zeta.p)
    @test p_l23.provenance === :Haverkort_thesis_2005
end

@testset "atomic_parameters — config normalisation accepts variants" begin
    # All of these should canonicalise to "3d8" and return identical results.
    p_canonical = atomic_parameters(:Ni, "3d8")
    @test atomic_parameters(:Ni, "3d^8").configuration == "3d8"
    @test atomic_parameters(:Ni, "3D8").configuration == "3d8"
    @test atomic_parameters(:Ni, " 3d8 ").configuration == "3d8"
    @test atomic_parameters(:Ni, "3d_8").configuration == "3d8"
    # Same numeric content.
    @test atomic_parameters(:Ni, "3d^8").Fdd.F2 == p_canonical.Fdd.F2

    # Multi-shell: "3d8 2p5" reorders to "2p5 3d8".
    # (Ni "2p5 3d9" exists in the dict — this just exercises normalisation.)
    norm = MOADyna.AtomicParameters._normalize_config("3d9 2p5")
    @test norm == "2p5 3d9"
end

@testset "atomic_parameters — primary form covers 4d block" begin
    p = atomic_parameters(:Pd, "4d8")
    @test p.configuration == "4d8"
    @test p.provenance === :Haverkort_thesis_2005
end

@testset "atomic_parameters — sugar form auto-resolves 3d vs 4d block" begin
    p3d = atomic_parameters(:Cu; charge = 2)
    @test p3d.configuration == "3d9"
    p4d = atomic_parameters(:Pd; charge = 2)
    @test occursin("4d", p4d.configuration)
end

@testset "atomic_parameters — sugar miss points at primary form" begin
    # Ti⁴⁺ ground would be 3d⁰ (closed shell) — not tabulated. The error
    # should point at the primary form and list available ground configs.
    saved_moad = get(ENV, "MOADYNA_COWAN", nothing)
    saved_ttmult = get(ENV, "TTMULT", nothing)
    saved_moad   !== nothing && delete!(ENV, "MOADYNA_COWAN")
    saved_ttmult !== nothing && delete!(ENV, "TTMULT")
    try
        err = nothing
        try
            atomic_parameters(:Ti; charge = 4)
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("ground", err.msg)
        @test occursin("atomic_parameters(:Ti", err.msg)
    finally
        saved_moad   !== nothing && (ENV["MOADYNA_COWAN"] = saved_moad)
        saved_ttmult !== nothing && (ENV["TTMULT"]    = saved_ttmult)
    end
end

@testset "atomic_parameters — primary form rejects unknown config" begin
    saved_moad = get(ENV, "MOADYNA_COWAN", nothing)
    saved_ttmult = get(ENV, "TTMULT", nothing)
    saved_moad   !== nothing && delete!(ENV, "MOADYNA_COWAN")
    saved_ttmult !== nothing && delete!(ENV, "TTMULT")
    try
        err = nothing
        try
            atomic_parameters(:Ni, "5g7")   # nonsense for Ni
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("Static coverage", err.msg)
    finally
        saved_moad   !== nothing && (ENV["MOADYNA_COWAN"] = saved_moad)
        saved_ttmult !== nothing && (ENV["TTMULT"]    = saved_ttmult)
    end
end

@testset "atomic_parameters — malformed config rejected" begin
    @test_throws ArgumentError atomic_parameters(:Ni, "garbage")
    @test_throws ArgumentError atomic_parameters(:Ni, "")
    @test_throws ArgumentError atomic_parameters(:Ni, "3x8")   # bad ℓ letter
end

@testset "atomic_parameters — closed-core tokens are stripped" begin
    # Ground state Ni²⁺ — both should resolve to the same row. The closed
    # 2p⁶ token is spectator core below the open 3d⁸ valence and must not
    # be interpreted as an L_{2,3} core hole.
    p_bare    = atomic_parameters(:Ni, "3d8")
    p_with_2p = atomic_parameters(:Ni, "2p6 3d8")
    # Direct `==` would mis-handle NaN (NaN != NaN); compare field-by-field
    # treating NaN as equal-to-NaN.
    @test isequal(p_bare, p_with_2p)
    # Both must show the closed-shell signature: no 2p hole anywhere.
    @test p_with_2p.configuration == "3d8"
    @test isnan(p_with_2p.Fpd.F2)
    @test isnan(p_with_2p.Gpd.G1)
    @test isnan(p_with_2p.Gpd.G3)
    @test isnan(p_with_2p.zeta.p)
    # Direct stripping check on the normaliser.
    @test MOADyna.AtomicParameters._normalize_config("2p6 3d8") == "3d8"
    @test MOADyna.AtomicParameters._normalize_config("3d10 4f2") == "4f2"
    # Closed shells *above* an open shell are kept (e.g. 2p5 3d10 in
    # Cu-like core-hole intermediates, where 3d is fully occupied but
    # the 2p hole is open).
    @test MOADyna.AtomicParameters._normalize_config("2p5 3d10") == "2p5 3d10"
end

@testset "atomic_parameters — configuration validation rejects malformed input" begin
    # Duplicate shell labels and over-capacity
    # occupancies (s≤2, p≤6, d≤10, f≤14, g≤18) must be rejected before
    # they reach Cowan.
    @test_throws ArgumentError atomic_parameters(:Ni, "3d8 3d2")    # duplicate
    @test_throws ArgumentError atomic_parameters(:Ni, "3d11")       # over-cap d
    @test_throws ArgumentError atomic_parameters(:Pr, "4f15")       # over-cap f
    @test_throws ArgumentError atomic_parameters(:Ni, "2p7")        # over-cap p
    @test_throws ArgumentError atomic_parameters(:Na, "3s3")        # over-cap s
end

@testset "atomic_parameters — scaled_80 reduces F^k>0 and G^k" begin
    raw = atomic_parameters(:Ni, "3d8"; scaling = :HF)
    sc  = atomic_parameters(:Ni, "3d8"; scaling = :scaled_80)
    @test sc.Fdd.F2 ≈ 0.8 * raw.Fdd.F2 atol = 1e-6
    @test sc.Fdd.F4 ≈ 0.8 * raw.Fdd.F4 atol = 1e-6
    @test sc.zeta.d ≈ raw.zeta.d atol = 1e-12   # ζ unscaled
    @test sc.r.r2 ≈ raw.r.r2 atol = 1e-12       # radial moments unscaled
    @test sc.provenance === :Haverkort_thesis_2005
    @test sc.scaling === :scaled_80
end

@testset "atomic_parameters — L23 row scales Gpd and Fpd" begin
    raw = atomic_parameters(:Ni, "2p5 3d9"; scaling = :HF)
    sc  = atomic_parameters(:Ni, "2p5 3d9"; scaling = :scaled_80)
    @test sc.Fpd.F2 ≈ 0.8 * raw.Fpd.F2 atol = 1e-6
    @test sc.Gpd.G1 ≈ 0.8 * raw.Gpd.G1 atol = 1e-6
    @test sc.Gpd.G3 ≈ 0.8 * raw.Gpd.G3 atol = 1e-6
    @test sc.zeta.p ≈ raw.zeta.p atol = 1e-12   # ζ_2p unscaled
end

@testset "atomic_parameters — invalid scaling rejected" begin
    @test_throws ArgumentError atomic_parameters(:Ni, "3d8"; scaling = :weird)
end

@testset "covered_elements / covered_configurations" begin
    elems = covered_elements()
    @test :Ni in elems
    @test :Cu in elems
    @test isa(elems, AbstractSet)

    confs = covered_configurations(:Ni)
    @test isa(confs, AbstractVector{String})
    @test "3d8" in confs
    @test "2p5 3d9" in confs
end
