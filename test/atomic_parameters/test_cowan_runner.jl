# test/atomic_parameters/test_cowan_runner.jl
#
# Validation-tier tests for the live Cowan runner. They self-skip when
# `MOAD_COWAN` (or `TTMULT`) is unset, so the unit suite stays green on
# machines without a Cowan binary.
#
# Reference values come from R. D. Cowan's RCN Mod 36 output (LANL build,
# user's `Slater_integrals/out36`). Cross-checked against the static
# Haverkort dictionary for shared (element, configuration) entries.

using Test
import MOAD
using MOAD: atomic_parameters, radial_wavefunction, radial_integral

const _MOAD_COWAN_PATH = let
    p = get(ENV, "MOAD_COWAN", get(ENV, "TTMULT", ""))
    isfile(p) ? p : ""
end

@testset "Cowan runner — error path: invalid binary" begin
    err = nothing
    try
        atomic_parameters(:Pr, "4f2";
                          cowan = "/tmp/definitely_not_a_cowan_binary_xyz")
    catch e
        err = e
    end
    @test err isa ArgumentError
    @test occursin("Cowan", err.msg)
end

@testset "Cowan runner — Pr 4f² (vs reference out36, f-shell return shape)" begin
    if _MOAD_COWAN_PATH == ""
        @info "Skipping live Cowan test: MOAD_COWAN / TTMULT not set or invalid"
    else
        p = atomic_parameters(:Pr, "4f2"; cowan = _MOAD_COWAN_PATH)
        # Reference (RCN Mod 36, LANL build):
        #   F²(4f,4f) = 0.8982247 Ry × 13.605693 = 12.222 eV
        #   F⁴(4f,4f) = 0.5634601 Ry × 13.605693 =  7.667 eV
        #   F⁶(4f,4f) = 0.4053223 Ry × 13.605693 =  5.515 eV
        #   ζ(4f)     = 0.00753   Ry × 13.605693 =  0.1024 eV (blume-watson)
        @test isapprox(p.Fff.F2, 12.222; atol = 0.01)
        @test isapprox(p.Fff.F4,  7.667; atol = 0.01)
        @test isapprox(p.Fff.F6,  5.515; atol = 0.01)
        @test isapprox(p.zeta.f,  0.1024; atol = 0.005)
        @test occursin("cowan_", String(p.provenance))
        @test p.scaling === :HF
        # f-shell return shape: Fff/Fdf/Gdf/zeta.{f,d} per the
        # documented schema. Ground 4f² has no 3d hole → Fdf/Gdf NaN.
        @test isnan(p.Fdf.F2)
        @test isnan(p.Fdf.F4)
        @test isnan(p.Gdf.G1)
        @test isnan(p.Gdf.G3)
        @test isnan(p.Gdf.G5)
        @test isnan(p.zeta.d)
    end
end

@testset "Cowan runner — Ni²⁺ 3d⁸ (matches Haverkort static dict)" begin
    if _MOAD_COWAN_PATH == ""
        @info "Skipping live Cowan test: MOAD_COWAN / TTMULT not set or invalid"
    else
        # The static path returns Haverkort 2005 values (Cowan RCN36K HF).
        # Since Ni "3d8" is in the static dict, atomic_parameters() will
        # never reach the Cowan runner for it — call _run_cowan_by_config
        # directly to verify the live runner reproduces the static-dict numbers.
        p_static = atomic_parameters(:Ni, "3d8")
        p_cowan  = MOAD.AtomicParameters._run_cowan_by_config(
            :Ni, "3d8";
            binary = _MOAD_COWAN_PATH,
            scaling = :HF)
        @test isapprox(p_cowan.Fdd.F2, p_static.Fdd.F2; atol = 0.05)
        @test isapprox(p_cowan.Fdd.F4, p_static.Fdd.F4; atol = 0.05)
        @test isapprox(p_cowan.zeta.d, p_static.zeta.d; atol = 0.005)
        # Radial moments should also agree to ~0.01 Å² / Å⁴.
        @test isapprox(p_cowan.r.r2, p_static.r.r2; atol = 0.01)
        @test isapprox(p_cowan.r.r4, p_static.r.r4; atol = 0.01)
        @test occursin("cowan_", String(p_cowan.provenance))
    end
end

@testset "Cowan runner — required-integral missing surfaces KeyError" begin
    # Regression guard: `_shape_to_nested` must not silently default to
    # 0.0 when the parser misses a required Slater integral. Build a
    # mock BlockData with the 3d-3d entries deliberately absent and
    # confirm `_shape_to_nested` throws `KeyError` rather than silently
    # producing F²(3d,3d) = 0.
    BD = MOAD.AtomicParameters.BlockData
    empty_block = BD(
        Dict{Tuple{String, String, Int}, Float64}(),  # fk: empty
        Dict{Tuple{String, String, Int}, Float64}(),  # gk: empty
        Dict{String, Float64}("3d" => 0.05),           # zeta: only 3d
        Dict{Tuple{String, Int}, Float64}(             # rk: 3d only
            ("3d", 2) => 0.5, ("3d", 4) => 0.4),
    )
    @test_throws KeyError MOAD.AtomicParameters._shape_to_nested(
        empty_block, :Ni, 2, :ground, Symbol("3d"), "3d8")
end

@testset "Cowan runner — keep_scratch=false reaps the scratch directory" begin
    # The runner reaps scratch with an explicit `try/finally`, not
    # `mktempdir(cleanup=true)` (which only fires at Julia exit). After a
    # successful call with `keep_scratch=false`, the scratch directory
    # must not exist.
    if _MOAD_COWAN_PATH == ""
        @info "Skipping live Cowan cleanup test: MOAD_COWAN / TTMULT not set"
    else
        # Run with keep_scratch=true to surface the path, then verify a
        # default-cleanup run (keep_scratch=false implicit) doesn't leave
        # one behind. We can't directly inspect the default scratch path
        # (the runner deliberately drops it), so the strategy is:
        #   1. Snapshot the temp-root entries before the call.
        #   2. Call without keep_scratch.
        #   3. Snapshot after the call. The set must be unchanged.
        tmproot = tempdir()
        before = Set(readdir(tmproot))
        atomic_parameters(:Ni, "3d8";
                          cowan = _MOAD_COWAN_PATH)   # static-dict hits first
        # Static dict short-circuits: confirm the runner is exercised
        # via _run_cowan_by_config directly.
        MOAD.AtomicParameters._run_cowan_by_config(
            :Ni, "3d8"; binary = _MOAD_COWAN_PATH, scaling = :HF)
        after = Set(readdir(tmproot))
        # Anything new in tmproot left behind by us is a leak. (Other
        # processes may add entries, but they wouldn't be removed —
        # we only assert no NEW jl_xxx-pattern dirs survive that don't
        # belong to anything else. We use a softer check: the diff size
        # should be 0 OR the new entries should not match `jl_*`
        # patterns we just created.)
        new_entries = setdiff(after, before)
        @test all(!startswith(e, "jl_") for e in new_entries) ||
              isempty(new_entries)

        # Direct keep_scratch=true smoke test: scratch_path is set.
        p_keep = MOAD.AtomicParameters._run_cowan(
            :Ni, 2, :ground;
            binary = _MOAD_COWAN_PATH,
            shell  = Symbol("3d"),
            scaling = :HF,
            keep_scratch = true)
        @test haskey(p_keep, :scratch_path)
        @test isdir(p_keep.scratch_path)
        # Clean up after ourselves.
        rm(p_keep.scratch_path; force = true, recursive = true)
    end
end

@testset "atomic_parameters — keep_scratch passes through to runner" begin
    # The public `atomic_parameters` accepts the `keep_scratch` kwarg so
    # the diagnostic in `_missing_required` is actionable through the
    # public API.
    if _MOAD_COWAN_PATH == ""
        @info "Skipping keep_scratch passthrough test: MOAD_COWAN / TTMULT not set"
    else
        # Use Pr 4f² which is *not* in the static dict, so the call is
        # forced through the live Cowan runner.
        p = atomic_parameters(:Pr, "4f2";
                              cowan = _MOAD_COWAN_PATH,
                              keep_scratch = true)
        @test haskey(p, :scratch_path)
        @test isdir(p.scratch_path)
        # The scratch tree should contain the in36 deck and out36 output.
        @test isfile(joinpath(p.scratch_path, "in36"))
        @test isfile(joinpath(p.scratch_path, "out36"))
        # Cleanup.
        rm(p.scratch_path; force = true, recursive = true)
    end
end

@testset "Cowan runner — Pr label defaults to spectroscopy convention" begin
    # Pr III vs Pr IV ambiguity is a non-issue — the
    # configuration is the lookup key and the Roman label is purely
    # cosmetic (Cowan ignores it numerically). Default is the
    # spectroscopy convention (`ion_stage = charge + 1`), so Pr³⁺ →
    # "Pr IV". The `ion_label` kwarg overrides only the deck label.
    deck_default = MOAD.AtomicParameters._build_in36(:Pr, 3, "4f2")
    deck_override = MOAD.AtomicParameters._build_in36(
        :Pr, 3, "4f2"; ion_label = "Pr III")
    @test occursin("Pr IV", deck_default)
    @test occursin("Pr III", deck_override)
    # The configuration token is identical in both decks (numerics
    # depend only on Z and config_str).
    @test occursin("4f2", deck_default)
    @test occursin("4f2", deck_override)
end

@testset "radial_wavefunction — Ni 3d⁸ (P(r) from tape2n; nIXS Bessel moments)" begin
    if _MOAD_COWAN_PATH == ""
        @info "Skipping live Cowan radial test: MOAD_COWAN / TTMULT not set or invalid"
    else
        rw = radial_wavefunction(:Ni, "3d8"; cowan = _MOAD_COWAN_PATH)
        # The 641-point RCN log mesh (Bohr), ending at r(mesh) ≈ 1910.724.
        @test length(rw.r) == 641
        @test isapprox(rw.r[end], 1910.724; atol = 0.01)
        @test issorted(rw.r)
        # All bound orbitals returned; 3d is the valence shell.
        @test Set(keys(rw.P)) == Set(["1s","2s","2p","3s","3p","3d"])
        P3d = rw.P["3d"]
        @test length(P3d) == 641
        # P_nl = r·R_nl is normalised: ∫P² dr = 1.
        @test isapprox(radial_integral(P3d, P3d, rw.r, 0; kind = :power, weight = :reduced),
                       1.0; atol = 1e-3)
        # nIXS Bessel moments at q = 4.5/a₀ — match the tabulated HF values
        # to 3 decimals (the whole point: a Cowan-native radial, no external data).
        @test isapprox(radial_integral(P3d, P3d, rw.r, 0; kind=:bessel, q=4.5, weight=:reduced),
                       0.070; atol = 0.002)
        @test isapprox(radial_integral(P3d, P3d, rw.r, 2; kind=:bessel, q=4.5, weight=:reduced),
                       0.161; atol = 0.002)
        @test isapprox(radial_integral(P3d, P3d, rw.r, 4; kind=:bessel, q=4.5, weight=:reduced),
                       0.086; atol = 0.002)
    end
end

@testset "radial_wavefunction — error path: no Cowan binary" begin
    err = nothing
    try
        radial_wavefunction(:Ni, "3d8"; cowan = "/tmp/definitely_not_a_cowan_binary_xyz")
    catch e
        err = e
    end
    @test err isa ArgumentError
end

@testset "radial parser — synthetic tape2n (no Cowan; CI coverage of the contract)" begin
    mesh, kmsh, ncsp = 5, 8, 2
    r_mesh = 4.0
    rfull  = vcat([0.0, 1.0, 2.0, 3.0, 4.0], zeros(kmsh - mesh))   # r[mesh+1:] = 0
    # ru is the potential array (not a mesh): negative and non-monotonic, so
    # the kmsh search cannot false-match on it (mirrors real Cowan output).
    rufull = vcat([-3.0, -1.0, -2.0, -0.5, -1.5], zeros(kmsh - mesh))
    pnl1s  = vcat([1.0, 2.0, 3.0, 4.0, 5.0], zeros(kmsh - mesh))
    pnl2s  = vcat([0.5, 1.5, 2.5, 3.5, 4.5], zeros(kmsh - mesh))
    # Fortran unformatted record: <hdr> r ru pnl(1s) pnl(2s) iw6, framed by
    # 4-byte little-endian length markers. Header content is opaque (the
    # parser is tail-anchored).
    payload = IOBuffer()
    write(payload, zeros(UInt8, 24))            # opaque header
    write(payload, rfull); write(payload, rufull)
    write(payload, pnl1s); write(payload, pnl2s)
    write(payload, Int32(7))                    # iw6
    pbytes = take!(payload)
    file = IOBuffer()
    write(file, Int32(length(pbytes))); write(file, pbytes); write(file, Int32(length(pbytes)))
    path = tempname() * ".tape2n"
    write(path, take!(file))

    fake_out36 = """
     Synthetic   nconf=  1    z= 28    ncores=  1    nvales=  1    ion= 2
    0rcn mod 36  ihf1= 2    mesh= $mesh   idb= $mesh   rdb=$r_mesh   r(mesh)=$r_mesh     irel= 1
      nl   wnl         ee
      1s    2.    -100.00000    1.0
      2s    2.     -10.00000    1.0
    """
    res = MOAD.AtomicParameters._parse_cowan_radial(fake_out36, path)
    @test res.r ≈ [0.0, 1.0, 2.0, 3.0, 4.0]
    @test Set(keys(res.P)) == Set(["1s", "2s"])
    @test res.P["1s"] ≈ [1.0, 2.0, 3.0, 4.0, 5.0]
    @test res.P["2s"] ≈ [0.5, 1.5, 2.5, 3.5, 4.5]

    # irel≥3 (fully relativistic) must be rejected with a clear error — the
    # tail layout differs (qnl present). Reuse the same binary; only out36 meta
    # changes. The irel guard fires before tape2n is even read.
    rel_out36 = replace(fake_out36, "irel= 1" => "irel= 4")
    @test_throws ErrorException MOAD.AtomicParameters._parse_cowan_radial(rel_out36, path)

    rm(path; force = true)
end
