@testset "atomic_parameters — static miss + no Cowan env → ArgumentError with hints" begin
    # Pr (4f) is not in the 3d/4d Haverkort static dict.
    # Guard against MOADYNA_COWAN / TTMULT being set in the user's environment
    # for this test by temporarily unsetting both.
    saved_moad = get(ENV, "MOADYNA_COWAN", nothing)
    saved_ttmult = get(ENV, "TTMULT", nothing)
    saved_moad   !== nothing && delete!(ENV, "MOADYNA_COWAN")
    saved_ttmult !== nothing && delete!(ENV, "TTMULT")
    try
        err = nothing
        try
            atomic_parameters(:Pr, "4f2")
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("MOADYNA_COWAN", err.msg) || occursin("TTMULT", err.msg)
        @test occursin("Static coverage", err.msg) || occursin("Haverkort", err.msg)
    finally
        saved_moad   !== nothing && (ENV["MOADYNA_COWAN"] = saved_moad)
        saved_ttmult !== nothing && (ENV["TTMULT"]    = saved_ttmult)
    end
end

@testset "atomic_parameters — invalid Cowan binary errors clearly" begin
    # The Cowan runner is fully implemented now; an invalid binary path
    # surfaces as an ArgumentError naming the missing file.
    err = nothing
    try
        atomic_parameters(:Pr, "4f2";
                          cowan = "/tmp/nonexistent_cowan_binary")
    catch e
        err = e
    end
    @test err isa ArgumentError
    @test occursin("Cowan", err.msg)
    @test occursin("not found", err.msg) || occursin("MOADYNA_COWAN", err.msg)
end

@testset "atomic_parameters — TTMULT env fallback engages the Cowan path" begin
    # Smoke-test the env resolution: TTMULT alone should drive _run_cowan
    # (which surfaces an ArgumentError on the bogus binary). MOADYNA_COWAN
    # takes precedence; clear it for this test.
    saved_moad = get(ENV, "MOADYNA_COWAN", nothing)
    saved_ttmult = get(ENV, "TTMULT", nothing)
    saved_moad !== nothing && delete!(ENV, "MOADYNA_COWAN")
    ENV["TTMULT"] = "/tmp/nonexistent_ttmult_binary"
    try
        err = nothing
        try
            atomic_parameters(:Pr, "4f2")
        catch e
            err = e
        end
        # Should fall through to the Cowan runner (which then errors on
        # the bogus binary), proving TTMULT was picked up.
        @test err isa ArgumentError
        @test occursin("Cowan", err.msg)
    finally
        saved_moad   !== nothing && (ENV["MOADYNA_COWAN"] = saved_moad)
        if saved_ttmult === nothing
            delete!(ENV, "TTMULT")
        else
            ENV["TTMULT"] = saved_ttmult
        end
    end
end
