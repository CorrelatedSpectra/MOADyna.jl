# Executes the runnable code blocks in README.md.
#
# The README is the first thing a new user runs, and nothing else in the build
# executes it: `docs/src/*.md` is covered by Documenter's @example blocks and
# `examples/*.jl` by the CI examples job, but README.md was unguarded and its
# NiO block silently rotted (see CHANGELOG 0.3.2).
#
# Convention: a ```julia block that STARTS WITH `using MOADyna` is a complete,
# self-contained script and is executed here. Blocks that do not (the `Pkg.add`
# installation snippet, the Quanty-interop fragment that reads a dump file the
# repo does not ship) are illustrative and skipped.

using Test

function _readme_runnable_blocks(readme)
    normalized = replace(readme, "\r\n" => "\n")
    blocks = [m.captures[1] for m in eachmatch(r"```julia\n(.*?)```"s, normalized)]
    return filter(b -> startswith(strip(b), "using MOADyna"), blocks)
end

@testset "README" begin
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    runnable = _readme_runnable_blocks(readme)

    @test length(runnable) == 2   # quick-start + NiO L2,3 XAS
    lf_readme = replace(readme, "\r\n" => "\n")
    @test all(_readme_runnable_blocks(text) == runnable for text in
              (lf_readme, replace(lf_readme, "\n" => "\r\n")))

    for (i, code) in enumerate(runnable)
        @testset "block $i runs" begin
            sandbox = Module(Symbol("READMEBlock", i))
            @test (Base.include_string(sandbox, code); true)
        end
    end
end
