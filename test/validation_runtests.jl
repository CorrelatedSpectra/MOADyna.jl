using Test
using MOAD

@testset "MOAD — validation" begin
    @testset "Shells" begin
        include("shells/validation/runtests.jl")
    end
    @testset "Spectroscopy" begin
        include("spectroscopy/validation/runtests.jl")
    end
    @testset "ED" begin
        include("ed/validation/runtests.jl")
    end
    @testset "Responses" begin
        include("responses/validation/runtests.jl")
    end
end
