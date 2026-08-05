using Test
using MOAD

@testset "Responses — validation" begin
    include("test_hubbard_dimer_gf.jl")
    include("test_dimer_finiteT_neutral.jl")
end
