using Test
using MOAD

@testset "ED — validation" begin
    include("test_bh_quspin.jl")
    include("test_holstein_quspin.jl")
end
