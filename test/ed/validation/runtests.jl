using Test
using MOADyna

@testset "ED — validation" begin
    include("test_bh_quspin.jl")
    include("test_holstein_quspin.jl")
end
