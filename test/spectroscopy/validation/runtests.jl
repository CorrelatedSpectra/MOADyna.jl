using Test
using MOADyna

@testset "Spectroscopy — validation" begin
    include("test_nio_xas.jl")
    include("test_sigma_chi_lehmann.jl")
end
