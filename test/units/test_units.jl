# MOAD.Units — energy conversions. Factors are exact (post-2019 SI), so one
# representative conversion validates the wiring; K→eV also exercises kB_eV.

using Test
using MOAD: convert_energy
using MOAD.Units

@testset "Units — energy conversions" begin
    @test convert_energy(300, :K => :eV) ≈ 300 * Units.kB_eV rtol = 1e-14
end
