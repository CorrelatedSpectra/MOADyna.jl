using Test
using MOAD

@testset "Shells" begin
    include("test_parse_tag.jl")
    include("test_shell_model.jl")
    include("test_operators.jl")
    include("test_angular_momentum.jl")
    include("test_interactions.jl")
    include("test_basis_dsl.jl")
    include("test_hubbard_acceptance.jl")
    include("test_rotations.jl")
    include("test_group_projected.jl")
    include("test_onsite_energies.jl")
    include("test_dipole.jl")
    include("test_multipole.jl")
    include("test_slater.jl")
end
