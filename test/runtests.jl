using Test
using MOADyna

@testset "MOADyna" begin
    @testset "Units" begin
        include("units/test_units.jl")
    end

    @testset "Algebra" begin
        include("algebra/test_sites.jl")
        include("algebra/test_hilbert.jl")
        include("algebra/test_operators.jl")
        include("algebra/test_operator_sum.jl")
        include("algebra/test_canonicalize.jl")
        include("algebra/test_algebra.jl")
        include("algebra/test_observables.jl")
        include("algebra/test_properties.jl")
        include("algebra/test_interactions.jl")
        include("algebra/test_rotate.jl")
        include("algebra/test_operator_io.jl")
    end

    @testset "Bases" begin
        include("bases/test_encoding.jl")
        include("bases/test_basis.jl")
        include("bases/test_restriction.jl")
        include("bases/test_compile.jl")
        include("bases/test_golden.jl")
        include("bases/test_embed.jl")
    end

    @testset "Shells" begin
        include("shells/runtests.jl")
    end

    @testset "QuantyIO" begin
        include("quantyio/test_parser.jl")
        include("quantyio/test_builder.jl")
        include("quantyio/test_wavefunction.jl")
        include("quantyio/test_hubbard_dimer.jl")
    end

    @testset "ED" begin
        include("ed/test_eigen.jl")
        include("ed/test_eigensystem_io.jl")
        include("ed/test_boson_layer.jl")
        include("ed/test_bose_hubbard.jl")
        include("ed/test_hubbard_holstein.jl")
    end

    @testset "PointGroups" begin
        include("pointgroups/test_group.jl")
    end

    include("diagnostics/runtests.jl")

    @testset "AtomicParameters" begin
        include("atomic_parameters/runtests.jl")
    end

    include("responses/runtests.jl")

    @testset "Spectroscopy" begin
        include("spectroscopy/test_types.jl")
        include("spectroscopy/test_defaults.jl")
        include("spectroscopy/test_kwarg_aliases.jl")
        include("spectroscopy/test_pretty_step.jl")
        include("spectroscopy/test_xas.jl")
        include("spectroscopy/test_helpers.jl")
        include("spectroscopy/test_re_broaden_table.jl")
        include("spectroscopy/test_io.jl")
        include("spectroscopy/test_rixs.jl")
        include("spectroscopy/test_fluorescence_yield.jl")
        include("spectroscopy/test_conductivity.jl")
        include("spectroscopy/test_structure_factor.jl")
        include("spectroscopy/test_finite_T_spectra.jl")
        include("spectroscopy/test_kubo.jl")
    end

    @testset "Gradients" begin
        include("gradients/test_affine_model.jl")
        include("gradients/test_groundstate_grad.jl")
        include("gradients/test_resolvent_jvp.jl")
        include("gradients/test_xas_jvp.jl")
        include("gradients/test_degeneracy_sector.jl")
        include("gradients/test_spectrum_pullback.jl")
        include("gradients/test_clustered_measure.jl")
        include("gradients/test_forward_fidelity.jl")
        include("gradients/test_fit_spectrum.jl")
    end

    include("test_aqua.jl")
end
