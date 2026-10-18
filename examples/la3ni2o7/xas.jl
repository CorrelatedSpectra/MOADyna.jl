# julia --project --threads=8 examples/la3ni2o7/xas.jl la3ni2o7-wannier results
include("model.jl")
using DelimitedFiles

function calculate_xas(datadir, output; krylovdim=600)
    mkpath(output)
    model, orbital_atoms = load_wannier(datadir)
    cluster, positions = bilayer_cluster(model, orbital_atoms)
    h, rotation = reduced_cluster(cluster)
    @info "Double cluster" atoms=size(positions, 2) orbitals=size(cluster.hamiltonian, 1) retained=size(h, 1)
    system = interacting_cluster(h)
    @info "Solving ground sector"
    states = eigen(system.H_gs, system.m; restrictions=system.ground,
                   n=4, tol=1e-8, krylovdim=60)
    @info "Ground energies" energies=states.values
    save_eigensystem(joinpath(output, "ground.h5"), states;
                     basis_id=MOADyna.Bases.basis_id(states.basis))
    @assert states.values[2]-states.values[1] < 1e-7
    @assert states.values[3]-states.values[1] > 1e-6 "increase n to include the complete ground manifold"
    omega = range(-15, 35; length=1001)
    @info "Calculating absorption tensor" krylovdim
    spectrum = xas(system.H_xas, system.T, states;
                   restrictions=system.excited, temperature=0.0, degen_tol=1e-7,
                   omega_grid=omega, Gamma=0.6, krylovdim, reorth=:none)
    save_spectra(spectrum, joinpath(output, "xas.h5"))
    intensity = hcat([-imag.(spectrum.tensor[a, a, :]) ./ π for a in 1:3]...)
    writedlm(joinpath(output, "xas.txt"), hcat(omega, intensity))
    open(joinpath(output, "calculation.toml"), "w") do io
        TOML.print(io, Dict("ground_energies_eV"=>states.values,
            "ground_dimension"=>length(states.basis),
            "cluster_spatial_orbitals"=>size(cluster.hamiltonian, 1),
            "direct_ligands"=>length(rotation.direct),
            "ligand_tail_coupling_eV"=>rotation.tail_coupling,
            "krylovdim"=>krylovdim, "reorth"=>"none", "Gamma_FWHM_eV"=>0.6,
            "ed_eV"=>system.ed, "eL_eV"=>system.eL))
    end
    return spectrum
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("supply the la3ni2o7-wannier data directory")
    calculate_xas(ARGS[1], length(ARGS) >= 2 ? ARGS[2] : "la3ni2o7-results")
end
