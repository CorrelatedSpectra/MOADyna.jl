using MOADyna
using LinearAlgebra
using TOML

"""Read the supplied 31-orbital La₃Ni₂O₇ Wannier model and orbital identities."""
function load_wannier(datadir)
    geometry = TOML.parsefile(joinpath(datadir, "geometry.toml"))
    centres = reduce(hcat, geometry["centres_fractional"])
    model = read_wannier90(joinpath(datadir, "model"); centres, interpolation=:ws)
    return model, geometry["orbital_atoms"]
end

"""Select the two Ni sites and the oxygen atoms within `radius` Å of either."""
function bilayer_cluster(model, orbital_atoms; radius=2.5)
    A = model.lattice
    atoms = model.metadata[:atoms].positions
    nickel = atoms[:, 4:5]
    centres = A * model.centres
    images = [Tuple(round.(Int, A \ (atoms[:, orbital_atoms[i]] - centres[:, i])))
              for i in eachindex(orbital_atoms)]

    # Geometric bounds include every periodic image inside the search sphere.
    bounds = radius .* [norm(row) for row in eachrow(inv(A))]
    oxygen = Set{Tuple{NTuple{3,Int},Int}}()
    for site in 6:12, ni in eachcol(nickel)
        centre = A \ (ni - atoms[:, site])
        ranges = [ceil(Int, centre[i]-bounds[i]):floor(Int, centre[i]+bounds[i]) for i in 1:3]
        for cell in Iterators.product(ranges...)
            norm(atoms[:, site] + A*collect(cell) - ni) <= radius &&
                push!(oxygen, (cell, site))
        end
    end
    selection = [(images[i], i) for i in 1:10]
    positions = [nickel[:, 1], nickel[:, 2]]
    for (cell, site) in sort!(collect(oxygen))
        push!(positions, atoms[:, site] + A*collect(cell))
        for i in findall(==(site), orbital_atoms)
            push!(selection, (cell .+ images[i], i))
        end
    end
    return finite_cluster(model, selection), reduce(hcat, positions)
end

"""Keep the ten direct ligand combinations and order each shell as real d harmonics."""
function reduced_cluster(cluster)
    rotated = ligand_basis(cluster, 1:10)
    length(rotated.direct) == 10 || error("this example requires ten independent ligand combinations")
    # WIEN2Wannier: xy,xz,yz,x²−y²,z². ShellModel: xy,yz,z²,xz,x²−y².
    permutation = vcat([1, 3, 5, 2, 4], [6, 8, 10, 7, 9],
                       [11, 13, 15, 12, 14], [16, 18, 20, 17, 19])
    h = rotated.hamiltonian[permutation, permutation]
    return h, rotated
end

"""Add atomic Ni interactions to the twenty-orbital double cluster (energies in eV)."""
function interacting_cluster(h; Udd=6.0, Upd=7.5, Delta=0.5, slater_scale=0.8)
    m = ShellModel([:Ni_2p, :NiA_3d, :NiB_3d, :LA_3d, :LB_3d])
    shells = (:NiA_3d, :NiB_3d, :LA_3d, :LB_3d)
    ed = (20Delta - 7*27Udd) / 35
    eL = (-15Delta + 7*8Udd) / 35
    cubic = copy(h)
    for (indices, mean_energy) in zip((1:5, 6:10, 11:15, 16:20), (ed, ed, eL, eL))
        shift = mean_energy - real(tr(cubic[indices, indices]))/5
        for i in indices
            cubic[i, i] += shift
        end
    end
    transform = zeros(ComplexF64, 40, 40)
    for (i, shell) in enumerate(shells)
        indices = 10i-9:10i
        transform[indices, indices] = to_real(m, shell)
    end
    spherical = transform * kron(cubic, Matrix{Float64}(I, 2, 2)) * transform'
    # Remove only rounding noise from the unitary transformations.
    tolerance = size(spherical, 1) * eps(Float64) * opnorm(spherical, Inf)
    @assert norm(spherical-spherical', Inf) <= tolerance
    spherical = (spherical+spherical')/2
    spherical[abs.(spherical) .< tolerance] .= 0
    modes = [(site_of(m, shell), i) for shell in shells for i in 1:10]
    H_gs = onebody(spherical, modes)
    Fdd = slater_scale .* (12.233, 7.597)
    for shell in (:NiA_3d, :NiB_3d)
        H_gs += coulomb(m, shell; U=Udd, F=Fdd) + 0.083LS(m, shell)
    end
    core_reference = -ed - 7Udd + 8Upd
    H_xas = H_gs + coulomb(m, :Ni_2p, :NiA_3d; U=Upd,
                           F=(slater_scale*7.720,), G=slater_scale .* (5.783, 3.290)) -
             6Upd*n(m, :NiA_3d) + 11.507LS(m, :Ni_2p) + core_reference
    T = dipole(m, :Ni_2p => :NiA_3d)
    ground = [nshells(m, :Ni_2p) == 6, total(m) == 41]
    excited = [nshells(m, :Ni_2p) == 5, total(m) == 41]
    return (; m, H_gs, H_xas, T, ground, excited, ed, eL, core_reference)
end
