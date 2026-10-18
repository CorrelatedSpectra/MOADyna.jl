# Tight-binding bands and an interacting cluster.
# Run: julia --project=. examples/10_wannier.jl [path/to/wannier90]

using MOADyna
using LinearAlgebra: Diagonal, I, norm

h0 = [0.0 -1.0; -1.0 2.0]
t = [0.0 0.0; -0.6 0.0]
model = TightBindingModel(
    Dict((0, 0, 0) => h0, (1, 0, 0) => t, (-1, 0, 0) => t');
    lattice=Diagonal([3.0, 5.0, 5.0]),
    centres=[0.0 0.5; 0.0 0.0; 0.0 0.0], labels=["d", "p"])
path = kpath(model, [0.0 0.5; 0.0 0.0; 0.0 0.0]; intervals=80, labels=["Γ", "X"])
bands = bandstructure(model, path)
println("Band-energy range (eV): ", extrema(bands.energies))

cluster = finite_cluster(model; cells=[(0, 0, 0), (1, 0, 0)])
sites = [FermionSite{2}(Symbol("orb$i")) for i in 1:4]
hilbert = Hilbert(s.name => s for s in sites)
modes = [(s, σ) for s in sites for σ in 1:2]
H0 = onebody(kron(cluster.hamiltonian, Matrix{Float64}(I, 2, 2)), modes)
H = H0 + 6.0 * n(sites[1], 1) * n(sites[1], 2)
sector = [n_fermion(hilbert) == 4,
          WeightedParticleCount(sites, repeat([1, -1], 4)) == 0]
gs = eigen(H, hilbert; restrictions=sector, n=1)
println("Cluster ground-state energy (eV): ", gs.values[1])

rotated = ligand_basis(cluster, [1])
println("Direct ligand orbitals: ", rotated.direct, "; remaining tail: ", rotated.tail)
println("Correlated-to-tail coupling (eV): ", rotated.coupling_residual)
grouped = supercell(model, Diagonal([2, 1, 1]))
println("Intercluster hopping norm (eV): ", norm(hopping(grouped, (1, 0, 0))))

# Optionally compare an existing Wannier calculation on its own path.
if !isempty(ARGS)
    seed = only(ARGS)
    imported = read_wannier90(seed)
    imported_path = read_wannier_path(seed; lattice=imported.lattice)
    reconstructed = bandstructure(imported, imported_path)
    reference = read_wannier_bands(seed; path=imported_path)
    println("Maximum Wannier band difference (eV): ",
            maximum(abs, reconstructed.energies - reference.energies))
end
