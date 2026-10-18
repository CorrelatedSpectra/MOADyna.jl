# Wannier models and clusters

```@meta
CurrentModule = MOADyna
```

A Wannier Hamiltonian can supply both a periodic band structure and the
one-electron part of an interacting cluster. This tutorial follows that route:
read a hopping model, compare its bands, select a cluster, and add interactions.
The self-contained example is `examples/10_wannier.jl`.

## Read a Wannier calculation

Pass the seedname, Hamiltonian filename, or directory containing one model:

```julia
using MOADyna

model = read_wannier90("calculation/wannier90")
path = read_wannier_path("calculation/wannier90"; lattice=model.lattice)
bands = bandstructure(model, path)
reference = read_wannier_bands("calculation/wannier90"; path)
maximum(abs, bands.energies - reference.energies)
```

Keep the `.win`, `.wout` and `_wsvec.dat` files beside `_hr.dat` when available.
The reader uses their lattice, orbital centres and interpolation settings.
The combined `_tb.dat` format is also supported. Energies stay in eV with the
source energy zero; spin and orbital ordering follow the input file.

Wannier90 interpolation can include centre-dependent Wigner–Seitz shifts.
MOADyna incorporates these shifts and degeneracy weights into the stored
hoppings. If `_wsvec.dat` is absent, the shifts can be reconstructed from the
lattice, mesh and final orbital centres. That reconstruction inherits the
precision of the printed centres. `model.metadata[:interpolation]` records the
route used. A bare hopping file with unknown settings requires an explicit
`interpolation=:plain` or `:ws` choice; see [`read_wannier90`](@ref).

The path reader prefers `_band.kpt` and `_band.labelinfo.dat`. Otherwise it
reconstructs the sampling from `.win`. Supplying that path to the reference-band
reader checks the point count and distances before comparing energies.

## A two-orbital chain

For a runnable example, construct a chain with two orbitals per cell. There is
a hopping within each cell and another between adjacent cells:

```@example wannier
using MOADyna
using LinearAlgebra: Diagonal, I, norm

h0 = [0.0 -1.0; -1.0 2.0]
t = [0.0 0.0; -0.6 0.0]
model = TightBindingModel(
    Dict((0, 0, 0) => h0, (1, 0, 0) => t, (-1, 0, 0) => t');
    lattice=Diagonal([3.0, 5.0, 5.0]),
    centres=[0.0 0.5; 0.0 0.0; 0.0 0.0], labels=["d", "p"])

path = kpath(model, [0.0 0.5; 0.0 0.0; 0.0 0.0];
             intervals=80, labels=["Γ", "X"])
bands = bandstructure(model, path)
size(bands.energies)
```

Rows of `bands.energies` label bands; columns label k points. Lattice vectors
are columns in Å, and the path vertices are fractional reciprocal coordinates.
Individual hopping blocks and Bloch matrices are available directly:

```@example wannier
hopping(model, (1, 0, 0)), bloch_hamiltonian(model, [0.25, 0.0, 0.0])
```

Load Plots to draw the bands:

```@example wannier
using Plots
plot(bands; label="Two-orbital chain", size=(560, 360), linewidth=2,
     guidefontsize=12, tickfontsize=11, legendfontsize=10)
```

## Compare with reference bands

The reference readers return the same [`BandStructure`](@ref) type:

```julia
dft = read_vaspkit_bands("calculation/PBAND_Ni.dat")
# For WIEN2k, the adjacent case.insp supplies the energy unit.
dft_wien = read_wien2k_bands("calculation/case.spaghetti_ene")
```

VASPKIT's `KLABELS` provides tick labels. Projected columns are accessible through
`dft.metadata[:columns]`, for example `dft.metadata[:columns]["dz2"]`.
WIEN2k bands retain the original band indices and source k coordinates in
metadata. When an `.insp` file is unavailable, specify `energy_unit=:Ry` or `:eV`.

Reference files containing only distances can be plotted on their original grid.
They need explicit fractional k coordinates before that grid can be used to
evaluate a new model. The `.eig` file from Wannier90 contains mesh-indexed
eigenvalues and is not a band-path file.

For an overlay, set the energy zero of each dataset from the calculation that
produced it. For example, if the Wannier energies are absolute and the DFT output
is already relative to the same Fermi level:

```julia
fermi_energy = 5.2  # eV; use the value from the calculation
p = plot(bands; energy_zero=fermi_energy, label="Wannier", color=:steelblue)
plot!(p, dft; energy_zero=0.0, label="DFT", color=:black, linestyle=:dash)
```

`energy_zero` shifts the displayed energies without changing the data. Each
curve keeps its own sampling and disconnected sections. Check that the two
calculations use the same physical k path and reciprocal cell before overlaying.

## Select an interacting cluster

Take two adjacent cells of the chain:

```@example wannier
cluster = finite_cluster(model; cells=[(0, 0, 0), (1, 0, 0)])
cluster.orbitals
```

The returned map gives `(cell, orbital)` for each row of the cluster matrix.
An arbitrary fragment can instead be selected with an ordered list such as
`finite_cluster(model, [((0,0,0), 1), ((0,0,0), 2), ((-1,0,0), 2)])`.

Our model contains spatial orbitals, so we explicitly add two spin modes per
orbital. The first orbital carries a local Hubbard interaction:

```@example wannier
sites = [FermionSite{2}(Symbol("orb$i")) for i in 1:4]
hilbert = Hilbert(s.name => s for s in sites)
modes = [(s, σ) for s in sites for σ in 1:2]
H0 = onebody(kron(cluster.hamiltonian, Matrix{Float64}(I, 2, 2)), modes)
H = H0 + 6.0 * n(sites[1], 1) * n(sites[1], 2)

sector = [n_fermion(hilbert) == 4,
          WeightedParticleCount(sites, repeat([1, -1], 4)) == 0]
gs = eigen(H, hilbert; restrictions=sector, n=1)
gs.values[1]
```

This is an ordinary `OperatorSum`, so the existing eigensolvers and spectroscopy
functions apply. For spinor Wannier data, map each input orbital directly to one
fermion mode. For a full-shell multiplet model, establish the correspondence to
the shell's orbital and spin order before adding its Coulomb and transition
operators; the trial projection labels alone do not fix that correspondence.

## Rotate the ligand orbitals

Select the first cluster orbital as a correlated centre. [`ligand_basis`](@ref)
rotates the environment so that its first block contains the combinations
coupled directly to this centre:

```@example wannier
rotated = ligand_basis(cluster, [1])
(direct=rotated.direct, tail=rotated.tail,
 residual=rotated.coupling_residual, tail_coupling=rotated.tail_coupling)
```

The transformation keeps every orbital and leaves the correlated centre
unchanged. `rotated.hamiltonian == rotated.U' * cluster.hamiltonian * rotated.U`
up to rounding; `U` has old orbitals along rows and new orbitals along columns,
as in [`rotate`](@ref). The tail remains part of the Hamiltonian and can couple
to the direct ligand block. Removing it would be an additional approximation.

## Group cells while retaining their connections

A supercell keeps the periodic hopping between clusters as well as the internal
cluster matrix:

```@example wannier
grouped = supercell(model, Diagonal([2, 1, 1]))
internal = hopping(grouped, (0, 0, 0))
connection = hopping(grouped, (1, 0, 0))
(size(internal), norm(connection), grouped.metadata[:orbital_map])
```

`grouped` is another `TightBindingModel`, with the supercell lattice and a map
back to the original orbitals. Its zero-translation block gives the open-cluster
matrix. Nonzero translations retain the connections needed to assemble larger
clusters or evaluate the periodic model in the reduced Brillouin zone.
