# From Wannier bands to bilayer XAS

```@meta
CurrentModule = MOADyna
```

This tutorial constructs a La₃Ni₂O₇ double cluster from a Wannier Hamiltonian,
then calculates its Ni L-edge absorption. It uses the tetragonal structure and
DFT inputs associated with [Chen et al., *Nature Communications* **15**, 9597
(2024)](https://doi.org/10.1038/s41467-024-53863-5). The hoppings and crystal-field
splittings come directly from DFT. The interaction strengths and charge-transfer
energy are separate model inputs.

## Input files and scripts

[Download the Wannier and DFT inputs](https://github.com/CorrelatedSpectra/MOADyna.jl/releases/download/v0.3.5/la3ni2o7-wannier.tar.gz)
(1.62 MiB) and unpack them:

```sh
tar -xzf la3ni2o7-wannier.tar.gz
```

The archive contains the 31-orbital hopping table, Wannier input and band output,
WIEN2Wannier projection definitions, crystal structure, and original DFT bands
and k path. Its README records their provenance and coordinate conventions.
The data is distributed separately from the package source.

The complete scripts are in
[`examples/la3ni2o7`](https://github.com/CorrelatedSpectra/MOADyna.jl/tree/main/examples/la3ni2o7).
From a MOADyna checkout, with Plots installed in the active environment, run:

```sh
julia --project=. --threads=4 examples/la3ni2o7/bands.jl la3ni2o7-wannier results
julia --project=. --threads=8 examples/la3ni2o7/xas.jl la3ni2o7-wannier results
julia --project=. examples/la3ni2o7/plot_xas.jl results
```

The band calculation is small. The full interacting cluster has 658,008 ground-sector
determinants and 548,340 core-excited determinants; allow several GB of memory
for its sparse matrices and Krylov vectors.

## Read the Wannier model and structure

The helper file defines the material-specific selection and orbital mapping:

```julia
include("examples/la3ni2o7/model.jl")
datadir = "la3ni2o7-wannier"
model, orbital_atoms = load_wannier(datadir)

size(model)                       # (31, 31)
model.lattice                     # primitive lattice vectors, columns in Å
model.metadata[:atoms].positions  # atomic positions from model.win, columns in Å
```

There are ten Ni d and 21 O p orbitals per primitive cell. `orbital_atoms` maps
each orbital to its atom in the original projection definitions. The atom labels
in this input happen to be integers; the importer preserves them as strings.

The centre file wraps Wannier centres into the home cell. The archive therefore
also supplies their unwrapped representatives in `geometry.toml`, which
`load_wannier` passes explicitly to [`read_wannier90`](@ref). These representatives
must agree with the translation indices in the hopping table. Wannier centres
can be displaced from nuclei; atomic positions select the fragment, while the
orbital map and cell representatives identify its Hamiltonian entries.

## Compare the bands

First reproduce the stored Wannier interpolation:

```julia
seed = joinpath(datadir, "model")
path = read_wannier_path(seed; lattice=model.lattice)
reference = read_wannier_bands(seed; path)
calculated = bandstructure(model, path)
maximum(abs, calculated.energies - reference.energies)
# approximately 3.8e-5 eV
```

The DFT calculation has a different sampling of the same path. `bands.jl` reads
its original `dft.klist_band`, converts the rational coordinates from the
conventional tetragonal reciprocal cell to Cartesian coordinates, and then to
the primitive reciprocal cell used by the Wannier Hamiltonian:

```math
\mathbf k = B_{\mathrm{conv}}\mathbf q,\qquad
\boldsymbol\kappa = A_{\mathrm{prim}}^T\mathbf k/(2\pi).
```

It evaluates MOADyna bands on these exact 201 DFT points. The stored Wannier
curves retain their own 1,546-point grid. All three use the original energy zero;
no alignment shift is fitted.

![DFT bands and Wannier interpolations](../assets/la3ni2o7_bands.svg)

[Download the band figure (PDF)](../assets/la3ni2o7_bands.pdf).

The near-coincident teal curves and open reference markers check the Wannier
import. Markers display a subset of the stored Wannier points; the numerical
comparison above uses all of them. Comparing with the grey DFT curves tests
the chosen Wannier subspace and interpolation.
DFT bands outside that subspace remain visible.

## Select two octahedra

The two Ni sites lie in adjacent layers and share one apical oxygen. Selecting
oxygen atoms within 2.5 Å of either Ni gives eleven distinct O sites:

```julia
cluster, positions = bilayer_cluster(model, orbital_atoms; radius=2.5)
size(positions, 2)             # 13 atoms: 2 Ni + 11 O
size(cluster.hamiltonian)      # (43, 43): 10 d + 33 p spatial orbitals
```

The helper enumerates periodic images inside the two spheres. For each selected
atom, it adds its explicitly assigned orbitals to an ordered `(cell, orbital)`
list and calls [`finite_cluster`](@ref). All hoppings between selected orbitals
are retained. Connections to omitted atoms are removed at the open boundary.
Increasing `radius` includes a larger oxygen neighbourhood while keeping the
same two Ni sites.

## Construct the ligand orbitals

Write the finite Hamiltonian as

```math
h=\begin{pmatrix}h_d&C^\dagger\\C&h_O\end{pmatrix}.
```

The columns of ``C`` describe which oxygen combinations couple directly to each
Ni d orbital. This ``33\times10`` matrix has rank ten, so its range supplies ten
spatial ligand orbitals. Both Ni shells must be treated together because they
share oxygen states:

```julia
rotation = ligand_basis(cluster, 1:10)
rotation.direct                    # 11:20
rotation.tail                      # 21:43
rotation.coupling_residual          # d-to-tail coupling, near numerical zero
rotation.tail_coupling              # ligand-to-tail coupling, about 1.51 eV
```

For full column rank, [`ligand_basis`](@ref) uses the polar factor
``Q=C(C^\dagger C)^{-1/2}``. Its columns follow the ordered d orbitals and the
direct coupling ``Q^\dagger C`` is Hermitian positive definite. The Ni orbitals
remain unchanged. The full returned transformation is unitary and retains every
oxygen orbital.

For the interacting model we keep the two d shells and these ten direct ligands:

```julia
h, rotation = reduced_cluster(cluster)
size(h)                            # (20, 20)
```

This last projection discards 23 oxygen combinations. They have no direct d
coupling, but they still couple to the retained ligands. Removing them is a
model approximation, and the coupling above is not negligible by construction.
A larger oxygen neighbourhood or additional ligand levels can be used to
investigate its effect.

The ideal dimer has D₄h symmetry about the shared oxygen; an individual Ni site
has C₄v symmetry. The retained Hamiltonian respects the dimer symmetry to
numerical precision. The site-labelled ligand orbitals can be combined into
even and odd dimer combinations when a parity basis is useful.

## Add interactions and set the charge-transfer energy

`reduced_cluster` puts each d-like shell in MOADyna's cubic order
``(xy,yz,z^2,xz,x^2-y^2)``. `interacting_cluster` adds spin and transforms the
one-electron matrix to spherical harmonics with [`to_real`](@ref), so the atomic
Coulomb, spin-orbit and dipole operators all use the same basis.

```julia
system = interacting_cluster(h; Udd=6.0, Upd=7.5, Delta=0.5, slater_scale=0.8)
```

The two Ni shells carry atomic Slater interactions and 3d spin-orbit coupling.
The ligand orbitals are noninteracting. The higher Slater integrals are scaled
to 80% of the atomic values used in the paper; 3d and 2p spin-orbit strengths
are 0.083 and 11.507 eV. Assigning atomic interactions to Ni-centred Wannier
orbitals is the usual effective-shell approximation.

We preserve the DFT hopping matrices and each shell's traceless onsite matrix.
The average d and ligand energies are replaced by ``\epsilon_d`` and
``\epsilon_L`` to impose

```math
\Delta = E(d^8d^8\underline L)-E(d^7d^8)
       = \epsilon_d-\epsilon_L+7U_{dd}.
```

Here a full ligand shell has 20 electrons. Choosing the ``d^7d^8L^{20}``
configuration centroid as zero also gives
``15\epsilon_d+20\epsilon_L+49U_{dd}=0``. These equations define the shell
means without altering the DFT crystal-field splittings. Configuration centroids
refer to the monopole energies, rather than individual multiplet levels.

For absorption on Ni A, an explicit 2p core hole changes the Hamiltonian by

```math
H_{\mathrm{XAS}}=H_{\mathrm{GS}}+H_{pd}
 -6U_{pd}N_{d,A}+\zeta_{2p}\mathbf l\cdot\mathbf s+C_0.
```

The subtraction removes the filled-core contribution already included in the
ground-state levels; the spectator Ni keeps its original levels. The full
``H_{pd}`` retains the core-valence multiplets. With
``C_0=-\epsilon_d-7U_{dd}+8U_{pd}``, the reference
``p^6d^7d^8L^{20}\to p^5d^8d^8L^{20}`` centroid transition is at zero.
The plotted energy is relative to this reference; an absolute edge position is
not determined by this model.

## Calculate absorption

There are 35 valence electrons initially and 36 after absorption. The example
keeps every determinant in those sectors:

```julia
states = eigen(system.H_gs, system.m; restrictions=system.ground,
               n=4, tol=1e-8, krylovdim=60)
@assert states.values[2] - states.values[1] < 1e-7
@assert states.values[3] - states.values[1] > 1e-6
omega = range(-15, 35; length=1001)
spec = xas(system.H_xas, system.T, states;
           restrictions=system.excited, temperature=0.0, degen_tol=1e-7,
           omega_grid=omega, Gamma=0.6, krylovdim=600, reorth=:none)
```

The lowest Kramers pair is averaged at zero temperature. The next pair checks
that the ground manifold is complete. `Gamma` is a Lorentzian full width at
half maximum in eV. The calculation returns the full Cartesian absorption
tensor, so choosing another polarization does not require another solve:

```julia
in_plane = (polarise(spec, [1, 0, 0]) + polarise(spec, [0, 1, 0])) / (2π)
out_of_plane = polarise(spec, [0, 0, 1]) / π
```

![Bilayer Ni L-edge absorption](../assets/la3ni2o7_xas.svg)

[Download the absorption figure (PDF)](../assets/la3ni2o7_xas.pdf).

The curves show absorption at one Ni site. The inversion-related second site
has the same linear-polarization spectrum; summing both sites doubles the
intensity. The example uses raw DFT hopping parameters, so this is an
illustrative cluster calculation rather than the fitted spectrum in the paper.

RIXS uses the same Hamiltonians, ground states and dipoles. For example, a
single incident energy near the main L₃ peak can be calculated with:

```julia
loss = rixs(system.H_gs, system.H_xas, system.T, system.T, states;
            restrictions_final=system.ground,
            restrictions_intermediate=system.excited,
            omega_in_grid=range(-5.6; step=1.0, length=1),
            omega_out_grid=range(0, 8; length=401),
            Gamma_intermediate=0.6, Gamma_final=0.1, krylovdim=600, reorth=:none,
            temperature=0.0, degen_tol=1e-7)
```

Both transition arguments use the absorption orientation; `rixs` applies the
outgoing adjoint internally. See [NiO: XAS, RIXS, and nIXS](@ref) for polarization
contractions and plotting. This longer calculation is not run by `xas.jl`.
Its final-state linewidth requires a separate Krylov-convergence check.
