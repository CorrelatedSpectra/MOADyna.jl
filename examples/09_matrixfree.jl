# Matrix-free ground state, charge response, and determinant selection.
# Run: julia --project=. --threads=4,0 examples/09_matrixfree.jl

using MOADyna
using LinearAlgebra: norm

L, t, U = 4, 1.0, 4.0
sites = [FermionSite{2}(Symbol("s$i")) for i in 1:L]
hilbert = Hilbert(s.name => s for s in sites)
hopping = sum(cdag(sites[i], σ) * c(sites[i+1], σ) for i in 1:L-1, σ in 1:2)
H = -t * (hopping + hopping') + U * sum(n(s, 1) * n(s, 2) for s in sites)
allowed = [n_fermion(hilbert) == L,
           WeightedParticleCount(sites, repeat([1, -1], L)) == 0]
full_basis = basis(hilbert, allowed...)

# The basis is explicit; Hamiltonian entries are generated during multiplication.
H_action = matrixfree(H, full_basis)
gs = eigen(H_action, full_basis; n=1, tol=1e-10)
psi0, Eg = gs.vectors[:, 1], gs.values[1]
println("Sector dimension: ", length(full_basis))
println("Ground-state energy: ", Eg)
println("Eigenstate residual: ", norm(H_action * psi0 - Eg * psi0))

# The sparse backend solves the same problem, useful as a small-system reference.
gs_sparse = eigen(H, full_basis; n=1)
@assert isapprox(Eg, gs_sparse.values[1]; atol=1e-9, rtol=0)

charge = sum((-1)^i * (n(sites[i], 1) + n(sites[i], 2)) for i in 1:L)
ops = [charge]
response = correlator(H_action, full_basis, ops, ops;
                      state=psi0, Eg, Γ=0.3, krylovdim=length(full_basis))
omega = range(0, 12; length=241)
grid = to_grid(response, omega)
broader = to_grid(response, omega; Γ=0.6)
intensity = -imag.(grid.data[1, 1, :]) / π
println("Charge response: ", length(intensity), " frequency points")

# Start selection from the single Néel determinant, then allow the whole sector.
neel_weights = vcat([isodd(i) ? [1, -1] : [-1, 1] for i in 1:L]...)
seed = basis(hilbert, allowed..., WeightedParticleCount(sites, neel_weights) == L)
selected = selected_eigen(H, seed; restrictions=allowed,
                          max_states=length(full_basis), residual_tol=1e-8)
@assert isapprox(selected.eigen.values[1], Eg; atol=1e-8, rtol=0)
println("Selected states: ", length(selected.basis))
println("Full allowed-space residual: ", selected.residual_norm)

# selected.eigen.vectors uses selected.basis, not the original seed coordinates.
# A spectrum needs its own response-space convergence; see selected_response.
