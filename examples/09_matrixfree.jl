# Matrix-free ground states and spectra with automatic basis management.
# Sparse storage remains the recommended default for routine spectroscopy.
# Run: julia --project=. --threads=4,0 examples/09_matrixfree.jl

using MOADyna

L, t, U = 4, 1.0, 4.0
sites = [FermionSite{2}(Symbol("s$i")) for i in 1:L]
hilbert = Hilbert(s.name => s for s in sites)
hopping = sum(cdag(sites[i], σ) * c(sites[i+1], σ) for i in 1:L-1, σ in 1:2)
H = -t * (hopping + hopping') + U * sum(n(s, 1) * n(s, 2) for s in sites)
allowed = [n_fermion(hilbert) == L,
           WeightedParticleCount(sites, repeat([1, -1], L)) == 0]

# A complete physical sector, with matrix entries generated during products.
gs = eigen(H, hilbert; restrictions=allowed, backend=:matrixfree, n=1)
println("Sector dimension: ", length(gs.basis))
println("Ground-state energy: ", gs.values[1])

# Adaptive selection starts from an automatic seed and checks omitted states.
selected = eigen(H, hilbert; restrictions=allowed, backend=:matrixfree,
                 adaptive=true, n=1, tol=1e-8)
println("Selected states: ", length(selected.basis))
println("Selected energy: ", selected.values[1])
println("Full allowed-space residual: ", selected.selection.residual_norm)

# Spectroscopy constructs its full response sector from explicit restrictions,
# independently of the selected ground-state determinant list.
charge = sum((-1)^i * (n(sites[i], 1) + n(sites[i], 2)) for i in 1:L)
omega = range(0, 12; length=241)
spec = xas(H, charge, selected; restrictions=allowed,
           omega_grid=omega, Gamma=0.3, krylovdim=36)
broader = re_broaden(spec; Γ=0.6)
intensity = -imag.(spec.tensor) / π
println("Charge response: ", length(intensity), " frequency points")

# Check seed stability for large adaptive calculations: a small residual does
# not certify that an unseeded, disconnected sector has no lower eigenstate.
# For adaptive response-space convergence, see selected_response.
