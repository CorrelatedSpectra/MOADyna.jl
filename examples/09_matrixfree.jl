# Ground state and charge spectrum of a four-site Hubbard chain.
# Run: julia --project=. --threads=4,0 examples/09_matrixfree.jl

using MOADyna

L, t, U = 4, 1.0, 4.0
sites = [FermionSite{2}(Symbol("s$i")) for i in 1:L]
hilbert = Hilbert(s.name => s for s in sites)
hopping = sum(cdag(sites[i], σ) * c(sites[i+1], σ) for i in 1:L-1, σ in 1:2)
H = -t * (hopping + hopping') + U * sum(n(s, 1) * n(s, 2) for s in sites)
sector = [n_fermion(hilbert) == L,
          WeightedParticleCount(sites, repeat([1, -1], L)) == 0]

# Solve in the complete half-filled, Sz=0 sector.
gs = eigen(H, hilbert; restrictions=sector, backend=:matrixfree, n=1)
println("Sector dimension: ", length(gs.basis))
println("Ground-state energy: ", gs.values[1])

# Staggered charge absorption and re-broadening.
charge = sum((-1)^i * (n(sites[i], 1) + n(sites[i], 2)) for i in 1:L)
omega = range(0, 12; length=241)
spec = xas(H, charge, gs; restrictions=sector,
           omega_grid=omega, Gamma=0.3, krylovdim=36)
broader = re_broaden(spec; Γ=0.6)
intensity = -imag.(spec.tensor) / π
println("Charge response: ", length(intensity), " frequency points")

# Grow the ground-state basis adaptively from an automatic initial sample.
selected = eigen(H, hilbert; restrictions=sector, backend=:matrixfree,
                 adaptive=true, n=1, tol=1e-8)
println("Selected states: ", length(selected.basis))
println("Selected energy: ", selected.values[1])
println("Full allowed-space residual: ", selected.selection.residual_norm)
