# =====================================================================
# Bose-Hubbard chain — bosonic on-site interaction
# =====================================================================
#
#   H = -t Σ_<ij> b†_i b_j + h.c.  +  (U/2) Σ_i n_i (n_i - 1)  -  μ Σ_i n_i

using MOADyna

const L = 4
const Nmax = 5      # bosonic occupation cutoff per site
const t_hop = 1.0
const U = 4.0
const μ = 1.0

sites = [BosonSite{Nmax}(Symbol("p$i")) for i = 1:L]
hilbert = Hilbert(s.name => s for s in sites)

# Hopping
H_hop = sum(bdag(sites[i]) * b(sites[i+1]) for i = 1:L-1)
H_hop = -t_hop * (H_hop + H_hop')

# On-site n(n-1) repulsion: n*n - n
H_int = (U/2) * sum(n_b(s) * n_b(s) - n_b(s) for s in sites)

# Chemical potential
H_chem = -μ * sum(n_b(s) for s in sites)

H = H_hop + H_int + H_chem

println("Bose-Hubbard chain L=$L, Nmax=$Nmax, t=$t_hop, U=$U, μ=$μ")
println("Number of distinct terms: ", length(H))
println("\nFirst few terms:")
for (i, term) in enumerate(H)
    i > 6 && break
    println("  ", term)
end
