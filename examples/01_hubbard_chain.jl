# =====================================================================
# Hubbard chain — 4-site spinful Hubbard model
# =====================================================================
#
# Demonstrates the operator-algebra and basis-construction surface on a
# generic FermionSite{2}-per-lattice-site Hubbard chain. Each site has 2
# fermionic modes, interpreted by the user as spin-up (mode 1) and
# spin-down (mode 2).
#
#   H = -t Σ_<ij>,σ c†_iσ c_jσ  +  h.c.  +  U Σ_i n_i↑ n_i↓

using MOAD
using SparseArrays: nnz

const L     = 4
const t_hop = 1.0
const U     = 4.0
const UP, DN = 1, 2

# Lattice sites and Hilbert space
sites   = [FermionSite{2}(Symbol("s$i")) for i = 1:L]
hilbert = Hilbert(s.name => s for s in sites)

# Hopping: nearest-neighbor, both spins, plus its h.c.
H_hop = sum(c'(sites[i], σ) * c(sites[i+1], σ) for i = 1:L-1, σ in (UP, DN))
H_hop = -t_hop * (H_hop + H_hop')

# On-site Coulomb: U n_↑ n_↓
H_U = U * sum(n(sites[i], UP) * n(sites[i], DN) for i = 1:L)

H = H_hop + H_U

println("Hubbard chain L=$L, t=$t_hop, U=$U")
println("Number of distinct terms in H: ", length(H))
println()
println(H)

# --- Restrict to half-filling and Sz=0 ---
#
# A bare FermionSite{N} carries no spin convention; to express "Sz" on
# fermionic modes we hand the basis explicit weights. With UP=1, DN=2
# per site, the weight pattern (+1, -1) on each site picks up the up
# minus down occupation, which is 2·Sz for spin-1/2 fermions.

sz_weights = repeat([1, -1], L)              # Σ wᵢ nᵢ = 2·Sz_total

restrictions = (
    n_fermion(hilbert) == L,                 # half-filling: L electrons in 2L modes
    WeightedParticleCount(sites, sz_weights) == 0,
)
println("\nRestrictions:")
for r in restrictions
    println("  ", r)
end

basis = EagerBasis(hilbert, restrictions...)
println("\nSector basis: $(length(basis)) states")

# Compile and assemble; hand H_sparse to KrylovKit / Arpack / dense eigen.
H_compiled = compile(H, basis)
H_sparse   = assemble(H_compiled, basis)
println("Sparse Hamiltonian: $(size(H_sparse)), nnz = $(nnz(H_sparse))")
