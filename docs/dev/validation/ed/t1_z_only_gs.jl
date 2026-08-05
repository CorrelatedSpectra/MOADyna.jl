# =====================================================================
# T1 — first end-to-end MOAD calculation
# =====================================================================
# 10-mode FermionSite{10} z_only model (same setup as z_only_compare.jl):
# build the Hamiltonian symbolically, enumerate the n_fermion == 8 sector
# as a sorted-vector basis, assemble the sparse matrix, and diagonalize
# (dense, since binomial(10,8) = 45 fits trivially).
#
# Reference: this is a sanity check on the assembled matrix being correct
# Hermitian and producing a sensible GS. Bit-for-bit Quanty cross-check is
# in z_only_compare.jl (operator-algebra layer); the matrix-element check
# here will come once we have a Quanty matrix dump to compare against.

using MOAD
using SparseArrays
using LinearAlgebra: eigvals, eigen, ishermitian, Hermitian

# ----- Parameters (same as z_only_compare.jl) -----
const U          = 6.0
const Delta      = 0.0
const dE_apin    = 0.20
const dE_apout   = 1.14
const t_sigma_in = 1.52
const t_sigma_out= 0.99

const nd  = 2.0
const nL  = 6.0
const dC  = U
const eL = (-0.0 - nd * Delta + nd * dC) / (nd + nL)
const ed = eL + Delta - dC

println("Parameters: ed=$ed, eL=$eL, U=$U")

# ----- Build Hamiltonian (z_only model) -----
s = FermionSite{10}(:s)
hilbert = Hilbert(:s => s)

H_Ni = ed * sum(n(s, m) for m in 1:4)
H_L  = eL * sum(n(s, m) for m in 5:10) +
       dE_apin  * sum(n(s, m) for m in 5:6) +
       dE_apout * sum(n(s, m) for m in 7:10)
H_U  = U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))

hop(i_set, j_set, t) = t * sum(cdag(s, i) * c(s, j) + cdag(s, j) * c(s, i)
                               for (i, j) in zip(i_set, j_set))

H_kin = (
      hop([1, 2], [7, 8],  t_sigma_out)
    - hop([1, 2], [5, 6],  t_sigma_in)
    + hop([3, 4], [5, 6],  t_sigma_in)
    - hop([3, 4], [9, 10], t_sigma_out)
)
H = H_Ni + H_L + H_U + H_kin
println("Hamiltonian: $(length(H)) terms")

# ----- Enumerate n=8 sector basis (Ni²⁺ with full ligand shell) -----
basis = EagerBasis(hilbert, n_fermion(hilbert) == 8)
println("Basis: $(length(basis)) states (expected $(binomial(10, 8)))")
@assert length(basis) == binomial(10, 8)

# ----- Compile and assemble -----
@time H_compiled = compile(H, basis)
@time M = assemble(H_compiled, basis)
println("Sparse matrix: $(size(M)), nnz=$(nnz(M))")

# Sanity: must be Hermitian.
M_dense = Matrix(M)
println("Hermitian? ", ishermitian(M_dense))
@assert ishermitian(M_dense) "Assembled matrix is not Hermitian"

# ----- Diagonalize, report GS energy and lowest few -----
eigs = sort(real.(eigvals(M_dense)))
println("\nLowest 5 eigenvalues:")
for (i, e) in enumerate(eigs[1:min(5, end)])
    println("  E_$i = $(round(e; digits = 8))")
end
println("\nGS = $(round(eigs[1]; digits = 8))")
println("Spectrum width: $(round(eigs[end] - eigs[1]; digits = 4))")
