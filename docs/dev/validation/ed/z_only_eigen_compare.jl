# =====================================================================
# z_only Eigensystem cross-check: MOADyna vs Quanty
# =====================================================================
#
# Phase-0 numerical validation of the full pipeline:
#
#   Quanty: Eigensystem on the z_only Hamiltonian (n_fermion=8 sector)
#           → z_only_quanty_eigvals.txt
#           → z_only_quanty_eigvecs.txt
#
#   MOADyna:   read_quanty_operator → compile → assemble → eigen
#           ≈
#           read_quanty_eigenvalues + read_quanty_wavefunctions
#
# Eigenvalues are compared to 1e-10. Eigenvectors are compared via
# projector Frobenius norm (eigvecs are defined only up to phase on
# non-degenerate levels and up to rotation within degenerate clusters).
#
# To regenerate the Quanty side:
#   cd docs/dev/validation
#   /path/to/Quanty z_only_eigen.Quanty
#
# Then run this script from the repo root:
#   julia --project=. docs/dev/validation/z_only_eigen_compare.jl

using MOADyna
using LinearAlgebra: norm, eigvals, Hermitian
using SparseArrays: nnz
using Printf

# ----- Paths --------------------------------------------------------------
const VALDIR = @__DIR__
const QUANTY_OP_DUMP    = joinpath(VALDIR, "z_only_quanty_output.txt")
const QUANTY_EIGVALS    = joinpath(VALDIR, "z_only_quanty_eigvals.txt")
const QUANTY_EIGVECS    = joinpath(VALDIR, "z_only_quanty_eigvecs.txt")

for p in (QUANTY_OP_DUMP, QUANTY_EIGVALS, QUANTY_EIGVECS)
    isfile(p) || error("Missing Quanty output: $p\n" *
                        "Run docs/dev/validation/z_only_eigen.Quanty " *
                        "(and z_only_print.Quanty for the operator dump) first.")
end

# ----- Build the Hamiltonian + basis on the MOADyna side ---------------------

const HILBERT = Hilbert(:s => FermionSite{10}(:s))
const SITE    = HILBERT[:s]
# Quanty modes 0..9 → MOADyna (:s, 1)..(:s, 10).
const MODE_MAP = i -> (:s, i + 1)

H = read_quanty_operator(QUANTY_OP_DUMP, HILBERT, MODE_MAP)
basis = EagerBasis(HILBERT, n_fermion(HILBERT) == 8)
@assert length(basis) == binomial(10, 8) == 45 "expected 45-state basis"

H_compiled = compile(H, basis)
H_sparse = assemble(H_compiled, basis)
@info "MOADyna assembled" basis_size = length(basis) nnz = nnz(H_sparse)

# ----- Run MOADyna's eigen ---------------------------------------------------

E_moad = eigen(H, basis; n = 5)
@info "MOADyna eigen done" values = E_moad.values

# ----- Read Quanty side ---------------------------------------------------

quanty_eigvals = read_quanty_eigenvalues(QUANTY_EIGVALS)
quanty_eigvecs = read_quanty_wavefunctions(QUANTY_EIGVECS, basis, MODE_MAP;
                                            eltype = Float64)

@info "Quanty side loaded" n_eigvals = length(quanty_eigvals) n_eigvecs = length(quanty_eigvecs)

# ----- Compare eigenvalues -----------------------------------------------

function _compare_eigvals(E_moad, quanty_eigvals)
    println("\n", "=" ^ 60)
    println("Eigenvalue comparison (n_fermion = 8 sector)")
    println("=" ^ 60)
    @printf "%-8s  %-22s  %-22s  %-12s\n" "k" "MOADyna" "Quanty" "|Δ|"
    println("-" ^ 70)
    max_diff = 0.0
    for k in 1:5
        Δ = abs(E_moad.values[k] - quanty_eigvals[k])
        @printf "%-8d  %22.15e  %22.15e  %.4e\n" k E_moad.values[k] quanty_eigvals[k] Δ
        max_diff = max(max_diff, Δ)
    end
    return max_diff
end
max_eigval_diff = _compare_eigvals(E_moad, quanty_eigvals)

# ----- Compare eigenvectors via projectors --------------------------------
#
# Eigenvectors agree only up to a global phase on non-degenerate levels and
# up to a unitary rotation within each degenerate cluster. The robust
# metric is the projector Frobenius norm:
#   non-degenerate: ‖ψ_M ψ_M' - ψ_Q ψ_Q'‖_F
#   degenerate:     ‖V_M V_M' - V_Q V_Q'‖_F  over the full block
#
# z_only's lowest 5 levels: GS at 0.0329 (non-degenerate),
#   triplet at 0.6491 (3-fold degenerate),
#   E_5 at 0.9077 (non-degenerate).

function _cluster_indices(λ::AbstractVector{<:Real}; tol = 1e-8)
    clusters = Vector{UnitRange{Int}}()
    n = length(λ)
    i = 1
    while i ≤ n
        j = i
        while j < n && abs(λ[j + 1] - λ[i]) ≤ tol
            j += 1
        end
        push!(clusters, i:j)
        i = j + 1
    end
    return clusters
end

# Build the Quanty eigvec matrix and compare cluster by cluster.
V_quanty = reduce(hcat, quanty_eigvecs)
V_moad = E_moad.vectors

function _compare_eigvecs(V_moad, V_quanty, λ)
    println("\n", "=" ^ 60)
    println("Eigenvector projector comparison")
    println("=" ^ 60)
    clusters = _cluster_indices(λ)
    println("Detected clusters (degenerate ranges): ", clusters)
    max_diff = 0.0
    for rng in clusters
        P_M = V_moad[:,   rng] * V_moad[:,   rng]'
        P_Q = V_quanty[:, rng] * V_quanty[:, rng]'
        fnorm = norm(P_M - P_Q)
        @printf "  cluster %s (size %d): ‖P_MOADyna - P_Quanty‖_F = %.4e\n" string(rng) length(rng) fnorm
        max_diff = max(max_diff, fnorm)
    end
    return max_diff
end
max_proj_diff = _compare_eigvecs(V_moad, V_quanty, E_moad.values)

# ----- Summary -----------------------------------------------------------

println("\n", "=" ^ 60)
println("Summary")
println("=" ^ 60)
@printf "Max |ΔE|                    : %.4e\n" max_eigval_diff
@printf "Max projector Frobenius diff: %.4e\n" max_proj_diff

if max_eigval_diff < 1e-10 && max_proj_diff < 1e-9
    println("\n✓ MOADyna and Quanty agree on the lowest 5 eigenpairs.")
else
    println("\n✗ Discrepancy exceeds tolerance. See per-level numbers above.")
    exit(1)
end
