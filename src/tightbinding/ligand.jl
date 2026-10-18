"""
    ligand_basis(h, correlated; atol=0.0, rtol=nothing)
    ligand_basis(cluster::FiniteCluster, correlated; kwargs...)

Rotate the ligand orbitals so their first block spans the direct coupling to the
ordered `correlated` indices. The correlated orbitals are unchanged and appear
first in the new basis. All ligand orbitals, including the remaining tail, are
retained, so this is a complete unitary transformation.

Returns a named tuple with `hamiltonian = U' * h * U`, the full `U` (old rows,
new columns), and index ranges `correlated`, `direct`, and `tail`. Use the same
`U` with [`rotate`](@ref) to transform other operators.

Singular values above `max(atol, rtol*smax)` define the direct block. The default
`rtol` is the coupling-matrix dimension times machine precision.
`coupling_residual` measures direct correlated-to-tail coupling; `tail_coupling`
measures the coupling between the direct ligand block and its tail. Discarding
the tail is a further approximation when either is nonzero.
"""
function ligand_basis(h::AbstractMatrix{<:Number}, correlated; atol::Real=0.0, rtol=nothing)
    n = size(h, 1)
    size(h, 2) == n || throw(DimensionMismatch("the one-electron Hamiltonian must be square"))
    all(isfinite, h) || throw(ArgumentError("Hamiltonian entries must be finite"))
    isapprox(h, h'; atol=1e-10, rtol=1e-10) || throw(ArgumentError("the Hamiltonian must be Hermitian"))
    indices = collect(correlated)
    !isempty(indices) && all(i -> i isa Integer && 1 <= i <= n, indices) ||
        throw(ArgumentError("supply correlated orbital indices within 1:$n"))
    allunique(indices) || throw(ArgumentError("correlated orbital indices must be distinct"))
    environment = setdiff(1:n, indices)
    nd, nl = length(indices), length(environment)
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative"))
    relative = rtol === nothing ? max(nd, nl) * eps(Float64) : rtol
    relative isa Real && isfinite(relative) && relative >= 0 ||
        throw(ArgumentError("rtol must be finite and nonnegative"))
    T = promote_type(Float64, eltype(h))
    U = zeros(T, n, n)
    for (new, old) in enumerate(indices)
        U[old, new] = 1
    end
    rank = 0
    if nl > 0
        coupling = Matrix{T}(h[environment, indices])
        factor = svd(coupling; full=true)
        threshold = max(atol, relative * maximum(factor.S; init=0.0))
        rank = count(>(threshold), factor.S)
        U[environment, nd+1:n] = factor.U
    end
    transformed = U' * h * U
    direct = nd+1:nd+rank
    tail = nd+rank+1:n
    return (; hamiltonian=transformed, U, correlated=1:nd, direct, tail,
            coupling_residual=norm(transformed[tail, 1:nd]),
            tail_coupling=norm(transformed[tail, direct]))
end

ligand_basis(cluster::FiniteCluster, correlated; kwargs...) =
    ligand_basis(cluster.hamiltonian, correlated; kwargs...)
