"""
    SelectedEigen

Eigenstates on a selected basis. `eigen.vectors` uses the coordinates of `basis`.
`residuals` contains full allowed-space residual norms, including omitted states;
`residual_norm` is their Euclidean norm, equivalently the block Frobenius norm.
`iterations` counts outer iterations and `candidate_count` counts nonzero
external rows in the last residual check (`nothing` if that check was incomplete).
Full-space Hermiticity is assumed;
each projected Hamiltonian is checked by `eigen`.
"""
struct SelectedEigen{E<:LinearAlgebra.Eigen}
    basis::EagerBasis
    eigen::E
    residuals::Vector{Float64}
    residual_norm::Float64
    iterations::Int
    candidate_count::Union{Nothing,Int}
end

"""
    SelectionConvergenceError <: Exception

Selection stopped before satisfying its requested residual tolerance.
`reason` identifies the stopping condition and `result` retains the latest
eigenstates and diagnostics. An infinite residual means the external check
could not finish within `max_candidates`; it is not a measured error.
"""
struct SelectionConvergenceError{R} <: Exception
    reason::Symbol
    result::R
end

function Base.showerror(io::IO, e::SelectionConvergenceError)
    print(io, "selected_eigen stopped at ", e.reason, " with ", length(e.result.basis),
          " states (residual norm = ", e.result.residual_norm, ")")
end

function _selected_eigenpairs(A, n, initial; backend, residual_tol, tol,
                              degen_tol, eig_maxiter, krylovdim, dense_below, threaded)
    b = A.ket
    action = backend === :matrixfree ? A : assemble(A.forward, b)
    requested = min(length(b), n + 1)
    while true
        seed = initial === nothing ? nothing : initial[:, 1:min(size(initial, 2), requested)]
        inner_tol = tol === nothing ? residual_tol / sqrt(2requested) : tol
        E = eigen(action, b; n=requested, x0=seed, tol=inner_tol, degen_tol,
                  maxiter=eig_maxiter, krylovdim=max(krylovdim, requested + 2),
                  dense_below, threaded)
        kept = min(n, length(b))
        while kept < requested && abs(E.values[kept + 1] - E.values[min(n, length(b))]) <= degen_tol
            kept += 1
        end
        if kept < requested || requested == length(b)
            return LinearAlgebra.Eigen(E.values[1:kept], E.vectors[:, 1:kept])
        end
        initial = E.vectors
        requested = min(length(b), max(requested + 1, 2requested))
    end
end

function _enrichment_rows(values, budget, available)
    weights = [norm(@view values[i, :]) for i in axes(values, 1)]
    order = sortperm(weights; rev=true)
    tail = zeros(Float64, length(order) + 1)
    for k in length(order):-1:1
        tail[k] = hypot(tail[k + 1], weights[order[k]])
    end
    count = 0
    while count < length(order) && tail[count + 1] > budget
        count += 1
    end
    return order[1:min(count, available)]
end

"""
    selected_eigen(H, seed; restrictions, max_states, residual_tol=1e-8,
                   n=1, backend=:matrixfree, kwargs...) -> SelectedEigen

Converge low-energy eigenstates without enumerating the full allowed basis.
`seed` is a small basis on the same Hilbert space. The required `restrictions`
keyword defines the allowed physical space (`nothing` permits all states);
restrictions used to construct the seed are not implicitly retained.

Each iteration solves an exact, fixed selected-space problem, evaluates
`H*Ψ-Ψ*Diagonal(E)` including omitted states, and adds determinants ranked by
their residual row norms. Already selected states are retained. `residual_tol`
bounds the full block Frobenius norm, and therefore every normalized linear
combination of the returned states. The lowest `n` states of the selected problem
are requested, including any projected degenerate cluster crossing that boundary
(`degen_tol=1e-8`). Residual convergence alone cannot certify the multiplicity
of the full allowed-space spectrum.

`max_states` caps the retained basis. `max_candidates=max_states` caps unique
generated external states before all cancellations are resolved. Each discovery
worker holds a single-column and a target-block accumulator, each capped at
`max_candidates`, in addition to the final merge and output storage. Temporary
storage consequently grows with the thread count. Exceeding either
limit or `maxiter=50` raises `SelectionConvergenceError` with the latest result.
An inner Krylov failure raises `ConvergenceError` with that projected solve's
diagnostics.

`backend` may be `:matrixfree` or `:sparse`. Inner solves reuse the previous
eigenvector block. `tol=nothing` allocates half the squared residual budget to
the inner solve; an explicit `tol` overrides that choice. `eig_maxiter=300`,
`krylovdim=20`, `dense_below=1024`, and `threaded=:auto` control the same kernels
as `eigen`. The dense threshold applies only to the sparse backend.

The Hamiltonian must be Hermitian throughout the allowed space. A small residual
certifies approximate eigenstates, not discovery of the global ground state or
all degenerate partners. Unseeded disconnected sectors are unreachable.
Ground-state convergence alone does not certify
a spectrum; transition and response spaces require their own convergence checks.
"""
function selected_eigen(H::OperatorSum, seed::EagerBasis;
                        restrictions, max_states::Int, max_candidates::Int=max_states,
                        residual_tol::Real=1e-8, n::Int=1, backend::Symbol=:matrixfree,
                        maxiter::Int=50, tol::Union{Real,Nothing}=nothing,
                        degen_tol::Real=1e-8, eig_maxiter::Int=300,
                        krylovdim::Int=20, dense_below::Int=1024, threaded=:auto)
    _validate_threading(threaded)
    backend in (:matrixfree, :sparse) || throw(ArgumentError("backend must be :matrixfree or :sparse"))
    1 <= length(seed) <= max_states || throw(ArgumentError("seed must contain 1:max_states states"))
    1 <= n <= max_states || throw(ArgumentError("n must lie in 1:max_states"))
    max_candidates > 0 && maxiter > 0 && eig_maxiter > 0 && krylovdim >= 3 ||
        throw(ArgumentError("candidate/iteration limits must be positive and krylovdim at least 3"))
    isfinite(residual_tol) && residual_tol > 0 || throw(ArgumentError("residual_tol must be finite and positive"))
    isfinite(degen_tol) && degen_tol >= 0 || throw(ArgumentError("degen_tol must be finite and nonnegative"))
    tol === nothing || (isfinite(tol) && tol > 0) || throw(ArgumentError("tol must be finite and positive"))
    b = _selected_basis(seed, seed.states; restrictions)
    A = matrixfree(H, b; threaded)
    initial = nothing
    for iteration in 1:maxiter
        E = _selected_eigenpairs(A, n, initial; backend, residual_tol, tol,
            degen_tol, eig_maxiter, krylovdim, dense_below, threaded)
        projected = A * E.vectors - E.vectors * Diagonal(E.values)
        external = try
            _external_action(A, E.vectors; restrictions, max_candidates, threaded)
        catch e
            e isa _CandidateLimit || rethrow()
            result = SelectedEigen(b, E, fill(Inf, length(E.values)), Inf, iteration, nothing)
            throw(SelectionConvergenceError(:max_candidates, result))
        end
        residuals = [hypot(norm(@view(projected[:, j])), norm(@view(external.values[:, j])))
                     for j in eachindex(E.values)]
        result = SelectedEigen(b, E, residuals, norm(residuals), iteration, size(external.values, 1))
        isfinite(result.residual_norm) || throw(SelectionConvergenceError(:nonfinite_residual, result))
        complete = length(E.values) >= n
        if result.residual_norm <= residual_tol && complete
            return result
        end
        !complete && isempty(external.states) &&
            throw(SelectionConvergenceError(:insufficient_states, result))
        iteration < maxiter || throw(SelectionConvergenceError(:maxiter, result))
        projected_norm = norm(projected)
        projected_norm < residual_tol || throw(SelectionConvergenceError(:inner_residual, result))
        budget = complete ? residual_tol * sqrt(max(0, 1 - (projected_norm / residual_tol)^2)) : 0.0
        rows = _enrichment_rows(external.values, budget, max_states - length(b))
        isempty(rows) && throw(SelectionConvergenceError(:max_states, result))
        states = copy(b.states)
        for row in rows
            append!(states, @view external.states[(row - 1) * b.nwords + 1:row * b.nwords])
        end
        enlarged = _selected_basis(b, states; restrictions)
        initial = _embed_block(E.vectors, b, enlarged)
        b = enlarged
        A = _rebind_action(A, b)
    end
    error("unreachable selection loop exit")
end
