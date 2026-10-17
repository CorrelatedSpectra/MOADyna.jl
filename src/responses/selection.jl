"""
    SelectedResponse

A selected basis, original source block, retained Lanczos recurrence, and
resolvent error bounds at `shifts`. Source columns keep their input order.
`projected_residuals` and `external_residuals` are block Frobenius norms;
`error_bounds` divides their combined norm by `abs(imag(shift))`.
`iterations` counts outer solves and `candidate_count` counts nonzero external
rows in their union (`nothing` for an incomplete check).

Use [`resolvent`](@ref) for selected-space vectors or [`to_grid`](@ref) for the
source-contracted response. New shifts and linewidths reuse the recurrence but
do not inherit a certificate. An all-zero source has `lanczos === nothing`.
The retained arrays describe one immutable calculation; modifying them invalidates
its certificates.
"""
struct SelectedResponse{T,L}
    basis::EagerBasis
    sources::Matrix{T}
    lanczos::L
    overlaps::Matrix{T}
    shifts::Vector{ComplexF64}
    projected_residuals::Vector{Float64}
    external_residuals::Vector{Float64}
    error_bounds::Vector{Float64}
    iterations::Int
    candidate_count::Union{Nothing,Int}
end

function _selected_response_result(b, B, L, shifts, projected, external, iteration, count)
    overlaps = L === nothing ? zeros(eltype(B), 0, size(B, 2)) :
        reduce(vcat, [L.V_basis[k]' * B for k in eachindex(L.α)])
    bounds = hypot.(projected, external) ./ abs.(imag.(shifts))
    return SelectedResponse(b, B, L, overlaps, shifts, projected, external, bounds, iteration, count)
end

"""
    resolvent(result::SelectedResponse, z)

Reconstruct the approximate `(z*I - H) \\ sources` in `result.basis` coordinates
from the retained recurrence. This performs no Hamiltonian products. Only shifts
listed in `result.shifts` have measured error bounds; evaluating a new shift
does not certify it.
"""
function resolvent(result::SelectedResponse, z::Number)
    isfinite(z) && !iszero(imag(z)) || throw(ArgumentError("z must be finite with nonzero imaginary part"))
    result.lanczos === nothing && return zeros(promote_type(eltype(result.sources), typeof(z)),
                                               size(result.sources))
    return _resolvent_from_lanczos(result.lanczos, z)
end

"""
    to_grid(result::SelectedResponse, ω; Γ, Eg=0)

Evaluate `sources' * resolvent(result, ω + Eg + im*Γ/2)` using small retained
matrices. `Γ` is the positive Lorentzian FWHM. Polarization channels retain their
input order. At checked shifts, the matrix error is bounded by
`opnorm(result.sources) * error_bound`, assuming a fixed exact source.
Other energies or linewidths can be evaluated without rerunning Lanczos, but
their accuracy has not been checked by this result.
"""
function to_grid(result::SelectedResponse, ω::AbstractVector{<:Real}; Γ::Real, Eg::Real=0)
    isfinite(Γ) && Γ > 0 && isfinite(Eg) && all(isfinite, ω) ||
        throw(ArgumentError("energies must be finite and Γ finite and positive"))
    n = size(result.sources, 2)
    data = zeros(ComplexF64, n, n, length(ω))
    L = result.lanczos
    if L !== nothing
        for (k, energy) in enumerate(ω)
            W = _block_resolvent_action(L.α, L.β, energy + Eg + im*Γ/2, L.R)
            mul!(view(data, :, :, k), adjoint(result.overlaps), W)
        end
    end
    return GridResponse(data, ω; Eg, Γ)
end

function _source_block(source::AbstractVecOrMat{<:Number}, basis)
    Base.require_one_based_indexing(source)
    size(source, 1) == length(basis) || throw(DimensionMismatch("source and basis disagree"))
    size(source, 2) > 0 && all(isfinite, source) ||
        throw(ArgumentError("sources must have at least one column and finite entries"))
    return Matrix{promote_type(Float64, eltype(source))}(reshape(source, length(basis), size(source, 2)))
end

function _response_candidates(A, B, L, shifts, restrictions, max_candidates, threaded, iteration)
    count = length(shifts)
    projected, external = fill(Inf, count), fill(Inf, count)
    weights = _StateAccumulator(Float64, A.ket.nwords, count, max_candidates)
    for (k, z) in enumerate(shifts)
        X = _resolvent_from_lanczos(L, z)
        projected[k] = norm(B - z * X + A * X)
        try
            outside = _external_action(A, X; restrictions, max_candidates, threaded)
            external[k] = norm(outside.values)
            for i in axes(outside.values, 1)
                state = @view outside.states[(i - 1) * A.ket.nwords + 1:i * A.ket.nwords]
                row = _accumulator_row!(weights, state)
                weights.values[(row - 1) * count + k] = norm(@view(outside.values[i, :])) / abs(imag(z))
            end
        catch e
            e isa _CandidateLimit || rethrow()
            result = _selected_response_result(A.ket, B, L, shifts, projected, external, iteration, nothing)
            throw(SelectionConvergenceError(:max_candidates, result))
        end
    end
    candidates = _finish_accumulator(weights)
    result = _selected_response_result(A.ket, B, L, shifts, projected, external,
                                        iteration, size(candidates.values, 1))
    return result, candidates
end

"""
    selected_response(H, seed, sources; restrictions, shifts, max_states,
                      error_tol=1e-6, kwargs...) -> SelectedResponse
    selected_response(H, source_basis, transitions, state; kwargs...)

Select a common basis for all source channels and sampled resolvents. `H` must
be a symbolic Hermitian Hamiltonian. `shifts` contains finite complex energies
with nonzero imaginary parts, for example `ω + Eg + im*Γ/2`. The required
`restrictions` define the allowed propagation space; `nothing` permits all states.

The three-argument form accepts a vector or source matrix on `seed`. The
four-argument form generates the complete nonzero support of symbolic transition
operators acting on `state` in `source_basis`, including states absent from that
basis. A vector of transitions and a matrix of states produce columns ordered
by transition, then by state. Destination restrictions may differ from the
source sector. Exactly cancelling paths are combined before support is chosen.

Each fixed-space solve uses full-reorthogonalized block Lanczos, with a scalar
recurrence for one channel. Selection adds determinants outside the solve and
never clips a matrix product. `error_tol` is an absolute Frobenius bound on the
resolvent action at **every** supplied shift:
`norm(X_exact - X) ≤ hypot(norm(r_projected), norm(r_external))/abs(imag(z))`.
Residuals use the original source, including any QR-deflated directions.
The bound assumes Hermiticity throughout the allowed space and exact supplied
sources and energies; it does not include a ground-state approximation error.

`backend=:matrixfree` (or `:sparse`) chooses the projected Hamiltonian storage.
`krylovdim=200` caps block-Lanczos iterations; `deflate_tol=eps(Float64)` controls
rank detection. A remaining projected residual above the budget fails explicitly;
increase `krylovdim` or reduce `deflate_tol` as appropriate. `maxiter=50` caps outer
selection iterations. `threaded=:auto`, `true`, or `false` controls multiplication
and discovery, as for `eigen`.

`max_states` caps the retained basis, and `max_candidates=max_states` caps unique
generated states before all cancellations are resolved, including the union over
shifts. Discovery uses per-worker temporary accumulators, as in `selected_eigen`.
Checks stream shifts but retain one scalar weight per candidate and shift.
Resource caps raise `SelectionConvergenceError` with the latest result; a cap
during initial transition support discovery has `result === nothing`.
All-zero sources return a zero response without a Lanczos solve.
"""
function selected_response(H::OperatorSum, seed::EagerBasis, sources::AbstractVecOrMat{<:Number};
                           restrictions, shifts, max_states::Int, max_candidates::Int=max_states,
                           error_tol::Real=1e-6, backend::Symbol=:matrixfree,
                           maxiter::Int=50, krylovdim::Int=200,
                           deflate_tol::Real=eps(Float64), threaded=:auto)
    _validate_threading(threaded)
    backend in (:matrixfree, :sparse) || throw(ArgumentError("backend must be :matrixfree or :sparse"))
    0 <= length(seed) <= max_states && max_states > 0 ||
        throw(ArgumentError("seed must contain at most max_states states"))
    max_candidates > 0 && maxiter > 0 && krylovdim > 0 ||
        throw(ArgumentError("candidate and iteration limits must be positive"))
    isfinite(error_tol) && error_tol > 0 || throw(ArgumentError("error_tol must be finite and positive"))
    isfinite(deflate_tol) && 0 < deflate_tol <= 1 ||
        throw(ArgumentError("deflate_tol must lie in (0, 1]"))
    zs = collect(ComplexF64, shifts)
    !isempty(zs) && all(z -> isfinite(z) && !iszero(imag(z)), zs) ||
        throw(ArgumentError("shifts must be finite with nonzero imaginary parts"))
    b = _selected_basis(seed, seed.states; restrictions)
    A = matrixfree(H, b; threaded)
    all(t -> isfinite(t.coef), A.forward.terms) || throw(ArgumentError("Hamiltonian coefficients must be finite"))
    B = Matrix{promote_type(eltype(A), eltype(sources), Float64)}(_source_block(sources, seed))
    if all(iszero, B)
        return _selected_response_result(b, B, nothing, zs, zeros(length(zs)), zeros(length(zs)), 0, 0)
    end
    for iteration in 1:maxiter
        action = backend === :matrixfree ? A : assemble(A.forward, b)
        ishermitian(action) || throw(ArgumentError("projected Hamiltonian must be Hermitian"))
        L = block_lanczos(action, B; krylovdim, max_iter=krylovdim, reorth=:full,
                          tol=0.0, deflate_tol, retain_basis=true, threaded)
        result, candidates = _response_candidates(A, B, L, zs, restrictions,
                                                   max_candidates, threaded, iteration)
        all(isfinite, result.error_bounds) || throw(SelectionConvergenceError(:nonfinite_residual, result))
        all(<=(error_tol), result.error_bounds) && return result
        inner = result.projected_residuals ./ abs.(imag.(zs))
        all(<(error_tol), inner) || throw(SelectionConvergenceError(:inner_residual, result))
        iteration < maxiter || throw(SelectionConvergenceError(:maxiter, result))
        budgets = error_tol .* sqrt.(max.(0.0, 1 .- (inner ./ error_tol).^2))
        rows = _enrichment_rows(candidates.values, budgets, max_states - length(b))
        isempty(rows) && throw(SelectionConvergenceError(:max_states, result))
        states = copy(b.states)
        for row in rows
            append!(states, @view candidates.states[(row - 1) * b.nwords + 1:row * b.nwords])
        end
        enlarged = _selected_basis(b, states; restrictions)
        B = _embed_block(B, b, enlarged)
        b = enlarged
        A = _rebind_action(A, b)
    end
    error("unreachable selection loop exit")
end

function selected_response(H::OperatorSum, source_basis::EagerBasis,
                           transitions::Union{OperatorSum,AbstractVector{<:OperatorSum}},
                           state::AbstractVecOrMat{<:Number}; restrictions,
                           max_states::Int, max_candidates::Int=max_states, threaded=:auto, kwargs...)
    max_states > 0 && max_candidates > 0 || throw(ArgumentError("state limits must be positive"))
    operators = transitions isa OperatorSum ? [transitions] : collect(transitions)
    isempty(operators) && throw(ArgumentError("at least one transition is required"))
    X = _source_block(state, source_basis)
    actions = [matrixfree(T, source_basis; threaded) for T in operators]
    T = promote_type(eltype(X), eltype.(actions)...)
    width, nw = size(X, 2), source_basis.nwords
    sources = _StateAccumulator(T, nw, width * length(actions), max_candidates)
    try
        for (k, action) in enumerate(actions)
            generated = _discover_action(action, X; exclude=nothing, restrictions,
                                           max_candidates, threaded)
            for i in axes(generated.values, 1)
                row = _accumulator_row!(sources, @view generated.states[(i - 1) * nw + 1:i * nw])
                for j in 1:width
                    sources.values[(row - 1) * sources.width + (k - 1) * width + j] = generated.values[i, j]
                end
            end
        end
    catch e
        e isa _CandidateLimit || rethrow()
        throw(SelectionConvergenceError(:max_candidates, nothing))
    end
    generated = _finish_accumulator(sources)
    size(generated.values, 1) <= max_states || throw(SelectionConvergenceError(:max_states, nothing))
    seed = _selected_basis(source_basis, generated.states; restrictions)
    return selected_response(H, seed, generated.values; restrictions, max_states,
                               max_candidates, threaded, kwargs...)
end
