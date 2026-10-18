"""
    Eigenstates

Eigenstates returned by `eigen(H, hilbert; restrictions, ...)`. `values` and
`vectors` expose the usual eigenvalues and eigenvectors; vector rows refer to
`basis`. `backend` records the operator representation. `selection` contains
[`SelectedEigen`](@ref) diagnostics for an adaptive solve and is `nothing`
otherwise. `eigen` gives the underlying `LinearAlgebra.Eigen`.
"""
struct Eigenstates{E<:LinearAlgebra.Eigen,S<:Union{Nothing,SelectedEigen}}
    basis::EagerBasis
    eigen::E
    backend::Symbol
    selection::S
end

function Base.getproperty(states::Eigenstates, name::Symbol)
    name in (:values, :vectors) && return getproperty(getfield(states, :eigen), name)
    return getfield(states, name)
end

Base.propertynames(::Eigenstates, private::Bool=false) =
    (:values, :vectors, :basis, :backend, :selection, :eigen)

"""
    eigen(H, hilbert::Hilbert; restrictions, backend=:sparse, adaptive=false,
          n=1, tol=1e-10, kwargs...) -> Eigenstates
    eigen(H, model::ShellModel; restrictions, kwargs...) -> Eigenstates

Construct a basis and solve for eigenstates, retaining their basis in the result.
`restrictions` defines the allowed occupations of each basis state; pass a single
restriction, a tuple/vector, or `nothing` for the unrestricted Hilbert space.
The default constructs the complete restricted basis and uses sparse storage.
`backend=:matrixfree` avoids storing matrix entries.

With `adaptive=true`, start with a small basis and enlarge it using
[`selected_eigen`](@ref). `tol` bounds the full allowed-space residual, including
omitted states. `which=:SR` is required. The default seed samples `krylovdim`
allowed states (or all of a smaller sector), without enumerating the full basis.
`seed_size` changes this count, `rng` changes the reproducible sampling sequence,
and `seed` supplies an explicit basis instead. Counting allowed configurations
can itself be costly for complicated overlapping restrictions. `seed_max_work`
limits counting/unranking calls (default `100_000`); exceeding it raises an
`ArgumentError` requesting an explicit seed or a larger budget. An explicit seed
bypasses counting. Automatic sampling cannot guarantee coverage of disconnected
sectors or discovery of the global ground state. Check stability under seed
enlargement and use physical symmetry sectors when available.

Adaptive resource controls are `max_states` (retained states), `max_candidates`
(generated states, defaults to `max_states`), and `selection_maxiter=50` (basis
enlargements). State/candidate limits are unset by default; set them to bound
memory use. Hitting a limit raises [`SelectionConvergenceError`](@ref).
`maxiter=300` always controls the inner eigensolver. `krylovdim`, `degen_tol`,
`dense_below`, and `threaded` have the same meaning as in `eigen(H, basis)`.
`x0` is available only for a fixed basis; adaptive solves reuse their preceding
eigenvectors automatically. Ground-state selection does not converge a response
space: spectroscopy requires its own physical sectors and accuracy checks.
"""
function LinearAlgebra.eigen(H::OperatorSum, h::Hilbert;
                             restrictions, backend::Symbol=:sparse, adaptive::Bool=false,
                             n::Int=1, which::Symbol=:SR, tol::Real=1e-10,
                             degen_tol::Real=1e-8, maxiter::Int=300,
                             krylovdim::Int=max(20, 2n + 10), dense_below::Int=1024,
                             x0=nothing, threaded=:auto,
                             seed::Union{Nothing,EagerBasis}=nothing,
                             seed_size::Union{Nothing,Int}=nothing,
                             rng::Union{Nothing,AbstractRNG}=nothing,
                             seed_max_work::Union{Nothing,Int}=nothing,
                             max_states::Union{Nothing,Int}=nothing,
                             max_candidates::Union{Nothing,Int}=max_states,
                             selection_maxiter::Union{Nothing,Int}=nothing)
    backend in (:sparse, :matrixfree) ||
        throw(ArgumentError("backend must be :sparse or :matrixfree"))
    _validate_threading(threaded)
    n > 0 || throw(ArgumentError("n must be positive"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    rs = _restriction_list(restrictions)
    if !adaptive
        all(isnothing, (seed, seed_size, rng, seed_max_work, max_states,
                        max_candidates, selection_maxiter)) ||
            throw(ArgumentError("seed and selection limits require adaptive=true"))
        b = basis(h, rs...)
        E = eigen(H, b; backend, n, which, tol, degen_tol, maxiter, krylovdim,
                  dense_below, x0, threaded)
        return Eigenstates(b, E, backend, nothing)
    end
    which === :SR || throw(ArgumentError("adaptive eigen supports only which=:SR"))
    x0 === nothing || throw(ArgumentError("use seed, not x0, for an adaptive solve"))
    limit = something(max_states, typemax(Int))
    candidates = something(max_candidates, typemax(Int))
    limit >= n || throw(ArgumentError("max_states must be at least n"))
    if seed === nothing
        count = seed_size === nothing ? min(krylovdim, limit) : seed_size
        1 <= count <= limit || throw(ArgumentError("seed_size must lie in 1:max_states"))
        seed, dimension = _sample_basis(h, rs, count;
            rng=something(rng, MersenneTwister(0)), max_work=something(seed_max_work, 100_000))
        dimension >= n || throw(ArgumentError("the allowed space contains fewer than n states"))
    else
        seed_size === nothing || throw(ArgumentError("specify seed or seed_size, not both"))
        _same_encoding(EncodingMap(h), seed.encoding) ||
            throw(ArgumentError("seed must use the supplied Hilbert space"))
    end
    selected = selected_eigen(H, seed; restrictions=rs, backend, n,
                              residual_tol=tol, max_states=limit,
                              max_candidates=candidates, maxiter=something(selection_maxiter, 50),
                              eig_maxiter=maxiter, krylovdim, degen_tol, dense_below, threaded)
    return Eigenstates(selected.basis, selected.eigen, backend, selected)
end

LinearAlgebra.eigen(H::OperatorSum, model::ShellModel; kwargs...) =
    eigen(H, model.hilbert; kwargs...)
