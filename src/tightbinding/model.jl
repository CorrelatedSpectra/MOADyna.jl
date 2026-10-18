"""
    TightBindingModel(hoppings; lattice=nothing, centres=nothing, labels=nothing,
                     metadata=Dict(), atol=1e-10, rtol=1e-10)

An orthonormal tight-binding model with `hoppings[R][i,j] = ⟨0,i|h|R,j⟩`.
Supply integer translation tuples `R` and square matrices, including both
directions of every bond. Hermiticity is checked; entries are retained as given.

Columns of `lattice` are direct lattice vectors in Å. Columns of `centres` are
orbital positions in fractional lattice coordinates, in the original cell
representatives. Hoppings are in eV. Geometry may be omitted for fractional-k
interpolation in the periodic gauge. Orbital order and spin content are explicit;
the constructor does not duplicate orbitals or subtract a chemical potential.

Input arrays are copied. Treat the stored arrays as read-only.
"""
struct TightBindingModel{D,T<:Number}
    translations::Vector{NTuple{D,Int}}
    hoppings::Vector{Matrix{T}}
    index::Dict{NTuple{D,Int},Int}
    lattice::Union{Nothing,Matrix{Float64}}
    centres::Union{Nothing,Matrix{Float64}}
    labels::Vector{String}
    metadata::Dict{Symbol,Any}
end

function TightBindingModel(hoppings::AbstractDict; lattice=nothing, centres=nothing, labels=nothing,
                           metadata=Dict(), atol::Real=1e-10, rtol::Real=1e-10)
    entries = collect(pairs(hoppings))
    isempty(entries) && throw(ArgumentError("hoppings must contain at least one translation"))
    first_R, first_h = first(entries)
    first_R isa Tuple && all(x -> x isa Integer, first_R) || throw(ArgumentError(
        "translations must be tuples of integers"))
    D = length(first_R)
    D > 0 || throw(ArgumentError("translations must have at least one component"))
    first_h isa AbstractMatrix || throw(ArgumentError("each hopping block must be a matrix"))
    n = size(first_h, 1)
    n > 0 || throw(ArgumentError("the model must contain at least one orbital"))
    all(entry -> last(entry) isa AbstractMatrix{<:Number}, entries) ||
        throw(ArgumentError("hopping blocks must be numeric matrices"))
    T = mapreduce(entry -> eltype(last(entry)), promote_type, entries; init=Float64)
    translations = NTuple{D,Int}[]
    blocks = Matrix{T}[]
    for (R, h) in entries
        R isa Tuple && length(R) == D && all(x -> x isa Integer, R) ||
            throw(ArgumentError("all translations must be integer tuples of length $D"))
        size(h) == (n, n) || throw(DimensionMismatch("hopping blocks must all be $n × $n"))
        all(isfinite, h) || throw(ArgumentError("hopping entries must be finite"))
        push!(translations, Tuple(Int.(R)))
        push!(blocks, Matrix{T}(h))
    end
    order = sortperm(translations)
    translations, blocks = translations[order], blocks[order]
    index = Dict(R => i for (i, R) in enumerate(translations))
    length(index) == length(translations) || throw(ArgumentError("duplicate translations"))
    isfinite(atol) && isfinite(rtol) && atol >= 0 && rtol >= 0 ||
        throw(ArgumentError("Hermiticity tolerances must be finite and nonnegative"))
    for (R, h) in zip(translations, blocks)
        reverse_index = get(index, map(-, R), 0)
        ok = reverse_index == 0 ? norm(h) <= atol :
             isapprox(h, blocks[reverse_index]'; atol, rtol)
        ok || throw(ArgumentError("hopping blocks at $R and $(map(-, R)) violate Hermiticity"))
    end
    A = _geometry_matrix(lattice, (D, D), "lattice")
    A === nothing || rank(A) == D || throw(ArgumentError("lattice vectors must be independent"))
    tau = _geometry_matrix(centres, (D, n), "centres")
    names = labels === nothing ? string.(1:n) : string.(collect(labels))
    length(names) == n || throw(DimensionMismatch("supply one label per orbital"))
    return TightBindingModel{D,T}(translations, blocks, index, A, tau, names,
                                  Dict{Symbol,Any}(metadata))
end

function _geometry_matrix(value, dims, name)
    value === nothing && return nothing
    value isa AbstractMatrix{<:Real} || throw(ArgumentError("$name must be a real matrix"))
    size(value) == dims || throw(DimensionMismatch("$name must have size $dims"))
    all(isfinite, value) || throw(ArgumentError("$name must contain finite values"))
    return Matrix{Float64}(value)
end

Base.size(model::TightBindingModel) = size(first(model.hoppings))
Base.size(model::TightBindingModel, dim::Integer) = size(first(model.hoppings), dim)

function Base.show(io::IO, model::TightBindingModel{D}) where {D}
    print(io, "TightBindingModel(", size(model, 1), " orbitals, ",
          length(model.translations), " translations, ", D, " dimensions)")
end

"""
    hopping(model, R)

Return a copy of the hopping matrix `⟨0|h|R⟩`, or a zero matrix when `R` is absent.
"""
function hopping(model::TightBindingModel{D,T}, R::NTuple{D,<:Integer}) where {D,T}
    i = get(model.index, R, 0)
    return i == 0 ? zeros(T, size(model)) : copy(model.hoppings[i])
end

function _fractional_k(model::TightBindingModel{D}, k, coordinates) where {D}
    length(k) == D || throw(DimensionMismatch("k must have $D components"))
    all(x -> x isa Real && isfinite(x), k) ||
        throw(ArgumentError("k must have finite real components"))
    q = Float64.(collect(k))
    coordinates === :fractional && return q
    coordinates === :cartesian || throw(ArgumentError(
        "coordinates must be :fractional or :cartesian"))
    model.lattice === nothing && throw(ArgumentError("Cartesian k points require a lattice"))
    return model.lattice' * q / (2π)
end

"""
    bloch_hamiltonian(model, k; coordinates=:fractional, gauge=:periodic)

Evaluate `sum_R exp(2πim * dot(k,R)) * hopping(model,R)`. Fractional k coordinates
refer to reciprocal vectors including `2π`; Cartesian coordinates are in Å⁻¹.

The default periodic gauge is the Wannier convention. With `gauge=:orbital`,
each entry also has the phase `exp(2πim * dot(k, centres[:,j]-centres[:,i]))`.
This requires orbital centres and leaves the band eigenvalues unchanged.
"""
function bloch_hamiltonian(model::TightBindingModel{D,T}, k; kwargs...) where {D,T}
    out = Matrix{promote_type(ComplexF64,T)}(undef, size(model))
    return bloch_hamiltonian!(out, model, k; kwargs...)
end

"""
    bloch_hamiltonian!(out, model, k; coordinates=:fractional, gauge=:periodic)

Overwrite a complex matrix with the Bloch Hamiltonian. `out` must have the model's
matrix size and must not alias a stored hopping block.
"""
function bloch_hamiltonian!(out::AbstractMatrix{<:Complex}, model::TightBindingModel, k;
                            coordinates=:fractional, gauge=:periodic)
    size(out) == size(model) || throw(DimensionMismatch("output must have size $(size(model))"))
    any(h -> Base.mightalias(out, h), model.hoppings) && throw(ArgumentError(
        "output must not alias a stored hopping block"))
    gauge in (:periodic, :orbital) || throw(ArgumentError("gauge must be :periodic or :orbital"))
    gauge === :orbital && model.centres === nothing &&
        throw(ArgumentError("the orbital gauge requires orbital centres"))
    q = _fractional_k(model, k, coordinates)
    fill!(out, 0)
    for (R, h) in zip(model.translations, model.hoppings)
        phase = cispi(2 * dot(q, R))
        @inbounds @simd for i in eachindex(out, h)
            out[i] += phase * h[i]
        end
    end
    if gauge === :orbital
        phases = cispi.(2 .* (model.centres' * q))
        @inbounds for j in axes(out, 2), i in axes(out, 1)
            out[i, j] *= conj(phases[i]) * phases[j]
        end
    end
    return out
end
