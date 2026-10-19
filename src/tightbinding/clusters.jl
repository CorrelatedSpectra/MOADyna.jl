"""
    FiniteCluster

An open cluster extracted from a tight-binding model. `hamiltonian` is its
one-electron matrix, and `orbitals` maps its indices to `(cell, orbital)` in the
parent model. Columns of `positions` are Cartesian positions in Å, or `nothing`
when the parent has incomplete geometry. Construct with [`finite_cluster`](@ref).
"""
struct FiniteCluster{D,T<:Number}
    hamiltonian::Matrix{T}
    orbitals::Vector{Tuple{NTuple{D,Int},Int}}
    positions::Union{Nothing,Matrix{Float64}}
    labels::Vector{String}
end

function _cell_tuple(cell, ::Val{D}) where {D}
    length(cell) == D || throw(DimensionMismatch("a cell translation must have $D components"))
    all(x -> x isa Integer, cell) || throw(ArgumentError("cell translations must be integral"))
    return ntuple(i -> Int(cell[i]), D)
end

"""
    finite_cluster(model; cells=[(0, ...)], orbitals=1:size(model,1))
    finite_cluster(model, selection)

Extract an open cluster. The keyword form selects the same orbitals in every
listed cell, in cell-major order. For an arbitrary fragment, supply an ordered
`selection` of `(cell, orbital)` pairs. Matrix entries are
`h[a,b] = hopping(model, cell_b-cell_a)[orbital_a,orbital_b]`.
"""
function finite_cluster(model::TightBindingModel{D};
                        cells=[ntuple(_ -> 0, D)], orbitals=1:size(model, 1)) where {D}
    return finite_cluster(model, [(cell, orbital) for cell in cells for orbital in orbitals])
end

function finite_cluster(model::TightBindingModel{D,T}, selection) where {D,T}
    chosen = Tuple{NTuple{D,Int},Int}[]
    for (cell, orbital) in selection
        orbital isa Integer && 1 <= orbital <= size(model, 1) ||
            throw(ArgumentError("orbital index must lie in 1:$(size(model, 1))"))
        push!(chosen, (_cell_tuple(cell, Val(D)), Int(orbital)))
    end
    isempty(chosen) && throw(ArgumentError("select at least one orbital"))
    allunique(chosen) || throw(ArgumentError("cluster orbitals must be distinct"))
    n = length(chosen)
    h = zeros(T, n, n)
    for b in 1:n, a in 1:n
        ra, i = chosen[a]
        rb, j = chosen[b]
        block = get(model.index, rb .- ra, 0)
        block == 0 || (h[a, b] = model.hoppings[block][i, j])
    end
    positions = if model.lattice === nothing || model.centres === nothing
        nothing
    else
        reduce(hcat, [model.lattice * (collect(r) + model.centres[:, i]) for (r, i) in chosen])
    end
    labels = [model.labels[i] for (_, i) in chosen]
    return FiniteCluster{D,T}(h, chosen, positions, labels)
end

"""
    onebody(cluster::FiniteCluster, modes)

Convert the cluster Hamiltonian to an `OperatorSum` using the explicitly ordered
fermionic `modes`, or the canonical order of a `Hilbert`.
"""
onebody(cluster::FiniteCluster, modes) = onebody(cluster.hamiltonian, modes)

function _supercell_geometry(S, D)
    S isa AbstractMatrix{<:Integer} || throw(ArgumentError("the supercell matrix must be integral"))
    size(S) == (D, D) || throw(DimensionMismatch("the supercell matrix must be $D × $D"))
    exact = Matrix{Rational{BigInt}}(S)
    determinant = det(exact)
    iszero(determinant) && throw(ArgumentError("the supercell matrix must be nonsingular"))
    count = Int(abs(determinant))
    inverse = inv(exact)
    adjugate = Matrix{BigInt}(count .* inverse)
    return count, inverse, adjugate
end

_coset_key(adjugate, count, cell) = Tuple(Int.(mod.(adjugate * collect(cell), count)))

function _cell_representatives(adjugate, count, ::Val{D}, supplied) where {D}
    cells = NTuple{D,Int}[]
    indices = Dict{NTuple{D,Int},Int}()
    if supplied === nothing
        push!(cells, ntuple(_ -> 0, D))
        indices[_coset_key(adjugate, count, cells[1])] = 1
        cursor = 1
        while length(cells) < count
            for axis in 1:D
                cell = ntuple(i -> cells[cursor][i] + (i == axis), D)
                key = _coset_key(adjugate, count, cell)
                if !haskey(indices, key)
                    push!(cells, cell)
                    indices[key] = length(cells)
                end
            end
            cursor += 1
        end
    else
        for cell in supplied
            push!(cells, _cell_tuple(cell, Val(D)))
        end
        length(cells) == count || throw(ArgumentError("supply exactly $count cell representatives"))
        for (i, cell) in enumerate(cells)
            key = _coset_key(adjugate, count, cell)
            haskey(indices, key) && throw(ArgumentError("representatives belong to the same supercell coset"))
            indices[key] = i
        end
    end
    return cells, indices
end

"""
    supercell(model, S; representatives=nothing, atol=1e-10, rtol=1e-10)

Reindex a periodic model into the integral supercell with lattice `model.lattice*S`.
Every hopping is retained. Optional `representatives` must contain exactly
`abs(det(S))` distinct primitive cells modulo `S`.

The result is another `TightBindingModel`. Its `metadata[:orbital_map]` gives
the parent `(cell, orbital)` for every new orbital. The zero-translation hopping
is the open-cluster Hamiltonian; nonzero translations connect different clusters.
Their sum at Γ is generally different from the open-cluster Hamiltonian.
`atol` and `rtol` are forwarded to the new model's Hermiticity check; supply the
same precision as the parent constructor when accepting rounded input.
"""
function supercell(model::TightBindingModel{D,T}, S;
                   representatives=nothing, atol::Real=1e-10, rtol::Real=1e-10) where {D,T}
    count, inverse, adjugate = _supercell_geometry(S, D)
    cells, indices = _cell_representatives(adjugate, count, Val(D), representatives)
    n = size(model, 1)
    total = Base.checked_mul(n, count)
    blocks = Dict{NTuple{D,Int},Matrix{T}}()
    for (a, cell) in enumerate(cells), (R, h) in zip(model.translations, model.hoppings)
        destination = cell .+ R
        b = indices[_coset_key(adjugate, count, destination)]
        delta = adjugate * collect(destination .- cells[b])
        all(x -> iszero(rem(x, count)), delta) || error("invalid supercell cell decomposition")
        L = Tuple(Int.(div.(delta, count)))
        out = get!(blocks, L) do
            zeros(T, total, total)
        end
        @inbounds for j in 1:n, i in 1:n
            out[(a-1)*n+i, (b-1)*n+j] += h[i, j]
        end
    end
    orbital_map = [(cell, i) for cell in cells for i in 1:n]
    lattice = model.lattice === nothing ? nothing : model.lattice * S
    fractional_transform = Float64.(inverse)
    centres = model.centres === nothing ? nothing :
        reduce(hcat, [fractional_transform * (collect(cell) + model.centres[:, i])
                      for (cell, i) in orbital_map])
    labels = [string(model.labels[i], "@", cell) for (cell, i) in orbital_map]
    metadata = merge(model.metadata, Dict(:supercell_matrix => copy(S), :orbital_map => orbital_map))
    return TightBindingModel(blocks; lattice, centres, labels, metadata, atol, rtol)
end
