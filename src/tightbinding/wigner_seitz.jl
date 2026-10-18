function _add_ws_entry!(blocks, R, i, j, value, n)
    iszero(value) && return nothing
    h = get!(blocks, R) do
        zeros(ComplexF64, n, n)
    end
    h[i, j] += value
    return nothing
end

function _nonempty_ws_blocks!(blocks, n)
    isempty(blocks) && (blocks[(0, 0, 0)] = zeros(ComplexF64, n, n))
    return blocks
end

function _expand_wsvec(hoppings, file, n)
    blocks = Dict{NTuple{3,Int},Matrix{ComplexF64}}()
    seen = Dict(R => falses(n, n) for R in keys(hoppings))
    open(file) do io
        readline(io)
        for _ in 1:length(hoppings)*n*n
            f = _w90_line(io)
            length(f) == 5 || throw(ArgumentError("wsvec records must contain R, m and n"))
            R = Tuple(parse.(Int, f[1:3]))
            i, j = parse.(Int, f[4:5])
            haskey(hoppings, R) || throw(ArgumentError("wsvec translation $R is absent from the Hamiltonian"))
            1 <= i <= n && 1 <= j <= n || throw(ArgumentError("wsvec orbital index out of range"))
            seen[R][i, j] && throw(ArgumentError("duplicate wsvec record at $R, $i, $j"))
            count = parse(Int, only(_w90_line(io)))
            count > 0 || throw(ArgumentError("wsvec degeneracies must be positive"))
            shifts = NTuple{3,Int}[]
            for _ in 1:count
                line = _w90_line(io)
                length(line) == 3 || throw(ArgumentError("wsvec shifts must have three components"))
                push!(shifts, Tuple(parse.(Int, line)))
            end
            allunique(shifts) || throw(ArgumentError("duplicate shifts in a wsvec record"))
            for shift in shifts
                _add_ws_entry!(blocks, R .+ shift, i, j, hoppings[R][i, j]/count, n)
            end
            seen[R][i, j] = true
        end
        all(all, values(seen)) || throw(ArgumentError("incomplete wsvec table"))
        all(line -> isempty(_w90_fields(line)), eachline(io)) || throw(ArgumentError("unexpected rows after wsvec table"))
    end
    return _nonempty_ws_blocks!(blocks, n)
end

function _nearest_ws_shifts(d, geometry, mesh, tolerance)
    B, inverse, row_norms, Q, R = geometry
    y = -(Q' * d)
    trial = MVector{3,Int}(0, 0, 0)
    for i in 3:-1:1
        remainder = sum(R[i, j] * trial[j] for j in i+1:3; init=0.0)
        trial[i] = round(Int, (y[i]-remainder)/R[i, i])
    end
    radius = norm(d + B * trial) + tolerance
    centre = -(inverse * d)
    lower = ceil.(Int, centre - radius * row_norms)
    upper = floor.(Int, centre + radius * row_norms)
    ranges = ntuple(i -> lower[i]:upper[i], 3)
    minimum_distance = Inf
    for shift in Iterators.product(ranges...)
        minimum_distance = min(minimum_distance, norm(d + B * SVector{3}(shift)))
    end
    shifts = NTuple{3,Int}[]
    for shift in Iterators.product(ranges...)
        distance = norm(d + B * SVector{3}(shift))
        abs(distance-minimum_distance) < tolerance && push!(shifts, Tuple(mesh .* shift))
    end
    isempty(shifts) && throw(ArgumentError("no Wigner–Seitz image found; check lattice precision and distance tolerance"))
    return shifts
end

function _reconstruct_ws(hoppings, lattice, centres, mesh, tolerance)
    A = SMatrix{3,3,Float64}(lattice)
    B = A * Diagonal(SVector{3,Int}(mesh))
    inverse = inv(B)
    row_norms = SVector{3}(ntuple(i -> norm(inverse[i, :]), 3))
    factor = qr(Matrix(B))
    geometry = (B, inverse, row_norms, SMatrix{3,3}(Matrix(factor.Q)), SMatrix{3,3}(factor.R))
    cartesian = A * centres
    n = size(centres, 2)
    blocks = Dict{NTuple{3,Int},Matrix{ComplexF64}}()
    for (R, h) in hoppings, j in 1:n, i in 1:n
        iszero(h[i, j]) && continue
        d = A * SVector{3}(R) + SVector{3}(cartesian[:, j] - cartesian[:, i])
        shifts = _nearest_ws_shifts(d, geometry, mesh, tolerance)
        for shift in shifts
            _add_ws_entry!(blocks, R .+ shift, i, j, h[i, j]/length(shifts), n)
        end
    end
    return _nonempty_ws_blocks!(blocks, n)
end
