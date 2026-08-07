"""
    expectation_table(sys, basis::AbstractBasis,
                      ops_list::AbstractVector{<:Pair{<:AbstractString, <:OperatorSum}};
                      digits::Int = 3,
                      header::Bool = true) -> String

Pretty-printed table of `⟨ψ_i | O_j | ψ_i⟩` for each (eigenstate, operator)
pair.

`sys` is the result of `MOADyna.ED.eigen(H, basis; n=...)` and provides
`sys.values::Vector{Float64}` and `sys.vectors::Matrix` (one eigenstate per
column). `ops_list` maps a column-name string to an `OperatorSum`; each
operator is `compile`d + `assemble`d on `basis` once. Each `OperatorSum` is
assumed Hermitian, so the real part of `⟨ψ|O|ψ⟩` is reported (the imaginary
part is dropped silently — the caller is responsible for handing in
Hermitian operators).

# Arguments
- `sys`        — eigensystem with `.values` and `.vectors`.
- `basis`      — `AbstractBasis` against which `sys` was diagonalized.
- `ops_list`   — `Vector` of `name => OperatorSum` pairs.

# Keyword arguments
| kwarg     | default | meaning                                    |
|-----------|---------|--------------------------------------------|
| `digits`  | `3`     | decimal places for E and ⟨O⟩ entries.      |
| `header`  | `true`  | emit a header row (`i  E  <op1>  <op2> …`).|

# Returns
A `String` with one row per eigenstate and one column per operator
(plus an `i` and `E` column on the left).
"""
function expectation_table(sys, basis::AbstractBasis,
                           ops_list::AbstractVector{<:Pair{<:AbstractString, <:OperatorSum}};
                           digits::Int = 3,
                           header::Bool = true)
    n_states = size(sys.vectors, 2)
    matrices = [assemble(compile(op, basis), basis) for (_, op) in ops_list]
    names    = String[String(name) for (name, _) in ops_list]

    # Compute everything first so column widths can size to content.
    fmt(x::Real) = string(round(float(x); digits = digits))
    i_strs = [string(i) for i in 1:n_states]
    E_strs = [fmt(sys.values[i]) for i in 1:n_states]
    val_strs = Matrix{String}(undef, n_states, length(ops_list))
    for j in eachindex(matrices)
        M = matrices[j]
        for i in 1:n_states
            ψ = @view sys.vectors[:, i]
            val_strs[i, j] = fmt(real(dot(ψ, M * ψ)))
        end
    end

    # Column widths — at least as wide as the header label, with one
    # space of padding so adjacent columns don't visually collide.
    pad = 2
    w_i = max(length("i"), maximum(length, i_strs; init = 1)) + pad
    w_E = max(length("E"), maximum(length, E_strs; init = 1)) + pad
    w_ops = [max(length(names[j]),
                 maximum(length, @view val_strs[:, j]; init = 1)) + pad
             for j in eachindex(names)]

    io = IOBuffer()
    if header
        print(io, rpad("i", w_i), rpad("E", w_E))
        for j in eachindex(names)
            print(io, rpad(names[j], w_ops[j]))
        end
        println(io)
    end
    for i in 1:n_states
        print(io, rpad(i_strs[i], w_i), rpad(E_strs[i], w_E))
        for j in eachindex(names)
            print(io, rpad(val_strs[i, j], w_ops[j]))
        end
        println(io)
    end
    return String(take!(io))
end
