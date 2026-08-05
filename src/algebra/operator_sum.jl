# =====================================================================
# OperatorSum — typed symbolic sum of operator terms
# =====================================================================
#
# Terms are keyed in an OrderedDict by their chain id (the tuple of
# LadderEntry's in the order they appear). The empty chain `()` is the
# constant / identity term — a bare scalar `α` lifts to `OperatorTerm(α, ())`.

"""
    Chain

A tuple of `LadderEntry`s in the order written (no normal-ordering).
The empty tuple `()` is the constant / identity term.
"""
const Chain = Tuple{Vararg{LadderEntry}}

"""
    OperatorTerm{T<:Number}

A single term in an `OperatorSum`, obtained by iterating the sum (not by
direct construction). Access its data via the `.coefficient` and `.chain`
fields:

```julia
for term in op
    println(term.coefficient, " × ", term.chain)
end
```

A single term in an `OperatorSum`: a coefficient and a chain.
"""
struct OperatorTerm{T<:Number}
    coefficient::T
    chain::Chain
end

coefficient(t::OperatorTerm) = t.coefficient
chain(t::OperatorTerm) = t.chain
id(t::OperatorTerm) = t.chain    # the chain itself serves as the dict key

Base.show(io::IO, t::OperatorTerm) = begin
    print(io, t.coefficient)
    if isempty(t.chain)
        print(io, " * I")
    else
        for e in t.chain
            print(io, " * ", e)
        end
    end
end

"""
    OperatorSum{T<:Number}

A sum of `OperatorTerm`s with deduplication by chain id. Lookup is O(1) hash
in an `OrderedDict`; iteration order is insertion order (deterministic).
"""
struct OperatorSum{T<:Number}
    terms::OrderedDict{Chain, OperatorTerm{T}}
end

OperatorSum{T}() where {T<:Number} = OperatorSum(OrderedDict{Chain, OperatorTerm{T}}())

# --- Construct a one-term sum (called from operators.jl) ---

function _make_oneterm(e::LadderEntry)
    chain::Chain = (e,)
    term = OperatorTerm(1.0, chain)
    d = OrderedDict{Chain, OperatorTerm{Float64}}()
    d[chain] = term
    return OperatorSum{Float64}(d)
end

# --- Identity / zero ---

"""
    one(::OperatorSum{T}) -> OperatorSum{T}
    one(::Type{OperatorSum{T}}) -> OperatorSum{T}

The identity operator: a one-term sum with empty chain `()` and coefficient `one(T)`.
"""
function Base.one(::Type{OperatorSum{T}}) where {T<:Number}
    d = OrderedDict{Chain, OperatorTerm{T}}()
    d[()] = OperatorTerm(one(T), ())
    return OperatorSum{T}(d)
end
Base.one(::OperatorSum{T}) where {T<:Number} = one(OperatorSum{T})

"""
    zero(::OperatorSum{T}) -> OperatorSum{T}

The zero operator: an empty sum.
"""
Base.zero(::OperatorSum{T}) where {T} = OperatorSum{T}()
Base.zero(::Type{OperatorSum{T}}) where {T<:Number} = OperatorSum{T}()

# --- Iteration / length ---

Base.length(op::OperatorSum) = length(op.terms)
Base.isempty(op::OperatorSum) = isempty(op.terms)
Base.iterate(op::OperatorSum, args...) = iterate(values(op.terms), args...)

# --- Display ---

function Base.show(io::IO, op::OperatorSum{T}) where {T}
    print(io, "OperatorSum{", T, "} with ", length(op), " term(s)")
    if length(op) <= 6
        for term in op
            print(io, "\n  ", term)
        end
    else
        for (i, term) in enumerate(op)
            i > 4 && (print(io, "\n  ⋮"); break)
            print(io, "\n  ", term)
        end
    end
end
