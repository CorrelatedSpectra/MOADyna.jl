# =====================================================================
# Algebra — +, *, scalar*, adjoint, chop on OperatorSum
# =====================================================================

# --- Addition: term-wise merge with promotion and zero-deletion ---

function Base.:+(a::OperatorSum{T₁}, b::OperatorSum{T₂}) where {T₁, T₂}
    T = promote_type(T₁, T₂)
    out = OperatorSum{T}()
    _add_inplace!(out, a)
    _add_inplace!(out, b)
    return out
end

function _add_inplace!(out::OperatorSum{T}, src::OperatorSum) where {T}
    for term in src
        _push!(out, term.coefficient, term.chain)
    end
    return out
end

# Push (coef, chain): merge with existing entry if any; delete if zero.
function _push!(out::OperatorSum{T}, coef, chain::Chain) where {T}
    existing = get(out.terms, chain, nothing)
    new_coef = if existing === nothing
        T(coef)
    else
        T(existing.coefficient + coef)
    end
    if iszero(new_coef)
        haskey(out.terms, chain) && delete!(out.terms, chain)
    else
        out.terms[chain] = OperatorTerm(new_coef, chain)
    end
    return out
end

Base.:-(a::OperatorSum) = (-1) * a
Base.:-(a::OperatorSum, b::OperatorSum) = a + (-b)

# --- Scalar arithmetic ---

function Base.:*(α::Number, a::OperatorSum{T}) where {T}
    iszero(α) && return OperatorSum{promote_type(T, typeof(α))}()
    Tnew = promote_type(T, typeof(α))
    out = OperatorSum{Tnew}()
    for term in a
        new_coef = Tnew(α * term.coefficient)
        out.terms[term.chain] = OperatorTerm(new_coef, term.chain)
    end
    return out
end

Base.:*(a::OperatorSum, α::Number) = α * a
Base.:/(a::OperatorSum, α::Number) = inv(α) * a

# --- Constant lift: scalar + OperatorSum ---

function Base.:+(a::OperatorSum{T}, α::Number) where {T}
    Tnew = promote_type(T, typeof(α))
    out = OperatorSum{Tnew}()
    _add_inplace!(out, a)
    _push!(out, α, ())
    return out
end

Base.:+(α::Number, a::OperatorSum) = a + α
Base.:-(a::OperatorSum, α::Number) = a + (-α)
Base.:-(α::Number, a::OperatorSum) = (-a) + α

# --- Multiplication: Cartesian product of terms, with Tier-2 canonicalization ---

function Base.:*(a::OperatorSum{T₁}, b::OperatorSum{T₂}) where {T₁, T₂}
    T = promote_type(T₁, T₂)
    out = OperatorSum{T}()
    for ta in a
        for tb in b
            base_chain = (ta.chain..., tb.chain...)
            base_coef = T(ta.coefficient * tb.coefficient)
            for (factor, canon_chain) in canonicalize(base_chain)
                _push!(out, base_coef * factor, canon_chain)
            end
        end
    end
    return out
end

# --- Adjoint ---

"""
    adjoint(a::OperatorSum) -> OperatorSum

Conjugate transpose. For each term:
- reverse the chain
- swap each entry's `kind` per `adjoint_kind` (c↔cdag, b↔bdag, S+↔S-, Sx/Sy/Sz unchanged)
- conjugate the coefficient
"""
function Base.adjoint(a::OperatorSum{T}) where {T}
    out = OperatorSum{T}()   # adjoint preserves T (real stays real, complex stays complex up to conj)
    for term in a
        rev_chain = _adjoint_chain(term.chain)
        new_coef = conj(term.coefficient)
        # Reversal can violate canonical order — re-canonicalize
        for (factor, canon_chain) in canonicalize(rev_chain)
            _push!(out, new_coef * factor, canon_chain)
        end
    end
    return out
end

function _adjoint_chain(c::Chain)
    return ntuple(i -> _adjoint_entry(c[end - i + 1]), length(c))
end

_adjoint_entry(e::LadderEntry) = LadderEntry(adjoint_kind(e.kind), e.site, e.label)

# --- Hermitian-conjugate convenience (+h.c.) ---

"""
    add_hc(a::OperatorSum) -> OperatorSum

Returns `a + a'`. Convenience for the very common "+h.c." pattern in
Hamiltonian construction. **Not** the Hermitian conjugate (that's `'`
or `adjoint(a)`).
"""
add_hc(a::OperatorSum) = a + adjoint(a)

# --- chop: drop terms with small coefficients ---

"""
    chop(a::OperatorSum; tol=1e-12) -> OperatorSum

Return a new `OperatorSum` containing every term of `a` whose coefficient
magnitude satisfies `|coefficient| ≥ tol`; terms with `|coefficient| < tol`
are dropped. Extends `Base.chop` for operator algebra.
"""
function Base.chop(a::OperatorSum{T}; tol = 1e-12) where {T}
    out = OperatorSum{T}()
    for term in a
        abs(term.coefficient) >= tol && (out.terms[term.chain] = term)
    end
    return out
end

# --- Equality (structural) ---

function Base.:(==)(a::OperatorSum, b::OperatorSum)
    length(a) == length(b) || return false
    for term in a
        other = get(b.terms, term.chain, nothing)
        other === nothing && return false
        term.coefficient == other.coefficient || return false
    end
    return true
end

"""
    isapprox(a::OperatorSum, b::OperatorSum; atol=1e-10, rtol=…) -> Bool

Term-by-term coefficient comparison up to tolerance. Two sums are approximately
equal iff every chain present in either operand has matching coefficients
(within `atol` / `rtol`); coefficients absent from one side are treated as zero
and must compare ≈ 0 against the other side.
"""
function Base.isapprox(a::OperatorSum, b::OperatorSum;
                       atol::Real = 0,
                       rtol::Real = atol > 0 ? 0 : sqrt(eps()))
    chains = union(keys(a.terms), keys(b.terms))
    for chain in chains
        ca = haskey(a.terms, chain) ? a.terms[chain].coefficient : 0
        cb = haskey(b.terms, chain) ? b.terms[chain].coefficient : 0
        isapprox(ca, cb; atol = atol, rtol = rtol) || return false
    end
    return true
end
