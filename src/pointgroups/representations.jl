# Wigner D^ℓ public surface, IR projector P_Γ, subduction D^ℓ↓G.

"""
    wignerd(g::GroupElement, ℓ::Int) -> Matrix{ComplexF64}

Return the `(2ℓ+1) × (2ℓ+1)` Wigner D-matrix `D^ℓ(g)` for the group
element `g` acting on an angular-momentum shell of rank `ℓ`. This is the
standard representation matrix used to transform spherical-harmonic states
`|ℓ m⟩` under a symmetry operation.

Conventions: active rotation, zyz Euler decomposition
`D^ℓ_{m'm}(α,β,γ) = e^{-im'α} d^ℓ_{m'm}(β) e^{-imγ}`; row index
`m' ∈ -ℓ..+ℓ`, column `m ∈ -ℓ..+ℓ`. Improper rotations (det = −1)
pick up the parity factor `(det g)^ℓ` per §7 (5).

To obtain `D^ℓ(g)` for every element of a group in one call, use
`wignerd_for(G, ℓ)`.
"""
wignerd(g::GroupElement, ℓ::Int) = _wignerd_matrix(g.matrix, ℓ)

"""
    wignerd_for(G::PointGroup, ℓ::Int) -> Vector{Matrix{ComplexF64}}

`D^ℓ(g)` for every element of `G`, in `G.elements` order.
"""
wignerd_for(G::PointGroup, ℓ::Int) = [_wignerd_matrix(e.matrix, ℓ) for e in G.elements]

"""
    subduce(G::PointGroup, ℓ::Int) -> Vector{Pair{Symbol, Int}}

Decompose the angular-momentum representation `D^ℓ` restricted to `G`
into Mulliken irreps with multiplicities. Multiplicities are computed
via complex constituents (per §9.2 step 1) — the factor-of-2 trap for
folded `E_j` cyclic-family irreps is avoided by summing constituent
multiplicities and asserting equality across conjugate pairs.

Returns a vector of `:label => m_Γ` pairs in `G.irreps` order, dropping
zero-multiplicity rows.
"""
function subduce(G::PointGroup, ℓ::Int)
    n_elems = length(G.elements)
    elem_to_class = Vector{Int}(undef, n_elems)
    for (k, C) in enumerate(G.classes), g in C
        elem_to_class[g] = k
    end
    χ_Dl = ComplexF64[tr(_wignerd_matrix(G.elements[g].matrix, ℓ)) for g in 1:n_elems]
    out = Pair{Symbol, Int}[]
    for ir in G.irreps
        # multiplicity per complex constituent
        ms = Int[]
        for cid in ir.complex_constituents
            cir = G.complex_irreps[cid]
            s = ComplexF64(0)
            for g in 1:n_elems
                s += conj(cir.characters[elem_to_class[g]]) * χ_Dl[g]
            end
            m = s / n_elems
            mi = round(Int, real(m))
            abs(m - mi) < 1e-6 || error(
                "subduce: complex multiplicity not integer for $(ir.label): $m")
            push!(ms, mi)
        end
        if length(ms) == 2
            ms[1] == ms[2] || error(
                "subduce: complex-constituent multiplicities differ for folded " *
                "$(ir.label): $(ms[1]) vs $(ms[2]); FS classification bug?")
        end
        m_Γ = ms[1]
        m_Γ > 0 && push!(out, ir.label => m_Γ)
    end
    return out
end

"""
    project(G::PointGroup, label::Symbol, ℓ::Int) -> Matrix{ComplexF64}

The Mulliken-IR projector `P_Γ` acting on `V_ℓ` (the `(2ℓ+1)`-dim
spherical-harmonic space). Built as a sum over complex constituents
per §6.5 / §9.2 (avoids the factor-of-2 cyclic-family trap). The
returned matrix is Hermitian and idempotent (`atol = 1e-9`).
"""
function project(G::PointGroup, label::Symbol, ℓ::Int)
    ir_idx = findfirst(ir -> ir.label === label, G.irreps)
    ir_idx === nothing && throw(ArgumentError(
        "no IR labelled :$label in group $(G.name); have " *
        join(string.(getfield.(G.irreps, :label)), ", ")))
    ir = G.irreps[ir_idx]
    n_elems = length(G.elements)
    elem_to_class = Vector{Int}(undef, n_elems)
    for (k, C) in enumerate(G.classes), g in C
        elem_to_class[g] = k
    end
    dim = 2ℓ + 1
    P = zeros(ComplexF64, dim, dim)
    for cid in ir.complex_constituents
        cir = G.complex_irreps[cid]
        Pγ = zeros(ComplexF64, dim, dim)
        for g in 1:n_elems
            Pγ .+= conj(cir.characters[elem_to_class[g]]) *
                   _wignerd_matrix(G.elements[g].matrix, ℓ)
        end
        Pγ .*= cir.dim / n_elems
        P .+= Pγ
    end
    return P
end
