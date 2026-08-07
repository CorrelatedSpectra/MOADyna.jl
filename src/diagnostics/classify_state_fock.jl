# =====================================================================
# Fock-space point-group classification of a many-body state
# =====================================================================
#
# Implements the deferred "LiftedRep apply": given a many-body coefficient
# vector ψ over an EagerBasis built from a ShellModel, compute the
# point-group matrix elements ⟨ψ|Û(g)|ψ⟩ for every group element g, then
# hand them to MOADyna.PointGroups.classify_state(::AbstractVector, G).
#
# The single-particle rotation acts on creation operators as
#   c†_a → Σ_r U_g[r, a] c†_r,
# block-diagonal over (shell, spin): within a shell of orbital ℓ the spatial
# Wigner matrix D^ℓ(g) (from MOADyna.PointGroups.lift) acts on the 2ℓ+1 orbital
# modes, identically for both spins (a spatial rotation does not touch the
# spin label). For a Slater determinant |occ⟩ (occupied global modes in
# ascending order) the rotated overlap with another determinant |tocc⟩ is
#   ⟨tocc|Û(g)|occ⟩ = det( U_g[tocc, occ] )
# with BOTH index lists in ascending global-mode order — this carries all
# fermion signs automatically (no manual interleave sign). Because U_g is
# block-diagonal, that submatrix is filled from the per-(shell,spin) Wigner
# blocks: entry (target r, source c) is D_block[a'(r), a(c)] when r,c lie in
# the same block, else 0.
#
# SCOPE / LIMITATION. This is a *spatial* (orbital) point-group action with
# the spin label held fixed; it is NOT a double-group / spinor
# representation. It classifies the spatial character of a state. For a
# Hamiltonian with spin–orbit coupling or a magnetic field Û(g) is not in
# general a symmetry, so use this as a spatial-symmetry diagnostic (or apply
# it to a spin-independent model), not as a full relativistic term symbol.

using ..PointGroups: PointGroup, lift
import ..PointGroups: classify_state
using ..Bases: get_index
using LinearAlgebra: det

# One (shell, spin) block: the per-orbital ModeEntry list (a = 1..2ℓ+1 ↔
# m = a-1-ℓ) for ONE spin, the shell index (→ Wigner cache slot), and the
# global bit address of each orbital mode (ascending-order key).
struct _LiftBlock
    s_idx::Int
    entries::Vector{ModeEntry}
    gbits::Vector{Int}
end

# Flat global bit address of a mode (word-aware: the encoding pads words, so
# this is NOT (mode-1); read it from the ModeEntry). Ordering modes by this
# matches the lexicographic bit order the basis is sorted by.
@inline _gbit(e::ModeEntry) = (e.word_idx - 1) * 64 + e.bit_offset

# Build the (shell, spin) blocks once. Local layout is m-major + dn-then-up:
# orbital a has local mode 2(a-1)+1 (dn) or 2(a-1)+2 (up).
function _lift_blocks(basis::AbstractBasis, m::ShellModel)
    enc = encoding(basis)
    blocks = _LiftBlock[]
    for (s_idx, shell) in enumerate(m.shells)
        ℓ = m.ell[shell]
        norb = 2ℓ + 1
        for σ in (1, 2)
            ents = Vector{ModeEntry}(undef, norb)
            gbits = Vector{Int}(undef, norb)
            for a in 1:norb
                e = enc.entries[(shell, (2 * (a - 1) + σ,))]
                ents[a] = e
                gbits[a] = _gbit(e)
            end
            push!(blocks, _LiftBlock(s_idx, ents, gbits))
        end
    end
    return blocks
end

# All size-k subsets of 1:n, as ascending Vector{Int}s (small n; no dep).
function _subsets(n::Int, k::Int)
    out = Vector{Vector{Int}}()
    k < 0 && return out
    k == 0 && return push!(out, Int[])
    k > n && return out
    idx = collect(1:k)
    while true
        push!(out, copy(idx))
        i = k
        while i ≥ 1 && idx[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        idx[i] += 1
        for j in (i + 1):k
            idx[j] = idx[j - 1] + 1
        end
    end
    return out
end

"""
    classify_state(ψ, basis, m, G; tol=1e-6)

Classify the spatial point-group symmetry of a many-body state `ψ`
(expanded over `basis`, an `EagerBasis` built from the `ShellModel` `m`)
under the point group `G`. Returns the same
`(weights, dominant_IR, dominant_weight)` named tuple as
[`classify_state`](@ref)`(matrix_elements, G)`.

The method builds the Fock-space representation `Û(g)` of each group element
from the single-particle Wigner matrices (`MOADyna.PointGroups.lift`), forms
`⟨ψ|Û(g)|ψ⟩` for every `g`, and decomposes the resulting class function into
Mulliken irreps.

# Scope
This is a **spatial / orbital** action with the spin label fixed — not a
double-group (spinor) representation. It reports the spatial irrep
character of `ψ`. For a model with spin–orbit coupling or a magnetic field
`Û(g)` is generally not a symmetry; use this as a spatial diagnostic (e.g.
on a spin-independent cubic Hamiltonian).

# Errors
- `DimensionMismatch` if `length(ψ) != length(basis)`.
- `ErrorException` if a rotated determinant leaves `basis` (the basis is not
  closed under the point-group action — e.g. an orbital-resolved restriction
  the group mixes); the projected overlap is then not a representation
  matrix element, so the routine refuses to silently proceed.
"""
function classify_state(ψ::AbstractVector, basis::AbstractBasis, m::ShellModel,
                        G::PointGroup; tol::Real = 1e-6)
    length(ψ) == length(basis) || throw(DimensionMismatch(
        "classify_state: length(ψ) = $(length(ψ)) != length(basis) = $(length(basis))"))
    nrm = sqrt(real(dot(ψ, ψ)))
    nrm > 0 || throw(ArgumentError("classify_state: ψ has zero norm"))
    me = _lift_matrix_elements(ψ ./ nrm, basis, m, G)
    return classify_state(me, G; tol = tol)
end

"""
    _lift_matrix_elements(ψ, basis, m, G) -> Vector{ComplexF64}

The point-group matrix elements `⟨ψ|Û(g)|ψ⟩` for every `g` in `G.elements`
order, via the Fock-space lifted apply. (`ψ` is used as given — the public
[`classify_state`](@ref) normalises first.) Exposed internally so the same
apply path can build a *subspace* character `χ_S(g) = Σ_i ⟨i|Û(g)|i⟩`.
"""
function _lift_matrix_elements(ψ::AbstractVector, basis::AbstractBasis,
                               m::ShellModel, G::PointGroup)
    rep = lift(G, [m.ell[s] for s in m.shells])
    blocks = _lift_blocks(basis, m)
    n_elems = length(G.elements)
    me = Vector{ComplexF64}(undef, n_elems)
    for g in 1:n_elems
        φ = _apply_lift(g, ψ, basis, rep, blocks)
        me[g] = dot(ψ, φ)
    end
    return me
end

# Amplitude below which a rotated determinant is treated as a true zero and
# skipped. This is a numerical-noise floor on the (exact, algebraic) overlap
# determinant — deliberately *separate* from the public classification `tol`,
# which is a tolerance on the final irrep weights. Kept tiny so the
# basis-closure check (`get_index == 0`) still fires on any real contribution.
const _DET_PRUNE = 1e-12

# φ = Û(g) ψ over the whole basis.
function _apply_lift(g::Int, ψ::AbstractVector, basis::AbstractBasis,
                     rep, blocks::Vector{_LiftBlock})
    φ = zeros(ComplexF64, length(ψ))
    nwords = basis.nwords
    nb = length(blocks)
    Ds = [rep.mode_action[(g, blk.s_idx)] for blk in blocks]   # per-block Wigner
    buf = Vector{UInt64}(undef, nwords)

    src_a = [Int[] for _ in 1:nb]   # occupied orbital indices per block (source)

    for i in 1:length(ψ)
        ψi = ψ[i]
        ψi == 0 && continue
        off = (i - 1) * nwords + 1

        # Source occupation per block.
        for bi in 1:nb
            empty!(src_a[bi])
            blk = blocks[bi]
            for a in eachindex(blk.entries)
                e = blk.entries[a]
                get_bit(basis.states, off, e.word_idx, e.bit_offset) == 1 &&
                    push!(src_a[bi], a)
            end
        end

        # Per-block candidate target orbital subsets (same count as source).
        choices = [_subsets(length(blocks[bi].entries), length(src_a[bi])) for bi in 1:nb]

        # Cartesian product over blocks via a mixed-radix counter.
        counts = ntuple(bi -> length(choices[bi]), nb)
        total = prod(counts; init = 1)
        sel = ones(Int, nb)
        for _ in 1:total
            tgt_a = ntuple(bi -> choices[bi][sel[bi]], nb)
            A = _overlap_det(blocks, Ds, src_a, tgt_a)
            if abs(A) > _DET_PRUNE
                _pack_target!(buf, basis, blocks, tgt_a)
                j = get_index(basis, buf)
                j == 0 && error(
                    "classify_state: a rotated determinant left the basis — the " *
                    "basis is not closed under the point-group action (an " *
                    "orbital-resolved restriction the group mixes?). Cannot form a " *
                    "representation matrix element; refusing to silently project.")
                φ[j] += A * ψi
            end
            # advance mixed-radix counter
            for bi in 1:nb
                sel[bi] += 1
                sel[bi] ≤ counts[bi] && break
                sel[bi] = 1
            end
        end
    end
    return φ
end

# ⟨tgt|Û(g)|src⟩ = det(U_g[tocc, occ]) with both occupations in ascending
# global-mode order. Build the ordered source/target mode lists (each entry
# tagged with its block and orbital), then fill the submatrix from the
# per-block Wigner matrices (zero across different blocks).
function _overlap_det(blocks, Ds, src_a, tgt_a)
    nb = length(blocks)
    # (gbit, block, orbital) for every occupied mode, then sort by gbit.
    src = Tuple{Int,Int,Int}[]
    tgt = Tuple{Int,Int,Int}[]
    for bi in 1:nb
        gb = blocks[bi].gbits
        for a in src_a[bi]; push!(src, (gb[a], bi, a)); end
        for a in tgt_a[bi]; push!(tgt, (gb[a], bi, a)); end
    end
    n = length(src)
    @assert length(tgt) == n
    n == 0 && return ComplexF64(1)        # vacuum overlap
    sort!(src; by = first)
    sort!(tgt; by = first)
    S = Matrix{ComplexF64}(undef, n, n)
    @inbounds for r in 1:n, c in 1:n
        (_, br, ar) = tgt[r]
        (_, bc, ac) = src[c]
        S[r, c] = br == bc ? Ds[br][ar, ac] : ComplexF64(0)
    end
    return det(S)
end

# Pack the target determinant into `buf` (an EagerBasis-shaped UInt64 slice).
function _pack_target!(buf, basis, blocks, tgt_a)
    fill!(buf, UInt64(0))
    @inbounds for bi in eachindex(blocks)
        blk = blocks[bi]
        for a in tgt_a[bi]
            e = blk.entries[a]
            buf[e.word_idx] |= UInt64(1) << e.bit_offset
        end
    end
    return buf
end
