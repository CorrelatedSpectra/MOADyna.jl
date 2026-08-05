# Reference character tables (production layer).
#
# Each entry is a curated, source-attributed character table for one
# (group, setting). The Burnside-computed table (teaching/diagnostics
# layer in `group.jl`) must match the reference up to a class
# permutation — `validate_against_reference!` enforces this at
# construction time.
#
# The reference table carries everything needed to produce canonical
# Mulliken-labelled output:
#   - class labels in canonical order (E, 8C3, 6C2, ...)
#   - class sizes
#   - per-class fingerprint = (size, order, det_sign, trace, anchor)
#     used to align computed-class indices to the canonical order
#   - IR labels (Mulliken)
#   - character matrix
#   - source provenance string
#
# Provenance sources are logged in the `source` string. The values in
# this file are widely-published character tables; correctness is
# enforced by Burnside-computed cross-validation at construction.
#
# Primary reference for all 32 crystallographic point groups:
#   Bilbao Crystallographic Server: Aroyo et al., Acta Cryst. A62,
#   115-128 (2006). https://www.cryst.ehu.es/rep/point
#   Cross-validated against Burnside-computed tables at construction time.
#
# Non-crystallographic groups (C5, D5, D4d, D6d, I, Ih) use standard
# published tables and are NOT cross-validated against Bilbao.

struct ReferenceCharacterTable
    group::Symbol
    setting::Symbol                                   # currently always :default
    classes::Vector{Symbol}                           # canonical class labels
    class_sizes::Vector{Int}
    class_fingerprints::Vector{NTuple{5, Any}}        # (size, order, det, trace, anchor)
    irreps::Vector{Symbol}                            # canonical Mulliken labels
    ir_dims::Vector{Int}
    characters::Matrix{Float64}                       # (n_irrep × n_class)
    source::String
end

# ─── Class-fingerprint primitives (carried over from previous file) ──

# Order of an orthogonal 3×3 matrix.
function _matrix_order(R::AbstractMatrix; max_order::Int=20)
    Rk = Matrix{Float64}(I(3))
    for k in 1:max_order
        Rk = Rk * R
        norm(Rk - I) < 1e-9 && return k
    end
    return max_order
end

function _primary_class_fingerprint(class_idx::Int, classes::Vector{Vector{Int}},
                                     elements::Vector{GroupElement})
    rep = elements[classes[class_idx][1]]
    R = rep.matrix
    size_c = length(classes[class_idx])
    order = _matrix_order(Matrix{Float64}(R))
    d = det(R) > 0 ? 1 : -1
    tr_r = round(real(tr(R)), digits=6)
    abs(tr_r) < 1e-9 && (tr_r = 0.0)   # collapse -0.0 → 0.0
    return (size_c, order, d, tr_r)
end

function _principal_class(classes::Vector{Vector{Int}}, elements::Vector{GroupElement})
    best = 0
    best_score = (-1, 0, 0)
    for k in eachindex(classes)
        rep = elements[classes[k][1]]
        det(rep.matrix) > 0.5 || continue
        ord = _matrix_order(Matrix{Float64}(rep.matrix))
        ord ≥ 2 || continue
        score = (-ord, length(classes[k]), k)
        if best == 0 || score < best_score
            best = k
            best_score = score
        end
    end
    return best
end

function _secondary_tag(class_idx::Int, principal_idx::Int,
                        classes::Vector{Vector{Int}}, multable::Matrix{Int})
    principal_idx == 0 && return 0
    class_idx == principal_idx && return 0
    elem_to_class = Vector{Int}(undef, size(multable, 1))
    for (k, C) in enumerate(classes), g in C
        elem_to_class[g] = k
    end
    counts = zeros(Int, length(classes))
    for g in classes[class_idx], h in classes[principal_idx]
        counts[elem_to_class[multable[g, h]]] += 1
    end
    pairs = sort(collect(enumerate(counts)); by=p -> -p[2])
    tag = 0
    for (idx, c) in pairs
        c == 0 && continue
        tag = tag * 100 + (idx * 10 + min(c, 9))
        tag > 10^9 && break
    end
    return tag
end

# Generator-tag anchors for σv/σd-style primary collisions.
_anchor_for_tag(tag::Symbol) = tag === :σv  ? 1001 :
                               tag === :σd  ? 1002 :
                               tag === :σh  ? 1003 :
                               tag === :C2x ? 1004 :
                               tag === :C2z ? 1005 :
                               tag === :C2y ? 1006 :
                               tag === :σyz ? 1010 :   # plane normal x
                               tag === :σxz ? 1011 :   # plane normal y
                               tag === :σxy ? 1012 :   # plane normal z
                               tag === :S4z ? 1020 :   # S4 class (S4³ gets complement 1021)
                               0

_complement_anchor(a::Int) = a == 1001 ? 1002 :
                              a == 1002 ? 1001 :
                              a == 1004 ? 1005 :
                              a == 1005 ? 1004 :
                              a == 1020 ? 1021 :
                              a == 1021 ? 1020 :
                              0

function _class_fingerprints(classes::Vector{Vector{Int}},
                              elements::Vector{GroupElement}, multable::Matrix{Int};
                              generators::Vector{GroupElement}=GroupElement[])
    primary = [_primary_class_fingerprint(k, classes, elements) for k in eachindex(classes)]
    dupes = Set{NTuple{4, Any}}()
    seen = Set{NTuple{4, Any}}()
    for p in primary
        p ∈ seen ? push!(dupes, p) : push!(seen, p)
    end
    isempty(dupes) && return [(primary[k]..., 0) for k in eachindex(classes)]

    gen_tag = zeros(Int, length(classes))
    for gen in generators
        idx = findfirst(e -> _isapprox(e.matrix, gen.matrix), elements)
        idx === nothing && continue
        anchor = _anchor_for_tag(gen.tag)
        anchor == 0 && continue
        for (k, C) in enumerate(classes)
            if idx ∈ C
                gen_tag[k] = anchor
                break
            end
        end
    end
    for k in eachindex(classes), j in eachindex(classes)
        k == j && continue
        primary[k] == primary[j] || continue
        gen_tag[j] != 0 && continue
        gen_tag[k] != 0 || continue
        gen_tag[j] = _complement_anchor(gen_tag[k])
    end

    principal = _principal_class(classes, elements)
    return [primary[k] ∈ dupes ?
            (primary[k]...,
             gen_tag[k] != 0 ? gen_tag[k] :
             _secondary_tag(k, principal, classes, multable)) :
            (primary[k]..., 0)
            for k in eachindex(classes)]
end

# ─── Curated reference tables ────────────────────────────────────────
#
# Source attribution: the character values are widely-published. The
# specific source listed per group is the one consulted to fix the
# canonical ordering of classes and IRs; multiple sources agree on
# the values modulo Mulliken labelling conventions.
#
# At construction time, the Burnside-computed character table for the
# group is matched against this reference (after class permutation
# alignment) and any mismatch raises `ArgumentError` — that's the
# primary correctness oracle. A typo in this file will surface as a
# loud failure on the affected group's first construction.

function _build_reference_tables()
    tables = Dict{Symbol, ReferenceCharacterTable}()

    # Helper to stamp source provenance + build the struct.
    function _ref(group, classes, sizes, sigs, irreps, dims, chars, source)
        ReferenceCharacterTable(group, :default,
                                classes, sizes, sigs, irreps, dims,
                                Float64.(chars), source)
    end

    # ─── C1 ─────────────────────────────────────────────────────────
    tables[:C1] = _ref(:C1,
        [:E], [1],
        [(1, 1, 1, 3.0, 0)],
        [:A], [1], reshape([1.0], 1, 1),
        "trivial group; standard textbook")

    # ─── Ci ─────────────────────────────────────────────────────────
    tables[:Ci] = _ref(:Ci,
        [:E, :i], [1, 1],
        [(1, 1, 1, 3.0, 0), (1, 2, -1, -3.0, 0)],
        [:Ag, :Au], [1, 1],
        [1.0  1.0;
         1.0 -1.0],
        "standard published table")

    # ─── Cs ─────────────────────────────────────────────────────────
    tables[:Cs] = _ref(:Cs,
        [:E, :σh], [1, 1],
        [(1, 1, 1, 3.0, 0), (1, 2, -1, 1.0, 0)],
        [:Aprime, :Adprime], [1, 1],
        [1.0  1.0;
         1.0 -1.0],
        "standard published table")

    # ─── C2 ─────────────────────────────────────────────────────────
    tables[:C2] = _ref(:C2,
        [:E, :C2], [1, 1],
        [(1, 1, 1, 3.0, 0), (1, 2, 1, -1.0, 0)],
        [:A, :B], [1, 1],
        [1.0  1.0;
         1.0 -1.0],
        "standard published table")

    # ─── C2v ────────────────────────────────────────────────────────
    tables[:C2v] = _ref(:C2v,
        [:E, :C2, :σv, :σd], [1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 2, 1, -1.0, 0),
         (1, 2, -1, 1.0, 1001),
         (1, 2, -1, 1.0, 1002)],
        [:A1, :A2, :B1, :B2], [1, 1, 1, 1],
        [1.0  1.0  1.0  1.0;
         1.0  1.0 -1.0 -1.0;
         1.0 -1.0  1.0 -1.0;
         1.0 -1.0 -1.0  1.0],
        "standard published table")

    # ─── C2h ────────────────────────────────────────────────────────
    tables[:C2h] = _ref(:C2h,
        [:E, :C2, :i, :σh], [1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 2, 1, -1.0, 0),
         (1, 2, -1, -3.0, 0),
         (1, 2, -1, 1.0, 0)],
        [:Ag, :Bg, :Au, :Bu], [1, 1, 1, 1],
        [1.0  1.0  1.0  1.0;
         1.0 -1.0  1.0 -1.0;
         1.0  1.0 -1.0 -1.0;
         1.0 -1.0 -1.0  1.0],
        "standard published table")

    # ─── C3v ────────────────────────────────────────────────────────
    tables[:C3v] = _ref(:C3v,
        [:E, :C3, :σv], [1, 2, 3],
        [(1, 1, 1, 3.0, 0),
         (2, 3, 1, 0.0, 0),
         (3, 2, -1, 1.0, 0)],
        [:A1, :A2, :E], [1, 1, 2],
        [1.0  1.0  1.0;
         1.0  1.0 -1.0;
         2.0 -1.0  0.0],
        "standard published table")

    # ─── C4v ────────────────────────────────────────────────────────
    tables[:C4v] = _ref(:C4v,
        [:E, :C4, :C2, :σv, :σd], [1, 2, 1, 2, 2],
        [(1, 1, 1, 3.0, 0),
         (2, 4, 1, 1.0, 0),
         (1, 2, 1, -1.0, 0),
         (2, 2, -1, 1.0, 1001),
         (2, 2, -1, 1.0, 1002)],
        [:A1, :A2, :B1, :B2, :E], [1, 1, 1, 1, 2],
        [1.0  1.0  1.0  1.0  1.0;
         1.0  1.0  1.0 -1.0 -1.0;
         1.0 -1.0  1.0  1.0 -1.0;
         1.0 -1.0  1.0 -1.0  1.0;
         2.0  0.0 -2.0  0.0  0.0],
        "standard published table")

    # ─── D2h ────────────────────────────────────────────────────────
    tables[:D2h] = _ref(:D2h,
        [:E, :C2z, :C2y, :C2x, :i, :σxy, :σxz, :σyz],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 2, 1, -1.0, 1005),
         (1, 2, 1, -1.0, 1006),
         (1, 2, 1, -1.0, 1004),
         (1, 2, -1, -3.0, 0),
         (1, 2, -1, 1.0, 1012),
         (1, 2, -1, 1.0, 1011),
         (1, 2, -1, 1.0, 1010)],
        [:Ag, :B1g, :B2g, :B3g, :Au, :B1u, :B2u, :B3u],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0;
         1.0  1.0 -1.0 -1.0  1.0  1.0 -1.0 -1.0;
         1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0;
         1.0 -1.0 -1.0  1.0  1.0 -1.0 -1.0  1.0;
         1.0  1.0  1.0  1.0 -1.0 -1.0 -1.0 -1.0;
         1.0  1.0 -1.0 -1.0 -1.0 -1.0  1.0  1.0;
         1.0 -1.0  1.0 -1.0 -1.0  1.0 -1.0  1.0;
         1.0 -1.0 -1.0  1.0 -1.0  1.0  1.0 -1.0],
        "standard published table; B1↔C2z, B2↔C2y, B3↔C2x")

    # ─── D4h ────────────────────────────────────────────────────────
    tables[:D4h] = _ref(:D4h,
        [:E, :C4, :C2, :C2pr, :C2dpr, :i, :S4, :σh, :σv, :σd],
        [1, 2, 1, 2, 2, 1, 2, 1, 2, 2],
        [(1, 1, 1, 3.0, 0),
         (2, 4, 1, 1.0, 0),
         (1, 2, 1, -1.0, 0),
         (2, 2, 1, -1.0, 1004),
         (2, 2, 1, -1.0, 1005),
         (1, 2, -1, -3.0, 0),
         (2, 4, -1, -1.0, 0),
         (1, 2, -1, 1.0, 0),
         (2, 2, -1, 1.0, 1001),
         (2, 2, -1, 1.0, 1002)],
        [:A1g, :A2g, :B1g, :B2g, :Eg, :A1u, :A2u, :B1u, :B2u, :Eu],
        [1, 1, 1, 1, 2, 1, 1, 1, 1, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0;
         1.0  1.0  1.0 -1.0 -1.0  1.0  1.0  1.0 -1.0 -1.0;
         1.0 -1.0  1.0  1.0 -1.0  1.0 -1.0  1.0  1.0 -1.0;
         1.0 -1.0  1.0 -1.0  1.0  1.0 -1.0  1.0 -1.0  1.0;
         2.0  0.0 -2.0  0.0  0.0  2.0  0.0 -2.0  0.0  0.0;
         1.0  1.0  1.0  1.0  1.0 -1.0 -1.0 -1.0 -1.0 -1.0;
         1.0  1.0  1.0 -1.0 -1.0 -1.0 -1.0 -1.0  1.0  1.0;
         1.0 -1.0  1.0  1.0 -1.0 -1.0  1.0 -1.0 -1.0  1.0;
         1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0;
         2.0  0.0 -2.0  0.0  0.0 -2.0  0.0  2.0  0.0  0.0],
        "standard published table; C2'↔axial (contains C2x), C2''↔diagonal")

    # ─── Td ─────────────────────────────────────────────────────────
    tables[:Td] = _ref(:Td,
        [:E, :C3, :C2, :S4, :σd], [1, 8, 3, 6, 6],
        [(1, 1, 1, 3.0, 0),
         (8, 3, 1, 0.0, 0),
         (3, 2, 1, -1.0, 0),
         (6, 4, -1, -1.0, 0),
         (6, 2, -1, 1.0, 0)],
        [:A1, :A2, :E, :T1, :T2], [1, 1, 2, 3, 3],
        [1.0  1.0  1.0  1.0  1.0;
         1.0  1.0  1.0 -1.0 -1.0;
         2.0 -1.0  2.0  0.0  0.0;
         3.0  0.0 -1.0  1.0 -1.0;
         3.0  0.0 -1.0 -1.0  1.0],
        "standard published table")

    # ─── O ──────────────────────────────────────────────────────────
    tables[:O] = _ref(:O,
        [:E, :C3, :C2sq, :C4, :C2pr], [1, 8, 3, 6, 6],
        [(1, 1, 1, 3.0, 0),
         (8, 3, 1, 0.0, 0),
         (3, 2, 1, -1.0, 0),
         (6, 4, 1, 1.0, 0),
         (6, 2, 1, -1.0, 0)],
        [:A1, :A2, :E, :T1, :T2], [1, 1, 2, 3, 3],
        [1.0  1.0  1.0  1.0  1.0;
         1.0  1.0  1.0 -1.0 -1.0;
         2.0 -1.0  2.0  0.0  0.0;
         3.0  0.0 -1.0  1.0 -1.0;
         3.0  0.0 -1.0 -1.0  1.0],
        "standard published table; C2sq=C4², C2pr=face-diag")

    # ─── Oh ─────────────────────────────────────────────────────────
    tables[:Oh] = _ref(:Oh,
        [:E, :C3, :C2pr, :C4, :C2sq, :i, :S4, :S6, :σh, :σd],
        [1, 8, 6, 6, 3, 1, 6, 8, 3, 6],
        [(1, 1, 1, 3.0, 0),
         (8, 3, 1, 0.0, 0),
         (6, 2, 1, -1.0, 0),
         (6, 4, 1, 1.0, 0),
         (3, 2, 1, -1.0, 0),
         (1, 2, -1, -3.0, 0),
         (6, 4, -1, -1.0, 0),
         (8, 6, -1, 0.0, 0),
         (3, 2, -1, 1.0, 0),
         (6, 2, -1, 1.0, 0)],
        [:A1g, :A2g, :Eg, :T1g, :T2g, :A1u, :A2u, :Eu, :T1u, :T2u],
        [1, 1, 2, 3, 3, 1, 1, 2, 3, 3],
        [1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0;
         1.0  1.0 -1.0 -1.0  1.0  1.0 -1.0  1.0  1.0 -1.0;
         2.0 -1.0  0.0  0.0  2.0  2.0  0.0 -1.0  2.0  0.0;
         3.0  0.0 -1.0  1.0 -1.0  3.0  1.0  0.0 -1.0 -1.0;
         3.0  0.0  1.0 -1.0 -1.0  3.0 -1.0  0.0 -1.0  1.0;
         1.0  1.0  1.0  1.0  1.0 -1.0 -1.0 -1.0 -1.0 -1.0;
         1.0  1.0 -1.0 -1.0  1.0 -1.0  1.0 -1.0 -1.0  1.0;
         2.0 -1.0  0.0  0.0  2.0 -2.0  0.0  1.0 -2.0  0.0;
         3.0  0.0 -1.0  1.0 -1.0 -3.0 -1.0  0.0  1.0  1.0;
         3.0  0.0  1.0 -1.0 -1.0 -3.0  1.0  0.0  1.0 -1.0],
        "standard published table; T1↔χ(C2pr)=-1, T2↔χ(C2pr)=+1")

    # ─── D2 ─────────────────────────────────────────────────────────────
    # Proper dihedral, order 4, 4 one-dimensional IRs.
    # All C2 classes share primary fingerprint (1,2,+1,-1.0); anchors
    # distinguish them. C2y has no dedicated generator tag in the minimal
    # generator set; adding (C2y, :C2y) to data.jl resolves it.
    tables[:D2] = _ref(:D2,
        [:E, :C2z, :C2y, :C2x], [1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 2, 1, -1.0, 1005),
         (1, 2, 1, -1.0, 1006),
         (1, 2, 1, -1.0, 1004)],
        [:A, :B1, :B2, :B3], [1, 1, 1, 1],
        [1.0  1.0  1.0  1.0;
         1.0  1.0 -1.0 -1.0;
         1.0 -1.0  1.0 -1.0;
         1.0 -1.0 -1.0  1.0],
        "standard published table; B1↔C2z, B2↔C2y, B3↔C2x")

    # ─── D3 ─────────────────────────────────────────────────────────────
    # Proper dihedral, order 6, 2 one-dim + 1 two-dim IRs.
    # All three classes have distinct primary fingerprints; no anchors needed.
    tables[:D3] = _ref(:D3,
        [:E, :C3, :C2pr], [1, 2, 3],
        [(1, 1, 1, 3.0, 0),
         (2, 3, 1, 0.0, 0),
         (3, 2, 1, -1.0, 0)],
        [:A1, :A2, :E], [1, 1, 2],
        [1.0  1.0  1.0;
         1.0  1.0 -1.0;
         2.0 -1.0  0.0],
        "standard published table")

    # ─── D3h ────────────────────────────────────────────────────────────
    # D3h = D3 × σh, order 12, 4 one-dim + 2 two-dim IRs.
    # Prime (') IRs are symmetric under σh; double-prime ('') are antisymmetric.
    # Julia symbol naming: Aprime/Adprime (matching Cs convention).
    # 2S3 trace: S3 = σh·C3z (σh = diag(1,1,-1)). C3z diagonal = (-1/2,-1/2,1),
    #   so S3 diagonal = (-1/2,-1/2,-1), trace = -2.
    tables[:D3h] = _ref(:D3h,
        [:E, :C3, :C2pr, :σh, :S3, :σv], [1, 2, 3, 1, 2, 3],
        [(1, 1, 1, 3.0, 0),
         (2, 3, 1, 0.0, 0),
         (3, 2, 1, -1.0, 0),
         (1, 2, -1, 1.0, 0),
         (2, 6, -1, -2.0, 0),
         (3, 2, -1, 1.0, 0)],
        [:A1prime, :A2prime, :Eprime, :A1dprime, :A2dprime, :Edprime],
        [1, 1, 2, 1, 1, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0;
         1.0  1.0 -1.0  1.0  1.0 -1.0;
         2.0 -1.0  0.0  2.0 -1.0  0.0;
         1.0  1.0  1.0 -1.0 -1.0 -1.0;
         1.0  1.0 -1.0 -1.0 -1.0  1.0;
         2.0 -1.0  0.0 -2.0  1.0  0.0],
        "standard published table; σh on z, C3 on z, C2' through x")

    # ─── D3d ────────────────────────────────────────────────────────────
    # D3d = D3 × i, order 12, 3 g + 3 u IRs.
    # 2S6 trace: S6 = i·C3, det = -1, tr(S6) = -tr(C3) = 0.
    # All six classes have distinct primary fingerprints; no anchors needed.
    tables[:D3d] = _ref(:D3d,
        [:E, :C3, :C2pr, :i, :S6, :σd], [1, 2, 3, 1, 2, 3],
        [(1, 1, 1, 3.0, 0),
         (2, 3, 1, 0.0, 0),
         (3, 2, 1, -1.0, 0),
         (1, 2, -1, -3.0, 0),
         (2, 6, -1, 0.0, 0),
         (3, 2, -1, 1.0, 0)],
        [:A1g, :A2g, :Eg, :A1u, :A2u, :Eu],
        [1, 1, 2, 1, 1, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0;
         1.0  1.0 -1.0  1.0  1.0 -1.0;
         2.0 -1.0  0.0  2.0 -1.0  0.0;
         1.0  1.0  1.0 -1.0 -1.0 -1.0;
         1.0  1.0 -1.0 -1.0 -1.0  1.0;
         2.0 -1.0  0.0 -2.0  1.0  0.0],
        "standard published table; C3 on z, C2' through x, σd planes contain C3 and bisect C2'")

    # ─── C5 ─────────────────────────────────────────────────────────────
    # Cyclic group of order 5.  All elements are in their own conjugacy class
    # (abelian group).  The 5 complex 1-D irreps fold into 3 displayed rows:
    # A (trivial) + E1 (pair {χ₁,χ₄}) + E2 (pair {χ₂,χ₃}).
    # Displayed characters are 2·Re of the complex constituents.
    # 2cos(2π/5) = (√5-1)/2 ≈ 0.618034;  2cos(4π/5) = -(√5+1)/2 ≈ -1.618034.
    # Secondary tags distinguish the two pairs of degenerate fingerprints
    # (classes C5/C5⁴ share primary fp, as do C5²/C5³).
    tables[:C5] = _ref(:C5,
        [:E, :C5, :C5sq, :C5cu, :C5fo],
        [1, 1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 5, 1, 1.618034, 0),
         (1, 5, 1, -0.618034, 41),
         (1, 5, 1, -0.618034, 51),
         (1, 5, 1, 1.618034, 11)],
        [:A, :E1, :E2], [1, 2, 2],
        [1.0   1.0        1.0       1.0       1.0;
         2.0   0.618034  -1.618034 -1.618034  0.618034;
         2.0  -1.618034   0.618034  0.618034 -1.618034],
        "standard published table; φ=(1+√5)/2, 2cos(2π/5)=φ-1≈0.618, 2cos(4π/5)=-φ≈-1.618")

    # ─── C5v ────────────────────────────────────────────────────────────
    # C5v = C5 + 5 vertical mirrors.  Order 10, 4 classes.
    # Canonical class order: E, 2C5, 2C5², 5σv.
    tables[:C5v] = _ref(:C5v,
        [:E, :C5, :C5sq, :σv],
        [1, 2, 2, 5],
        [(1, 1, 1, 3.0, 0),
         (2, 5, 1, 1.618034, 0),
         (2, 5, 1, -0.618034, 0),
         (5, 2, -1, 1.0, 0)],
        [:A1, :A2, :E1, :E2], [1, 1, 2, 2],
        [1.0   1.0        1.0       1.0;
         1.0   1.0        1.0      -1.0;
         2.0   0.618034  -1.618034   0.0;
         2.0  -1.618034   0.618034   0.0],
        "standard published table; C5 on z, σv through x")

    # ─── C5h ────────────────────────────────────────────────────────────
    # C5h = C5 × Cs.  Order 10, 10 classes (all singleton, abelian).
    # Canonical class order: E, C5, C5², C5³, C5⁴, σh, S5, S5⁷, S5³, S5⁹.
    # S5^k = σh·(C5z)^k.  Traces of S5 family:
    #   S5¹:  tr = 2cos(2π/5) - 1 ≈ -0.381966
    #   S5⁷:  tr = 2cos(14π/5) - 1 = 2cos(4π/5) - 1 ≈ -2.618034
    #   S5³:  tr = 2cos(6π/5)  - 1 = 2cos(6π/5)  - 1 ≈ -2.618034
    #   S5⁹:  tr = 2cos(18π/5) - 1 = 2cos(2π/5)  - 1 ≈ -0.381966
    # Secondary tags from _class_fingerprints distinguish the 4 pairs.
    # 6 displayed rows: A', E1', E2', A'', E1'', E2'' (prime=even under σh).
    let c1 = 0.618034, c2 = -1.618034
        tables[:C5h] = _ref(:C5h,
            [:E, :C5, :C5sq, :C5cu, :C5fo, :σh, :S5, :S5se, :S5cu, :S5ni],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [(1, 1, 1, 3.0, 0),
             (1, 5, 1, 1.618034, 0),
             (1, 5, 1, -0.618034, 71),
             (1, 5, 1, -0.618034, 81),
             (1, 5, 1, 1.618034, 11),
             (1, 2, -1, 1.0, 0),
             (1, 10, -1, -0.381966, 61),
             (1, 10, -1, -2.618034, 91),
             (1, 10, -1, -2.618034, 101),
             (1, 10, -1, -0.381966, 31)],
            [:Aprime, :E1prime, :E2prime, :Adprime, :E1dprime, :E2dprime],
            [1, 2, 2, 1, 2, 2],
            [1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0;
             2.0   c1    c2   c2   c1    2.0   c1    c2   c2   c1;
             2.0   c2    c1   c1   c2    2.0   c2    c1   c1   c2;
             1.0   1.0   1.0   1.0   1.0  -1.0  -1.0  -1.0  -1.0  -1.0;
             2.0   c1    c2   c2   c1   -2.0  -c1   -c2  -c2  -c1;
             2.0   c2    c1   c1   c2   -2.0  -c2   -c1  -c1  -c2],
            "standard published table; prime=even under σh; c1=2cos(2π/5), c2=2cos(4π/5)")
    end

    # ─── D5 ─────────────────────────────────────────────────────────────
    # Proper dihedral, order 10, 4 classes.
    # Canonical class order: E, 2C5, 2C5², 5C2'.
    tables[:D5] = _ref(:D5,
        [:E, :C5, :C5sq, :C2pr],
        [1, 2, 2, 5],
        [(1, 1, 1, 3.0, 0),
         (2, 5, 1, 1.618034, 0),
         (2, 5, 1, -0.618034, 0),
         (5, 2, 1, -1.0, 0)],
        [:A1, :A2, :E1, :E2], [1, 1, 2, 2],
        [1.0   1.0        1.0        1.0;
         1.0   1.0        1.0       -1.0;
         2.0   0.618034  -1.618034   0.0;
         2.0  -1.618034   0.618034   0.0],
        "standard published table; C5 on z, C2' through x")

    # ─── D5h ────────────────────────────────────────────────────────────
    # D5h = D5 × Cs.  Order 20, 8 classes.
    # Canonical class order: E, 2C5, 2C5², 5C2', σh, 2S5, 2S5³, 5σv.
    # 8 IRs: A1', A2', E1', E2', A1'', A2'', E1'', E2''.
    # Character matrix derived from D5 ⊗ {σh symmetry}.
    let c1 = 0.618034, c2 = -1.618034
        tables[:D5h] = _ref(:D5h,
            [:E, :C5, :C5sq, :C2pr, :σh, :S5, :S5cu, :σv],
            [1, 2, 2, 5, 1, 2, 2, 5],
            [(1, 1, 1, 3.0, 0),
             (2, 5, 1, 1.618034, 0),
             (2, 5, 1, -0.618034, 0),
             (5, 2, 1, -1.0, 0),
             (1, 2, -1, 1.0, 0),
             (2, 10, -1, -0.381966, 0),
             (2, 10, -1, -2.618034, 0),
             (5, 2, -1, 1.0, 0)],
            [:A1prime, :A2prime, :E1prime, :E2prime,
             :A1dprime, :A2dprime, :E1dprime, :E2dprime],
            [1, 1, 2, 2, 1, 1, 2, 2],
            [1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0;
             1.0   1.0   1.0  -1.0   1.0   1.0   1.0  -1.0;
             2.0   c1    c2    0.0   2.0   c1    c2    0.0;
             2.0   c2    c1    0.0   2.0   c2    c1    0.0;
             1.0   1.0   1.0   1.0  -1.0  -1.0  -1.0  -1.0;
             1.0   1.0   1.0  -1.0  -1.0  -1.0  -1.0   1.0;
             2.0   c1    c2    0.0  -2.0  -c1   -c2    0.0;
             2.0   c2    c1    0.0  -2.0  -c2   -c1    0.0],
            "standard published table; prime=even under σh; c1=2cos(2π/5), c2=2cos(4π/5)")
    end

    # ─── D5d ────────────────────────────────────────────────────────────
    # D5d = D5 × Ci.  Order 20, 8 classes.
    # Canonical class order: E, 2C5, 2C5², 5C2, i, 2S10, 2S10³, 5σd.
    # S10  = i·C5,  trace = -1.618034 = c2.
    # S10³ = i·C5², trace = +0.618034 = c1.
    # 8 IRs: A1g, A2g, E1g, E2g, A1u, A2u, E1u, E2u.
    # For g-type: χ(S10=i·C5) = χ(C5), χ(S10³=i·C5²) = χ(C5²).
    # For u-type: χ(S10) = -χ(C5), χ(S10³) = -χ(C5²).
    let c1 = 0.618034, c2 = -1.618034
        tables[:D5d] = _ref(:D5d,
            [:E, :C5, :C5sq, :C2pr, :i, :S10, :S10cu, :σd],
            [1, 2, 2, 5, 1, 2, 2, 5],
            [(1, 1, 1, 3.0, 0),
             (2, 5, 1, 1.618034, 0),
             (2, 5, 1, -0.618034, 0),
             (5, 2, 1, -1.0, 0),
             (1, 2, -1, -3.0, 0),
             (2, 10, -1, -1.618034, 0),
             (2, 10, -1, 0.618034, 0),
             (5, 2, -1, 1.0, 0)],
            [:A1g, :A2g, :E1g, :E2g, :A1u, :A2u, :E1u, :E2u],
            [1, 1, 2, 2, 1, 1, 2, 2],
            [1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0;
             1.0   1.0   1.0  -1.0   1.0   1.0   1.0  -1.0;
             2.0   c1    c2    0.0   2.0   c1    c2    0.0;
             2.0   c2    c1    0.0   2.0   c2    c1    0.0;
             1.0   1.0   1.0   1.0  -1.0  -1.0  -1.0  -1.0;
             1.0   1.0   1.0  -1.0  -1.0  -1.0  -1.0   1.0;
             2.0   c1    c2    0.0  -2.0  -c1   -c2    0.0;
             2.0   c2    c1    0.0  -2.0  -c2   -c1    0.0],
            "standard published table; g=even under i, u=odd; c1=2cos(2π/5), c2=2cos(4π/5)")
    end

    # ─── D4d ────────────────────────────────────────────────────────────
    # Anti-prism group, order 16, 7 classes.
    # Canonical class order: E, 2S8, 2C4, 2S8³, C2, 4C2', 4σd.
    # 7 IRs: A1, A2, B1, B2, E1, E2, E3.
    # S8 trace = 2cos(π/4) - 1 = √2 - 1 ≈ 0.414214
    # S8³ trace = 2cos(3π/4) - 1 = -√2 - 1 ≈ -2.414214
    # E1 uses √2; E2 uses 0; E3 uses -√2.
    let sq2 = sqrt(2.0)
        tables[:D4d] = _ref(:D4d,
            [:E, :S8, :C4, :S8cu, :C2, :C2pr, :σd],
            [1, 2, 2, 2, 1, 4, 4],
            [(1, 1, 1, 3.0, 0),
             (2, 8, -1, 0.414214, 0),
             (2, 4, 1, 1.0, 0),
             (2, 8, -1, -2.414214, 0),
             (1, 2, 1, -1.0, 0),
             (4, 2, 1, -1.0, 0),
             (4, 2, -1, 1.0, 0)],
            [:A1, :A2, :B1, :B2, :E1, :E2, :E3], [1, 1, 1, 1, 2, 2, 2],
            [1.0   1.0   1.0   1.0   1.0   1.0   1.0;
             1.0   1.0   1.0   1.0   1.0  -1.0  -1.0;
             1.0  -1.0   1.0  -1.0   1.0   1.0  -1.0;
             1.0  -1.0   1.0  -1.0   1.0  -1.0   1.0;
             2.0   sq2   0.0  -sq2  -2.0   0.0   0.0;
             2.0   0.0  -2.0   0.0   2.0   0.0   0.0;
             2.0  -sq2   0.0   sq2  -2.0   0.0   0.0],
            "standard published table; S8 on z, C2' through x, σd bisecting C2' pairs")
    end

    # ─── D6d ────────────────────────────────────────────────────────────
    # Anti-prism group, order 24, 9 classes.
    # Canonical class order: E, 2S12, 2C6, 2S4, 2C3, 2S12⁵, C2, 6C2', 6σd.
    # 9 IRs: A1, A2, B1, B2, E1, E2, E3, E4, E5.
    # S12 trace = 2cos(π/6) - 1 = √3 - 1 ≈ 0.732051
    # S12⁵ trace = 2cos(5π/6) - 1 = -√3 - 1 ≈ -2.732051
    let sq3 = sqrt(3.0)
        tables[:D6d] = _ref(:D6d,
            [:E, :S12, :C6, :S4, :C3, :S12fi, :C2, :C2pr, :σd],
            [1, 2, 2, 2, 2, 2, 1, 6, 6],
            [(1, 1, 1, 3.0, 0),
             (2, 12, -1, 0.732051, 0),
             (2, 6, 1, 2.0, 0),
             (2, 4, -1, -1.0, 0),
             (2, 3, 1, 0.0, 0),
             (2, 12, -1, -2.732051, 0),
             (1, 2, 1, -1.0, 0),
             (6, 2, 1, -1.0, 0),
             (6, 2, -1, 1.0, 0)],
            [:A1, :A2, :B1, :B2, :E1, :E2, :E3, :E4, :E5], [1, 1, 1, 1, 2, 2, 2, 2, 2],
            [1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0;
             1.0   1.0   1.0   1.0   1.0   1.0   1.0  -1.0  -1.0;
             1.0  -1.0   1.0  -1.0   1.0  -1.0   1.0   1.0  -1.0;
             1.0  -1.0   1.0  -1.0   1.0  -1.0   1.0  -1.0   1.0;
             2.0   1.0  -1.0  -2.0  -1.0   1.0   2.0   0.0   0.0;
             2.0  -1.0  -1.0   2.0  -1.0  -1.0   2.0   0.0   0.0;
             2.0   sq3   1.0   0.0  -1.0  -sq3  -2.0   0.0   0.0;
             2.0  -sq3   1.0   0.0  -1.0   sq3  -2.0   0.0   0.0;
             2.0   0.0  -2.0   0.0   2.0   0.0  -2.0   0.0   0.0],
            "standard published table; S12 on z, C2' through x, σd bisecting C2' pairs")
    end

    # ─── I ──────────────────────────────────────────────────────────────
    # Icosahedral rotation group, order 60, 5 classes.
    # Canonical class order: E, 12C5, 12C5², 20C3, 15C2.
    # 5 IRs: A, T1, T2, G, H (dims 1,3,3,4,5).
    # φ = (1+√5)/2 ≈ 1.618034;  1-φ = -1/φ ≈ -0.618034.
    # Note: χ(T1,C5) = φ, χ(T1,C5²) = 1-φ (standard convention).
    #       χ(T2,C5) = 1-φ, χ(T2,C5²) = φ.
    let phi = (1 + sqrt(5)) / 2, psi = (1 - sqrt(5)) / 2
        tables[:I] = _ref(:I,
            [:E, :C5, :C5sq, :C3, :C2pr],
            [1, 12, 12, 20, 15],
            [(1, 1, 1, 3.0, 0),
             (12, 5, 1, 1.618034, 0),
             (12, 5, 1, -0.618034, 0),
             (20, 3, 1, 0.0, 0),
             (15, 2, 1, -1.0, 0)],
            [:A, :T1, :T2, :G, :H], [1, 3, 3, 4, 5],
            [1.0   1.0   1.0   1.0   1.0;
             3.0   phi   psi   0.0  -1.0;
             3.0   psi   phi   0.0  -1.0;
             4.0  -1.0  -1.0   1.0   0.0;
             5.0   0.0   0.0  -1.0   1.0],
            "standard published table; phi=(1+sqrt(5))/2")
    end

    # ─── Ih ─────────────────────────────────────────────────────────────
    # Full icosahedral group, order 120, 10 classes.
    # Canonical class order: E, 12C5, 12C5², 20C3, 15C2, i, 12S10, 12S10³, 20S6, 15σ.
    # 10 IRs: Ag, T1g, T2g, Gg, Hg, Au, T1u, T2u, Gu, Hu.
    # S10 trace ≈ -1.618034 (S10 = i·C5); S10³ trace ≈ 0.618034.
    # S6 trace = 0 (S6 = i·C3, det=-1, tr=0).
    # σ trace = 1 (σ = i·C2, det=-1, tr=1).
    # g-irreps: same sign for i as for E (and for paired operations).
    # u-irreps: opposite sign for i.
    let phi = (1 + sqrt(5)) / 2, psi = (1 - sqrt(5)) / 2
        tables[:Ih] = _ref(:Ih,
            [:E, :C5, :C5sq, :C3, :C2pr, :i, :S10, :S10cu, :S6, :σ],
            [1, 12, 12, 20, 15, 1, 12, 12, 20, 15],
            [(1, 1, 1, 3.0, 0),
             (12, 5, 1, 1.618034, 0),
             (12, 5, 1, -0.618034, 0),
             (20, 3, 1, 0.0, 0),
             (15, 2, 1, -1.0, 0),
             (1, 2, -1, -3.0, 0),
             (12, 10, -1, -1.618034, 0),
             (12, 10, -1, 0.618034, 0),
             (20, 6, -1, 0.0, 0),
             (15, 2, -1, 1.0, 0)],
            [:Ag, :T1g, :T2g, :Gg, :Hg, :Au, :T1u, :T2u, :Gu, :Hu],
            [1, 3, 3, 4, 5, 1, 3, 3, 4, 5],
            [1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0;
             3.0   phi   psi   0.0  -1.0   3.0   phi   psi   0.0  -1.0;
             3.0   psi   phi   0.0  -1.0   3.0   psi   phi   0.0  -1.0;
             4.0  -1.0  -1.0   1.0   0.0   4.0  -1.0  -1.0   1.0   0.0;
             5.0   0.0   0.0  -1.0   1.0   5.0   0.0   0.0  -1.0   1.0;
             1.0   1.0   1.0   1.0   1.0  -1.0  -1.0  -1.0  -1.0  -1.0;
             3.0   phi   psi   0.0  -1.0  -3.0  -phi  -psi   0.0   1.0;
             3.0   psi   phi   0.0  -1.0  -3.0  -psi  -phi   0.0   1.0;
             4.0  -1.0  -1.0   1.0   0.0  -4.0   1.0   1.0  -1.0   0.0;
             5.0   0.0   0.0  -1.0   1.0  -5.0   0.0   0.0   1.0  -1.0],
            "standard published table; phi=(1+sqrt(5))/2")
    end

    # ─── C3 ─────────────────────────────────────────────────────────────
    # Cyclic order 3. C3 and C3² share primary fingerprint (1,3,1,0,0);
    # secondary tag 11 on C3² distinguishes them.
    # Canonical class order: E, C3, C3². Folded E IR: dim=2.
    tables[:C3] = _ref(:C3,
        [:E, :C3, :C3sq], [1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 3, 1, 0.0, 0),
         (1, 3, 1, 0.0, 11)],
        [:A, :E], [1, 2],
        [1.0  1.0  1.0;
         2.0 -1.0 -1.0],
        "standard published table")

    # ─── C4 ─────────────────────────────────────────────────────────────
    # Cyclic order 4. C4 and C4³ share primary fingerprint; secondary tag
    # 11 on C4³ distinguishes it. Folded E IR: dim=2.
    # Canonical class order: E, C4, C2, C4³.
    tables[:C4] = _ref(:C4,
        [:E, :C4, :C2, :C4cu], [1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 4, 1, 1.0, 0),
         (1, 2, 1, -1.0, 0),
         (1, 4, 1, 1.0, 11)],
        [:A, :B, :E], [1, 1, 2],
        [1.0  1.0  1.0  1.0;
         1.0 -1.0  1.0 -1.0;
         2.0  0.0 -2.0  0.0],
        "standard published table")

    # ─── C6 ─────────────────────────────────────────────────────────────
    # Cyclic order 6. C3 and C3² distinguished by secondary tags 41/61;
    # C6 and C6⁵ by secondary tags 0/11. Two folded E IRs: E1, E2.
    # Canonical: E, C6, C3, C2, C3², C6⁵.
    tables[:C6] = _ref(:C6,
        [:E, :C6, :C3, :C2, :C3sq, :C6fi], [1, 1, 1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 6, 1, 2.0, 0),
         (1, 3, 1, 0.0, 41),
         (1, 2, 1, -1.0, 0),
         (1, 3, 1, 0.0, 61),
         (1, 6, 1, 2.0, 11)],
        [:A, :B, :E1, :E2], [1, 1, 2, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0;
         1.0 -1.0  1.0 -1.0  1.0 -1.0;
         2.0  1.0 -1.0 -2.0 -1.0  1.0;
         2.0 -1.0 -1.0  2.0 -1.0 -1.0],
        "standard published table")

    # ─── C3h ────────────────────────────────────────────────────────────
    # C3 × Cs, order 6. C3 (tag 0) and C3² (tag 11) distinguished by
    # secondary tags; S3 = C3·σh (tag 61) and S3⁵ = σh·C3² (tag 31).
    # Canonical: E, C3, C3², σh, S3, S3⁵.
    tables[:C3h] = _ref(:C3h,
        [:E, :C3, :C3sq, :σh, :S3, :S3fi], [1, 1, 1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 3, 1, 0.0, 0),
         (1, 3, 1, 0.0, 11),
         (1, 2, -1, 1.0, 0),
         (1, 6, -1, -2.0, 61),
         (1, 6, -1, -2.0, 31)],
        [:Aprime, :Eprime, :Adprime, :Edprime], [1, 2, 1, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0;
         2.0 -1.0 -1.0  2.0 -1.0 -1.0;
         1.0  1.0  1.0 -1.0 -1.0 -1.0;
         2.0 -1.0 -1.0 -2.0  1.0  1.0],
        "standard published table")

    # ─── C4h ────────────────────────────────────────────────────────────
    # C4 × Ci, order 8. C4/C4³ distinguished by tags 0/11;
    # S4³ (tag 61, = C4·i) and S4 (tag 31, = C4³·i) by tags.
    # Canonical: E, C4, C2, C4³, i, S4³, σh, S4.
    tables[:C4h] = _ref(:C4h,
        [:E, :C4, :C2, :C4cu, :i, :S4cu, :σh, :S4], [1, 1, 1, 1, 1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 4, 1, 1.0, 0),
         (1, 2, 1, -1.0, 0),
         (1, 4, 1, 1.0, 11),
         (1, 2, -1, -3.0, 0),
         (1, 4, -1, -1.0, 61),
         (1, 2, -1, 1.0, 0),
         (1, 4, -1, -1.0, 31)],
        [:Ag, :Bg, :Eg, :Au, :Bu, :Eu], [1, 1, 2, 1, 1, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0;
         1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0;
         2.0  0.0 -2.0  0.0  2.0  0.0 -2.0  0.0;
         1.0  1.0  1.0  1.0 -1.0 -1.0 -1.0 -1.0;
         1.0 -1.0  1.0 -1.0 -1.0  1.0 -1.0  1.0;
         2.0  0.0 -2.0  0.0 -2.0  0.0  2.0  0.0],
        "standard published table")

    # ─── S6 ─────────────────────────────────────────────────────────────
    # C3 × Ci, order 6. C3 (tag 0) and C3² (tag 11) distinguished;
    # S6⁵ (secondary tag 21) and S6 (secondary tag 41) distinguished.
    # Canonical: E, C3, C3², i, S6⁵, S6.
    tables[:S6] = _ref(:S6,
        [:E, :C3, :C3sq, :i, :S6fi, :S6], [1, 1, 1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 3, 1, 0.0, 0),
         (1, 3, 1, 0.0, 11),
         (1, 2, -1, -3.0, 0),
         (1, 6, -1, 0.0, 21),
         (1, 6, -1, 0.0, 41)],
        [:Ag, :Eg, :Au, :Eu], [1, 2, 1, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0;
         2.0 -1.0 -1.0  2.0 -1.0 -1.0;
         1.0  1.0  1.0 -1.0 -1.0 -1.0;
         2.0 -1.0 -1.0 -2.0  1.0  1.0],
        "standard published table")

    # ─── C6h ────────────────────────────────────────────────────────────
    # C6 × Ci, order 12. Secondary tags distinguish all colliding pairs:
    # C3 (tag 71) vs C3² (tag 121); C6 (tag 0) vs C6⁵ (tag 11);
    # S3⁵ (tag 61) vs S3 (tag 31); S6⁵ (tag 91) vs S6 (tag 111).
    # Canonical: E, C6, C3, C2, C3², C6⁵, i, S3⁵, S6⁵, σh, S6, S3.
    tables[:C6h] = _ref(:C6h,
        [:E, :C6, :C3, :C2, :C3sq, :C6fi, :i, :S3fi, :S6fi, :σh, :S6, :S3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [(1, 1, 1, 3.0, 0),
         (1, 6, 1, 2.0, 0),
         (1, 3, 1, 0.0, 71),
         (1, 2, 1, -1.0, 0),
         (1, 3, 1, 0.0, 121),
         (1, 6, 1, 2.0, 11),
         (1, 2, -1, -3.0, 0),
         (1, 6, -1, -2.0, 61),
         (1, 6, -1, 0.0, 91),
         (1, 2, -1, 1.0, 0),
         (1, 6, -1, 0.0, 111),
         (1, 6, -1, -2.0, 31)],
        [:Ag, :Bg, :E1g, :E2g, :Au, :Bu, :E1u, :E2u], [1, 1, 2, 2, 1, 1, 2, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0;
         1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0;
         2.0  1.0 -1.0 -2.0 -1.0  1.0  2.0  1.0 -1.0 -2.0 -1.0  1.0;
         2.0 -1.0 -1.0  2.0 -1.0 -1.0  2.0 -1.0 -1.0  2.0 -1.0 -1.0;
         1.0  1.0  1.0  1.0  1.0  1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0;
         1.0 -1.0  1.0 -1.0  1.0 -1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0;
         2.0  1.0 -1.0 -2.0 -1.0  1.0 -2.0 -1.0  1.0  2.0  1.0 -1.0;
         2.0 -1.0 -1.0  2.0 -1.0 -1.0 -2.0  1.0  1.0 -2.0  1.0  1.0],
        "standard published table")

    # ─── D4 ─────────────────────────────────────────────────────────────
    # Proper dihedral, order 8. 2C2' (axial C2x, anchor 1004) and 2C2''
    # (diagonal, complement anchor 1005) disambiguated by generator tag.
    # Canonical: E, 2C4, C2, 2C2', 2C2''.
    tables[:D4] = _ref(:D4,
        [:E, :C4, :C2, :C2pr, :C2dpr], [1, 2, 1, 2, 2],
        [(1, 1, 1, 3.0, 0),
         (2, 4, 1, 1.0, 0),
         (1, 2, 1, -1.0, 0),
         (2, 2, 1, -1.0, 1004),
         (2, 2, 1, -1.0, 1005)],
        [:A1, :A2, :B1, :B2, :E], [1, 1, 1, 1, 2],
        [1.0  1.0  1.0  1.0  1.0;
         1.0  1.0  1.0 -1.0 -1.0;
         1.0 -1.0  1.0  1.0 -1.0;
         1.0 -1.0  1.0 -1.0  1.0;
         2.0  0.0 -2.0  0.0  0.0],
        "standard published table; C2'↔axial (C2x), C2''↔diagonal")

    # ─── D2d ────────────────────────────────────────────────────────────
    # Anti-prism dihedral, order 8. All 5 primary fingerprints are unique;
    # no anchor surgery required.
    # Canonical: E, 2S4, C2, 2C2', 2σd.
    tables[:D2d] = _ref(:D2d,
        [:E, :S4, :C2, :C2pr, :σd], [1, 2, 1, 2, 2],
        [(1, 1, 1, 3.0, 0),
         (2, 4, -1, -1.0, 0),
         (1, 2, 1, -1.0, 0),
         (2, 2, 1, -1.0, 0),
         (2, 2, -1, 1.0, 0)],
        [:A1, :A2, :B1, :B2, :E], [1, 1, 1, 1, 2],
        [1.0  1.0  1.0  1.0  1.0;
         1.0  1.0  1.0 -1.0 -1.0;
         1.0 -1.0  1.0  1.0 -1.0;
         1.0 -1.0  1.0 -1.0  1.0;
         2.0  0.0 -2.0  0.0  0.0],
        "standard published table")

    # ─── D6 ─────────────────────────────────────────────────────────────
    # Proper dihedral, order 12. 3C2' (C2x, anchor 1004) and 3C2''
    # (C6·C2x, complement anchor 1005) disambiguated by generator tag.
    # Canonical: E, 2C6, 2C3, C2, 3C2', 3C2''.
    tables[:D6] = _ref(:D6,
        [:E, :C6, :C3, :C2, :C2pr, :C2dpr], [1, 2, 2, 1, 3, 3],
        [(1, 1, 1, 3.0, 0),
         (2, 6, 1, 2.0, 0),
         (2, 3, 1, 0.0, 0),
         (1, 2, 1, -1.0, 0),
         (3, 2, 1, -1.0, 1004),
         (3, 2, 1, -1.0, 1005)],
        [:A1, :A2, :B1, :B2, :E1, :E2], [1, 1, 1, 1, 2, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0;
         1.0  1.0  1.0  1.0 -1.0 -1.0;
         1.0 -1.0  1.0 -1.0  1.0 -1.0;
         1.0 -1.0  1.0 -1.0 -1.0  1.0;
         2.0  1.0 -1.0 -2.0  0.0  0.0;
         2.0 -1.0 -1.0  2.0  0.0  0.0],
        "standard published table; C2'↔axial (C2x), C2''↔diagonal")

    # ─── C6v ────────────────────────────────────────────────────────────
    # Pyramidal, order 12. σv planes contain C2x axis (anchor 1001); σd
    # planes are complement (anchor 1002).
    # Canonical: E, 2C6, 2C3, C2, 3σv, 3σd.
    tables[:C6v] = _ref(:C6v,
        [:E, :C6, :C3, :C2, :σv, :σd], [1, 2, 2, 1, 3, 3],
        [(1, 1, 1, 3.0, 0),
         (2, 6, 1, 2.0, 0),
         (2, 3, 1, 0.0, 0),
         (1, 2, 1, -1.0, 0),
         (3, 2, -1, 1.0, 1001),
         (3, 2, -1, 1.0, 1002)],
        [:A1, :A2, :B1, :B2, :E1, :E2], [1, 1, 1, 1, 2, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0;
         1.0  1.0  1.0  1.0 -1.0 -1.0;
         1.0 -1.0  1.0 -1.0  1.0 -1.0;
         1.0 -1.0  1.0 -1.0 -1.0  1.0;
         2.0  1.0 -1.0 -2.0  0.0  0.0;
         2.0 -1.0 -1.0  2.0  0.0  0.0],
        "standard published table; σv contains C2' axes, σd bisects them")

    # ─── D6h ────────────────────────────────────────────────────────────
    # D6 × Ci, order 24. 3C2'/3C2'' resolved by generator anchors 1004/1005.
    # Two σ-plane classes use secondary tags from class-product:
    # σd = C2x·i class (secondary tag 96), σv = C2x·C6·i class (tag 86).
    # Canonical: E, 2C6, 2C3, C2, 3C2', 3C2'', i, 2S3, 2S6, σh, 3σd, 3σv.
    tables[:D6h] = _ref(:D6h,
        [:E, :C6, :C3, :C2, :C2pr, :C2dpr, :i, :S3, :S6, :σh, :σd, :σv],
        [1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3],
        [(1, 1, 1, 3.0, 0),
         (2, 6, 1, 2.0, 0),
         (2, 3, 1, 0.0, 0),
         (1, 2, 1, -1.0, 0),
         (3, 2, 1, -1.0, 1004),
         (3, 2, 1, -1.0, 1005),
         (1, 2, -1, -3.0, 0),
         (2, 6, -1, -2.0, 0),
         (2, 6, -1, 0.0, 0),
         (1, 2, -1, 1.0, 0),
         (3, 2, -1, 1.0, 96),
         (3, 2, -1, 1.0, 86)],
        [:A1g, :A2g, :B1g, :B2g, :E1g, :E2g,
         :A1u, :A2u, :B1u, :B2u, :E1u, :E2u],
        [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
        [1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0;
         1.0  1.0  1.0  1.0 -1.0 -1.0  1.0  1.0  1.0  1.0 -1.0 -1.0;
         1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0;
         1.0 -1.0  1.0 -1.0 -1.0  1.0  1.0 -1.0  1.0 -1.0 -1.0  1.0;
         2.0  1.0 -1.0 -2.0  0.0  0.0  2.0  1.0 -1.0 -2.0  0.0  0.0;
         2.0 -1.0 -1.0  2.0  0.0  0.0  2.0 -1.0 -1.0  2.0  0.0  0.0;
         1.0  1.0  1.0  1.0  1.0  1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0;
         1.0  1.0  1.0  1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0  1.0  1.0;
         1.0 -1.0  1.0 -1.0  1.0 -1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0;
         1.0 -1.0  1.0 -1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0  1.0 -1.0;
         2.0  1.0 -1.0 -2.0  0.0  0.0 -2.0 -1.0  1.0  2.0  0.0  0.0;
         2.0 -1.0 -1.0  2.0  0.0  0.0 -2.0  1.0  1.0 -2.0  0.0  0.0],
        "standard published table; σd=secondary-tag 96, σv=secondary-tag 86")

    # ─── T ──────────────────────────────────────────────────────────────
    # Chiral tetrahedral, order 12. 4C3 (secondary tag 0) and 4C3² (tag
    # 3914) distinguished by class-product secondary tags. Folded E: dim=2.
    # Canonical: E, 4C3, 4C3², 3C2.
    tables[:T] = _ref(:T,
        [:E, :C3, :C3sq, :C2], [1, 4, 4, 3],
        [(1, 1, 1, 3.0, 0),
         (4, 3, 1, 0.0, 0),
         (4, 3, 1, 0.0, 3914),
         (3, 2, 1, -1.0, 0)],
        [:A, :E, :T], [1, 2, 3],
        [1.0  1.0  1.0  1.0;
         2.0 -1.0 -1.0  2.0;
         3.0  0.0  0.0 -1.0],
        "standard published table; E folded from complex {omega, omega^2} pair")

    # ─── Th ─────────────────────────────────────────────────────────────
    # T × Ci, order 24. C3² (tag 3914) distinguished from C3 (tag 0);
    # S6 (tag 89) and S6⁵ (tag 7944) distinguished by class-product tags.
    # Canonical: E, 4C3, 4C3², 3C2, i, 4S6, 4S6⁵, 3σh.
    tables[:Th] = _ref(:Th,
        [:E, :C3, :C3sq, :C2, :i, :S6, :S6fi, :σh], [1, 4, 4, 3, 1, 4, 4, 3],
        [(1, 1, 1, 3.0, 0),
         (4, 3, 1, 0.0, 0),
         (4, 3, 1, 0.0, 3914),
         (3, 2, 1, -1.0, 0),
         (1, 2, -1, -3.0, 0),
         (4, 6, -1, 0.0, 89),
         (4, 6, -1, 0.0, 7944),
         (3, 2, -1, 1.0, 0)],
        [:Ag, :Eg, :Tg, :Au, :Eu, :Tu], [1, 2, 3, 1, 2, 3],
        [1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0;
         2.0 -1.0 -1.0  2.0  2.0 -1.0 -1.0  2.0;
         3.0  0.0  0.0 -1.0  3.0  0.0  0.0 -1.0;
         1.0  1.0  1.0  1.0 -1.0 -1.0 -1.0 -1.0;
         2.0 -1.0 -1.0  2.0 -2.0  1.0  1.0 -2.0;
         3.0  0.0  0.0 -1.0 -3.0  0.0  0.0  1.0],
        "standard published table; E folded; S6 tag 89, S6fi tag 7944")

    # ─── S4 ─────────────────────────────────────────────────────────
    # Order 4, classes E, S4, C2, S4³. S4 and S4³ share primary
    # fingerprint (1, 4, -1, -1.0); the :S4z generator anchor (1020)
    # tags the S4 class, complement (1021) tags S4³. 3 displayed IRs:
    # A, B, E (E is the FS=0 folded pair from γ_1, γ_3).
    tables[:S4] = _ref(:S4,
        [:E, :S4, :C2, :S4three], [1, 1, 1, 1],
        [(1, 1,  1,  3.0, 0),
         (1, 4, -1, -1.0, 1020),
         (1, 2,  1, -1.0, 0),
         (1, 4, -1, -1.0, 1021)],
        [:A, :B, :E], [1, 1, 2],
        [1.0  1.0  1.0  1.0;
         1.0 -1.0  1.0 -1.0;
         2.0  0.0 -2.0  0.0],
        "standard published table; E folded from FS=0 γ_1, γ_3 pair")

    return tables
end

const REFERENCE_CHARACTER_TABLES = _build_reference_tables()

# ─── Lookup: assign canonical labels by matching to reference ────────

# Returns (labels, class_perm) where class_perm[i] = canonical_class_idx.
# `labels` is a vector of Mulliken Symbols ordered as `folds`.
#
# Behavior:
#   - If `name` is NOT in REFERENCE_CHARACTER_TABLES, returns
#     (nothing, nothing). The caller falls back to the auto-namer; the
#     resulting IRrep.provenance will be `:computed_auto`.
#   - If `name` IS in the table, matching MUST succeed. Any failure
#     (class-fingerprint mismatch, character row mismatch) raises
#     ArgumentError — the reference data is part of the production
#     contract, and silently falling back to auto-naming would let
#     production-mode `expand_clm` proceed on an unverified group.
function _assign_reference_labels(name::Symbol,
                                   folds::Vector{Vector{Int}},
                                   complex_chars::Vector{Tuple{Int, Vector{ComplexF64}}},
                                   classes::Vector{Vector{Int}},
                                   elements::Vector{GroupElement},
                                   multable::Matrix{Int};
                                   generators::Vector{GroupElement}=GroupElement[])
    haskey(REFERENCE_CHARACTER_TABLES, name) || return nothing, nothing
    ref = REFERENCE_CHARACTER_TABLES[name]
    fingerprints = _class_fingerprints(classes, elements, multable;
                                        generators=generators)

    n_class = length(classes)
    n_class == length(ref.class_fingerprints) || throw(ArgumentError(
        "reference-table match failed for :$name — computed class count " *
        "$n_class differs from reference's $(length(ref.class_fingerprints)). " *
        "Either the generator data in `data.jl` is wrong, or the reference " *
        "table needs updating."))

    # Align computed-class indices to reference's canonical class order.
    class_perm = zeros(Int, n_class)
    for (canonical_idx, ref_fp) in enumerate(ref.class_fingerprints)
        match_idx = findfirst(==(ref_fp), fingerprints)
        match_idx === nothing && throw(ArgumentError(
            "reference-table match failed for :$name — computed classes do " *
            "not contain a fingerprint matching reference class " *
            ":$(ref.classes[canonical_idx]) = $ref_fp. Computed fingerprints: " *
            "$fingerprints."))
        class_perm[match_idx] = canonical_idx
    end
    any(class_perm .== 0) && throw(ArgumentError(
        "reference-table match for :$name produced an incomplete class " *
        "permutation; some computed classes have no reference match."))

    # For each fold (Mulliken-displayed IR), find the matching reference row.
    labels = Vector{Symbol}(undef, length(folds))
    used = falses(length(ref.irreps))
    for (k, fold) in enumerate(folds)
        d_real = length(fold) == 1 ? complex_chars[fold[1]][1] :
                                     2 * complex_chars[fold[1]][1]
        χ_disp = sum(complex_chars[i][2] for i in fold)
        χ_aligned = Vector{Float64}(undef, n_class)
        for (computed_idx, canonical_idx) in enumerate(class_perm)
            χ_aligned[canonical_idx] = real(χ_disp[computed_idx])
        end
        match_row = nothing
        for (row_idx, ref_label) in enumerate(ref.irreps)
            used[row_idx] && continue
            ref.ir_dims[row_idx] == d_real || continue
            ok = true
            for col in 1:n_class
                abs(χ_aligned[col] - ref.characters[row_idx, col]) < 1e-4 ||
                    (ok = false; break)
            end
            ok || continue
            match_row = row_idx
            break
        end
        match_row === nothing && throw(ArgumentError(
            "reference-table match failed for :$name — computed IR #$k " *
            "(real_dim=$d_real, characters≈$(round.(χ_aligned, digits=3))) " *
            "matches no remaining row in the reference table. " *
            "Either the reference table has a typo or the Burnside computation diverged."))
        used[match_row] = true
        labels[k] = ref.irreps[match_row]
    end
    return labels, class_perm
end

# Validate that the Burnside-computed character table for a constructed
# group matches the reference (after class permutation alignment).
# Raises ArgumentError on mismatch — the primary correctness oracle for
# both the reference data and the Burnside computation.
function _validate_against_reference(name::Symbol, labels::Vector{Symbol},
                                      complex_chars::Vector{Tuple{Int, Vector{ComplexF64}}},
                                      folds::Vector{Vector{Int}},
                                      class_perm::Vector{Int})
    haskey(REFERENCE_CHARACTER_TABLES, name) || return
    ref = REFERENCE_CHARACTER_TABLES[name]
    n_class = length(class_perm)
    for (k, fold) in enumerate(folds)
        χ_disp = sum(complex_chars[i][2] for i in fold)
        χ_aligned = Vector{Float64}(undef, n_class)
        for (computed_idx, canonical_idx) in enumerate(class_perm)
            χ_aligned[canonical_idx] = real(χ_disp[computed_idx])
        end
        ref_row_idx = findfirst(==(labels[k]), ref.irreps)
        ref_row_idx === nothing && error(
            "internal error: label $(labels[k]) not found in reference table for $name")
        for col in 1:n_class
            abs(χ_aligned[col] - ref.characters[ref_row_idx, col]) < 1e-6 ||
                error("character-table mismatch: group=$name, IR=$(labels[k]), " *
                      "class $col: computed=$(χ_aligned[col]), " *
                      "reference=$(ref.characters[ref_row_idx, col]). " *
                      "Either the reference table has a typo or the Burnside " *
                      "computation diverged.")
        end
    end
end

# ─── Public introspection helpers ────────────────────────────────────

"""
    has_reference_labels(name::Symbol) -> Bool

Return `true` if `name` has a curated reference character table in
`REFERENCE_CHARACTER_TABLES`. Groups with reference labels carry the
canonical Mulliken labels (A1g, T2u, ...) in `pointgroup(name).irreps`
and are eligible for production-mode `expand_clm`. Groups without a
reference table fall back to auto-generated structural labels (suitable
for teaching / exploratory use, not for production-style API stability).
"""
has_reference_labels(name::Symbol) = haskey(REFERENCE_CHARACTER_TABLES, name)

"""
    reference_label_groups() -> Vector{Symbol}

Return the sorted list of point-group names that have a curated reference
character table. These are the groups eligible for production-mode
`expand_clm` (without `experimental=true`): their irreps carry canonical
Mulliken labels (A1g, T2u, …) matched against a sourced table. Use
`has_reference_labels(name)` to test a single group name.
"""
reference_label_groups() = sort(collect(keys(REFERENCE_CHARACTER_TABLES)))

