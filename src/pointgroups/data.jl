# Embedded generator data, lattice metrics, default-source registry.
#
# Hand-typed Cartesian generator sets per group; the full element list is
# recovered via BFS closure in `group.jl`. The spgrep / libmsym
# embedded-blob path (chapter §2.3) adds fully-ordered element lists with
# explicit setting variants and serves as the cross-source agreement
# oracle (§2.4 Oracle 2). Those blobs are produced by dev_scripts and are
# populated when bumping the source version.

const _I3 = SMatrix{3,3,Float64}(1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0)

# 3×3 rotation matrix from axis-angle (active rotation, Cartesian).
function _axis_angle(axis::AbstractVector{<:Real}, angle::Real)
    n = axis / norm(axis)
    c, s = cos(angle), sin(angle)
    nx, ny, nz = n[1], n[2], n[3]
    SMatrix{3,3,Float64}(
        c + nx^2*(1-c),     nx*ny*(1-c) - nz*s, nx*nz*(1-c) + ny*s,
        ny*nx*(1-c) + nz*s, c + ny^2*(1-c),     ny*nz*(1-c) - nx*s,
        nz*nx*(1-c) - ny*s, nz*ny*(1-c) + nx*s, c + nz^2*(1-c),
    )
end

# Reflection through a plane with normal n.
function _reflection(n::AbstractVector{<:Real})
    nh = n / norm(n)
    SMatrix{3,3,Float64}(I - 2 * nh * nh')
end

const _INVERSION = SMatrix{3,3,Float64}(-1.0, 0.0, 0.0,
                                        0.0, -1.0, 0.0,
                                        0.0, 0.0, -1.0)

# Crystal-system metric for the spgrep → Cartesian transform (§2.2 (2.1)).
const CRYSTAL_SYSTEM_METRIC = Dict{Symbol, SMatrix{3,3,Float64,9}}(
    :triclinic    => _I3,
    :monoclinic   => _I3,
    :orthorhombic => _I3,
    :tetragonal   => _I3,
    :cubic        => _I3,
    :trigonal_hex => SMatrix{3,3,Float64}(1.0, 0.0, 0.0,
                                          -0.5, sqrt(3)/2, 0.0,
                                          0.0, 0.0, 1.0),
    :hexagonal    => SMatrix{3,3,Float64}(1.0, 0.0, 0.0,
                                          -0.5, sqrt(3)/2, 0.0,
                                          0.0, 0.0, 1.0),
)

# Group → minimal generator list, in Cartesian. BFS closure in group.jl
# recovers the full element list. Each generator is annotated with its
# Schoenflies-type tag.
#
# All 32 crystallographic groups + selected molecular groups (Cnv, Dn,
# Dnh, Dnd for n=5,6 + Ih) seeded by hand. Closure verifies these match
# the standard reference tables order-of-group |G|.
function _make_generators()
    gens = Dict{Symbol, Vector{Tuple{SMatrix{3,3,Float64,9}, Symbol}}}()

    # C2z, C3z, ..., Cn rotations about z.
    Cnz(n::Int) = _axis_angle([0.0, 0.0, 1.0], 2π / n)

    # Rotations about Cartesian axes.
    C2x = _axis_angle([1.0, 0.0, 0.0], π)
    C2y = _axis_angle([0.0, 1.0, 0.0], π)

    # Body-diagonal C3 (cubic groups).
    C3_111 = _axis_angle([1.0, 1.0, 1.0], 2π/3)

    # Mirrors.
    σh_xy = _reflection([0.0, 0.0, 1.0])  # horizontal (xy plane) ─ normal = ẑ
    σv_xz = _reflection([0.0, 1.0, 0.0])  # vertical containing x ─ normal = ŷ
    σv_yz = _reflection([1.0, 0.0, 0.0])  # vertical containing y ─ normal = x̂
    # Diagonal mirror at 45° in xy plane (Td σd).
    σd_xy = _reflection([1.0, -1.0, 0.0]) # plane x=y

    i = _INVERSION

    # ─── Order-1 / order-2 small groups ──────────────────────────────
    gens[:C1]  = []
    gens[:Ci]  = [(i,    :i)]
    gens[:Cs]  = [(σh_xy, :σh)]
    gens[:C2]  = [(_axis_angle([0.0, 0.0, 1.0], π), :C2z)]
    gens[:C2v] = [(_axis_angle([0.0, 0.0, 1.0], π), :C2z),
                  (σv_xz, :σv)]
    gens[:C2h] = [(_axis_angle([0.0, 0.0, 1.0], π), :C2z),
                  (i,     :i)]

    # ─── Cyclic / cyclic-with-mirrors families ───────────────────────
    gens[:C3]  = [(Cnz(3), :C3z)]
    gens[:C3v] = [(Cnz(3), :C3z), (σv_xz, :σv)]
    gens[:C3h] = [(Cnz(3), :C3z), (σh_xy, :σh)]
    gens[:C4]  = [(Cnz(4), :C4z)]
    gens[:C4v] = [(Cnz(4), :C4z), (σv_xz, :σv)]
    gens[:C4h] = [(Cnz(4), :C4z), (i, :i)]
    gens[:C5]  = [(Cnz(5), :C5z)]
    gens[:C5v] = [(Cnz(5), :C5z), (σv_xz, :σv)]
    gens[:C5h] = [(Cnz(5), :C5z), (σh_xy, :σh)]
    gens[:C6]  = [(Cnz(6), :C6z)]
    gens[:C6v] = [(Cnz(6), :C6z), (σv_xz, :σv)]
    gens[:C6h] = [(Cnz(6), :C6z), (i, :i)]

    # ─── S2n improper-rotation families (Ci ≡ S2 already covered) ────
    gens[:S4] = [(σh_xy * Cnz(4), :S4z)]
    gens[:S6] = [(σh_xy * Cnz(6), :S6z)]

    # ─── Dihedral families ───────────────────────────────────────────
    # D2: C2y added as redundant anchored generator (BFS closure unchanged;
    # anchors the C2z/C2y/C2x class disambiguation in the reference-table
    # fingerprint matcher — same pattern as D2h below).
    gens[:D2]  = [(_axis_angle([0.0, 0.0, 1.0], π), :C2z),
                  (C2x, :C2x),
                  (_axis_angle([0.0, 1.0, 0.0], π), :C2y)]
    # D2h: tagged C2 generators along all three axes plus the three σ
    # planes (redundant for closure; anchors the C2_a / σ_a class
    # disambiguation in the reference-table fingerprint matcher).
    gens[:D2h] = [(_axis_angle([0.0, 0.0, 1.0], π), :C2z),
                  (C2x, :C2x),
                  (_axis_angle([0.0, 1.0, 0.0], π), :C2y),
                  (i, :i),
                  (SMatrix{3,3,Float64}(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,-1.0), :σxy),
                  (SMatrix{3,3,Float64}(1.0,0.0,0.0, 0.0,-1.0,0.0, 0.0,0.0,1.0), :σxz),
                  (SMatrix{3,3,Float64}(-1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0), :σyz)]
    gens[:D2d] = [(σh_xy * Cnz(4), :S4z), (C2x, :C2x)]

    gens[:D3]  = [(Cnz(3), :C3z), (C2x, :C2x)]
    gens[:D3h] = [(Cnz(3), :C3z), (C2x, :C2x), (σh_xy, :σh)]
    gens[:D3d] = [(Cnz(3), :C3z), (C2x, :C2x), (i, :i)]

    gens[:D4]  = [(Cnz(4), :C4z), (C2x, :C2x)]
    # D4h: σv generator added (redundant for closure but anchors the
    # σv ↔ σd class disambiguation in the reference-table fingerprint matcher).
    gens[:D4h] = [(Cnz(4), :C4z), (C2x, :C2x), (σv_xz, :σv), (i, :i)]
    gens[:D4d] = [(σh_xy * Cnz(8), :S8z), (C2x, :C2x)]

    gens[:D5]  = [(Cnz(5), :C5z), (C2x, :C2x)]
    gens[:D5h] = [(Cnz(5), :C5z), (C2x, :C2x), (σh_xy, :σh)]
    gens[:D5d] = [(Cnz(5), :C5z), (C2x, :C2x), (i, :i)]

    gens[:D6]  = [(Cnz(6), :C6z), (C2x, :C2x)]
    gens[:D6h] = [(Cnz(6), :C6z), (C2x, :C2x), (i, :i)]
    gens[:D6d] = [(σh_xy * Cnz(12), :S12z), (C2x, :C2x)]

    # ─── Cubic / tetrahedral / octahedral ────────────────────────────
    gens[:T]  = [(C3_111, :C3_111), (_axis_angle([0.0, 0.0, 1.0], π), :C2z)]
    gens[:Td] = [(C3_111, :C3_111), (_axis_angle([0.0, 0.0, 1.0], π), :C2z), (σd_xy, :σd)]
    gens[:Th] = [(C3_111, :C3_111), (_axis_angle([0.0, 0.0, 1.0], π), :C2z), (i, :i)]
    gens[:O]  = [(C3_111, :C3_111), (Cnz(4), :C4z)]
    gens[:Oh] = [(C3_111, :C3_111), (Cnz(4), :C4z), (i, :i)]

    # ─── Icosahedral ─────────────────────────────────────────────────
    # z-aligned-C5 orientation: C5 axis along z; C3 face axis derived
    # by rotating the canonical (1,1,1) face-centroid direction through
    # the R_y rotation that maps the canonical (1,0,φ) C5-axis to z.
    # With (s, c) = (1, φ)/√(1+φ²), R_y(-α)·(1,1,1) = (c−s, 1, s+c).
    # Angle between C5z and the resulting C3 face axis is the standard
    # arccos(φ²/(√3·√(1+φ²))) ≈ 37.38°. BFS closure: |I|=60, |Ih|=120.
    let
        φ  = (1 + sqrt(5)) / 2
        s  = 1 / sqrt(1 + φ^2)
        c  = φ / sqrt(1 + φ^2)
        c3_axis = [c - s, 1.0, s + c]
        gens[:I]  = [(Cnz(5), :C5z), (_axis_angle(c3_axis, 2π/3), :C3_face)]
        gens[:Ih] = [(Cnz(5), :C5z), (_axis_angle(c3_axis, 2π/3), :C3_face), (i, :i)]
    end

    return gens
end

const POINTGROUP_GENERATORS = _make_generators()

# Default operation-list source per group. The embedded generators above
# are used for all groups; spgrep / libmsym blobs are populated in
# dev_scripts. This map records the canonical blob source per group.
const DEFAULT_SOURCE = Dict{Symbol, Symbol}(
    :C1 => :spgrep, :Ci => :spgrep, :Cs => :spgrep,
    :C2 => :spgrep, :C2v => :spgrep, :C2h => :spgrep,
    :C3 => :spgrep, :C3v => :spgrep, :C3h => :spgrep,
    :C4 => :spgrep, :C4v => :spgrep, :C4h => :spgrep,
    :C6 => :spgrep, :C6v => :spgrep, :C6h => :spgrep,
    :S4 => :spgrep, :S6 => :spgrep,
    :D2 => :spgrep, :D2h => :spgrep, :D2d => :spgrep,
    :D3 => :spgrep, :D3h => :spgrep, :D3d => :spgrep,
    :D4 => :spgrep, :D4h => :spgrep, :D4d => :spgrep,
    :D6 => :spgrep, :D6h => :spgrep, :D6d => :spgrep,
    :T => :spgrep, :Td => :spgrep, :Th => :spgrep,
    :O => :spgrep, :Oh => :spgrep,
    # libmsym-only / molecular families
    :C5 => :libmsym, :C5v => :libmsym, :C5h => :libmsym,
    :D5 => :libmsym, :D5h => :libmsym, :D5d => :libmsym,
    :I => :libmsym, :Ih => :libmsym,
)
