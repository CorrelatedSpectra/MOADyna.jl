# Named-lookup constructor and frame-rotation helper.

# Rotation angles around z for non-default settings.
# D3h/D3d/C3v: 30° (π/6). D6h: 15° (π/12).
const _SETTING_ANGLES = Dict{Tuple{Symbol, Symbol}, Float64}(
    (:D3h, :y) => π/6,
    (:D3d, :y) => π/6,
    (:C3v, :y) => π/6,
    (:D6h, :y) => π/12,
)

"""
    pointgroup(name::Symbol; source=nothing, setting=:default) -> PointGroup

Construct a point group by Schoenflies symbol. The implementation uses
hand-typed Cartesian generators in `POINTGROUP_GENERATORS` closed via
BFS to recover the full element list.

The `source` keyword names the embedded operation-list blob the group
should be built from (`:spgrep` for crystallographic, `:libmsym` for
molecular). Both blobs are reserved data slots; the present
implementation does not consume them yet, so passing any non-`nothing`
value is rejected with a clear error rather than silently ignored.

Settings (`:default`, `:y`, …) are pinned per group; only the listed
combinations build successfully. The default setting matches ITA / the
standard molecular convention as appropriate for the group.

For the four trigonal/hexagonal groups D3h, D3d, C3v, D6h the `:y`
setting is supported: it returns the default group rotated by 30° (D3h,
D3d, C3v) or 15° (D6h) around z, placing the σv/C2' axes along the
y-axis rather than the x-axis. Same abstract group — same character
table and Mulliken labels — different Cartesian frame.
"""
function pointgroup(name::Symbol; source::Union{Symbol, Nothing}=nothing,
                    setting::Symbol=:default)
    haskey(POINTGROUP_GENERATORS, name) || throw(ArgumentError(
        "unknown point group :$name; supported groups: " *
        join(sort(collect(keys(POINTGROUP_GENERATORS))), ", ")))
    # The `source` keyword names the embedded operation-list blob to use
    # (spgrep for crystallographic, libmsym for molecular). Both blobs are
    # reserved data slots; the present implementation always constructs
    # from the hand-typed Cartesian generator set in `POINTGROUP_GENERATORS`.
    # Non-`nothing` values are rejected to avoid silently ignoring user intent.
    if source !== nothing
        throw(ArgumentError(
            "the `source` keyword is reserved; the embedded spgrep / libmsym " *
            "operation-list blobs are not yet populated. Omit the keyword to " *
            "use the hand-typed generator path."))
    end

    if setting !== :default
        haskey(_SETTING_ANGLES, (name, setting)) || throw(ArgumentError(
            "no :$setting variant for :$name; supported non-default settings: " *
            join(["$(k[1])=>$(k[2])" for k in sort(collect(keys(_SETTING_ANGLES)))], ", ")))
        G_default = pointgroup(name; source=source, setting=:default)
        angle = _SETTING_ANGLES[(name, setting)]
        c, s = cos(angle), sin(angle)
        # Active rotation matrix R_z(angle): columns are images of x̂, ŷ, ẑ.
        Rz = SMatrix{3,3,Float64}(
             c, s, 0.0,
            -s, c, 0.0,
            0.0, 0.0, 1.0)
        G_rotated = rotate(G_default, Rz)
        G_rotated.setting = setting
        return G_rotated
    end

    gens_data = POINTGROUP_GENERATORS[name]
    return _construct_pointgroup(name, setting, gens_data)
end

"""
    rotate(G::PointGroup, R::AbstractMatrix) -> PointGroup

Return a new `PointGroup` whose elements are the conjugates
`R · g · R^{-1}` of `G`'s elements. `R` must be a 3×3 orthogonal matrix
(`‖R^T R - I‖ < 1e-10`). Useful for placing a group in a non-standard
orientation (e.g. quantisation axis off the z-axis).
"""
function rotate(G::PointGroup, R::AbstractMatrix)
    size(R) == (3, 3) || throw(ArgumentError("R must be 3×3"))
    Rs = SMatrix{3,3,Float64,9}(R)
    norm(Rs' * Rs - I) < 1e-10 || throw(ArgumentError(
        "R is not orthogonal (‖R^T R - I‖ = $(norm(Rs' * Rs - I)))"))
    Rinv = inv(Rs)
    rotated_gens = [(Rs * gen.matrix * Rinv, gen.tag) for gen in G.generators]
    return _construct_pointgroup(G.name, G.setting, rotated_gens)
end
