# UX: character_table accessor, pretty-print at REPL,
# computed-vs-reference comparison helper.

using Printf

"""
    character_table(G::PointGroup; source::Symbol=:auto)

Return the character table of `G` as a `NamedTuple` with fields:

- `classes::Vector{Symbol}`           — class labels in column order
- `class_sizes::Vector{Int}`          — `|C_k|` per class
- `irreps::Vector{Symbol}`            — IR labels in row order
- `ir_dims::Vector{Int}`              — `dim(Γ)` per IR
- `characters::Matrix{Float64}`       — `(n_irrep × n_class)` real character matrix
- `source::Symbol`                    — `:reference` or `:computed`
- `provenance::String`                — human-readable source attribution

`source=:reference` returns the curated published table for groups in
`reference_label_groups()`; column order matches the canonical
published convention. `source=:computed` returns the table in MOAD's
internal BFS class-discovery order, with row labels exactly as they
appear in `G.irreps`. `:auto` (default) picks `:reference` when
available, else `:computed`.
"""
function character_table(G::PointGroup; source::Symbol=:auto)
    src = source === :auto ?
          (haskey(REFERENCE_CHARACTER_TABLES, G.name) ? :reference : :computed) :
          source
    if src === :reference
        haskey(REFERENCE_CHARACTER_TABLES, G.name) || throw(ArgumentError(
            "character_table: no reference table for :$(G.name); pass " *
            "`source=:computed` to get the Burnside-computed table instead."))
        ref = REFERENCE_CHARACTER_TABLES[G.name]
        return (classes=ref.classes, class_sizes=ref.class_sizes,
                irreps=ref.irreps, ir_dims=ref.ir_dims,
                characters=copy(ref.characters), source=:reference,
                provenance=ref.source)
    elseif src === :computed
        n_class = length(G.classes)
        n_ir = length(G.irreps)
        chars = Matrix{Float64}(undef, n_ir, n_class)
        for i in 1:n_ir, j in 1:n_class
            chars[i, j] = real(G.irreps[i].characters[j])
        end
        return (classes=Symbol[G.elements[G.classes[k][1]].tag for k in 1:n_class],
                class_sizes=Int[length(C) for C in G.classes],
                irreps=Symbol[ir.label for ir in G.irreps],
                ir_dims=Int[ir.real_dim for ir in G.irreps],
                characters=chars, source=:computed,
                provenance="MOAD Burnside-Dixon class-operator algorithm; classes in BFS-discovery order")
    else
        throw(ArgumentError(
            "character_table: source must be :reference, :computed, or :auto, got :$src"))
    end
end

# Round to nearest integer if very close, else 3 decimals.
function _format_char(x::Real)
    abs(x - round(x)) < 1e-9 && return string(Int(round(x)))
    return @sprintf("%.3f", x)
end

# Pretty-print a character_table NamedTuple.
function _print_character_table(io::IO, ct::NamedTuple)
    n_class = length(ct.classes)
    n_ir = length(ct.irreps)

    # Column headers: "n·label" or just "label" when size = 1.
    class_strs = [ct.class_sizes[j] == 1 ? string(ct.classes[j]) :
                  string(ct.class_sizes[j], string(ct.classes[j]))
                  for j in 1:n_class]
    ir_strs   = string.(ct.irreps)
    char_strs = [_format_char(ct.characters[i, j]) for i in 1:n_ir, j in 1:n_class]

    label_w = max(maximum(length, ir_strs), 3)
    col_w = [max(length(class_strs[j]),
                 maximum(length(char_strs[i, j]) for i in 1:n_ir))
             for j in 1:n_class]

    # Top header
    print(io, " " ^ (label_w + 1), "│")
    for j in 1:n_class
        print(io, " ", lpad(class_strs[j], col_w[j]))
    end
    println(io)
    # Separator
    print(io, "─" ^ (label_w + 1), "┼")
    for j in 1:n_class
        print(io, "─" ^ (col_w[j] + 1))
    end
    println(io)
    # Rows
    for i in 1:n_ir
        print(io, lpad(ir_strs[i], label_w), " │")
        for j in 1:n_class
            print(io, " ", lpad(char_strs[i, j], col_w[j]))
        end
        println(io)
    end
end

"""
    print_character_table(G::PointGroup; source::Symbol=:auto)
    print_character_table(io::IO, G::PointGroup; source::Symbol=:auto)

Pretty-print the character table of `G` — irrep rows against
conjugacy-class columns — to `stdout` (or to `io`).

The table itself comes from [`character_table`](@ref
MOAD.PointGroups.character_table); `source` selects its provenance and is
forwarded unchanged (`:auto` by default).

# Example
```julia
print_character_table(pointgroup(:Oh))
```
"""
print_character_table(G::PointGroup; source::Symbol=:auto) =
    _print_character_table(stdout, character_table(G; source=source))
print_character_table(io::IO, G::PointGroup; source::Symbol=:auto) =
    _print_character_table(io, character_table(G; source=source))

# REPL display.
function Base.show(io::IO, ::MIME"text/plain", G::PointGroup)
    has_ref = haskey(REFERENCE_CHARACTER_TABLES, G.name)
    prov = has_ref ? "reference" : "computed_auto"
    println(io, "PointGroup(:", G.name, ", |G|=", length(G.elements),
            ", ", length(G.irreps), " IRs, labels=", prov, ")")
    println(io)
    _print_character_table(io, character_table(G))
    println(io)
    if has_ref
        ref = REFERENCE_CHARACTER_TABLES[G.name]
        print(io, "Source: ", ref.source)
    else
        print(io, "Labels auto-named; for canonical Mulliken labels add a ",
              "`ReferenceCharacterTable` for :$(G.name) in `reference_tables.jl`.")
    end
end

"""
    character_table_compare([io::IO,] G::PointGroup)

Print the curated reference character table and the Burnside-computed
table side by side. Useful as a teaching diagnostic — for
reference-listed groups, MOAD already cross-validates the two tables
at construction (mismatch raises), so this primarily highlights the
class-permutation alignment between published canonical order and
MOAD's BFS-discovery order. Throws if `G` has no reference table.
"""
function character_table_compare(io::IO, G::PointGroup)
    haskey(REFERENCE_CHARACTER_TABLES, G.name) || throw(ArgumentError(
        "character_table_compare: no reference table for :$(G.name) — " *
        "nothing to compare. Use `print_character_table(G; source=:computed)` " *
        "to inspect the Burnside-computed table alone."))
    println(io, "── Reference (canonical column order) ──")
    _print_character_table(io, character_table(G; source=:reference))
    println(io)
    println(io, "── Computed (BFS class-discovery order) ──")
    _print_character_table(io, character_table(G; source=:computed))
    println(io)
    println(io, "Note: the two tables describe the same algebra. Columns are")
    println(io, "permuted; rows match by Mulliken label (validated at")
    println(io, "construction). Class permutation is recovered via the")
    println(io, "fingerprint matcher in `reference_tables.jl`.")
    return nothing
end

character_table_compare(G::PointGroup) = character_table_compare(stdout, G)
