#!/usr/bin/env julia
# parse_haverkort.jl — read ref/Quanty_scripts/slater_integrals.tex and emit
# src/atomic_parameters/haverkort_data.jl with the const HAVERKORT_PARAMETERS
# dict. Re-run when the tex source updates.
#
# Source: M. W. Haverkort, PhD thesis, Universität zu Köln (2005),
#         appendix table "Slater integrals for 3d and 4d elements".
#         arXiv:cond-mat/0505214
# Computed with R. D. Cowan's RCN36K Hartree-Fock atomic-structure code
# (Cowan 1981). RAW HF values — no scaling applied.
#
# Schema (nested NamedTuple):
#   key = (element::Symbol, configuration::String)
#       The configuration string is the canonical, normalised form
#       (lowercase shell letters, no `^`, single-space-separated shells,
#       shells in standard order); e.g. "3d8" or "2p5 3d9".
#       Examples:
#         (:Ni, "3d8")       → 2p^6 3d^8 ground configuration of Ni²⁺
#         (:Ni, "2p5 3d9")   → 2p^5 3d^9 L_{2,3}-edge intermediate state
#   val = NamedTuple with fields
#       Fdd           :: NamedTuple (F2 :: Float64, F4 :: Float64)        (eV)  — d-d direct
#       Fpd           :: NamedTuple (F2 :: Float64)                       (eV)  — 2p-d direct
#       Gpd           :: NamedTuple (G1 :: Float64, G3 :: Float64)        (eV)  — 2p-d exchange
#       zeta          :: NamedTuple (d :: Float64, p :: Float64)          (eV)  — spin-orbit per shell
#       r             :: NamedTuple (r2 :: Float64, r4 :: Float64)        (Å^n; NaN ⇔ flagged)
#       unreliable    :: Bool
#       configuration :: String, e.g. "3d8" or "2p5 3d9" (== the dict key's second field)
#       provenance    :: Symbol = :Haverkort_thesis_2005
#
# Ground-configuration rows (no 2p hole) carry NaN in Fpd.F2, Gpd.G1, Gpd.G3,
# and zeta.p. A row marked "*" by Haverkort (r^2 > 10 Å² or r^4 > 10 Å⁴)
# carries r.r2 = NaN, r.r4 = NaN, unreliable = true. The Slater integrals on
# those rows should not be trusted in a solid (per the thesis text).

const TEX_FILE = joinpath(@__DIR__, "..", "ref", "Quanty_scripts",
                          "slater_integrals.tex")
const OUT_FILE = joinpath(@__DIR__, "..", "src", "atomic_parameters",
                          "haverkort_data.jl")

# Group number (= number of (n-1)d + ns valence electrons) for each tabulated
# element. Charge = group - n_d_in_configuration.
const GROUP_NUMBER = Dict{Symbol, Int}(
    # 3d block (and the two pre-3d s-block entries K, Ca with tabulated d^N)
    :K  => 1, :Ca => 2,
    :Sc => 3, :Ti => 4, :V  => 5, :Cr => 6, :Mn => 7,
    :Fe => 8, :Co => 9, :Ni => 10, :Cu => 11, :Zn => 12,
    # 4d block (Rb, Sr same convention)
    :Rb => 1, :Sr => 2,
    :Y  => 3, :Zr => 4, :Nb => 5, :Mo => 6, :Tc => 7,
    :Ru => 8, :Rh => 9, :Pd => 10, :Ag => 11, :Cd => 12,
)

"""
    parse_number_pair(int_part, frac_part) -> Float64

Parse the `r@{.}l` two-cell pattern from the tex (e.g. `\$12\$&\$233\$`)
into a single Float64. `int_part` and `frac_part` are the inner contents
of the two `\$...\$` groups. Returns NaN if either cell is `*`.
"""
function parse_number_pair(int_part::AbstractString, frac_part::AbstractString)
    a = strip(int_part)
    b = strip(frac_part)
    if a == "*" || b == "*" || isempty(a) || isempty(b)
        return NaN
    end
    return parse(Float64, a * "." * b)
end

"""
    extract_dollar_groups(s) -> Vector{String}

Return the contents of every \$...\$ group in `s`, in order.
"""
function extract_dollar_groups(s::AbstractString)
    out = String[]
    for m in eachmatch(r"\$([^$]*)\$", s)
        push!(out, m.captures[1])
    end
    return out
end

"""
    parse_config_cell(cell) -> (n_d::Int, has_2p_hole::Bool, config_str::String)

The first column of every data row encodes the configuration as
`\$2p^{6}\$ \$3d^{N\\phantom{0}}\$` (ground, no 2p hole) or
`\$2p^{5}\$ \$3d^{N\\phantom{0}}\$` (L_{2,3}-edge intermediate).
4d analogues use `4d^{N}`.
"""
function parse_config_cell(cell::AbstractString)
    # Pull out the two dollar-groups.
    grps = extract_dollar_groups(cell)
    @assert length(grps) >= 2 "config cell missing dollar groups: $cell"
    p_part = grps[1]   # e.g. "2p^{6}"
    d_part = grps[2]   # e.g. "3d^{8\\phantom{0}}" or "3d^{10}"

    # 2p hole?
    m_p = match(r"2p\^\{(\d+)\}", p_part)
    @assert m_p !== nothing "could not parse 2p shell from: $p_part"
    n_2p = parse(Int, m_p.captures[1])
    has_2p_hole = (n_2p == 5)

    # d-count and shell label (3d or 4d).
    m_d = match(r"([34])d\^\{(\d+)(?:\\phantom\{0\})?\}", d_part)
    @assert m_d !== nothing "could not parse nd shell from: $d_part"
    shell = m_d.captures[1]   # "3" or "4"
    n_d = parse(Int, m_d.captures[2])

    # Canonical key form: no `^`, single space, shells in standard order.
    # 2p (n=2,ℓ=p) before 3d/4d, so the order is already correct.
    config_str = if has_2p_hole
        "2p5 $(shell)d$n_d"
    else
        "$(shell)d$n_d"
    end
    return n_d, has_2p_hole, config_str
end

"""
    parse_data_row(line) -> (n_d, has_2p_hole, config_str, values::NamedTuple, unreliable::Bool)

Parse a single tex data row (`\$2p^{...}\$ \$Xd^{...}\$& ... \\\\`).
Splits on `&`, then walks the cells.
"""
function parse_data_row(line::AbstractString)
    # Strip trailing "\\" and any "\vline" tag.
    body = replace(line, r"\\vline\s*\\\\\s*$" => "")
    body = replace(body, r"\\\\\s*$" => "")
    body = strip(body)

    raw_cells = split(body, '&')

    # Expand multicolumn{2}{c}{X}: occupies two grid columns. We map each raw
    # `&`-separated cell into either one slot (regular cell) or two slots
    # tagged with the special `MC` marker carrying the inner content. After
    # this expansion there are exactly 19 slots: 1 config + 9*(int, frac).
    # Slot type: either an AbstractString (regular cell) or a
    # `(:MC, inner::String, starred::Bool)` tuple covering BOTH grid columns
    # of the r@{.}l pair it spans.
    slots = Any[]
    for c in raw_cells
        m = match(r"\\multicolumn\{2\}\{c\}\{(.*?)\}", c)
        if m !== nothing
            inner = strip(m.captures[1])
            starred = (inner == "\$*\$" || inner == "*")
            push!(slots, (:MC, inner, starred))
            push!(slots, (:MC, inner, starred))   # placeholder for the second column
        else
            push!(slots, c)
        end
    end
    @assert length(slots) >= 19 "row has too few slots ($(length(slots))): $line"

    # Column 1: configuration label.
    n_d, has_2p_hole, config_str = parse_config_cell(slots[1])

    # Helper: read an r@{.}l number pair occupying slots[i_lo:i_hi].
    function read_pair(i_lo::Int, i_hi::Int)
        a, b = slots[i_lo], slots[i_hi]

        # Case A: BOTH halves are the same multicolumn placeholder.
        if a isa Tuple && a[1] === :MC && b isa Tuple && b[1] === :MC
            inner = a[2]
            starred = a[3]
            return (NaN, starred)
        end

        # Case B: Rb's deviant single-column phantom-then-star formatting,
        # e.g. "\phantom{0}& \$*\$ \phantom{0}".
        a_str = a isa AbstractString ? a : ""
        b_str = b isa AbstractString ? b : ""
        if occursin(r"\\phantom\{0\}", a_str) && occursin("\$*\$", b_str)
            return (NaN, true)
        end

        # Case C: standard r@{.}l number pair.
        a_groups = extract_dollar_groups(a_str)
        b_groups = extract_dollar_groups(b_str)
        if isempty(a_groups) || isempty(b_groups)
            return (NaN, false)
        end
        return (parse_number_pair(a_groups[end], b_groups[1]), false)
    end

    r2, r2_star = read_pair(2, 3)
    r4, r4_star = read_pair(4, 5)
    unreliable = r2_star || r4_star

    zeta_d, _   = read_pair(6, 7)
    F2_dd, _    = read_pair(8, 9)
    F4_dd, _    = read_pair(10, 11)
    zeta_2p, _  = read_pair(12, 13)
    F2_pd, _    = read_pair(14, 15)
    G1_pd, _    = read_pair(16, 17)
    G3_pd, _    = read_pair(18, 19)

    # Ground-state rows (2p^6) have no core-hole couplings — coerce to NaN
    # rather than rely on whatever read_pair returned for empty cells.
    if !has_2p_hole
        zeta_2p = NaN
        F2_pd = NaN
        G1_pd = NaN
        G3_pd = NaN
    end

    vals = (
        Fdd = (F2 = F2_dd, F4 = F4_dd),
        Fpd = (F2 = F2_pd,),
        Gpd = (G1 = G1_pd, G3 = G3_pd),
        zeta = (d = zeta_d, p = zeta_2p),
        r = (r2 = r2, r4 = r4),
        unreliable = unreliable,
        configuration = config_str,
        provenance = :Haverkort_thesis_2005,
    )
    return n_d, has_2p_hole, config_str, vals
end

"""
    parse_haverkort_tex(path) -> Dict

Walk the tex file.  Element name comes from the table-header line
"<Symbol>& \\multicolumn{2}{c}{\$r^2\$} ...".  Data rows are recognised
by starting with `\$2p^{`.
"""
function parse_haverkort_tex(path::AbstractString)
    out = Dict{Tuple{Symbol, String}, NamedTuple}()
    current_element::Union{Nothing, Symbol} = nothing
    n_rows = 0
    n_collisions = 0

    open(path, "r") do io
        for line in eachline(io)
            sline = strip(line)
            isempty(sline) && continue

            # Element-header detection: a short token before "&" that is one
            # of our known elements.
            if occursin(r"\\multicolumn\{2\}\{c\}\{\$r\^2\$\}", sline)
                m_el = match(r"^\s*([A-Z][a-z]?)\s*&", sline)
                if m_el !== nothing
                    sym = Symbol(m_el.captures[1])
                    if haskey(GROUP_NUMBER, sym)
                        current_element = sym
                    else
                        @warn "unrecognised element header: $(m_el.captures[1])"
                    end
                end
                continue
            end

            # Data-row detection.
            if startswith(sline, raw"$2p^{")
                @assert current_element !== nothing "data row before any element header"
                _, _, config_str, vals = parse_data_row(sline)
                key = (current_element, config_str)
                if haskey(out, key)
                    n_collisions += 1
                    @warn "duplicate key, keeping first: $key"
                else
                    out[key] = vals
                end
                n_rows += 1
            end
        end
    end
    @info "parsed $n_rows rows, kept $(length(out)) unique keys, $n_collisions collisions"
    return out
end

# ---------------------------------------------------------------------------
# Emit Julia source.
# ---------------------------------------------------------------------------

"""
    fmt_float(x) -> String

Format a Float64 for pretty inclusion in the emitted file.
NaN → "NaN"; finite values keep up to 4 fractional digits, no trailing zeros
beyond the source precision.
"""
function fmt_float(x::Float64)
    isnan(x) && return "NaN"
    s = string(x)
    return s
end

function fmt_value(v::NamedTuple)
    parts = String[]
    push!(parts, "Fdd=(F2=$(fmt_float(v.Fdd.F2)), F4=$(fmt_float(v.Fdd.F4)))")
    push!(parts, "Fpd=(F2=$(fmt_float(v.Fpd.F2)),)")
    push!(parts, "Gpd=(G1=$(fmt_float(v.Gpd.G1)), G3=$(fmt_float(v.Gpd.G3)))")
    push!(parts, "zeta=(d=$(fmt_float(v.zeta.d)), p=$(fmt_float(v.zeta.p)))")
    push!(parts, "r=(r2=$(fmt_float(v.r.r2)), r4=$(fmt_float(v.r.r4)))")
    push!(parts, "unreliable=$(v.unreliable)")
    push!(parts, "configuration=$(repr(v.configuration))")
    push!(parts, "provenance=$(repr(v.provenance))")
    return "(" * join(parts, ", ") * ")"
end

# Rank elements in periodic-table order (3d block first, then 4d).
const ELEMENT_ORDER = Dict{Symbol, Int}(
    :K=>1, :Ca=>2, :Sc=>3, :Ti=>4, :V=>5, :Cr=>6, :Mn=>7,
    :Fe=>8, :Co=>9, :Ni=>10, :Cu=>11, :Zn=>12,
    :Rb=>13, :Sr=>14, :Y=>15, :Zr=>16, :Nb=>17, :Mo=>18, :Tc=>19,
    :Ru=>20, :Rh=>21, :Pd=>22, :Ag=>23, :Cd=>24,
)

# Sort keys (element, config) so the emitted file lists ground before
# core-hole rows and high-charge before low-charge within each element.
# We sort secondarily by (has_2p_hole_flag, -nd, config) which is a stable
# proxy for (edge, charge ascending within edge).
function _sort_key_for_config(config::AbstractString)
    has_2p = startswith(config, "2p5")
    # Pull the trailing nd^N integer (works for "3d8", "2p5 3d9", etc.).
    parts = split(config)
    last_part = parts[end]
    m = match(r"^([0-9])([spdfg])([0-9]+)$", last_part)
    n_val = m === nothing ? 0 : parse(Int, m.captures[3])
    return (has_2p ? 1 : 0, -n_val, config)
end

function key_sort_index(k::Tuple{Symbol, String})
    el, cfg = k
    has_2p, neg_n, _ = _sort_key_for_config(cfg)
    return (ELEMENT_ORDER[el], has_2p, neg_n, cfg)
end

function emit_julia(dict, outpath::AbstractString)
    mkpath(dirname(outpath))
    open(outpath, "w") do io
        println(io, "# AUTO-GENERATED by dev_scripts/parse_haverkort.jl — do not hand-edit.")
        println(io, "#")
        println(io, "# Source: M. W. Haverkort, PhD thesis, Universität zu Köln (2005),")
        println(io, "#         appendix \"Slater integrals for 3d and 4d elements\".")
        println(io, "#         arXiv:cond-mat/0505214")
        println(io, "# Computed with R. D. Cowan's RCN36K Hartree-Fock atomic-structure code")
        println(io, "# (Cowan, \"The Theory of Atomic Structure and Spectra\", 1981).")
        println(io, "# RAW HF values — no scaling applied.")
        println(io, "#")
        println(io, "# Schema (nested NamedTuple):")
        println(io, "#   key = (element::Symbol, configuration::String)")
        println(io, "#         The configuration string is the canonical, normalised form")
        println(io, "#         (lowercase shell letters, no `^`, single-space-separated shells).")
        println(io, "#   val = NamedTuple with fields")
        println(io, "#         Fdd        :: (F2, F4)        (eV)  — d-d direct")
        println(io, "#         Fpd        :: (F2,)           (eV)  — 2p-d direct (NaN on ground)")
        println(io, "#         Gpd        :: (G1, G3)        (eV)  — 2p-d exchange (NaN on ground)")
        println(io, "#         zeta       :: (d, p)          (eV)  — spin-orbit per shell (zeta.p NaN on ground)")
        println(io, "#         r          :: (r2, r4)        (Å^n; NaN ⇔ Haverkort flagged it >10)")
        println(io, "#         unreliable :: Bool            — row was *-flagged in the thesis")
        println(io, "#         configuration :: String       — same as key's second field (e.g. \"3d8\" or \"2p5 3d9\")")
        println(io, "#         provenance :: Symbol          — :Haverkort_thesis_2005")
        println(io, "#")
        println(io, "# Configuration keys: \"3d^N\" / \"4d^N\" rows are ground;")
        println(io, "# \"2p5 3d^N\" / \"2p5 4d^N\" rows are L_{2,3} core-hole intermediate states.")
        println(io)
        println(io, "const HAVERKORT_PARAMETERS = Dict{Tuple{Symbol, String}, NamedTuple}(")
        sorted_keys = sort(collect(keys(dict)); by = key_sort_index)
        for key in sorted_keys
            v = dict[key]
            println(io, "    (:$(key[1]), $(repr(key[2]))) => ", fmt_value(v), ",")
        end
        println(io, ")")
    end
end

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------

dict = parse_haverkort_tex(TEX_FILE)
emit_julia(dict, OUT_FILE)
println("Emitted $(length(dict)) entries to $(OUT_FILE)")

# ---------------------------------------------------------------------------
# Inline reference checks — fire on every run so a tex edit can't silently
# corrupt the canonical artifact.
# ---------------------------------------------------------------------------

let
    ni2 = dict[(:Ni, "3d8")]
    @assert isapprox(ni2.Fdd.F2, 12.233; atol=1e-3) "Ni2+ Fdd.F2 mismatch: $(ni2.Fdd.F2)"
    @assert isapprox(ni2.Fdd.F4,  7.597; atol=1e-3) "Ni2+ Fdd.F4 mismatch: $(ni2.Fdd.F4)"
    @assert isapprox(ni2.zeta.d, 0.083; atol=1e-3) "Ni2+ zeta.d mismatch: $(ni2.zeta.d)"
    @assert isnan(ni2.Fpd.F2)            "Ni2+ ground row should have no Fpd.F2"
    @assert ni2.unreliable === false     "Ni2+ ground should not be flagged"
    @assert ni2.configuration == "3d8"
    @assert ni2.provenance === :Haverkort_thesis_2005

    cu2 = dict[(:Cu, "3d9")]
    @assert isapprox(cu2.Fdd.F2, 12.854; atol=1e-3) "Cu2+ Fdd.F2 mismatch: $(cu2.Fdd.F2)"
    @assert isapprox(cu2.Fdd.F4,  7.980; atol=1e-3) "Cu2+ Fdd.F4 mismatch: $(cu2.Fdd.F4)"
    @assert cu2.configuration == "3d9"

    # L23 intermediate state for Ni2+ : 2p^5 3d^9
    ni2_l23 = dict[(:Ni, "2p5 3d9")]
    @assert isapprox(ni2_l23.zeta.p, 11.507; atol=1e-3) "Ni2+ L23 zeta.p mismatch"
    @assert ni2_l23.configuration == "2p5 3d9"

    # A starred row: K (Z=19) 3d^1 ground (charge = 1 - 1 = 0).
    k0 = dict[(:K, "3d1")]
    @assert k0.unreliable === true   "K0 3d^1 row should be * flagged"
    @assert isnan(k0.r.r2) && isnan(k0.r.r4)
end
println("Reference assertions passed.")
