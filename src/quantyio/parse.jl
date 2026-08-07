# =====================================================================
# Parser — read Quanty's print(operator) text format
# =====================================================================
#
# Quanty's print(::OperatorType) format (real-coefficient case):
#
#   Operator: CrAn  [or other tag]
#   QComplex         =          0
#   MaxLength        =          2
#   NFermionic modes =         10
#   NBosonic modes   =          0
#
#   Operator of Length   2
#   QComplex      =          0
#   N             =          4
#   C  0 A  2 |  1.50000000000000E+00
#   C  1 A  3 |  1.50000000000000E+00
#   ...
#
#   Operator of Length   4
#   QComplex      =          0
#   N             =          1
#   C  1 C  0 A  1 A  0 | -4.00000000000000E+00
#
# Each chain line is a sequence of `C n` and `A n` tokens followed by `|`
# and a single coefficient (for QComplex=0). Quanty stores chains in
# normal-ordered form (creations left, annihilations right; descending
# index sort within each side).

"""
    ParsedTerm

A single operator term parsed from a Quanty dump:

- `kinds`       — vector of `:cdag` / `:c` symbols, in chain order
- `indices`     — vector of Quanty mode integers (0-indexed) in chain order
- `coefficient` — `ComplexF64`. For real (`QComplex=0`) blocks the imaginary
                  part is exactly zero; the builder collapses such operators
                  to a real `OperatorSum{Float64}` automatically.
"""
struct ParsedTerm
    kinds::Vector{Symbol}
    indices::Vector{Int}
    coefficient::ComplexF64
end

"""
    ParsedOperator

The contents of one Quanty operator dump, returned by `parse_quanty_operator`.

# Fields
- `terms::Vector{ParsedTerm}` — flat list of all ladder chains across every
  `Operator of Length N` block in the dump.
- `max_length::Int` — largest chain length declared in the top-level
  `MaxLength` header; `0` if the header was absent.
- `nfermion::Int` — number of fermionic modes from the `NFermionic modes`
  header; `0` if absent.
- `nboson::Int` — number of bosonic modes from the `NBosonic modes` header;
  `0` if absent.

Pass a `ParsedOperator` to `build_operator` (together with a `Hilbert` and a
`mode_map`) to obtain a MOADyna `OperatorSum`.  For the common single-file
workflow, `read_quanty_operator` wraps both steps.
"""
struct ParsedOperator
    terms::Vector{ParsedTerm}
    max_length::Int
    nfermion::Int
    nboson::Int
end

# --- Top-level parse ---

"""
    parse_quanty_operator(text::AbstractString) -> ParsedOperator

Parse the full text of a Quanty operator dump (output of Quanty's
`print(operator)`) and return a `ParsedOperator` intermediate.

The function runs a two-step pipeline:

1. **`parse_quanty_operator`** (this function) — reads the text, collects
   every ladder chain across all `Operator of Length N` blocks, and returns a
   `ParsedOperator` carrying the flat term list plus the bookkeeping scalars
   `max_length`, `nfermion`, and `nboson`.

2. **`build_operator(parsed, hilbert, mode_map)`** — converts the
   `ParsedOperator` into a MOADyna `OperatorSum` using the supplied `Hilbert`
   space and mode mapping.

For the common single-file workflow, `read_quanty_operator(path, hilbert,
mode_map)` wraps both steps and returns the final `OperatorSum` directly.
"""
function parse_quanty_operator(text::AbstractString)
    return parse_quanty_operator(IOBuffer(text))
end

function parse_quanty_operator(io::IO)
    terms = ParsedTerm[]
    max_length = 0
    nfermion = 0
    nboson = 0

    # Per-block bookkeeping for N-count validation
    expected_block_n = -1                  # N declared in the current block header
    block_terms_seen = 0
    block_label = "<top>"
    block_qc = 0                           # current block's QComplex flag (0 = real, 1 = complex)

    function _check_block_count()
        if expected_block_n >= 0 && block_terms_seen != expected_block_n
            throw(ArgumentError(
                "Quanty parse: block $block_label declared N=$expected_block_n " *
                "but parser found $block_terms_seen term(s)"))
        end
    end

    for line in eachline(io)
        line = strip(line)
        isempty(line) && continue

        # Top-level operator header lines
        if startswith(line, "MaxLength")
            max_length = max(max_length, _parse_header_int(line))
            continue
        elseif startswith(line, "NFermionic modes")
            nfermion = _parse_header_int(line)
            continue
        elseif startswith(line, "NBosonic modes")
            nboson = _parse_header_int(line)
            continue
        elseif startswith(line, "Operator of Length")
            # Close out the previous block before opening the next
            _check_block_count()
            block_label = line
            expected_block_n = -1
            block_terms_seen = 0
            block_qc = 0                            # default until per-block override
            continue
        elseif startswith(line, "N ") || startswith(line, "N\t")
            # Per-block "N = N_terms" declaration
            expected_block_n = _parse_header_int(line)
            continue
        elseif startswith(line, "QComplex")
            # Per-block QComplex flag: 0 = real (one float per chain line),
            # 1 = complex (two floats: real + imag). Mixed (2) is rejected
            # for now — it requires a per-line indicator that no NiO test
            # currently exercises. Top-level "QComplex" header (before any
            # "Operator of Length" block) sets the default for subsequent
            # blocks until they override it.
            qc = _parse_header_int(line)
            if qc == 0 || qc == 1
                block_qc = qc
            else
                throw(ArgumentError(
                    "Quanty parse: block $block_label has QComplex=$qc; " *
                    "MOADyna.QuantyIO supports QComplex ∈ {0, 1}. Mixed (2) " *
                    "blocks are not supported."))
            end
            continue
        end

        # Chain lines (contain a `|` separator). For these we MUST succeed
        # in parsing — silent skips would corrupt the operator sum.
        if occursin('|', line)
            term = _parse_chain_line(line, block_qc)
            if term === nothing
                throw(ArgumentError(
                    "Quanty parse: malformed chain line in block $block_label:\n" *
                    "  \"" * line * "\"\n" *
                    "Lines must look like \"C i [C j ...] A k [A l ...] | <real_coef>\"."))
            end
            push!(terms, term)
            block_terms_seen += 1
        end
    end

    # Final-block validation
    _check_block_count()

    return ParsedOperator(terms, max_length, nfermion, nboson)
end

# --- Per-line chain parser ---

# Match a `C i C j ... A k A l | <coef>` line and produce a ParsedTerm.
# Returns nothing for lines that don't match (defensive — non-chain lines
# are filtered out earlier but we double-check here).
#
# `qc = 0` (real block):    one Float64 after `|`  → ParsedTerm with imag=0.
# `qc = 1` (complex block): two Float64s          → ParsedTerm with the
#                            second number as the imaginary part.
function _parse_chain_line(line::AbstractString, qc::Int = 0)
    # Split off the coefficient
    parts = split(line, '|')
    length(parts) == 2 || return nothing
    chain_part = strip(parts[1])
    coef_part  = strip(parts[2])

    # Tokenize the chain part: alternating (C|A) (number).
    # An empty chain is a length-0 (constant) operator — Quanty emits these
    # as `|  <coef>` for the identity coefficient. We accept these and build
    # them as `coef · 1` in the OperatorSum.
    tokens = split(chain_part)
    iseven(length(tokens)) || return nothing

    kinds   = Symbol[]
    indices = Int[]
    i = 1
    while i <= length(tokens)
        t = tokens[i]
        if t == "C"
            push!(kinds, :cdag)
        elseif t == "A"
            push!(kinds, :c)
        else
            return nothing  # not a chain line
        end
        idx = tryparse(Int, tokens[i + 1])
        idx === nothing && return nothing
        push!(indices, idx)
        i += 2
    end

    coef_tokens = split(coef_part)
    coef = if qc == 1
        length(coef_tokens) == 2 || return nothing
        re = tryparse(Float64, coef_tokens[1])
        im = tryparse(Float64, coef_tokens[2])
        (re === nothing || im === nothing) && return nothing
        complex(re, im)
    else
        length(coef_tokens) == 1 || return nothing
        re = tryparse(Float64, coef_tokens[1])
        re === nothing && return nothing
        complex(re, 0.0)
    end

    return ParsedTerm(kinds, indices, coef)
end

# Split a header line like "MaxLength = 2 (largest number ...)"
# Quanty appends a parenthesised description after the value; take the
# first whitespace-delimited token after `=`.
function _parse_header_int(line::AbstractString)
    eq = findfirst('=', line)
    eq === nothing && return 0
    after = strip(line[(eq + 1):end])
    tokens = split(after; limit = 2)
    isempty(tokens) && return 0
    n = tryparse(Int, tokens[1])
    return n === nothing ? 0 : n
end

# --- Convenience: parse just one length block (testing) ---

"""
    parse_quanty_block(text)

Parse a single chain block (one or more `C/A` lines after one
`Operator of Length N` header). Returns the same `ParsedOperator` shape
as `parse_quanty_operator`, but only the term entries are populated;
header fields stay at zero.
"""
parse_quanty_block(text::AbstractString) = parse_quanty_operator(text)
