"""
    MOADyna.AtomicParameters

Atomic Slater-Condon parameters (Fᵏ, Gᵏ, ζ) and radial moments (⟨r²⟩, ⟨r⁴⟩)
for transition-metal and lanthanide configurations.

# Public API — two forms

The lookup is keyed by **electron configuration** (the unambiguous physical
quantity). Two equivalent entry points:

1. **Primary (configuration string).** Pass the electron configuration
   directly:

   ```julia
   atomic_parameters(:Ni, "3d8")           # 2p⁶ 3d⁸ ground (Ni²⁺)
   atomic_parameters(:Ni, "2p5 3d9")       # L_{2,3} core-hole intermediate
   atomic_parameters(:Pr, "4f2")           # Pr³⁺ ground
   atomic_parameters(:Pr, "3d9 4f3")       # M_{4,5} intermediate
   ```

   Configuration strings are normalised internally — `"3d^8"`, `"3D8"`,
   `"3d_8"`, `" 3d8 "` all canonicalise to `"3d8"`.

2. **Sugar (charge-keyed, GROUND STATE ONLY).** Resolves to a ground
   configuration via a small lookup table built automatically from the
   static dict's coverage:

   ```julia
   atomic_parameters(:Ni; charge=2)              # → "3d8"
   ```

   The sugar form returns ground configurations only. For core-hole or
   any other intermediate configuration (e.g. L_{2,3} `"2p5 3d9"`,
   M_{4,5} `"3d9 4f3"`), use the primary form. If `(element, charge)`
   has no ground entry, an `ArgumentError` lists the available ground
   configurations for that `element` and points at the primary form.

Both forms accept the same kwargs (`scaling`, `cowan`).

# Return shape

For d-block elements (3d / 4d / 5d):

```
(Fdd  = (F2, F4),                  # d-d direct (eV)
 Fpd  = (F2,),                     # 2p-d direct  (NaN ⇔ no 2p hole in config)
 Gpd  = (G1, G3),                  # 2p-d exchange (NaN ⇔ no 2p hole)
 zeta = (d, p),                    # spin-orbit per shell (eV; NaN where absent)
 r    = (r2, r4),                  # ⟨rᵏ⟩ on the valence shell (Å^k)
 unreliable    :: Bool,            # `*`-flagged in Haverkort thesis
 configuration :: String,          # canonical normalised form
 provenance    :: Symbol,          # data source
 scaling       :: Symbol)          # scaling actually applied
```

For f-block elements (4f / 5f):

```
(Fff  = (F2, F4, F6),              # f-f direct (eV)
 Fdf  = (F2, F4),                  # 3d-f direct (NaN ⇔ no 3d hole in config)
 Gdf  = (G1, G3, G5),              # 3d-f exchange (NaN ⇔ no 3d hole)
 zeta = (f, d),                    # spin-orbit per shell (NaN where absent)
 r    = (r2, r4),                  # ⟨rᵏ⟩ on the valence shell (Å^k)
 unreliable, configuration, provenance, scaling)
```

# Naming rule

Slater-integral fields follow `(F|G)<orbital_pair>` and spin-orbit fields
follow `zeta.<orbital_letter>`:

  * `<orbital_pair>` is the two shell *types* involved
    (e.g. `dd` for d-d self-interaction, `pd` for 2p-d core-valence,
    `ff` for f-f, `df` for 3d-f).
  * Homo-shell pairs (`dd`, `ff`) carry only direct (Fᵏ) integrals.
  * Hetero-shell pairs (`pd`, `df`) carry both direct (Fᵏ) and
    exchange (Gᵏ).
  * `zeta.<letter>` uses a single orbital letter (`d`, `p`, `f`).

# Lookup chain

  1. Static dict from Haverkort thesis (Cowan RCN36K HF, RAW).
     Returns `provenance = :Haverkort_thesis_2005`.
  2. (Optional) live Cowan run if `MOADYNA_COWAN` (or `TTMULT`) env var
     is set OR the `cowan` kwarg points at a binary. Returns
     `provenance = :cowan_<basename>`.
  3. `ArgumentError` listing the available configurations.

# Citations

Static data: M. W. Haverkort, "Spin and orbital degrees of freedom in
transition metal oxides and oxide thin films studied by soft x-ray
absorption spectroscopy", PhD thesis, Universität zu Köln (2005),
arXiv:cond-mat/0505214.

Computed values follow R. D. Cowan, *The Theory of Atomic Structure and
Spectra* (UC Press, 1981).

# Live Cowan binaries

MOADyna does NOT bundle Cowan binaries. Recommended sources:

  - C. McGuinness's TCD distribution:
    https://www.tcd.ie/Physics/people/Cormac.McGuinness/Cowan
  - C. J. Titus's `ttmult` build:
    https://bitbucket.org/cjtitus/ttmult

Build from source against your local Fortran toolchain (the McGuinness/TCD
source builds cleanly on macOS, Linux, and Apple Silicon; pre-built
x86_64 binaries from third-party sources require Rosetta on M-series Macs).
"""
module AtomicParameters

include("haverkort_data.jl")    # provides const HAVERKORT_PARAMETERS
include("cowan_runner.jl")      # provides _run_cowan(...)

export atomic_parameters, radial_wavefunction, covered_elements, covered_configurations

# ---------------------------------------------------------------------
# Configuration normalisation
# ---------------------------------------------------------------------

# Standard ordering for shells: (n, ℓ_index) ascending. ℓ index follows
# the spectroscopic order s, p, d, f, g (= 0, 1, 2, 3, 4).
const _SHELL_LETTER_ORDER = Dict{Char, Int}(
    's' => 0, 'p' => 1, 'd' => 2, 'f' => 3, 'g' => 4,
)

# Closed-shell capacities: 2(2ℓ+1) for ℓ = 0..4.
const _SHELL_CAPACITY = Dict{Char, Int}(
    's' => 2, 'p' => 6, 'd' => 10, 'f' => 14, 'g' => 18,
)

"""
    _normalize_config(s) -> String

Canonicalise a user-supplied configuration string into MOADyna's normalised
form: lowercase shell letters, no `^`/`_` separators, single-space-
separated shell groups, shells sorted in standard `(n, ℓ)` ascending
order. Closed-shell tokens lying *below* an open valence shell are
dropped — they describe spectator core that contributes nothing
observable, and stripping them makes "2p6 3d8" canonically equivalent
to "3d8". Examples:

```
"3d^8"        → "3d8"
"3D8"         → "3d8"
" 3d8 "       → "3d8"
"3d8 2p5"     → "2p5 3d8"     # reordered (2p5 is open, kept)
"2p^5 3d^9"   → "2p5 3d9"
"3d_9 4f_3"   → "3d9 4f3"
"2p6 3d8"     → "3d8"         # 2p6 is closed core below open 3d → dropped
"3d10 4f2"    → "4f2"         # 3d10 is closed core below open 4f → dropped
"2p5 3d10"    → "2p5 3d10"    # 2p5 is open; 3d10 closed but above → kept
```

Each shell group must match `<n><letter><electrons>` after stripping
`^` and `_`. Throws `ArgumentError` on an unparseable token, on
duplicate shell labels, or on over-capacity occupancies (s≤2, p≤6,
d≤10, f≤14, g≤18).
"""
function _normalize_config(s::AbstractString)
    original = String(s)
    # Lowercase, drop the optional `^` and `_` separators.
    cleaned = lowercase(replace(replace(original, "^" => ""), "_" => ""))
    tokens = split(cleaned)
    isempty(tokens) && throw(ArgumentError(
        "AtomicParameters: empty configuration string."))
    parsed = Tuple{Int, Char, Int, String}[]   # (n, letter, electrons, token)
    seen = Set{Tuple{Int, Char}}()
    for tok in tokens
        m = match(r"^([1-9])([spdfg])([0-9]+)$", tok)
        m === nothing && throw(ArgumentError(
            "AtomicParameters: cannot parse configuration token \"$tok\". " *
            "Expected `<n><letter><electrons>` (e.g. \"3d8\", \"2p5\")."))
        n = parse(Int, m.captures[1])
        letter = m.captures[2][1]
        electrons = parse(Int, m.captures[3])
        # Validation: no duplicate shell labels (e.g. "3d8 3d2").
        if (n, letter) in seen
            throw(ArgumentError(
                "AtomicParameters: configuration \"$original\" has duplicate " *
                "shell label \"$n$letter\"."))
        end
        push!(seen, (n, letter))
        # Validation: capacity (s≤2, p≤6, d≤10, f≤14, g≤18).
        cap = _SHELL_CAPACITY[letter]
        if electrons > cap
            throw(ArgumentError(
                "AtomicParameters: configuration \"$original\": shell " *
                "\"$n$letter\" has count $electrons exceeding capacity $cap."))
        end
        push!(parsed, (n, letter, electrons, "$n$letter$electrons"))
    end
    sort!(parsed; by = t -> (t[1], _SHELL_LETTER_ORDER[t[2]]))
    # Strip closed-shell tokens that lie *below* the highest open shell.
    # A token is "open" iff 1 ≤ count < capacity; "closed" iff count == capacity.
    # Spectator closed-core (e.g. 2p6 below 3d8) carries no observable, so we
    # drop it for canonical equivalence with the open-only form.
    kept = _strip_closed_core(parsed)
    return join((t[4] for t in kept), " ")
end

"""
    _strip_closed_core(parsed) -> Vector

Given a `(n, letter, electrons, token)` tuple list sorted by (n, ℓ),
drop closed-shell tokens that lie strictly below the highest open
shell. If no open shell exists, return the list unchanged so callers
still see the user's intended configuration (and downstream lookup
can decide what to do).
"""
function _strip_closed_core(parsed::Vector{Tuple{Int, Char, Int, String}})
    # Find the (n, ℓ) of the highest open shell.
    highest_open_idx = 0
    for (i, t) in enumerate(parsed)
        n, letter, electrons, _ = t
        cap = _SHELL_CAPACITY[letter]
        if 1 <= electrons < cap
            highest_open_idx = i
        end
    end
    highest_open_idx == 0 && return parsed   # no open shell anywhere
    n_open, letter_open, _, _ = parsed[highest_open_idx]
    open_rank = (n_open, _SHELL_LETTER_ORDER[letter_open])
    out = Tuple{Int, Char, Int, String}[]
    for t in parsed
        n, letter, electrons, _ = t
        cap = _SHELL_CAPACITY[letter]
        rank = (n, _SHELL_LETTER_ORDER[letter])
        is_closed = (electrons == cap)
        if is_closed && rank < open_rank
            continue   # spectator closed-core below the open valence; drop
        end
        push!(out, t)
    end
    return out
end

# ---------------------------------------------------------------------
# Sugar resolver: (element, charge) -> ground configuration
# ---------------------------------------------------------------------

# Core-hole signatures recognised when decomposing a configuration.
# Used only to *exclude* core-hole rows from the sugar lookup; the
# sugar form (`charge=`) returns ground configurations only.
#   (2, 'p', 5) → 2p5 hole (L_{2,3} edge).
#   (3, 'd', 9) → 3d9 hole (M_{4,5} edge).
const _CORE_HOLE_SIGNATURES = ((2, 'p', 5), (3, 'd', 9))

"""
    _config_components(config) -> (core_hole_or_nothing, valence_n, valence_letter, valence_count, valence_group)

Decompose a normalised config into core-hole (if any) and valence shells.
Returns:

  * `core_hole`: `nothing` if no recognised core-hole shell is present,
    otherwise an `(n, letter)` tuple — `(2, 'p')` for an L_{2,3} hole,
    `(3, 'd')` for an M_{4,5} hole.
  * `valence_group`: the highest-`n`, highest-ℓ shell that is *not* the
    core-hole shell — used as the "valence" reference for charge math.
  * `valence_count`: electron count in that valence shell.

Throws `ArgumentError` on an empty / unrecognised config.
"""
function _config_components(config::AbstractString)
    tokens = split(config)
    isempty(tokens) && throw(ArgumentError("empty configuration"))
    # Parse every shell token.
    shells = Tuple{Int, Char, Int}[]
    for tok in tokens
        m = match(r"^([1-9])([spdfg])([0-9]+)$", tok)
        m === nothing && throw(ArgumentError("unrecognised shell: $tok"))
        push!(shells, (parse(Int, m.captures[1]), m.captures[2][1],
                       parse(Int, m.captures[3])))
    end
    # Sort by (n, ℓ) ascending. Highest-(n, ℓ) shell is the valence;
    # any other shell that matches a registered core-hole signature
    # (e.g. "2p5", "3d9") is the core hole.
    sorted = sort(shells; by = t -> (t[1], _SHELL_LETTER_ORDER[t[2]]))
    valence_shell = sorted[end]
    core_hole = nothing
    for s in sorted[1:end - 1]
        for expected in _CORE_HOLE_SIGNATURES
            if s == expected
                core_hole = (s[1], s[2])
                break
            end
        end
        core_hole !== nothing && break
    end
    return core_hole, valence_shell
end

# Group-number lookup (= number of (n-1)d + ns + nf valence electrons in
# the neutral atom, using the convention that the highest-(n,ℓ) shell
# absorbs all valence electrons). Charge is computed as
# `group_number(element) - electrons_in_valence_shell` — valid for the
# rows in the static dict (and the Cowan templates).
const _GROUP_NUMBER = Dict{Symbol, Int}(
    :K  => 1, :Ca => 2,
    :Sc => 3, :Ti => 4, :V => 5, :Cr => 6, :Mn => 7,
    :Fe => 8, :Co => 9, :Ni => 10, :Cu => 11, :Zn => 12,
    :Rb => 1, :Sr => 2,
    :Y => 3, :Zr => 4, :Nb => 5, :Mo => 6, :Tc => 7,
    :Ru => 8, :Rh => 9, :Pd => 10, :Ag => 11, :Cd => 12,
)

# Build the (element, charge) → ground-configuration table from the
# static dict at module load. Only ground rows (no core hole) are
# included; core-hole channels live in `HAVERKORT_PARAMETERS` and are
# reachable solely via the primary configuration-string form.
#
# Rule: ground row `nd^N` for element with group number G has formal
# charge `G - N`.
function _build_ground_charge_lookup()
    table = Dict{Tuple{Symbol, Int}, String}()
    for (key, _) in HAVERKORT_PARAMETERS
        elem, config = key
        haskey(_GROUP_NUMBER, elem) || continue
        core_hole, (_, _, ve) = _config_components(config)
        core_hole === nothing || continue   # skip core-hole rows
        charge = _GROUP_NUMBER[elem] - ve
        sub_key = (elem, charge)
        if haskey(table, sub_key)
            # Two distinct ground configurations for the same
            # (element, charge) would be a real ambiguity — shouldn't
            # happen for the current Haverkort coverage. Warn and
            # keep the first.
            @warn "AtomicParameters: duplicate ground (element, charge) " *
                  "$(sub_key) maps to multiple configs " *
                  "($(table[sub_key]), $config); keeping first."
        else
            table[sub_key] = config
        end
    end
    return table
end

const _GROUND_CONFIG_LOOKUP = _build_ground_charge_lookup()

"""
    _default_ground_config(element, charge) -> Union{String, Nothing}

Resolve `(element, charge)` to its tabulated ground configuration
string, or `nothing` if no ground row is in the static dict. The
sugar form (`charge=` kwarg) is restricted to ground states by
design — core-hole and other configurations require the primary
configuration-string form.
"""
function _default_ground_config(element::Symbol, charge::Int)
    return get(_GROUND_CONFIG_LOOKUP, (element, charge), nothing)
end

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

"""
    atomic_parameters(element::Symbol, configuration::AbstractString;
                      scaling=:HF, cowan=nothing) -> NamedTuple
    atomic_parameters(element::Symbol; charge::Int,
                      scaling=:HF, cowan=nothing) -> NamedTuple

Look up atomic Slater-Condon parameters and spin-orbit constants for the
requested electron configuration. See the `MOADyna.AtomicParameters` module
docstring for the full return shape, naming rule, and lookup chain.

# Lookup forms

1. **PRIMARY** (configuration string, unambiguous):
   ```julia
   atomic_parameters(:Ni, "3d8")           # Ni²⁺ ground
   atomic_parameters(:Ni, "2p5 3d9")       # L_{2,3} core-hole intermediate
   atomic_parameters(:Pr, "3d9 4f3")       # M_{4,5} core-hole intermediate
   ```
2. **SUGAR** (charge-keyed, GROUND STATE ONLY):
   ```julia
   atomic_parameters(:Ni; charge=2)        # → resolves to "3d8"
   ```
   For core-hole or any other configuration, use the primary form.

# Arguments

- `element` : atomic species, e.g. `:Ni`, `:Pr`.
- `configuration` (positional, primary form) : electron configuration
  string. Whitespace, case, and `^` / `_` separators are normalised
  internally — `"3d^8"`, `"3D8"`, `"3d_8"` all resolve to `"3d8"`.
- `charge` (kwarg, sugar form) : ionic charge; resolves to the
  tabulated ground configuration for `(element, charge)`.

# Common kwargs

- `scaling::Symbol = :HF` — `:HF` for raw Hartree-Fock; `:scaled_80`
  multiplies F^k(k>0) and G^k by 0.8.
- `cowan` — explicit path to a Cowan RCN binary. If unset, falls back
  to `ENV["MOADYNA_COWAN"]` then `ENV["TTMULT"]`.
- `keep_scratch::Bool = false` — forensic-debug flag, only meaningful
  on the live-Cowan path. When `true`, the scratch directory holding
  the in36 deck and out36 output is preserved and its absolute path is
  reported in the returned NamedTuple as `scratch_path`. When `false`
  (default), the scratch tree is removed in a `finally` block on
  every exit path. Has no effect on the static-dict path.

# Throws

  * `ArgumentError` on a malformed configuration string.
  * `ArgumentError` on a sugar-form lookup that doesn't resolve to a
    ground row in the static dict — message lists the available
    ground configurations for that element and points at the primary
    form for core-hole / other configurations.
  * `ArgumentError` on an unsupported scaling.
  * `ArgumentError` if no entry is found and no Cowan binary is
    available.
"""
function atomic_parameters(element::Symbol, configuration::AbstractString;
                           scaling::Symbol = :HF,
                           cowan = nothing,
                           keep_scratch::Bool = false)
    norm_config = _normalize_config(configuration)
    key = (element, norm_config)

    # Step 1: static dict
    if haskey(HAVERKORT_PARAMETERS, key)
        params = HAVERKORT_PARAMETERS[key]
        scaled = _apply_scaling(params, scaling)
        return merge(scaled, (scaling = scaling,))
    end

    # Step 2: live Cowan
    cowan_path = _resolve_cowan_path(cowan)
    if cowan_path !== nothing
        return _run_cowan_by_config(element, norm_config;
                                    binary = cowan_path,
                                    scaling = scaling,
                                    keep_scratch = keep_scratch)
    end

    # Step 3: error with helpful hint
    available = sort([k[2] for k in keys(HAVERKORT_PARAMETERS) if k[1] === element])
    throw(ArgumentError("""
        atomic_parameters: no parameters available for (element=$element, \
        configuration=\"$norm_config\").

        Static coverage for $element: $(isempty(available) ? "(none)" : join(available, ", ")).

        Static data is from the Haverkort thesis (3d / 4d block, ground + \
        L_{2,3} edge). To extend coverage, install a Cowan RCN binary \
        (recommended: TCD archive at \
        https://www.tcd.ie/Physics/people/Cormac.McGuinness/Cowan or \
        cjtitus/ttmult at https://bitbucket.org/cjtitus/ttmult), then \
        either set ENV[\"MOADYNA_COWAN\"] (or ENV[\"TTMULT\"]) to its path, \
        or pass `cowan = \"...\"` as a kwarg.
        """))
end

"""
    radial_wavefunction(element::Symbol, configuration::AbstractString;
                        cowan = nothing) -> (r, P)

Hartree–Fock radial wavefunctions from a live Cowan RCN run, for the
requested `configuration` (e.g. `"3d8"`). Returns a named tuple `(r, P)`:

- `r::Vector{Float64}` — the radial mesh in Bohr (RCN's logarithmic grid).
- `P::Dict{String, Vector{Float64}}` — each bound orbital's reduced radial
  function ``P_{n\\ell}(r) = r\\,R_{n\\ell}(r)`` (normalised ``\\int P^2\\,dr = 1``),
  keyed by label (`"1s"`, …, `"3d"`).

This is **opt-in**: it is the only entry point that reads the radial function
(parsed from RCN's binary `tape2n`), and [`atomic_parameters`](@ref) never
computes it. Typical use — build nIXS Bessel moments with
[`radial_integral`](@ref MOADyna.Shells.radial_integral):

```julia
rw = radial_wavefunction(:Ni, "3d8")          # needs ENV["MOADYNA_COWAN"] or cowan=...
Rj = Dict(k => radial_integral(rw.P["3d"], rw.P["3d"], rw.r, k;
                               kind = :bessel, q = 4.5, weight = :reduced)
          for k in (0, 2, 4))
```

Requires a Cowan RCN binary: set `ENV["MOADYNA_COWAN"]` (or `ENV["TTMULT"]`) to
its path, or pass `cowan = "..."`.
"""
function radial_wavefunction(element::Symbol, configuration::AbstractString;
                             cowan = nothing)
    cowan_path = _resolve_cowan_path(cowan)
    cowan_path === nothing && throw(ArgumentError(
        "radial_wavefunction needs a Cowan RCN binary: set ENV[\"MOADYNA_COWAN\"] " *
        "(or ENV[\"TTMULT\"]) to its path, or pass cowan = \"...\"."))
    norm_config = _normalize_config(configuration)
    res = _run_cowan_by_config(element, norm_config;
                               binary = cowan_path, scaling = :HF, radial = true)
    return res.radial
end

function atomic_parameters(element::Symbol;
                           charge::Int,
                           scaling::Symbol = :HF,
                           cowan = nothing,
                           keep_scratch::Bool = false)
    config_str = _default_ground_config(element, charge)
    if config_str !== nothing
        return atomic_parameters(element, config_str;
                                 scaling = scaling, cowan = cowan,
                                 keep_scratch = keep_scratch)
    end

    # No static-dict ground entry for (element, charge). Try the
    # Cowan path with a templated ground configuration if a binary
    # is available.
    cowan_path = _resolve_cowan_path(cowan)
    if cowan_path !== nothing
        templated = _cowan_template_ground_config(element, charge)
        if templated !== nothing
            return atomic_parameters(element, templated;
                                     scaling = scaling, cowan = cowan,
                                     keep_scratch = keep_scratch)
        end
    end

    avail_grounds = sort([v for ((e, _), v) in _GROUND_CONFIG_LOOKUP if e === element])
    hint = if isempty(avail_grounds)
        "no ground entries tabulated for $element."
    else
        "available ground configurations for $element: " *
        join(avail_grounds, ", ") * "."
    end
    throw(ArgumentError("""
        atomic_parameters: no ground parameters available for \
        (element=$element, charge=$charge).

        The sugar form (`charge=`) returns ground configurations only. \
        For core-hole or other configurations (e.g. \"2p5 3d9\", \
        \"3d9 4f3\"), use the primary form: \
        `atomic_parameters(:$element, \"<config>\")`.

        $hint

        Alternatively, provide a Cowan binary \
        (`ENV[\"MOADYNA_COWAN\"]` / `ENV[\"TTMULT\"]` / `cowan = \"...\"`).
        """))
end

"""
    covered_elements() -> Set{Symbol}

Return the set of elements (`Symbol`) with at least one entry in the
static Haverkort parameter dictionary.

!!! note "Static dictionary only"
    Coverage reflects the built-in Haverkort lookup table, **not** the
    live Cowan runner. To probe what the Cowan runner can produce for a
    given element, call `atomic_parameters` directly.
"""
function covered_elements()
    return Set(k[1] for k in keys(HAVERKORT_PARAMETERS))
end

"""
    covered_configurations(element::Symbol) -> Vector{String}

Return the canonical configuration strings available in the static
Haverkort dictionary for `element`, sorted lexicographically.  Returns
an empty `Vector{String}` (no error) when `element` has no entry in the
dictionary.

# Arguments
- `element` — element symbol, e.g. `:Ni` or `:Co`.

!!! note "Static dictionary only"
    Coverage reflects the built-in Haverkort lookup table, **not** the
    live Cowan runner. To probe what the Cowan runner can produce for a
    given element, call `atomic_parameters` directly.
"""
function covered_configurations(element::Symbol)
    return sort([k[2] for k in keys(HAVERKORT_PARAMETERS) if k[1] == element])
end

# ---------------------------------------------------------------------
# Cowan adapter — by-configuration entry point
# ---------------------------------------------------------------------

"""
    _run_cowan_by_config(element, normalised_config; binary, scaling)

Invoke the Cowan runner with an explicit configuration string. Wraps
`_run_cowan` (which still accepts the legacy `(charge, edge)` arg pair
for its built-in templates) by passing the config via
`configuration_override` — Cowan ignores the `(charge, edge)` tuple in
that path. Charge supplied to the in36 deck is back-derived from the
configuration so the deck label is sensible (Cowan ignores it
numerically; this is purely cosmetic).
"""
function _run_cowan_by_config(element::Symbol, norm_config::AbstractString;
                              binary::AbstractString,
                              scaling::Symbol,
                              keep_scratch::Bool = false,
                              radial::Bool = false)
    # Translate config → (charge, edge, shell) hints for the runner.
    core_hole, (vn, vl, ve) = _config_components(norm_config)
    valence_for_charge = core_hole === nothing ? ve : (ve - 1)
    group = get(_GROUP_NUMBER, element, 0)
    cosmetic_charge = if group > 0
        group - valence_for_charge
    else
        # No group entry — use a sensible default.
        # (Cowan ignores it numerically; this only affects the ion label.)
        0
    end
    # Pick the edge symbol that drives the runner's "is this a core-hole
    # config?" detection. Anything not :ground forces it to honour
    # `configuration_override`.
    edge_sym = if core_hole === nothing
        :ground
    elseif core_hole == (2, 'p')
        :L23
    elseif core_hole == (3, 'd')
        :M45
    else
        :other
    end
    shell_sym = Symbol("$vn$vl")
    return _run_cowan(element, cosmetic_charge, edge_sym;
                      binary = binary,
                      shell = shell_sym,
                      scaling = scaling,
                      configuration_override = norm_config,
                      keep_scratch = keep_scratch,
                      radial = radial)
end

"""
    _cowan_template_ground_config(element, charge) -> Union{String, Nothing}

Return a templated ground configuration string for the Cowan runner's
known blocks (3d / 4d / 4f / 5d). Returns `nothing` if the runner has
no template for this element — caller should fall back to its own
error path. The sugar form is restricted to ground states; core-hole
templates are unreachable from this entry point.
"""
function _cowan_template_ground_config(element::Symbol, charge::Int)
    Z = get(MOADYNA_AP_ATOMIC_NUMBER(), element, nothing)
    Z === nothing && return nothing
    # 3d block
    if 19 <= Z <= 30
        n_d = Z - 18 - charge
        n_d >= 0 || return nothing
        return _normalize_config("3d$n_d")
    elseif 37 <= Z <= 48
        n_d = Z - 36 - charge
        n_d >= 0 || return nothing
        return _normalize_config("4d$n_d")
    elseif 57 <= Z <= 71
        n_f = Z - 54 - charge
        (0 <= n_f <= 14) || return nothing
        return _normalize_config("4f$n_f")
    elseif 72 <= Z <= 80
        n_d = Z - 68 - charge
        n_d >= 0 || return nothing
        return _normalize_config("5d$n_d")
    end
    return nothing
end

# Indirection so the AtomicParameters module sees the runner's
# ATOMIC_NUMBER without re-importing it.
MOADYNA_AP_ATOMIC_NUMBER() = ATOMIC_NUMBER

# ---------------------------------------------------------------------
# Internal: Cowan binary resolution
# ---------------------------------------------------------------------

"""
    _resolve_cowan_path(cowan_kwarg) -> Union{String, Nothing}

Resolve the Cowan RCN binary path. Order: explicit kwarg, then
`ENV["MOADYNA_COWAN"]`, then `ENV["TTMULT"]`. Returns `nothing` if none
are set.
"""
function _resolve_cowan_path(cowan_kwarg)
    cowan_kwarg !== nothing && return cowan_kwarg
    haskey(ENV, "MOADYNA_COWAN") && return ENV["MOADYNA_COWAN"]
    haskey(ENV, "TTMULT")     && return ENV["TTMULT"]
    return nothing
end

# ---------------------------------------------------------------------
# Internal: scaling
# ---------------------------------------------------------------------

# Field names inside the nested NamedTuple that hold scalable
# Slater-Condon integrals. These are F^k(k>0) and G^k entries on
# d-block (Fdd/Fpd/Gpd) and f-block (Fff/Fdf/Gdf) returns.
const _SCALABLE_GROUPS = (:Fdd, :Fpd, :Gpd, :Fff, :Fdf, :Gdf)

function _apply_scaling(params::NamedTuple, scaling::Symbol)
    if scaling === :HF
        return params
    elseif scaling === :scaled_80
        return _scale_named(params, 0.8)
    else
        throw(ArgumentError("scaling = $scaling not supported. " *
            "Use :HF (no scaling) or :scaled_80 (80% reduction on F^k(k>0) + G^k)."))
    end
end

"""
    _scale_subgroup(nt, factor) -> NamedTuple

Multiply every numeric entry of `nt` by `factor`, preserving NaN.
Used on the Fdd / Fpd / Gpd / Fff / Fdf / Gdf nested NamedTuples.
"""
function _scale_subgroup(nt::NamedTuple, factor::Real)
    names = keys(nt)
    vals = map(names) do n
        v = getfield(nt, n)
        v isa Number ? v * factor : v
    end
    return NamedTuple{names}(vals)
end

function _scale_named(params::NamedTuple, factor::Real)
    names = keys(params)
    vals = map(names) do n
        v = getfield(params, n)
        if n in _SCALABLE_GROUPS && v isa NamedTuple
            return _scale_subgroup(v, factor)
        else
            return v
        end
    end
    return NamedTuple{names}(vals)
end

end # module
