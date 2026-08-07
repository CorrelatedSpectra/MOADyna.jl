# src/atomic_parameters/cowan_runner.jl
#
# Live Cowan RCN runner. Generates an in36 deck for the requested
# (element, charge, edge, shell) configuration, runs RCN in a temporary
# scratch directory, parses the resulting out36 for F^k, G^k, ζ, and
# ⟨r^k⟩, and shapes the result into the same nested NamedTuple that the
# static-dict path emits.
#
# Parser approach (more robust than mretegan/atomic-parameters'
# parameters.py:286-306 compact-summary tokenizer):
#
#   * Slater integrals are read from the explicit
#     `( shellA, shellB)  k  <value> ryd  =  <value> cm-1  frac` table
#     in the "slater integrals" block. Pair (shellA, shellB) and the F^k
#     vs G^k column (left vs right of the line) disambiguate every entry,
#     so we never depend on the variable-width compact summary line which
#     mretegan parses with `tokens[4::2]` (brittle for multi-word labels
#     like "3d9 4f3").
#   * Spin-orbit ζ values come from the "blume-watson" column of the
#     `--zeta--` table.
#   * Radial moments ⟨r²⟩, ⟨r⁴⟩ come from the `(r+2)`, `(r+4)` columns of
#     the orbital table at the head of each block.
#
# The runner writes a single-configuration deck per call, so out36
# contains exactly one block (no ambiguity about which block to harvest).

# ---------------------------------------------------------------------
# Physical conversion factors
# ---------------------------------------------------------------------

const _RYDBERG_TO_EV = 13.605693122994  # CODATA 2018, eV/Ry
const _BOHR_TO_ANG   = 0.529177210903   # CODATA 2018, Å/a₀
const _BOHR2_TO_ANG2 = _BOHR_TO_ANG^2   # ≈ 0.28002852
const _BOHR4_TO_ANG4 = _BOHR_TO_ANG^4   # ≈ 0.07841596

# ---------------------------------------------------------------------
# Atomic-number lookup (kept small — covers the static dict's elements
# plus the lanthanides we care about for live runs).
# ---------------------------------------------------------------------

const ATOMIC_NUMBER = Dict{Symbol, Int}(
    :H => 1,  :He => 2,  :Li => 3,  :Be => 4,  :B => 5,  :C => 6,
    :N => 7,  :O => 8,   :F => 9,   :Ne => 10,
    :Na => 11, :Mg => 12, :Al => 13, :Si => 14, :P => 15, :S => 16,
    :Cl => 17, :Ar => 18, :K => 19, :Ca => 20,
    # 3d block
    :Sc => 21, :Ti => 22, :V => 23, :Cr => 24, :Mn => 25,
    :Fe => 26, :Co => 27, :Ni => 28, :Cu => 29, :Zn => 30,
    # 4d block (incl. Rb, Sr)
    :Rb => 37, :Sr => 38, :Y => 39, :Zr => 40, :Nb => 41,
    :Mo => 42, :Tc => 43, :Ru => 44, :Rh => 45, :Pd => 46,
    :Ag => 47, :Cd => 48,
    # Lanthanides (4f block) — relevant for live Cowan runs
    :La => 57, :Ce => 58, :Pr => 59, :Nd => 60, :Pm => 61,
    :Sm => 62, :Eu => 63, :Gd => 64, :Tb => 65, :Dy => 66,
    :Ho => 67, :Er => 68, :Tm => 69, :Yb => 70, :Lu => 71,
    # 5d block (relevant for live Cowan runs of heavy TMs)
    :Hf => 72, :Ta => 73, :W => 74, :Re => 75, :Os => 76,
    :Ir => 77, :Pt => 78, :Au => 79, :Hg => 80,
)

# Roman numerals 1..20 — Cowan labels ions by ion-stage = charge + 1.
const _ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII",
                "XIX", "XX")

# ---------------------------------------------------------------------
# NAMES table — kept for reference / future use. The current parser
# reads the explicit `( shellA, shellB) k ... ` Slater-integral lines so
# it never has to map RCN's compact-summary positional output to
# physical names. Retained so a downstream tool that wants to consume
# the compact summary (faster but brittle) has the column ordering.
#
# Each value is a tuple of column names as they appear in RCN's
# compact-summary line for the corresponding configuration class.
# ---------------------------------------------------------------------

const NAMES = Dict{Symbol, Tuple{Vararg{String}}}(
    # f-valence with one core particle (port from mretegan parameters.py:158-188)
    :s_with_one_particle_and_f => (
        "F2(nf,nf)", "F4(nf,nf)", "F6(nf,nf)",
        "zeta(nf)", "G3(ns,nf)",
    ),
    :p_with_one_particle_and_f => (
        "F2(nf,nf)", "F4(nf,nf)", "F6(nf,nf)",
        "zeta(np)", "zeta(nf)",
        "F2(np,nf)", "G2(np,nf)", "G4(np,nf)",
    ),
    :d_with_one_particle_and_f => (
        "F2(nf,nf)", "F4(nf,nf)", "F6(nf,nf)",
        "zeta(nd)", "zeta(nf)",
        "F2(nd,nf)", "F4(nd,nf)",
        "G1(nd,nf)", "G3(nd,nf)", "G5(nd,nf)",
    ),
    # d-valence (transition metals) ----------------------------------
    :p_with_one_particle_and_d => (
        "F2(nd,nd)", "F4(nd,nd)",
        "zeta(np)", "zeta(nd)",
        "F2(np,nd)", "G1(np,nd)", "G3(np,nd)",
    ),
    :p_with_multiple_particles_and_d => (
        "F2(nd,nd)", "F4(nd,nd)",
        "zeta(np)", "zeta(nd)",
        "F2(np,nd)", "G1(np,nd)", "G3(np,nd)",
    ),
    :s_with_one_particle_and_d => (
        "F2(nd,nd)", "F4(nd,nd)",
        "zeta(nd)", "G2(ns,nd)",
    ),
)

# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

"""
    _run_cowan(element, charge, edge; binary, shell, scaling,
               configuration_override=nothing, keep_scratch=false,
               ion_label=nothing)

Run a Cowan RCN calculation for the requested `(element, charge, edge,
shell)` configuration and return a nested NamedTuple of Slater-Condon
parameters with the same shape the static-dict path produces.

If `configuration_override` is set, it bypasses the built-in
configuration template and uses the literal Cowan-syntax string (e.g.
`"3d9 4f3"`).

If `keep_scratch=true`, the scratch directory is preserved and the
absolute path is reported in the returned NamedTuple as
`scratch_path`. Otherwise the scratch tree is removed in a `finally`
block — both on successful return and on exception. (Switched from
`mktempdir(; cleanup=...)` because that variant only fires at Julia
exit; we want the scratch reaped per call.)

# Ion label

The Roman-numeral label written into the in36 deck (e.g.
`"Pr III"`) is **purely cosmetic** — Cowan ignores it numerically.
By default we use the spectroscopy convention (`ion_stage = charge + 1`,
so neutral = `I`, monocation = `II`, …; e.g. Pr⁰ = `Pr I`, Pr³⁺ = `Pr IV`).
Pass `ion_label="Pr III"` to override the label only; the numerics will
be identical.
"""
function _run_cowan(element::Symbol, charge::Int, edge::Symbol;
                    binary::AbstractString,
                    shell::Symbol = Symbol("3d"),
                    scaling::Symbol = :HF,
                    configuration_override::Union{AbstractString, Nothing} = nothing,
                    keep_scratch::Bool = false,
                    radial::Bool = false,
                    ion_label::Union{AbstractString, Nothing} = nothing)
    # 1. Validate binary path.
    isfile(binary) || throw(ArgumentError(
        "Cowan binary not found at $(binary). " *
        "Set ENV[\"MOADYNA_COWAN\"] / ENV[\"TTMULT\"] or pass `cowan = ...`."))

    haskey(ATOMIC_NUMBER, element) || throw(ArgumentError(
        "Cowan runner: no atomic number registered for element=$element. " *
        "Extend ATOMIC_NUMBER in src/atomic_parameters/cowan_runner.jl."))

    # 2. Build the in36 deck (single configuration per call).
    config_str = configuration_override === nothing ?
        _config_string(element, charge, edge, shell) :
        String(configuration_override)
    deck = _build_in36(element, charge, config_str; ion_label = ion_label)

    # 3. Run Cowan inside a fresh scratch directory. Use try/finally
    #    so the scratch is reaped on every exit path (the prior
    #    `mktempdir(cleanup=true)` only fires at Julia exit).
    scratch = mktempdir()
    try
        in36_path  = joinpath(scratch, "in36")
        out36_path = joinpath(scratch, "out36")
        write(in36_path, deck)

        cd(scratch) do
            run(pipeline(`$binary`; stdout = devnull, stderr = devnull); wait = true)
        end

        isfile(out36_path) || throw(ErrorException(
            "Cowan run did not produce out36 in $(scratch). Deck was:\n$(deck)"))

        out36_text = read(out36_path, String)

        # 4. Parse the (single) configuration block.
        block = _parse_block(out36_text)

        # 5. Shape into the nested NamedTuple expected by atomic_parameters.
        params = _shape_to_nested(block, element, charge, edge, shell, config_str)

        # 6. Apply scaling and add provenance.
        scaled = _apply_scaling(params, scaling)
        banner = _banner_tag(binary)
        result = merge(scaled,
                       (provenance = Symbol(:cowan_, banner),
                        scaling = scaling))

        # Optional radial-wavefunction extraction (parsed while the scratch dir
        # — and its binary `tape2n` — still exist). Off by default: the radial
        # function is rarely needed and is the only reason to read `tape2n`.
        if radial
            radial_data = _parse_cowan_radial(out36_text, joinpath(scratch, "tape2n"))
            result = merge(result, (radial = radial_data,))
        end

        if keep_scratch
            return merge(result, (scratch_path = scratch,))
        else
            return result
        end
    finally
        if !keep_scratch
            rm(scratch; force = true, recursive = true)
        end
    end
end

# ---------------------------------------------------------------------
# Optional radial-wavefunction extraction (Cowan tape2n)
# ---------------------------------------------------------------------
#
# RCN writes the bound radial functions P_nl(r) = r·R_nl(r) to the binary
# file `tape2n` (a single Fortran *unformatted* sequential record, normally
# consumed by RCN2/RCG). The formatted `out36` carries only the ⟨r^k⟩
# moments and the mesh metadata — not the tabulated P(r). We read the
# radial functions from `tape2n` directly.
#
# The record is `write(2) <header scalars>, r, ru, ((pnl(i,m)...)), iw6`,
# where `r`, `ru` and each `pnl(:,m)` are length-`kmsh` REAL*8 arrays (kmsh
# is a build-time parameter ≥ the physical `mesh`; entries past `mesh` are
# zero). The three arrays sit at the *tail* of the record, so we anchor the
# parse from the end and never have to decode the variable header. `kmsh`
# is recovered by requiring `r[mesh]` to equal the `r(mesh)` value printed
# in out36. Little-endian REAL*8 (the platform RCN runs on).

"""
    _parse_out36_radial_meta(out36) -> (mesh, ncsp, orbitals, r_mesh, irel)

From RCN out36 text, parse the radial mesh size `mesh`, the number of
bound orbitals `ncsp = ncores + nvales`, their ordered labels (e.g.
`["1s","2s","2p","3s","3p","3d"]`), the largest mesh radius `r(mesh)`
(Bohr), and the relativistic flag `irel` — everything needed to anchor
and validate the `tape2n` parse.
"""
function _parse_out36_radial_meta(out36::AbstractString)
    # Fortran float grammar (tolerates D/E exponents): D/d → E before parse.
    _ffloat(s) = parse(Float64, replace(replace(s, 'D' => 'E'), 'd' => 'e'))

    m_mesh = match(r"mesh=\s*(\d+)", out36)
    m_mesh === nothing && throw(ErrorException("Cowan radial: 'mesh=' not found in out36"))
    mesh = parse(Int, m_mesh.captures[1])

    m_rm = match(r"r\(mesh\)=\s*([-+0-9.EeDd]+)", out36)
    m_rm === nothing && throw(ErrorException("Cowan radial: 'r(mesh)=' not found in out36"))
    r_mesh = _ffloat(m_rm.captures[1])

    m_nc = match(r"ncores=\s*(\d+)", out36)
    m_nv = match(r"nvales=\s*(\d+)", out36)
    (m_nc === nothing || m_nv === nothing) &&
        throw(ErrorException("Cowan radial: 'ncores='/'nvales=' not found in out36"))
    ncsp = parse(Int, m_nc.captures[1]) + parse(Int, m_nv.captures[1])

    # Relativistic flag. The tail-anchored parser is validated only for the
    # non-relativistic record (irel<3: a single large-component `pnl`); the
    # Dirac path (irel≥3) also carries `qnl` and is not supported here.
    m_ir = match(r"irel=\s*(\d+)", out36)
    irel = m_ir === nothing ? 1 : parse(Int, m_ir.captures[1])

    # Ordered orbital labels from the "nl wnl ee az …" table: rows like
    # "  3d    9.    -2.98036 …". Take the first `ncsp` (the table repeats).
    orbitals = String[]
    for line in split(out36, '\n')
        mo = match(r"^\s{2,}([1-9][spdfgh])\s+[\d.]+\s+-?\d", line)
        mo === nothing && continue
        push!(orbitals, mo.captures[1])
        length(orbitals) == ncsp && break
    end
    length(orbitals) == ncsp ||
        throw(ErrorException("Cowan radial: found $(length(orbitals)) orbital labels, expected $ncsp"))
    return (mesh, ncsp, orbitals, r_mesh, irel)
end

"""
    _read_tape2n(path, ncsp, mesh, r_mesh) -> (r, pnl)

Tail-anchored parse of an RCN `tape2n` record. Returns the radial mesh
`r` (length `mesh`, Bohr) and `pnl`, a `Vector` of `ncsp` radial functions
`P_nl(r) = r·R_nl(r)` (each length `mesh`), in the orbital order RCN
writes them (`ns..ncspvs`, i.e. `1..ncsp` for the RCN mod-36 deck, valence
last). `kmsh` is auto-detected by requiring the recovered `r[1:mesh]` to be
a finite, strictly increasing mesh ending at `r_mesh`. Assumes little-endian
REAL*8 and 4-byte Fortran record-length markers (gfortran default).
"""
function _read_tape2n(path::AbstractString, ncsp::Int, mesh::Int, r_mesh::Float64)
    (ENDIAN_BOM == 0x04030201) || throw(ErrorException(
        "Cowan radial: tape2n parsing is implemented for little-endian hosts only " *
        "(RCN's REAL*8 records); this host is big-endian."))
    raw = read(path)
    length(raw) ≥ 8 || throw(ErrorException("Cowan radial: tape2n too small ($(length(raw)) bytes)"))
    # Strip the 4-byte leading/trailing Fortran record-length markers.
    lead  = reinterpret(Int32, raw[1:4])[1]
    trail = reinterpret(Int32, raw[end-3:end])[1]
    (lead == trail && Int(lead) + 8 == length(raw)) ||
        throw(ErrorException("Cowan radial: tape2n is not a single unformatted record with " *
                             "4-byte length markers (markers $(lead)/$(trail), file " *
                             "$(length(raw)) bytes); 8-byte markers / multi-record streams " *
                             "are not supported."))
    rec = @view raw[5:end-4]
    n = length(rec); d = 8
    getf64(off, len) = reinterpret(Float64, rec[off+1:off+len*d])   # off is 0-based bytes

    # A candidate `kmsh` is accepted only if the recovered length-`kmsh`
    # r-array is a genuine Cowan mesh: r[1:mesh] finite, non-negative,
    # strictly increasing and ending at r_mesh, AND the padding r[mesh+1:kmsh]
    # is (numerically) zero — Cowan zero-pads past the physical mesh. The
    # zero-padding test is what rejects a false match on the adjacent `ru`
    # (potential) array, which can itself look monotonic.
    function valid_kmsh(rfull, cand)
        head = @view rfull[1:mesh]
        all(isfinite, head) || return false
        head[1] ≥ 0 || return false
        all(>(0), diff(head)) || return false
        isapprox(head[end], r_mesh; rtol = 1e-4) || return false
        if cand > mesh
            pad = @view rfull[mesh+1:cand]
            all(x -> abs(x) ≤ 1e-8 * max(r_mesh, 1.0), pad) || return false
        end
        return true
    end

    # Layout from the record tail: … [r:kmsh][ru:kmsh][pnl:kmsh*ncsp][iw6:int4].
    kmsh = 0
    cand = mesh
    while (2 + ncsp) * cand * d + 4 ≤ n
        r_off = n - ((2 + ncsp) * cand * d + 4)
        rfull = collect(getf64(r_off, cand))
        if valid_kmsh(rfull, cand)
            kmsh = cand; break
        end
        cand += 1
    end
    kmsh == 0 && throw(ErrorException(
        "Cowan radial: could not locate a valid radial mesh in tape2n " *
        "(ncsp=$ncsp, mesh=$mesh, r(mesh)=$r_mesh) — binary format mismatch?"))

    r_off   = n - ((2 + ncsp) * kmsh * d + 4)
    pnl_off = r_off + 2 * kmsh * d
    r   = collect(getf64(r_off, kmsh))[1:mesh]
    pnl = [collect(getf64(pnl_off + (m - 1) * kmsh * d, kmsh))[1:mesh] for m in 1:ncsp]
    # Sanity: every written orbital is finite and not identically zero
    # (catches an orbital-count / layout mismapping).
    for (m, P) in enumerate(pnl)
        (all(isfinite, P) && any(!iszero, P)) || throw(ErrorException(
            "Cowan radial: orbital $m parsed as non-finite or all-zero — " *
            "tape2n layout mismatch (ncsp=$ncsp, kmsh=$kmsh)?"))
    end
    return r, pnl
end

"""
    _parse_cowan_radial(out36, tape2n_path) -> (r, P)

Combine the out36 metadata and the `tape2n` binary into the public radial
return: `r` (Bohr) and `P`, a `Dict` mapping each orbital label (`"3d"`, …)
to its `P_nl(r) = r·R_nl(r)` on the mesh `r` (normalised `∫P² dr = 1`).
"""
function _parse_cowan_radial(out36::AbstractString, tape2n_path::AbstractString)
    isfile(tape2n_path) || throw(ErrorException(
        "Cowan radial: tape2n not found at $(tape2n_path) (did RCN write it?)"))
    mesh, ncsp, orbitals, r_mesh, irel = _parse_out36_radial_meta(out36)
    irel < 3 || throw(ErrorException(
        "Cowan radial: out36 reports irel=$irel (fully relativistic); the tape2n " *
        "radial parser supports only non-relativistic records (irel<3, single " *
        "large-component pnl)."))
    r, pnl = _read_tape2n(tape2n_path, ncsp, mesh, r_mesh)
    P = Dict{String, Vector{Float64}}(orbitals[m] => pnl[m] for m in 1:ncsp)
    return (r = r, P = P)
end

# ---------------------------------------------------------------------
# Deck generation
# ---------------------------------------------------------------------

# Cowan's IONST = ion stage = charge + 1 (neutral atom is "I", singly
# ionised "II", etc.). Configurations follow Cowan's tokenized syntax
# (whitespace-separated `<n><l><electrons>` groups, e.g. "2p5 3d9").

"""
    _config_string(element, charge, edge, shell) -> String

Built-in configuration templates. Supports:

  * `(:ground, :3d|:4d)`   → `"<shell>^N"` with `N = Z - <core_size> - charge`.
  * `(:L23, :3d|:4d)`      → `"2p5 <shell>^{N+1}"`.
  * `(:ground, :4f)`       → `"4f^{N}"` for trivalent lanthanides
                              (`N = Z - 54 - charge`, requires `charge >= 2`).
  * `(:M45, :4f)`          → `"3d9 4f^{N+1}"`.

For anything outside this set, pass `configuration_override` explicitly.
"""
function _config_string(element::Symbol, charge::Int, edge::Symbol, shell::Symbol)
    Z = ATOMIC_NUMBER[element]
    if shell === Symbol("3d") || shell === Symbol("4d")
        n_d = _valence_count_d(Z, charge)
        n_d >= 0 || throw(ArgumentError(
            "Cowan runner: derived n_d=$n_d < 0 for element=$element, charge=$charge."))
        if edge === :ground
            return "$(string(shell))$(n_d)"
        elseif edge === :L23
            return "2p5 $(string(shell))$(n_d + 1)"
        else
            throw(ArgumentError(
                "Cowan runner: edge=$edge not templated for shell=$shell. " *
                "Pass `configuration_override` or extend _config_string."))
        end
    elseif shell === Symbol("4f")
        # Trivalent lanthanide ground: 4f^(Z-57). For other charge states,
        # n_f = Z - 54 - charge (assumes 6s/5d removed first, valid for
        # most common Ln^{2+}/^{3+}/^{4+} cases).
        n_f = Z - 54 - charge
        (0 <= n_f <= 14) || throw(ArgumentError(
            "Cowan runner: derived n_f=$n_f outside [0,14] for element=$element, " *
            "charge=$charge. Pass `configuration_override`."))
        if edge === :ground
            return "4f$(n_f)"
        elseif edge === :M45
            return "3d9 4f$(n_f + 1)"
        elseif edge === :L23
            throw(ArgumentError(
                "Cowan runner: edge=:L23 (2p edge) for 4f-shell lanthanides " *
                "is not standard XAS. Use edge=:M45 (3d edge) or pass " *
                "`configuration_override`."))
        else
            throw(ArgumentError(
                "Cowan runner: edge=$edge not templated for shell=4f. " *
                "Pass `configuration_override` or extend _config_string."))
        end
    else
        throw(ArgumentError(
            "Cowan runner: shell=$shell not templated. " *
            "Pass `configuration_override` or extend _config_string."))
    end
end

"""
    _valence_count_d(Z, charge) -> Int

Number of d-electrons in the simplest filling rule used by Cowan
multiplet calculations: ignore the (n+1)s electrons and put everything
above the noble-gas core into nd. K..Zn → nd = Z - 18 - charge;
Rb..Cd → nd = Z - 36 - charge. Returns negative if charge exceeds
valence; caller validates.
"""
function _valence_count_d(Z::Int, charge::Int)
    if 19 <= Z <= 30           # K..Zn (3d block)
        return Z - 18 - charge
    elseif 37 <= Z <= 48       # Rb..Cd (4d block)
        return Z - 36 - charge
    elseif 72 <= Z <= 80       # Hf..Hg (5d block)
        return Z - 68 - charge   # Hf 5d²: 72-68-0=4? No, Hf is 5d²6s², so cation Hf⁴⁺=5d⁰. Use Z - 68 - charge → Hf⁴⁺: 72-68-4=0 ✓.
    else
        throw(ArgumentError(
            "Cowan runner: no d-block valence-count rule for Z=$Z. " *
            "Pass `configuration_override`."))
    end
end

"""
    _build_in36(element, charge, config_str; ion_label=nothing) -> String

Generate a one-configuration RCN deck. Format (column-oriented):

```
21 -9    2   10  0.2    5.e-08    1.e-11-2   190    1.0 0.65  0.0  0.0   -6
   <Z>   <ION><LABEL padded to 21 chars>< config>
   -1
```

where `Z` and `ION = charge + 1` are right-aligned in 5-character fields,
the label is `<element symbol> <Roman numeral>` left-justified in a
21-character field, and the configuration string follows immediately.
This matches the deck format used by McGuinness's TCD distribution and
mretegan/atomic-parameters' templates.

# Ion label convention (cosmetic only; Cowan ignores it numerically)

The default label uses the **spectroscopy convention**: `ion_stage =
charge + 1`, so neutral atom = "I", singly ionised = "II", and so on
(e.g. Pr⁰ → "Pr I", Pr³⁺ → "Pr IV"). Pass an explicit `ion_label`
string to override the label only — F^k, G^k, ζ, and ⟨r^k⟩ depend
solely on Z and the configuration string, never on this label.
"""
function _build_in36(element::Symbol, charge::Int, config_str::AbstractString;
                     ion_label::Union{AbstractString, Nothing} = nothing)
    Z = ATOMIC_NUMBER[element]
    ion_stage = charge + 1
    1 <= ion_stage <= length(_ROMAN) || throw(ArgumentError(
        "Cowan runner: charge=$charge → ion_stage=$ion_stage outside [1,20]."))
    label = ion_label === nothing ?
        string(element, " ", _ROMAN[ion_stage]) :
        String(ion_label)
    # Fixed-width: %5d %5d %-21s %s   (1+5+5+21 = 32 chars before config)
    label_padded = rpad(label, 21)
    z_field = lpad(string(Z), 5)
    ion_field = lpad(string(ion_stage), 5)
    header = "21 -9    2   10  0.2    5.e-08    1.e-11-2   190    1.0 0.65  0.0  0.0   -6"
    record = string(z_field, ion_field, label_padded, config_str)
    return string(header, "\n", record, "\n   -1\n")
end

# ---------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------

"""
    BlockData

Holds the parsed contents of one Cowan single-configuration block.
"""
struct BlockData
    fk::Dict{Tuple{String,String,Int}, Float64}   # (shellA, shellB, k) → value in Ry
    gk::Dict{Tuple{String,String,Int}, Float64}   # (shellA, shellB, k) → value in Ry
    zeta::Dict{String, Float64}                   # shell label "3d" → value in Ry
    rk::Dict{Tuple{String,Int}, Float64}          # (shell, k) → ⟨r^k⟩ in a₀^k
end

"""
    _parse_block(text) -> BlockData

Parse a Cowan out36 single-configuration text block. Walks the file
once and harvests:

  * `( shellA, shellB)  k  value ryd  =  value cm-1  frac  k  value ryd ...`
    lines from the "slater integrals" section.
  * `nl  wnl  blume-watson(ryd)  blume-watson(cm-1) ...` lines from the
    `--zeta--` table.
  * `nl  wnl  ee  az  (r-3)..(r+6)` lines from the orbital table.
"""
function _parse_block(text::AbstractString)
    fk = Dict{Tuple{String,String,Int}, Float64}()
    gk = Dict{Tuple{String,String,Int}, Float64}()
    zeta = Dict{String, Float64}()
    rk = Dict{Tuple{String,Int}, Float64}()

    lines = split(text, '\n')

    in_orbital_table = false
    in_zeta_table    = false

    for (idx, raw) in enumerate(lines)
        line = String(raw)

        # ----- Orbital table (provides ⟨r^k⟩ in a₀^k) -----
        if occursin(r"^\s*nl\s+wnl\s+ee\s+az\s+\(r-3\)", line)
            in_orbital_table = true
            continue
        end
        if in_orbital_table
            m = match(r"^\s*([0-9][spdfg])\s+", line)
            if m === nothing
                # blank or non-orbital line → end of orbital table
                if !isempty(strip(line))
                    in_orbital_table = false
                end
            else
                shell_label = m.captures[1]
                # Expect 12 columns: nl wnl ee az (r-3) (r-2) (r-1) (r+1) (r+2) (r+3) (r+4) (r+6)
                tokens = split(strip(line))
                if length(tokens) >= 12
                    # tokens[1]=nl, [2]=wnl, [3]=ee, [4]=az,
                    # [5]=(r-3), [6]=(r-2), [7]=(r-1), [8]=(r+1),
                    # [9]=(r+2), [10]=(r+3), [11]=(r+4), [12]=(r+6)
                    rk[(shell_label, 2)] = _parse_float(tokens[9])
                    rk[(shell_label, 4)] = _parse_float(tokens[11])
                end
            end
        end

        # ----- Zeta table -----
        if occursin(r"^\s*nl\s+wnl\s+----blume-watson", line)
            in_zeta_table = true
            continue
        end
        if in_zeta_table
            m = match(r"^\s*([0-9][spdfg])\s+", line)
            if m === nothing
                if !isempty(strip(line)) && !occursin(r"\(ryd\)", line)
                    in_zeta_table = false
                end
            else
                shell_label = m.captures[1]
                tokens = split(strip(line))
                # tokens: nl wnl bw_ryd bw_cm-1 rvi_ryd rvi_cm-1 ...
                if length(tokens) >= 4
                    zeta[shell_label] = _parse_float(tokens[3])
                end
            end
        end

        # ----- Slater integrals lines -----
        # Shape: " ( shellA, shellB)  <k>  <Fk> ryd  =  <Fk> cm-1  <frac>  <k>  <Gk> ryd  =  <Gk> cm-1  <frac>"
        m = match(r"^\s*\(\s*([0-9][spdfg])\s*,\s*([0-9][spdfg])\s*\)\s+([0-9]+)\s+(\S+)\s+ryd\s*=\s*\S+\s+cm-1\s+\S+\s+([0-9]+)\s+(\S+)\s+ryd",
                  line)
        if m !== nothing
            sa, sb = String(m.captures[1]), String(m.captures[2])
            kF = parse(Int, m.captures[3])
            valF = _parse_float(m.captures[4])
            kG = parse(Int, m.captures[5])
            valG = _parse_float(m.captures[6])
            fk[(sa, sb, kF)] = valF
            # G^0 entries are dummies (always 0.0); skip them.
            if !(kG == 0 && valG == 0.0)
                gk[(sa, sb, kG)] = valG
            end
        end
    end

    return BlockData(fk, gk, zeta, rk)
end

# Cowan's printout sometimes runs adjacent fixed-width fields together
# (e.g. "-26.30124-0.07993") so we accept whatever Float64 parser tolerates.
function _parse_float(tok::AbstractString)
    # Strip non-numeric prefix runs that the regex captured cleanly anyway.
    return parse(Float64, tok)
end

# ---------------------------------------------------------------------
# Shape into nested NamedTuple
# ---------------------------------------------------------------------

"""
    _shape_to_nested(block, element, charge, edge, shell, config) -> NamedTuple

Map the parsed `BlockData` into the nested NamedTuple layout used
throughout AtomicParameters. Missing entries (e.g. F^k(2p,nd) on a
ground configuration) are filled with `NaN` to match the static-dict
convention.

The shell letter dictates which pair we read:

  * shell `:3d`/`:4d`/`:5d` → Fdd from `(<shell>,<shell>)` k=2,4;
                              Fpd from `(2p,<shell>)` k=2;
                              Gpd from `(2p,<shell>)` k=1,3;
                              zeta.d from `<shell>`, zeta.p from `2p`.
  * shell `:4f`/`:5f` → Fff from `(<shell>,<shell>)` k=2,4,6;
                        zeta.f from `<shell>`. (Core-hole channels for
                        f-shells live in different fields and are added
                        as we extend coverage.)

Radial moments ⟨r²⟩, ⟨r⁴⟩ are converted Bohr → Å and reported only for
the valence shell.
"""
function _shape_to_nested(block::BlockData, element::Symbol, charge::Int,
                          edge::Symbol, shell::Symbol, config::AbstractString)
    shell_str = string(shell)
    is_d_shell = shell_str in ("3d", "4d", "5d")
    is_f_shell = shell_str in ("4f", "5f")

    # Parse the configuration into a (shell_label → electron_count) map
    # so we can decide which integrals are *required* (open-shell, must
    # be present in the parse) vs *optional* (closed-shell or absent
    # core hole; documented NaN).
    occ = _config_occupancy(config)

    # ----- spin-orbit ζ ------------------------------------------------
    # Required: ζ for the valence shell (always present for partially-
    # occupied valence). Required: ζ for any core-hole shell (which by
    # definition is partially occupied). Optional otherwise.
    zeta_val_d = if is_d_shell
        _shell_open(occ, shell_str) ?
            _zeta_required(block, shell_str, config) :
            _zeta_optional(block, shell_str)
    else
        NaN
    end
    zeta_val_f = if is_f_shell
        _shell_open(occ, shell_str) ?
            _zeta_required(block, shell_str, config) :
            _zeta_optional(block, shell_str)
    else
        NaN
    end
    zeta_val_p = _shell_open(occ, "2p") ?
        _zeta_required(block, "2p", config) :
        _zeta_optional(block, "2p")
    # ζ for 3d as a *core hole* (only meaningful when the valence is
    # f-shell and the deck declares a 3d^9 hole).
    zeta_val_3d_core = _shell_open(occ, "3d") && shell_str != "3d" ?
        _zeta_required(block, "3d", config) :
        _zeta_optional(block, "3d")

    # ----- radial moments ⟨r²⟩, ⟨r⁴⟩ -----------------------------------
    # Required for the valence shell (for any open shell with > 0
    # electrons); optional otherwise.
    r2 = if _shell_open(occ, shell_str)
        _get_required_rk(block, shell_str, 2, config) * _BOHR2_TO_ANG2
    else
        _rk_optional(block, shell_str, 2) * _BOHR2_TO_ANG2
    end
    r4 = if _shell_open(occ, shell_str)
        _get_required_rk(block, shell_str, 4, config) * _BOHR4_TO_ANG4
    else
        _rk_optional(block, shell_str, 4) * _BOHR4_TO_ANG4
    end

    # ----- Slater integrals -------------------------------------------
    # Required-vs-optional rule:
    #   For a shell pair (a, b):
    #     * homo pair (a==b): F^k is required if shell `a` has ≥ 2
    #       electrons (multielectron shell), optional otherwise.
    #     * hetero pair (a != b): F^k / G^k are required if BOTH shells
    #       are present in the configuration (i.e. either open or both
    #       open). When any of the two shells is absent / closed in the
    #       deck, the integral isn't emitted by RCN — report NaN.
    has_2p_hole = _shell_open(occ, "2p")
    has_3d_hole = is_f_shell && _shell_open(occ, "3d")

    if is_d_shell
        # Fdd: required when the d-shell is partially occupied with ≥ 2
        # electrons (otherwise direct integral is undefined).
        nd = get(occ, shell_str, 0)
        F2_dd = _slater_value(block.fk, (shell_str, shell_str, 2), config;
                              required = (nd >= 2))
        F4_dd = _slater_value(block.fk, (shell_str, shell_str, 4), config;
                              required = (nd >= 2))

        # Fpd / Gpd: required iff the deck declares a 2p hole AND the
        # valence shell has at least one electron.
        require_pd = has_2p_hole && nd >= 1
        F2_pd = has_2p_hole ?
            _slater_value(block.fk, ("2p", shell_str, 2), config; required = require_pd) :
            NaN
        G1_pd = has_2p_hole ?
            _slater_value(block.gk, ("2p", shell_str, 1), config; required = require_pd) :
            NaN
        G3_pd = has_2p_hole ?
            _slater_value(block.gk, ("2p", shell_str, 3), config; required = require_pd) :
            NaN

        return (Fdd = (F2 = F2_dd, F4 = F4_dd),
                Fpd = (F2 = F2_pd,),
                Gpd = (G1 = G1_pd, G3 = G3_pd),
                zeta = (d = zeta_val_d,
                        p = has_2p_hole ? zeta_val_p : NaN),
                r = (r2 = r2, r4 = r4),
                unreliable = false,
                configuration = _format_config_pretty(config))
    elseif is_f_shell
        nf = get(occ, shell_str, 0)
        F2_ff = _slater_value(block.fk, (shell_str, shell_str, 2), config;
                              required = (nf >= 2))
        F4_ff = _slater_value(block.fk, (shell_str, shell_str, 4), config;
                              required = (nf >= 2))
        F6_ff = _slater_value(block.fk, (shell_str, shell_str, 6), config;
                              required = (nf >= 2))

        # M45 (3d core hole interacting with f valence): Fdf, Gdf.
        require_df = has_3d_hole && nf >= 1
        F2_df = has_3d_hole ?
            _slater_value(block.fk, ("3d", shell_str, 2), config; required = require_df) :
            NaN
        F4_df = has_3d_hole ?
            _slater_value(block.fk, ("3d", shell_str, 4), config; required = require_df) :
            NaN
        G1_df = has_3d_hole ?
            _slater_value(block.gk, ("3d", shell_str, 1), config; required = require_df) :
            NaN
        G3_df = has_3d_hole ?
            _slater_value(block.gk, ("3d", shell_str, 3), config; required = require_df) :
            NaN
        G5_df = has_3d_hole ?
            _slater_value(block.gk, ("3d", shell_str, 5), config; required = require_df) :
            NaN

        return (Fff = (F2 = F2_ff, F4 = F4_ff, F6 = F6_ff),
                Fdf = (F2 = F2_df, F4 = F4_df),
                Gdf = (G1 = G1_df, G3 = G3_df, G5 = G5_df),
                zeta = (f = zeta_val_f, d = has_3d_hole ? zeta_val_3d_core : NaN),
                r = (r2 = r2, r4 = r4),
                unreliable = false,
                configuration = _format_config_pretty(config))
    else
        throw(ArgumentError(
            "Cowan runner: shell=$shell not supported in _shape_to_nested."))
    end
end

# ----- Required / optional helpers ---------------------------------------

"""
    _config_occupancy(config) -> Dict{String, Int}

Map each shell label in `config` (e.g. "3d", "2p", "4f") to its
electron count. Tokens that don't match `<n><letter><electrons>` are
silently ignored — `config` should already be in canonical form when
it reaches this point.
"""
function _config_occupancy(config::AbstractString)
    occ = Dict{String, Int}()
    for tok in split(config)
        m = match(r"^([0-9])([spdfg])([0-9]+)$", String(tok))
        m === nothing && continue
        label = String(m.captures[1]) * String(m.captures[2])
        occ[label] = parse(Int, m.captures[3])
    end
    return occ
end

# Closed-shell capacities by ℓ (s, p, d, f, g).
const _COWAN_SHELL_CAPACITY = Dict{Char, Int}(
    's' => 2, 'p' => 6, 'd' => 10, 'f' => 14, 'g' => 18,
)

# A shell is "open" iff its electron count satisfies 1 ≤ n < capacity —
# i.e. it has at least one electron AND at least one hole. Closed shells
# (n == 0 or n == capacity) carry no core-hole physics: a fully-occupied
# core listed in the deck (e.g. "2p6") generates no observable splitting,
# and an absent shell has no integrals at all. (A bare `n > 0` test
# would mis-classify closed-core configurations such as "2p6 3d8" as
# L_{2,3} core-hole channels.)
function _shell_open(occ::Dict{String, Int}, label::AbstractString)
    s = String(label)
    n = get(occ, s, 0)
    n == 0 && return false
    # Extract the ℓ letter from the label (last char of "3d", "2p", "4f", ...).
    letter = s[end]
    cap = get(_COWAN_SHELL_CAPACITY, letter, typemax(Int))
    return n < cap
end

"""
    _slater_value(table, key, config; required) -> Float64

Look up the Slater integral `key = (shellA, shellB, k)` in the parsed
table. If `required` is true and the entry is missing, throw a
`KeyError` whose message names the missing integral and the
configuration that produced the parse. If `required` is false and the
entry is missing, return `NaN` (documented-absent slot). Returned
finite values are converted Ry → eV.
"""
function _slater_value(table::Dict{Tuple{String, String, Int}, Float64},
                       key::Tuple{String, String, Int},
                       config::AbstractString;
                       required::Bool)
    if haskey(table, key)
        return _ry_to_ev(table[key])
    end
    required && _missing_required(_slater_label(key), config)
    return NaN
end

# Used by zeta/⟨r^k⟩ helpers; throws if the entry is missing.
function _zeta_required(block::BlockData, shell::AbstractString,
                        config::AbstractString)
    haskey(block.zeta, String(shell)) || _missing_required("zeta($shell)", config)
    return block.zeta[String(shell)] * _RYDBERG_TO_EV
end
function _zeta_optional(block::BlockData, shell::AbstractString)
    haskey(block.zeta, String(shell)) || return NaN
    return block.zeta[String(shell)] * _RYDBERG_TO_EV
end
function _get_required_rk(block::BlockData, shell::AbstractString, k::Int,
                          config::AbstractString)
    sh = String(shell)
    haskey(block.rk, (sh, k)) || _missing_required("<r^$k>($sh)", config)
    return block.rk[(sh, k)]
end
function _rk_optional(block::BlockData, shell::AbstractString, k::Int)
    sh = String(shell)
    return get(block.rk, (sh, k), NaN)
end

function _slater_label(key::Tuple{String, String, Int})
    a, b, k = key
    return a == b ? "F$k($a,$b)" : "F$k($a,$b)/G$k($a,$b)"
end

function _missing_required(name::AbstractString, config::AbstractString)
    throw(KeyError(
        "Cowan runner: required atomic parameter `$name` not found in " *
        "the parsed RCN output for configuration \"$config\". " *
        "This typically signals a parse failure — check the out36 file " *
        "(rerun with `keep_scratch=true` to inspect)."))
end

_ry_to_ev(x::Real) = isnan(x) ? NaN : x * _RYDBERG_TO_EV

# Return the canonical normalised form of a Cowan-deck configuration
# string ("3d8", "2p5 3d9", ...). Matches the static-dict convention
# so the `configuration` field in the returned NamedTuple is consistent
# across the static and live-Cowan paths.
function _format_config_pretty(s::AbstractString)
    parts = split(strip(lowercase(String(s))))
    out = String[]
    for p in parts
        m = match(r"^([0-9])([spdfg])([0-9]+)$", p)
        if m === nothing
            push!(out, p)
        else
            push!(out, string(m.captures[1], m.captures[2], m.captures[3]))
        end
    end
    return join(out, " ")
end

# ---------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------

"""
    _banner_tag(binary) -> Symbol

Derive a short provenance tag from the binary path's basename. We don't
exec the binary just to read a banner — RCN doesn't print one without
a deck — so we use the filename, e.g. `rcnlanl` → `:rcnlanl`.
"""
function _banner_tag(binary::AbstractString)
    name = lowercase(basename(String(binary)))
    # Strip common suffixes.
    name = replace(name, r"\.exe$" => "")
    return Symbol(name)
end
