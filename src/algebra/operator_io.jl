# =====================================================================
# Operator I/O — save / load OperatorSum to HDF5
# =====================================================================
#
# Format (v"1"):
#   /version          attribute, "1"
#   /coefficients     Vector{ComplexF64} of length N_terms
#   /chains/term_<i>  group per term, with:
#     kinds           Vector{String}   (e.g. "c", "cdag")
#     site_names      Vector{String}
#     label_kind      String           ("int1" for FermionSite single-Int tuples)
#     labels          Vector{Int}      (one per ladder entry; valid for "int1")
#
# Current scope: FermionSite only. The schema reserves label_kind values
# "empty" / "intvec" for future Boson / Spin extension; non-fermion sites
# raise an ArgumentError at save time.

const _OPERATOR_IO_VERSION = "1"

"""
    save_operator(file::AbstractString, op::OperatorSum)

Save an `OperatorSum` to an HDF5 file. Supports `FermionSite` only;
raises `ArgumentError` if `op` contains any non-fermion ladder entry.

Round-trips with [`load_operator`](@ref) when the loader is given a
`Hilbert` containing the same site names and types as the saved op.
"""
function save_operator(file::AbstractString, op::OperatorSum)
    for term in op
        for entry in term.chain
            entry.site isa FermionSite || throw(ArgumentError(
                "operator I/O supports FermionSite only at present; " *
                "Boson/Spin support is not yet implemented"))
        end
    end

    HDF5.h5open(file, "w") do f
        HDF5.attributes(f)["version"] = _OPERATOR_IO_VERSION

        n_terms = length(op)
        coeffs = Vector{ComplexF64}(undef, n_terms)
        chains_grp = HDF5.create_group(f, "chains")

        for (i, term) in enumerate(op)
            coeffs[i] = ComplexF64(term.coefficient)
            tg = HDF5.create_group(chains_grp, "term_$(i)")
            n_entries = length(term.chain)
            kinds = Vector{String}(undef, n_entries)
            site_names = Vector{String}(undef, n_entries)
            labels = Vector{Int}(undef, n_entries)
            for (j, entry) in enumerate(term.chain)
                kinds[j] = string(entry.kind)
                site_names[j] = string(name(entry.site))
                labels[j] = Int(entry.label[1])
            end
            tg["kinds"]      = kinds
            tg["site_names"] = site_names
            tg["label_kind"] = "int1"
            tg["labels"]     = labels
        end

        f["coefficients"] = coeffs
    end
    return file
end

"""
    load_operator(file::AbstractString, hilbert::Hilbert) -> OperatorSum

Load an `OperatorSum` written by [`save_operator`](@ref). Site names in
the file are resolved to site objects via `hilbert`.

Raises `ArgumentError` if the file's `/version` attribute is unrecognized
or if a saved chain reports an unsupported `label_kind`. Raises
`KeyError` if a saved site name is not present in `hilbert`.
"""
function load_operator(file::AbstractString, hilbert::Hilbert)
    HDF5.h5open(file, "r") do f
        version = read(HDF5.attributes(f)["version"])
        version == _OPERATOR_IO_VERSION || throw(ArgumentError(
            "operator I/O: unrecognized version \"$version\" (this MOADyna supports \"$_OPERATOR_IO_VERSION\")"))

        coeffs = read(f["coefficients"])
        chains_grp = f["chains"]

        n_groups = length(keys(chains_grp))
        n_groups == length(coeffs) || throw(ArgumentError(
            "operator I/O: /chains has $(n_groups) term groups but /coefficients has $(length(coeffs)) entries"))

        out = OperatorSum{ComplexF64}()
        for i in 1:length(coeffs)
            tg = chains_grp["term_$(i)"]
            label_kind = read(tg["label_kind"])
            label_kind == "int1" || throw(ArgumentError(
                "operator I/O: unsupported label_kind \"$label_kind\" (this MOADyna expects \"int1\" for FermionSite)"))

            kinds      = read(tg["kinds"])
            site_names = read(tg["site_names"])
            labels     = read(tg["labels"])

            length(kinds) == length(site_names) == length(labels) || throw(ArgumentError(
                "operator I/O: chain term_$(i) has inconsistent array lengths " *
                "(kinds=$(length(kinds)), site_names=$(length(site_names)), labels=$(length(labels)))"))

            entries = LadderEntry[]
            for j in eachindex(kinds)
                site_key = Symbol(site_names[j])
                site = hilbert.sites[site_key]   # KeyError if missing
                site isa FermionSite || throw(ArgumentError(
                    "operator I/O supports FermionSite only; site :$site_key in `hilbert` has type $(typeof(site))"))
                push!(entries, LadderEntry(Symbol(kinds[j]), site, (Int(labels[j]),)))
            end

            chain = Tuple(entries)
            out.terms[chain] = OperatorTerm(ComplexF64(coeffs[i]), chain)
        end
        return out
    end
end
