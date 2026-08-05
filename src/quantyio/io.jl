# =====================================================================
# IO — file and directory readers
# =====================================================================

"""
    read_quanty_operator(path, hilbert, mode_map) -> OperatorSum

Read a Quanty operator dump from a file and build the corresponding
MOAD `OperatorSum`. Equivalent to:

    parsed = parse_quanty_operator(read(path, String))
    build_operator(parsed, hilbert, mode_map)
"""
function read_quanty_operator(path::AbstractString, hilbert, mode_map)
    parsed = open(parse_quanty_operator, path, "r")
    return build_operator(parsed, hilbert, mode_map)
end

"""
    read_quanty_operators(dir, hilbert, mode_map; pattern=r".*") -> Dict{String, OperatorSum}

Read every file in `dir` whose name matches `pattern`, build the
corresponding MOAD operator for each. Returned dict is keyed by
filename (basename, without directory).

Useful for the PyQuanty pattern: dump observables from Quanty into
a directory, then load them all in one call.
"""
function read_quanty_operators(dir::AbstractString, hilbert, mode_map; pattern::Regex = r".*")
    out = Dict{String, OperatorSum}()
    for fname in readdir(dir)
        path = joinpath(dir, fname)
        isfile(path) || continue
        occursin(pattern, fname) || continue
        out[fname] = read_quanty_operator(path, hilbert, mode_map)
    end
    return out
end
