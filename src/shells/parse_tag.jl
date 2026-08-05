# src/shells/parse_tag.jl

const _ORBITAL_CHARS = ('s', 'p', 'd', 'f', 'g', 'h')
const _ORBITAL_ELL = Dict{Char,Int}('s'=>0, 'p'=>1, 'd'=>2, 'f'=>3, 'g'=>4, 'h'=>5)

"""
    _parse_shell_tag(tag::Symbol) -> NamedTuple

Parse `:<atom>_<n><orbital>` into `(atom, n, orbital, ell)`.

- `atom` is any leading letter-digit-underscore identifier (or empty).
- `n` is the principal quantum number (0 if unspecified).
- `orbital` is `s`, `p`, `d`, `f`, `g`, or `h`.
- `ell` is the orbital angular momentum (0..5).

The `atom` and `n` fields have **no physical meaning** — they are user
labels. Only the orbital character drives the mode-allocation count.
"""
function _parse_shell_tag(tag::Symbol)
    s = String(tag)

    # Find the trailing orbital character
    isempty(s) && throw(ArgumentError("empty shell tag"))
    orbital_char = s[end]
    orbital_char in _ORBITAL_CHARS || throw(ArgumentError(
        "shell tag :$tag has unrecognised trailing orbital character '$orbital_char'. " *
        "Expected one of s, p, d, f, g, h. " *
        "Shell tags follow the form <atom>_<n><orbital>, e.g., :Ni_3d, :O_2p, :Sm_4f."))
    orbital = Symbol(orbital_char)
    ell = _ORBITAL_ELL[orbital_char]

    # Strip the orbital character; the body is "<atom>_*<n>" with optional atom and digits.
    body = s[1:end-1]

    # Trailing-digit-run idiom: the principal quantum number is the run of
    # digits at the end of `body`, with any number of underscores between it
    # and the atom label. Empty digits ⇒ n = 0; empty atom ⇒ atom = "".
    m = match(r"^(.*?)_*(\d*)$", body)
    atom = m.captures[1]
    n = isempty(m.captures[2]) ? 0 : parse(Int, m.captures[2])

    return (atom = atom, n = n, orbital = orbital, ell = ell)
end
