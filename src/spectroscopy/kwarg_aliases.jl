# =====================================================================
# Greek ↔ ASCII kwarg-alias resolver
# =====================================================================
#
# Project-wide naming convention: every user-facing Greek-letter kwarg also
# accepts its ASCII Latin spelling. Specifying both at once raises
# ArgumentError. The canonical Greek name is the one shown in
# docstrings; the ASCII alias resolves to the same value.
#
# This file holds the small helper used by every public Spectroscopy
# entry point. The pattern at the call site is:
#
#     function xas(H, basis, T, ψ;
#                  Γ = nothing, Gamma = nothing,
#                  ω_grid = nothing, omega_grid = nothing,
#                  ...)
#         Γ_resolved      = _resolve_pair(Γ, Gamma; default = DEFAULTS.Γ_xas, name = "Γ")
#         ω_grid_resolved = _resolve_pair(ω_grid, omega_grid; default = :auto, name = "ω_grid")
#         ...
#     end

"""
    _resolve_pair(greek, ascii; default, name) -> value

Return whichever of `greek` (canonical Greek-named kwarg) and `ascii`
(ASCII Latin alias) was passed in. The sentinel for "not passed" is
`nothing`.

- Both `nothing` → returns `default`.
- Exactly one non-`nothing` → returns that value.
- Both non-`nothing` → raises `ArgumentError` mentioning `name`.

`name` is a human-readable string (typically the canonical Greek form,
e.g. `"Γ"`, `"ω_grid"`) used only to construct the error message.
"""
function _resolve_pair(greek, ascii; default, name::AbstractString)
    if greek !== nothing && ascii !== nothing
        throw(ArgumentError(
            "Specify $name only once: provided both Greek and ASCII Latin forms"))
    end
    return greek !== nothing ? greek :
           ascii !== nothing ? ascii : default
end
