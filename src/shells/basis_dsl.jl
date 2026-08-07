# src/shells/basis_dsl.jl
#
# Shell-keyed basis-restriction DSL. Provides:
#   - nshells(m, :s)              — ParticleCount on one shell
#   - nshells(m, :s1, :s2, ...)   — ParticleCount on multiple shells
#   - total(m)                    — ParticleCount on all shells (alias for n_fermion(m.hilbert))
#   - EagerBasis(m::ShellModel, restrictions...) — convenience overload
#
# All restrictions are positional (NOT kwargs); Julia kwargs are
# name=value pairs and cannot carry arbitrary equality expressions
# like `nshells(m, :s) == 6`.

"""
    nshells(m::ShellModel, shell::Symbol) -> ParticleCount{Fermionic}

Conserved-quantity builder for the particle count on one shell.
Use in basis restrictions:

```julia
basis = EagerBasis(m, nshells(m, :Ni_2p) == 6)
```
"""
nshells(m::ShellModel, shell::Symbol) =
    ParticleCount{Fermionic}([site_of(m, shell)])

"""
    nshells(m::ShellModel, shells::Vararg{Symbol}) -> ParticleCount{Fermionic}

Conserved-quantity builder summing particle counts over the named
shells. Combining via `+` on the result of single-shell `nshells` calls
also works (existing `+(::ParticleCount, ::ParticleCount)` overload).

Calling `nshells(m)` with no shells is rejected (would silently build an
empty `ParticleCount` and leave the basis unrestricted). Use `total(m)`
for the all-shells sum instead.
"""
function nshells(m::ShellModel, shells::Vararg{Symbol})
    isempty(shells) && throw(ArgumentError(
        "nshells(m) with no shells: would build an empty ParticleCount and " *
        "silently leave the basis unrestricted. Use `total(m)` for the " *
        "sum across all shells, or pass at least one shell symbol."))
    return ParticleCount{Fermionic}([site_of(m, s) for s in shells])
end

"""
    total(m::ShellModel) -> ParticleCount{Fermionic}

Conserved-quantity builder for total particle count across all shells
in the model. Sums occupation over every shell (equivalent to
`nshells(m, m.shells...)` and to `n_fermion(m.hilbert)`). Use with
`==` or `∈` to restrict the basis:

```julia
basis(m, total(m) == 18)   # exactly 18 electrons across all shells
```
"""
total(m::ShellModel) = n_fermion(m.hilbert)

# --- EagerBasis convenience overload ---

"""
    EagerBasis(m::ShellModel, restrictions...) -> EagerBasis

Build a basis on the underlying Hilbert with shell-aware restriction syntax.

```julia
basis_gs = EagerBasis(m, nshells(m, :Ni_2p) == 6,
                        nshells(m, :Ni_3d) + nshells(m, :L_3d) == 16)
```

Restrictions are positional (NOT kwargs). The method delegates to
`EagerBasis(m.hilbert, restrictions...)`.
"""
EagerBasis(m::ShellModel, restrictions...) =
    EagerBasis(m.hilbert, restrictions...)

# --- basis() convenience overload for ShellModel ---

"""
    basis(m::ShellModel, restrictions...; lazy::Bool = false) -> AbstractBasis

Convenience overload of `MOADyna.Bases.basis` for `ShellModel`. Equivalent
to `basis(m.hilbert, restrictions...; lazy=lazy)`.
"""
basis(m::ShellModel, restrictions...; lazy::Bool = false) =
    basis(m.hilbert, restrictions...; lazy = lazy)
