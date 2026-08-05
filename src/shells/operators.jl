# src/shells/operators.jl
#
# Shell-keyed operator dispatch. Each method here adds a new dispatch
# for ShellModel arguments to functions imported from MOAD.Algebra
# (n, Sx, Sy, Sz, Splus, Sminus). The functions themselves are
# imported in Shells.jl with `import` (not `using`) so that the
# methods we add here extend the original Algebra function rather
# than creating a separate one.
#
# All operators follow the same multi-dispatch pattern:
#   op(m::ShellModel)                          → sum over all shells
#   op(m::ShellModel, shell::Symbol)           → on one shell
#   op(m::ShellModel, sel::Pair{Symbol, Int})  → single orbital by m value
#   op(m::ShellModel, sel::Pair{Symbol, <:AbstractVector})  → orbital subset
#   op(m::ShellModel, shells::Vararg{Symbol})  → sum over named shells

# --- shell-keyed n ---

"""
    n(m::ShellModel, shell::Symbol) -> OperatorSum

Total fermion number on a shell. Equivalent to `n(site_of(m, shell))`,
which sums n over all modes of the underlying FermionSite.
"""
n(m::ShellModel, shell::Symbol) = n(site_of(m, shell))

"""
    n(m::ShellModel) -> OperatorSum

Total fermion number across all shells in the model.
"""
n(m::ShellModel) = sum(n(m, s) for s in m.shells)

"""
    n(m::ShellModel, sel::Pair{Symbol, Int}) -> OperatorSum

Number on a single orbital within a shell, identified by m value.
For a d-shell, `n(m, :Ni_3d => 0)` gives the number on the m=0 orbital
(both spins).
"""
function n(m::ShellModel, sel::Pair{Symbol, <:Integer})
    shell, m_value = sel.first, sel.second
    site = site_of(m, shell)
    ell = ell_of(m, shell)
    dn_idx, up_idx = _orbital_mode_pair(ell, m_value)
    return n(site, dn_idx) + n(site, up_idx)
end

"""
    n(m::ShellModel, sel::Pair{Symbol, <:AbstractVector}) -> OperatorSum

Number on a subset of orbitals within a shell, identified by a list
of m values.
"""
function n(m::ShellModel, sel::Pair{Symbol, <:AbstractVector})
    shell, m_values = sel.first, sel.second
    return sum(n(m, shell => mv) for mv in m_values)
end

"""
    n(m::ShellModel, shells::Vararg{Symbol}) -> OperatorSum

Number summed over the named shells.
"""
function n(m::ShellModel, shells::Vararg{Symbol})
    length(shells) >= 1 || return n(m)
    return sum(n(m, s) for s in shells)
end

# --- shell-keyed Sz ---
#
# Delegates to the Algebra primitive Sz(site::FermionSite, orbital_pairs).

"""
    Sz(m::ShellModel, shell::Symbol) -> OperatorSum

Total z-spin on a shell:

    Sz = (1/2) Σ_m (n(m, ↑) − n(m, ↓))

where the sum runs over all 2ℓ+1 orbital m values of the shell.
Delegates to `Sz(site::FermionSite, orbital_pairs)` in MOAD.Algebra.
"""
function Sz(m::ShellModel, shell::Symbol)
    site = site_of(m, shell)
    ell = ell_of(m, shell)
    pairs = [_orbital_mode_pair(ell, mv) for mv in _m_values(ell)]
    return Sz(site, pairs)
end

"""
    Sz(m::ShellModel) -> OperatorSum

Total z-spin summed over all shells.
"""
Sz(m::ShellModel) = sum(Sz(m, s) for s in m.shells)

"""
    Sz(m::ShellModel, sel::Pair{Symbol, <:Integer}) -> OperatorSum

z-spin on a single orbital identified by m value.
"""
function Sz(m::ShellModel, sel::Pair{Symbol, <:Integer})
    shell, m_value = sel.first, sel.second
    site = site_of(m, shell)
    ell = ell_of(m, shell)
    dn_idx, up_idx = _orbital_mode_pair(ell, m_value)
    return (1//2) * (n(site, up_idx) - n(site, dn_idx))
end

"""
    Sz(m::ShellModel, sel::Pair{Symbol, <:AbstractVector}) -> OperatorSum

z-spin on an orbital subset.
"""
Sz(m::ShellModel, sel::Pair{Symbol, <:AbstractVector}) =
    sum(Sz(m, sel.first => mv) for mv in sel.second)

"""
    Sz(m::ShellModel, shells::Vararg{Symbol}) -> OperatorSum

z-spin summed over the named shells.
"""
function Sz(m::ShellModel, shells::Vararg{Symbol})
    length(shells) >= 1 || return Sz(m)
    return sum(Sz(m, s) for s in shells)
end

# --- shell-keyed Splus / Sminus ---
#
# Delegates to the Algebra primitive Splus(site::FermionSite, orbital_pairs).

"""
    Splus(m::ShellModel, shell::Symbol) -> OperatorSum

Total raising operator on a shell:

    Splus = Σ_m c†(m, ↑) c(m, ↓)

summed over all 2ℓ+1 orbital m values.
Delegates to `Splus(site::FermionSite, orbital_pairs)` in MOAD.Algebra.
"""
function Splus(m::ShellModel, shell::Symbol)
    site = site_of(m, shell)
    ell = ell_of(m, shell)
    pairs = [_orbital_mode_pair(ell, mv) for mv in _m_values(ell)]
    return Splus(site, pairs)
end

"""
    Splus(m::ShellModel) -> OperatorSum

Total raising operator across all shells.
"""
Splus(m::ShellModel) = sum(Splus(m, s) for s in m.shells)

"""
    Splus(m::ShellModel, sel::Pair{Symbol, <:Integer}) -> OperatorSum

Raising operator on a single orbital identified by m value:

    c†(m, ↑) c(m, ↓)
"""
function Splus(m::ShellModel, sel::Pair{Symbol, <:Integer})
    shell, m_value = sel.first, sel.second
    site = site_of(m, shell)
    ell = ell_of(m, shell)
    dn_idx, up_idx = _orbital_mode_pair(ell, m_value)
    return cdag(site, up_idx) * c(site, dn_idx)
end

"""
    Splus(m::ShellModel, sel::Pair{Symbol, <:AbstractVector}) -> OperatorSum

Raising operator on an orbital subset.
"""
Splus(m::ShellModel, sel::Pair{Symbol, <:AbstractVector}) =
    sum(Splus(m, sel.first => mv) for mv in sel.second)

"""
    Splus(m::ShellModel, shells::Vararg{Symbol}) -> OperatorSum

Raising operator summed over the named shells.
"""
function Splus(m::ShellModel, shells::Vararg{Symbol})
    length(shells) >= 1 || return Splus(m)
    return sum(Splus(m, s) for s in shells)
end

# Sminus = adjoint(Splus) for every dispatch form.

"""
    Sminus(m::ShellModel, args...) -> OperatorSum

Lowering operator: Hermitian conjugate of `Splus`. Forms parallel to `Splus`.
"""
Sminus(m::ShellModel, args...) = adjoint(Splus(m, args...))

# --- shell-keyed Sx, Sy ---
#
# Sx = (1/2)(Splus + Sminus);  Sy = (-im/2)(Splus - Sminus)
#
# Same dispatch pattern via the args... forwarding form.

"""
    Sx(m::ShellModel, args...) -> OperatorSum

x-component of total spin. Same dispatch as `Sz`.
"""
Sx(m::ShellModel, args...) =
    (1//2) * (Splus(m, args...) + Sminus(m, args...))

"""
    Sy(m::ShellModel, args...) -> OperatorSum

y-component of total spin. Same dispatch as `Sz`.
"""
Sy(m::ShellModel, args...) =
    (-im/2) * (Splus(m, args...) - Sminus(m, args...))

# --- shell-keyed Ssqr = Sx² + Sy² + Sz² ---
#
# Extends Ssqr from Algebra (imported in Shells.jl) with ShellModel dispatch.

"""
    Ssqr(m::ShellModel, args...) -> OperatorSum

Total spin squared, `S² = Sx² + Sy² + Sz²`. Same dispatch table as
`Sx`/`Sy`/`Sz`: `args...` may be empty (total over all shells), a
single shell `Symbol`, a `Pair{Symbol, Int}` (single orbital), a
`Pair{Symbol, Vector{Int}}` (orbital subset), or several `Symbol`s
(multi-shell sum).

Extends `Ssqr(site::FermionSite, orbital_pairs)` from MOAD.Algebra.
"""
function Ssqr(m::ShellModel, args...)
    sx = Sx(m, args...)
    sy = Sy(m, args...)
    sz = Sz(m, args...)
    return sx*sx + sy*sy + sz*sz
end
