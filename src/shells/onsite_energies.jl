# src/shells/onsite_energies.jl
#
# Anchor-based linear solver for shell onsite energies (ZSA-style).
#
# Each anchor pairs a configuration (n_a = N_a, n_b = N_b, ...) with its
# target total energy. We strip off the known interaction contributions
# (intra-shell U via N(N-1)/2 and cross-shell pair interactions via N_a·N_b)
# from each anchor target, then solve the linear system
#
#     Σ_s occ[s] · ε_s  =  E_anchor
#                          − Σ_s occ[s](occ[s] − 1)/2 · U[s]
#                          − Σ_(a,b)∈pairs occ[a] · occ[b] · U_pair[(a,b)]
#
# for the unknowns ε_s. With #anchors == #shells the system is square and
# `\` returns the unique solution; with more anchors it returns the
# least-squares fit.

"""
    onsite_energies(m::ShellModel;
                    anchors::AbstractVector{<:Pair{<:NamedTuple, <:Real}},
                    U::NamedTuple = NamedTuple(),
                    pairs::Tuple   = (),
                    shells         = nothing) -> NamedTuple

Anchor-based solver for shell onsite energies ε_s.

Each anchor pairs a configuration `NamedTuple` (e.g. `(Ni_3d=8, L_3d=10)`)
with its target total energy. The solver subtracts known interaction
corrections from each anchor target, then solves the resulting linear
system `Σ_s occ[s] · ε_s = residual` for the per-shell ε_s.

# Keyword arguments

- `anchors` — vector of `config => energy` pairs. Configuration entries
  default to 0 for shells absent from the NamedTuple.
- `U` — NamedTuple of intra-shell Hubbard `U` scales, keyed by shell tag.
  Each ε_s row picks up the correction `occ[s](occ[s] − 1)/2 · U[s]`.
  Shells absent from `U` contribute zero (the bare onsite, no Hubbard).
- `pairs` — tuple of `Pair{Tuple{Symbol,Symbol}, <:Real}` cross-shell
  density-density couplings, e.g. `((:Ni_2p, :Ni_3d) => 6.5,)`. Each
  pair contributes `occ[a] · occ[b] · U_pair` to the residual.
- `shells` — optional ordering of shells to solve for (defaults to
  `m.shells`).

# Returns

A NamedTuple `(s1 = ε_1, s2 = ε_2, ...)` with one entry per shell in
`shells` (or `m.shells`).

# Example

```julia
m = ShellModel([:Ni_3d, :L_3d])
es = onsite_energies(m;
    U = (Ni_3d = 7.3,),
    anchors = [(Ni_3d=8, L_3d=10) => 0.0,
               (Ni_3d=9, L_3d=9)  => 4.7])
# es.Ni_3d ≈ -41.18888…, es.L_3d ≈ 12.51111…
```
"""
function onsite_energies(m::ShellModel;
                         anchors::AbstractVector{<:Pair{<:NamedTuple, <:Real}},
                         U::NamedTuple   = NamedTuple(),
                         pairs::Tuple    = (),
                         shells          = nothing)
    target_shells = shells === nothing ? collect(m.shells) : collect(shells)
    n_shells  = length(target_shells)
    n_anchors = length(anchors)

    # Validation: empty anchor list is meaningless.
    if n_anchors == 0
        throw(ArgumentError("onsite_energies: `anchors` must be non-empty " *
                            "(got 0 anchors for $(n_shells) unknown ε's)."))
    end

    # Validation: under-determined system (more unknowns than equations).
    if n_anchors < n_shells
        throw(ArgumentError("onsite_energies: under-determined system — " *
                            "got $(n_anchors) anchor(s) for $(n_shells) " *
                            "unknown ε's. Need at least $(n_shells) anchors."))
    end

    # Validation: every shell mentioned in any anchor or in `shells` kwarg
    # must exist in the model; anchors must not mention shells outside the
    # solver's `target_shells` set (those would silently be ignored).
    target_set = Set(target_shells)
    model_set  = Set(m.shells)
    for s in target_shells
        if !(s in model_set)
            throw(ArgumentError("onsite_energies: shell `$(s)` from " *
                                "`shells` kwarg is not in `m.shells` " *
                                "($(collect(m.shells)))."))
        end
    end
    for (i, anchor_pair) in enumerate(anchors)
        config, _ = anchor_pair
        for k in keys(config)
            if !(k in model_set)
                throw(ArgumentError("onsite_energies: anchor #$(i) mentions " *
                                    "shell `$(k)` which is not in `m.shells` " *
                                    "($(collect(m.shells)))."))
            end
            if !(k in target_set)
                throw(ArgumentError("onsite_energies: anchor #$(i) mentions " *
                                    "shell `$(k)` which is not in the solver " *
                                    "set `shells` ($(target_shells))."))
            end
        end
    end

    A = zeros(Float64, n_anchors, n_shells)
    b = zeros(Float64, n_anchors)

    for (i, anchor_pair) in enumerate(anchors)
        config, E_anchor = anchor_pair
        residual = float(E_anchor)

        # Coefficient row + intra-shell U corrections.
        for (j, s) in enumerate(target_shells)
            occ_s = float(get(config, s, 0))
            A[i, j] = occ_s
            U_s = float(get(U, s, 0))
            residual -= occ_s * (occ_s - 1) / 2 * U_s
        end

        # Cross-shell pair corrections: occ[sa] * occ[sb] * U_pair.
        # (Note: avoid naming clash with the RHS vector `b` above.)
        for pair_entry in pairs
            (sa, sb), U_pair = pair_entry
            occ_a = float(get(config, sa, 0))
            occ_b = float(get(config, sb, 0))
            residual -= occ_a * occ_b * float(U_pair)
        end

        b[i] = residual
    end

    eps = A \ b
    return NamedTuple{Tuple(target_shells)}(Tuple(eps))
end
