# =====================================================================
# rotate — single-particle basis change on many-body operators
# =====================================================================

using LinearAlgebra: I, opnorm

"""
    rotate(op::OperatorSum, h_in::Hilbert, U::AbstractMatrix;
           h_out::Hilbert = h_in, project::Bool = false) -> OperatorSum

Apply a single-particle basis change to a many-body operator. `U` is the
basis-change matrix in standard physics convention (Sakurai):

    U[j, i] = ⟨e_old_j | e_new_i⟩

Rows index the old basis, columns index the new basis. `U` is `N_old × N_new`.
Substitution rule applied to `op` (written in the old/input basis):

    c_old_j   →  Σ_i      U[j, i]  · c_new_i
    c†_old_j  →  Σ_i conj(U[j, i]) · c†_new_i

`h_in` and `h_out` fix the `(site_name, label_tuple) → flat global mode
index` ordering. Default `h_out = h_in` for the square-`U` case. Arbitrary
non-orthonormal `U` is rejected.

# Extended help

**Quanty convention**: this follows Sakurai, which is the Hermitian conjugate
of Quanty's `Rotate(...)` convention. If you have a Quanty `rotMat`, pass
`adjoint(rotMat)` here. For real-valued matrices (e.g. cubic-harmonic
transforms) the two conventions agree.

**Two regimes via the `project` kwarg**:

- Default `project = false` — basis change. `U` must be square and unitary
  (`‖U†U − I‖₂ < 1e-10`). Anticommutation preserved; round-trip with `U'`
  reproduces the input operator.

- `project = true` — projection / downfolding / irrep restriction. `U` must
  be an isometry (`size(U,1) ≥ size(U,2)` and `‖U†U − I_{N_out}‖₂ < 1e-10`).
  The result is the effective operator on the projected sub-Hilbert;
  anticommutation is projected, not preserved, and round-trip is not
  guaranteed.
"""
function rotate(op::OperatorSum, h_in::Hilbert, U::AbstractMatrix;
                h_out::Hilbert = h_in, project::Bool = false)
    n_in  = sum(encoding_bits(s) for s in values(h_in.sites);  init = 0)
    n_out = sum(encoding_bits(s) for s in values(h_out.sites); init = 0)
    size(U, 1) == n_in || throw(DimensionMismatch(
        "rotate: U has $(size(U, 1)) rows but expected $n_in " *
        "(rows index the input/old basis modes)."))

    if !project
        size(U, 1) == size(U, 2) || throw(ArgumentError(
            "rotate (basis change) requires square U; got $(size(U, 1))×$(size(U, 2)). " *
            "Pass project=true for orthonormal projection / downfolding."))
        size(U, 2) == n_out || throw(DimensionMismatch(
            "rotate: U has $(size(U, 2)) columns but h_out has $n_out modes."))
    else
        size(U, 2) == n_out || throw(DimensionMismatch(
            "rotate (project=true): U has $(size(U, 2)) columns but h_out has $n_out modes."))
        n_in >= n_out || throw(ArgumentError(
            "rotate (project=true) requires N_in ≥ N_out; got $(n_in) < $(n_out)."))
    end

    residual = opnorm(U' * U - I, 2)
    residual < 1e-10 || throw(ArgumentError(
        "rotate: U is not " * (project ? "isometric" : "unitary") *
        " to 1e-10 (‖U†U − I‖₂ = $residual). Columns of U must be orthonormal. " *
        (project ? "" : "Pass project=true if you intended an isometric projection.")))

    in_idx    = _rotate_global_index_map(h_in)
    out_modes = _rotate_global_mode_list(h_out)

    out = zero(OperatorSum{Complex{Float64}})
    for term in op
        out += _rotate_term(term, U, in_idx, out_modes)
    end
    return out
end

function _rotate_global_index_map(h::Hilbert)
    map = Dict{Tuple{Symbol, Any}, Int}()
    g = 0
    for (key, site) in h.sites
        for label in mode_labels_canonical(site)
            g += 1
            map[(key, label)] = g
        end
    end
    return map
end

function _rotate_global_mode_list(h::Hilbert)
    out = Tuple{AbstractSite, Any}[]
    for site in values(h.sites)
        for label in mode_labels_canonical(site)
            push!(out, (site, label))
        end
    end
    return out
end

function _rotate_term(term::OperatorTerm, U::AbstractMatrix, in_idx, out_modes)
    accum = term.coefficient * one(OperatorSum{Complex{Float64}})
    for entry in term.chain
        if entry.kind === :c || entry.kind === :cdag
            j = in_idx[(name(entry.site), entry.label)]
            replacement = zero(OperatorSum{Complex{Float64}})
            for i in 1:length(out_modes)
                site_out, label_out = out_modes[i]
                coeff = entry.kind === :cdag ? conj(U[j, i]) : U[j, i]
                iszero(coeff) && continue
                if entry.kind === :cdag
                    replacement += coeff * cdag(site_out, label_out...)
                else
                    replacement += coeff * c(site_out, label_out...)
                end
            end
            accum *= replacement
        else
            throw(ArgumentError(
                "rotate currently supports only fermionic c/cdag entries; " *
                "got kind=$(entry.kind). Bosonic / spin support is a separate task."))
        end
    end
    return accum
end
