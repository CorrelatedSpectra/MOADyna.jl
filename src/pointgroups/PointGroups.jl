"""
    MOAD.PointGroups

Point-group representation theory for crystal-field and term-symbol
analysis. Public surface (chapter §17):

- Construction: [`pointgroup`](@ref), [`rotate`](@ref).
- Types: [`GroupElement`](@ref), [`IRrep`](@ref), [`PointGroup`](@ref).
- Representation theory: [`subduce`](@ref), [`project`](@ref),
  [`wignerd`](@ref).
- Crystal field: [`expand_clm`](@ref MOAD.PointGroups.expand_clm), [`expand_clm_central`](@ref MOAD.PointGroups.expand_clm_central),
  [`nparams`](@ref).
- Term-symbol classification: [`classify_subspace`](@ref),
  [`classify_state`](@ref).
- Label provenance: [`has_reference_labels`](@ref),
  [`reference_label_groups`](@ref).

Internal / experimental (not top-level exported; access via
`MOAD.PointGroups.<name>`):

- `ComplexIR` — absolutely-irreducible complex representation; carried
  by `IRrep.complex_constituents`. Mainly for advanced introspection.
- `LiftedRep`, `lift` — single-particle rotation matrices per
  `(element, shell)`. The Fock-space action on a many-body state is
  built from these and exposed as `MOAD.classify_state(ψ, basis, m, G)`
  (in `MOAD.Diagnostics`, which can reach the `Shells`/`Bases` layers
  that `lift` itself cannot); `lift`/`LiftedRep` stay module-qualified
  as the underlying primitive.
"""
module PointGroups

using LinearAlgebra
using StaticArrays
using Random
using WignerSymbols: wigner3j

# `rotate` is shared with MOAD.Algebra; extend the same generic function so
# `MOAD` re-exports a single name with methods on both `PointGroup` (here)
# and `OperatorSum` (Algebra). Algebra is included first in MOAD.jl.
import ..Algebra: rotate

include("data.jl")
include("wigner.jl")
include("group.jl")
include("reference_tables.jl")
include("construction.jl")
include("representations.jl")
include("applications.jl")
include("display.jl")

export GroupElement, IRrep, PointGroup
export pointgroup, rotate
export subduce, project, wignerd
export expand_clm, expand_clm_central, nparams
export classify_subspace, classify_state
export has_reference_labels, reference_label_groups
export character_table, print_character_table, character_table_compare
# `LiftedRep` and `lift` are not top-level exported — they are the
# single-particle primitive behind `MOAD.classify_state(ψ, basis, m, G)`
# (the many-body Fock-space apply lives in `MOAD.Diagnostics`). Reach the
# primitive via `MOAD.PointGroups.lift` / `MOAD.PointGroups.LiftedRep`.

end # module
