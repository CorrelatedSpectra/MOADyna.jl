# =====================================================================
# NiO L₂,₃ XAS — compact-basis (n_2p = 5 only) workflow
# =====================================================================
#
# This is the same physics as `05_nio_xas.jl`, computed on a smaller
# Hilbert sector. The canonical workflow uses
#
#     basis_xas = basis(m, n_2p ∈ 5:6, total == 24)        # 310 states
#
# which embeds |g⟩ into the n_2p = 6 sector (190 states) so that the
# dipole T can lower it into n_2p = 5 (120 states, where Lanczos runs).
# Because H_XAS is block-diagonal in n_2p, the n_2p = 6 block is never
# visited by the recurrence — it costs ~2× memory and ~2× assembly time
# at production scale (~10⁷ states), and matvec FLOPs are not affected.
#
# This example builds the n_2p = 5 sector only, applies T·|g⟩ across
# bases (compute on the union basis, then project amplitudes down to
# the n_2p = 5 sector), and runs block-Lanczos directly on the smaller
# space. The two examples produce identical spectra to FP tolerance.
#
#   Reference: M. Haverkort et al., Phys. Rev. B 85, 165113 (2012);
#              same parameters as `05_nio_xas.jl`.
#
# Output (saved next to this script):
#   nio_xas_compact_x.txt, nio_xas_compact_y.txt, nio_xas_compact_z.txt
#       three-column files with [ω, Re χ, Im χ] for each polarisation.

using MOADyna
using DelimitedFiles

# ---------------------------------------------------------------------
# Shell registry and physical parameters (eV) — identical to 05_nio_xas.jl
# ---------------------------------------------------------------------
m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])

F2dd, F4dd        = 11.14, 6.87
F2pd, G1pd, G3pd  = 6.67, 4.92, 2.80
zeta_3d, zeta_2p  = 0.081, 11.51
tenDq, tenDqL     = 0.56, 1.44
Veg, Vt2g         = 2.06, 1.21
Udd, Upd, Delta   = 7.3, 8.5, 4.7
Bz, Hz            = 1e-6, 0.120

# ---------------------------------------------------------------------
# Onsite energies — same anchor solver as 05_nio_xas.jl
# ---------------------------------------------------------------------
es_gs = onsite_energies(m;
    shells  = (:Ni_3d, :L_3d),
    U       = (Ni_3d = Udd,),
    anchors = [(Ni_3d = 8, L_3d = 10) => 0.0,
               (Ni_3d = 9, L_3d = 9)  => Delta])

es_xas = onsite_energies(m;
    U       = (Ni_3d = Udd,),
    pairs   = ((:Ni_2p, :Ni_3d) => Upd,),
    anchors = [(Ni_2p = 6, Ni_3d = 8, L_3d = 10) => 0.0,
               (Ni_2p = 6, Ni_3d = 9, L_3d = 9)  => Delta,
               (Ni_2p = 5, Ni_3d = 9, L_3d = 10) => 0.0])

# ---------------------------------------------------------------------
# Hamiltonian — same as 05_nio_xas.jl
# ---------------------------------------------------------------------
H_common = coulomb(m, :Ni_3d; U = Udd, F = (F2dd, F4dd)) +
           zeta_3d * LS(m, :Ni_3d) +
           tenDq  * Akm(m, :Ni_3d, :Oh, [0.6, -0.4]) +
           tenDqL * Akm(m, :L_3d,  :Oh, [0.6, -0.4]) +
           Veg  * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :Eg)  +
           Vt2g * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :T2g) +
           Bz * (2 * Sz(m, :Ni_3d) + Lz(m, :Ni_3d)) +
           Hz * Sz(m, :Ni_3d)

H_GS = H_common + es_gs.Ni_3d * n(m, :Ni_3d) + es_gs.L_3d * n(m, :L_3d)

H_XAS = H_common +
        coulomb(m, :Ni_2p, :Ni_3d; U = Upd, F = (F2pd,), G = (G1pd, G3pd)) +
        zeta_2p * LS(m, :Ni_2p) +
        es_xas.Ni_2p * n(m, :Ni_2p) +
        es_xas.Ni_3d * n(m, :Ni_3d) +
        es_xas.L_3d  * n(m, :L_3d)

T = dipole(m, :Ni_2p => :Ni_3d)

# ---------------------------------------------------------------------
# Bases — three layers
# ---------------------------------------------------------------------
# basis_gs    : ground-state sector (n_2p = 6, 18 valence electrons).
# basis_xas_6 : the n_2p = 6 sector of basis_xas — same as basis_gs in
#               this example; used as the staging ground for T·|g⟩.
# basis_xas_5 : the n_2p = 5 sector of basis_xas — where Lanczos runs.
# basis_xas   : the union (n_2p ∈ 5:6); used as a scratch basis to hold
#               the embedded |g⟩ so that T can act on it across sectors.
#
# Production-scale note: only basis_xas_5 needs to live as an assembled
# H. The union basis is constructed but never used to assemble H.
basis_gs    = basis(m,
    nshells(m, :Ni_2p) == 6,
    nshells(m, :Ni_3d) + nshells(m, :L_3d) == 18)        # 190 states
basis_xas_5 = basis(m,
    nshells(m, :Ni_2p) == 5,
    total(m) == 24)                                      # 120 states
basis_xas   = basis(m,
    nshells(m, :Ni_2p) in 5:6,
    total(m) == 24)                                      # 310 states (scratch)

@info "NiO basis sizes" GS = length(basis_gs) XAS_5 = length(basis_xas_5) XAS_union = length(basis_xas)

# ---------------------------------------------------------------------
# Ground state — on basis_gs
# ---------------------------------------------------------------------
gs = eigen(H_GS, basis_gs; n = 3)
Eg = gs.values[1]
@info "NiO ground state" Eg = round(Eg; digits = 6) gap = round(gs.values[2] - Eg; digits = 6)

# ---------------------------------------------------------------------
# Cross-basis T·|g⟩ → land directly in basis_xas_5
# ---------------------------------------------------------------------
# Step 1: embed |g⟩ from basis_gs into the union basis_xas (since
#         basis_gs ⊂ basis_xas, this is the standard `embed`).
# Step 2: apply T (assembled on basis_xas) to land in n_2p = 5 amplitudes.
# Step 3: project the resulting vector down to basis_xas_5 by walking
#         basis_xas_5's states and copying the matching amplitudes.
#         This is just a permutation/restriction; the amplitudes outside
#         basis_xas_5 in T·|g⟩ are zero by construction (T lowers n_2p
#         by 1 from 6 to 5).
#
# `project_to_subset` below is a small downscope helper; the public
# `embed` requires source ⊂ target, but here we need the opposite. Once
# MOADyna ships a built-in down-projection this can be replaced.

"""
    project_to_subset(psi_full, basis_full, basis_sub) -> Vector

Project `psi_full` (defined on `basis_full`) onto a subset basis
`basis_sub ⊂ basis_full`. Equivalent to the adjoint of `embed`. Errors
if any state in `basis_sub` is not present in `basis_full`.
"""
function project_to_subset(psi_full::AbstractVector,
                           basis_full::EagerBasis,
                           basis_sub::EagerBasis)
    length(psi_full) == length(basis_full) || throw(DimensionMismatch(
        "project_to_subset: vector length $(length(psi_full)) does not match basis_full size $(length(basis_full))"))
    psi_sub = zeros(eltype(psi_full), length(basis_sub))
    for i in 1:length(basis_sub)
        state = get_state(basis_sub, i)
        j = get_index(basis_full, state)
        j > 0 || throw(ArgumentError(
            "project_to_subset: basis_sub state $i not present in basis_full"))
        psi_sub[i] = psi_full[j]
    end
    return psi_sub
end

psi_g_full = embed(gs.vectors[:, 1], basis_gs => basis_xas)

# ---------------------------------------------------------------------
# Hamiltonian on the compact basis
# ---------------------------------------------------------------------
H_xas_5_sp = assemble(compile(H_XAS, basis_xas_5), basis_xas_5)

# ---------------------------------------------------------------------
# Spectrum loop: pre-applied source → 3-arg `xas` entry point.
# ---------------------------------------------------------------------
# `xas(H, basis, source; ...)` accepts a pre-applied Lanczos starting
# vector and runs the same block-Lanczos + continued-fraction pipeline
# as the 4-arg form, just skipping the internal T·ψ step. We use it
# here because T was applied across bases (basis_xas → basis_xas_5)
# rather than on the working basis — exactly the cross-basis dipole
# workflow PyQuanty's `CreateSpectra(psi=..., H=...)` covers.
#
# Polarisations are looped explicitly: each call returns a 1-D
# SpectraTensor (one source vector → one χ_aa(ω) channel). The
# off-diagonal a≠b cross-correlations are not needed in this example.

omega_grid = range(-15.0, 25.0; length = 801)
Gamma      = 0.6                                         # Lorentzian FWHM (eV)
krylovdim  = 200

specs = Vector{Any}(undef, 3)
for (idx, name) in enumerate(("x", "y", "z"))
    # T_a in basis_xas (310 × 310 sparse). The matrix is small at this
    # scale, but at production scale this assemble would dominate cost
    # if done on the union basis — it can equivalently be assembled as
    # a rectangular operator from basis_gs to basis_xas_5 (future work).
    T_xas_sp = assemble(compile(T[idx], basis_xas), basis_xas)

    # T·|g⟩ on the union basis, then drop into basis_xas_5.
    psi_T_full = T_xas_sp * psi_g_full
    psi_T_5    = project_to_subset(psi_T_full, basis_xas, basis_xas_5)

    spec = xas(H_xas_5_sp, basis_xas_5, psi_T_5;
               omega_grid = omega_grid,
               Gamma      = Gamma,
               Eg         = Eg,
               krylovdim  = krylovdim)
    χ = spec.tensor                                      # already 1-D (n_ω,)

    specs[idx] = χ
    out = hcat(collect(omega_grid), real.(χ), imag.(χ))
    writedlm(joinpath(@__DIR__, "nio_xas_compact_$name.txt"), out)
    @info "Polarisation $name" peak = round(maximum(-imag.(χ)); digits = 6) ω_at_peak = omega_grid[argmax(-imag.(χ))]
end

# Polarisation-summed (isotropic) intensity.
chi_iso = sum(specs) ./ 3
@info "Isotropic peak" peak = round(maximum(-imag.(chi_iso)); digits = 6) ω_at_peak = omega_grid[argmax(-imag.(chi_iso))]

@info "Done. XAS spectra written to nio_xas_compact_{x,y,z}.txt next to this script."
