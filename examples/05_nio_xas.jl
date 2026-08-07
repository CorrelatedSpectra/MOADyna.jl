# =====================================================================
# NiO L₂,₃ XAS — native multiplet calculation
# =====================================================================
#
# Charge-transfer multiplet calculation for the NiO L₂,₃ X-ray absorption
# edge. The model is:
#
#   - Ni 3d shell (10 modes)
#   - Ni 2p core shell (6 modes)
#   - O 2p ligand band L_3d (10 modes), modeled as a single bath orbital
#     per d-symmetry channel
#
# All operators are built from MOADyna primitives — no QuantyIO crutch.
# Slater-Condon Coulomb on both 3d and 2p-3d, atomic spin-orbit on
# both shells, octahedral crystal field on Ni and on the ligand,
# eg/t2g hybridization with anchor-based ZSA onsite energies.
#
#   Reference: M. Haverkort et al., Phys. Rev. B 85, 165113 (2012);
#              parameters here follow the canonical NiO Quanty script.
#
# Output (saved next to this script):
#   nio_xas_x.txt, nio_xas_y.txt, nio_xas_z.txt
#       three-column files with [ω, Re χ, Im χ] for each polarisation.
#       The XAS intensity is -Im χ.

using MOADyna
using DelimitedFiles

# ---------------------------------------------------------------------
# Shell registry and physical parameters (eV)
# ---------------------------------------------------------------------
m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])

F2dd, F4dd        = 11.14, 6.87        # d-d Slater integrals
F2pd, G1pd, G3pd  = 6.67, 4.92, 2.80   # 2p-3d Slater integrals (F², G¹, G³)
zeta_3d, zeta_2p  = 0.081, 11.51       # atomic spin-orbit
tenDq, tenDqL     = 0.56, 1.44         # cubic crystal-field on Ni / ligand
Veg, Vt2g         = 2.06, 1.21         # Ni-ligand hybridization
Udd, Upd, Delta   = 7.3, 8.5, 4.7      # Hubbard U + charge-transfer
Bz, Hz            = 1e-6, 0.120        # micro-Zeeman + magnetic field on Ni 3d

# ---------------------------------------------------------------------
# Onsite energies — anchor solver
# ---------------------------------------------------------------------
# Ground-state subsystem (Ni_2p stays full → no anchor needed for it).
es_gs = onsite_energies(m;
    shells  = (:Ni_3d, :L_3d),
    U       = (Ni_3d = Udd,),
    anchors = [(Ni_3d = 8, L_3d = 10) => 0.0,
               (Ni_3d = 9, L_3d = 9)  => Delta])

# XAS subsystem: all three shells, with Upd cross-shell coupling.
es_xas = onsite_energies(m;
    U       = (Ni_3d = Udd,),
    pairs   = ((:Ni_2p, :Ni_3d) => Upd,),
    anchors = [(Ni_2p = 6, Ni_3d = 8, L_3d = 10) => 0.0,
               (Ni_2p = 6, Ni_3d = 9, L_3d = 9)  => Delta,
               (Ni_2p = 5, Ni_3d = 9, L_3d = 10) => 0.0])

# ---------------------------------------------------------------------
# Hamiltonian — common piece (everything except Ni_2p-specific terms)
# ---------------------------------------------------------------------
H_common = coulomb(m, :Ni_3d; U = Udd, F = (F2dd, F4dd)) +
           zeta_3d * LS(m, :Ni_3d) +
           tenDq  * Akm(m, :Ni_3d, :Oh, [0.6, -0.4]) +
           tenDqL * Akm(m, :L_3d,  :Oh, [0.6, -0.4]) +
           Veg  * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :Eg)  +
           Vt2g * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :T2g) +
           Bz * (2 * Sz(m, :Ni_3d) + Lz(m, :Ni_3d)) +
           Hz * Sz(m, :Ni_3d)

# Ground-state Hamiltonian (Ni_2p full, treated as inert).
H_GS = H_common + es_gs.Ni_3d * n(m, :Ni_3d) + es_gs.L_3d * n(m, :L_3d)

# XAS Hamiltonian (with 2p core hole + Upd cross-shell + 2p spin-orbit).
H_XAS = H_common +
        coulomb(m, :Ni_2p, :Ni_3d; U = Upd, F = (F2pd,), G = (G1pd, G3pd)) +
        zeta_2p * LS(m, :Ni_2p) +
        es_xas.Ni_2p * n(m, :Ni_2p) +
        es_xas.Ni_3d * n(m, :Ni_3d) +
        es_xas.L_3d  * n(m, :L_3d)

# Cartesian dipole vector [T_x, T_y, T_z] for the Ni 2p → 3d transition.
T = dipole(m, :Ni_2p => :Ni_3d)

# ---------------------------------------------------------------------
# Bases — sector-restricted
# ---------------------------------------------------------------------
basis_gs  = basis(m,
    nshells(m, :Ni_2p) == 6,
    nshells(m, :Ni_3d) + nshells(m, :L_3d) == 18)        # 190 states
basis_xas = basis(m,
    nshells(m, :Ni_2p) in 5:6,
    total(m) == 24)                                      # 310 states
# `basis_xas` covers BOTH n_2p = 6 (190 states, where |g⟩ embeds) and
# n_2p = 5 (120 states, where T·|g⟩ lands and Lanczos runs). The
# n_2p = 6 block is "embed scaffolding" — it is never visited during
# Lanczos because H_XAS is block-diagonal in n_2p, so matvec FLOPs are
# unaffected. But it costs ~2× memory and ~2× assembly time. For
# production-scale problems where this matters, see
# `07_nio_xas_compact.jl` for the n_2p = 5-only workflow.

@info "NiO basis sizes" GS = length(basis_gs) XAS = length(basis_xas)

# ---------------------------------------------------------------------
# Ground state
# ---------------------------------------------------------------------
gs = eigen(H_GS, basis_gs; n = 3)
Eg = gs.values[1]
@info "NiO ground state" Eg = round(Eg; digits = 6) gap = round(gs.values[2] - Eg; digits = 6)

# Embed ground state into the larger XAS basis (n_2p ∈ 5:6).
psi0 = embed(gs.vectors[:, 1], basis_gs => basis_xas)

# ---------------------------------------------------------------------
# XAS spectra — three Cartesian polarisations
# ---------------------------------------------------------------------
H_XAS_sp   = assemble(compile(H_XAS, basis_xas), basis_xas)
omega_grid = range(-15.0, 25.0; length = 801)
Gamma      = 0.6                                         # Lorentzian FWHM (eV)

specs = Vector{Any}(undef, 3)
for (idx, name) in enumerate(("x", "y", "z"))
    specs[idx] = xas(H_XAS_sp, basis_xas, T[idx], psi0;
                     omega_grid = omega_grid,
                     Gamma      = Gamma,
                     Eg         = Eg,
                     krylovdim  = 200)
    χ = specs[idx].tensor
    out = hcat(collect(omega_grid), real.(χ), imag.(χ))
    writedlm(joinpath(@__DIR__, "nio_xas_$name.txt"), out)
    @info "Polarisation $name" peak = round(maximum(-imag.(χ)); digits = 6) ω_at_peak = omega_grid[argmax(-imag.(χ))]
end

# Polarisation-summed (isotropic) intensity.
chi_iso = sum(s.tensor for s in specs) ./ 3
@info "Isotropic peak" peak = round(maximum(-imag.(chi_iso)); digits = 6) ω_at_peak = omega_grid[argmax(-imag.(chi_iso))]

@info "Done. XAS spectra written to nio_xas_{x,y,z}.txt next to this script."
