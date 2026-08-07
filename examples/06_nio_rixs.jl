# =====================================================================
# NiO L₃ RIXS — native multiplet calculation
# =====================================================================
#
# Resonant inelastic X-ray scattering at the NiO L₃ edge using the same
# multiplet model as `05_nio_xas.jl`. The RIXS amplitude is
#
#   F(ω_in, ω_out) = ⟨g| T_out† G_f(ω_out) T_out G_i(ω_in) T_in |g⟩
#
# where G_i is the resolvent of the intermediate-state Hamiltonian (with
# 2p core hole) and G_f is the resolvent of the final-state Hamiltonian
# (no core hole). MOADyna's `rixs` builds both resolvents via block Lanczos
# and contracts them on the (ω_in, ω_out) grid.
#
# In this example: T_in = T_x (incident x-polarised), T_out = T_y
# (emitted y-polarised). The intensity map is -Im F(ω_in, ω_out).
#
# Output (saved next to this script):
#   nio_rixs_xy.txt  — (n_ω_in, n_ω_out) grid of -Im F values, plus a
#                      header row of ω_out and a header column of ω_in.

using MOADyna
using DelimitedFiles

# ---------------------------------------------------------------------
# Same multiplet setup as 05_nio_xas.jl
# ---------------------------------------------------------------------
m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])

F2dd, F4dd        = 11.14, 6.87
F2pd, G1pd, G3pd  = 6.67, 4.92, 2.80
zeta_3d, zeta_2p  = 0.081, 11.51
tenDq, tenDqL     = 0.56, 1.44
Veg, Vt2g         = 2.06, 1.21
Udd, Upd, Delta   = 7.3, 8.5, 4.7
Bz, Hz            = 1e-6, 0.120

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

H_common = coulomb(m, :Ni_3d; U = Udd, F = (F2dd, F4dd)) +
           zeta_3d * LS(m, :Ni_3d) +
           tenDq  * Akm(m, :Ni_3d, :Oh, [0.6, -0.4]) +
           tenDqL * Akm(m, :L_3d,  :Oh, [0.6, -0.4]) +
           Veg  * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :Eg)  +
           Vt2g * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :T2g) +
           Bz * (2 * Sz(m, :Ni_3d) + Lz(m, :Ni_3d)) +
           Hz * Sz(m, :Ni_3d)

# Ground-state Hamiltonian (n_2p = 6 sector).
H_GS = H_common + es_gs.Ni_3d * n(m, :Ni_3d) + es_gs.L_3d * n(m, :L_3d)

# Intermediate state (2p core hole present).
H_int = H_common +
        coulomb(m, :Ni_2p, :Ni_3d; U = Upd, F = (F2pd,), G = (G1pd, G3pd)) +
        zeta_2p * LS(m, :Ni_2p) +
        es_xas.Ni_2p * n(m, :Ni_2p) +
        es_xas.Ni_3d * n(m, :Ni_3d) +
        es_xas.L_3d  * n(m, :L_3d)

# Final state (core hole filled). Same shell physics as GS, but built
# on the larger basis_xas so the resolvent G_f(ω_out) acts in the right
# space. Ni_2p onsite is set to 0 here so the n_2p = 6 sector matches
# H_GS exactly (which has no explicit n(:Ni_2p) term — implicit zero).
# The n_2p = 5 sector of H_final is unphysical (no core-hole potential)
# but is not exercised: T_out brings the propagated state back to
# n_2p = 6 before G_f acts on it.
H_final = H_common +
          es_gs.Ni_3d * n(m, :Ni_3d) +
          es_gs.L_3d  * n(m, :L_3d)

# Cartesian dipole vector — same operator drives both absorption and
# emission (with appropriate polarisations).
T = dipole(m, :Ni_2p => :Ni_3d)

# ---------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------
basis_gs  = basis(m,
    nshells(m, :Ni_2p) == 6,
    nshells(m, :Ni_3d) + nshells(m, :L_3d) == 18)
basis_xas = basis(m,
    nshells(m, :Ni_2p) in 5:6,
    total(m) == 24)

# ---------------------------------------------------------------------
# Ground state, then embed into the larger RIXS basis
# ---------------------------------------------------------------------
gs   = eigen(H_GS, basis_gs; n = 3)
Eg   = gs.values[1]
psi0 = embed(gs.vectors[:, 1], basis_gs => basis_xas)
@info "NiO ground state" Eg = round(Eg; digits = 6)

# ---------------------------------------------------------------------
# RIXS — incident x, emitted y, on a grid across the L_3 main edge
# ---------------------------------------------------------------------
H_int_sp   = assemble(compile(H_int,   basis_xas), basis_xas)
H_final_sp = assemble(compile(H_final, basis_xas), basis_xas)

# 11 incident energies covering the L_3 resonance (-6 to 0 eV from
# absorption-edge centroid), 851 emission energies for energy-loss
# resolution (0 to 8.5 eV from elastic line).
omega_in_grid  = range(-6.0, 0.0; length = 11)
omega_out_grid = range(-0.5, 8.0; length = 851)

@info "Running RIXS on $(length(omega_in_grid)) × $(length(omega_out_grid)) grid (T_in=Tx, T_out=Ty) …"
@time result = rixs(H_final_sp, H_int_sp, basis_xas, T[1], T[2], psi0;
                    omega_in_grid    = omega_in_grid,
                    omega_out_grid   = omega_out_grid,
                    Gamma_intermediate = 0.6,    # core-hole lifetime FWHM
                    Gamma_final        = 0.1,    # valence broadening FWHM
                    Eg                 = Eg,
                    krylovdim          = 200)

intensity = -imag.(result.tensor)                # (n_ω_in, n_ω_out)

# Save: header row = ω_out, header col = ω_in, body = -Im F.
out = vcat(reshape(vcat(NaN, collect(omega_out_grid)), 1, :),
           hcat(collect(omega_in_grid), intensity))
writedlm(joinpath(@__DIR__, "nio_rixs_xy.txt"), out)

# Quick summary: peak position in the (ω_in, ω_out) plane.
peak_idx     = argmax(intensity)
peak_ω_in    = omega_in_grid[peak_idx[1]]
peak_ω_out   = omega_out_grid[peak_idx[2]]
peak_value   = intensity[peak_idx]
@info "RIXS peak" peak_value ω_in = peak_ω_in ω_out = peak_ω_out

@info "Done. RIXS map written to nio_rixs_xy.txt next to this script."
