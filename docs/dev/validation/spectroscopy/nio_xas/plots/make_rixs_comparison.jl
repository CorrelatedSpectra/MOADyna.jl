# =====================================================================
# NiO RIXS — apples-to-apples comparison plot at fixed incident energies
# =====================================================================
#
# Re-runs MOAD's RIXS calculation at the same `(ω_in, ω_out)` grid as
# PyQuanty's reference (`RIXSxy.txt`) and Quanty's
# (`RIXSxy_Quanty.txt`), and overlays line cuts at a handful of
# incident energies across the L_3 peak.
#
# All three codes use the same operators (T_in = T_x, T_out = T_y,
# H_intermediate = XASHamiltonian, H_final = Hamiltonian) and the same
# broadenings (Γ_int = 0.6 eV, Γ_final = 0.1 eV).
#
# Run:
#   julia --project=docs/dev/validation/spectroscopy/nio_xas/plots \
#         docs/dev/validation/spectroscopy/nio_xas/plots/make_rixs_comparison.jl

using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Printf
using MOAD
using MOAD: rixs, eigen
using Plots

const HERE   = @__DIR__
const NIO    = dirname(HERE)
const OP_DIR  = joinpath(NIO, "operators")
const REF_DIR = joinpath(NIO, "reference")

# ---------------------------------------------------------------------
# Setup: same site layout + GS as XAS comparison
# ---------------------------------------------------------------------

s_2p = FermionSite{6}(:p)
s_3d = FermionSite{10}(:d)
s_Ld = FermionSite{10}(:L)
h    = Hilbert(:p => s_2p, :d => s_3d, :L => s_Ld)
map_fn = i -> i < 6  ? (:p, i + 1) :
              i < 16 ? (:d, i - 5) :
                       (:L, i - 15)

basis_gs = EagerBasis(h,
    n_fermion([s_2p]) == 6,
    n_fermion([s_2p, s_3d, s_Ld]) == 24)
H_gs_op = read_quanty_operator(joinpath(OP_DIR, "Hamiltonian.txt"), h, map_fn)
H_gs_sp = assemble(compile(H_gs_op, basis_gs), basis_gs)
gs = eigen(H_gs_sp, basis_gs; n = 3, which = :SR)
Eg = gs.values[1]
ψg_in_gs = gs.vectors[:, 1]
println("MOAD Eg = $(round(Eg; digits = 6)) eV")

basis_xas = EagerBasis(h,
    n_fermion([s_2p]) ∈ 5:6,
    n_fermion([s_2p, s_3d, s_Ld]) == 24)

ψg_in_xas = zeros(ComplexF64, length(basis_xas))
for i in 1:length(basis_xas)
    state = get_state(basis_xas, i)
    j = get_index(basis_gs, state)
    j > 0 && (ψg_in_xas[i] = ψg_in_gs[j])
end

# Operators on the larger XAS basis.
H_op       = read_quanty_operator(joinpath(OP_DIR, "Hamiltonian.txt"),     h, map_fn)
H_xas_op   = read_quanty_operator(joinpath(OP_DIR, "XASHamiltonian.txt"),  h, map_fn)
T_x        = read_quanty_operator(joinpath(OP_DIR, "TXASx.txt"),           h, map_fn)
T_y        = read_quanty_operator(joinpath(OP_DIR, "TXASy.txt"),           h, map_fn)
H_final_sp = assemble(compile(H_op,     basis_xas), basis_xas)
H_int_sp   = assemble(compile(H_xas_op, basis_xas), basis_xas)

# ---------------------------------------------------------------------
# Run MOAD rixs on the same grid as PyQuanty / Quanty
# ---------------------------------------------------------------------
# PyQuanty: inc_params(emin=-6, emax=0, ne=11, eta=0.3) → ω_in step 0.6,
#                                                       Γ_int = 2·η = 0.6
#           emit_params(emin=-0.5, emax=8, ne=851, eta=0.05) →
#                                            ω_out step 0.01, Γ_fin = 0.1

ω_in_grid  = range(-6.0, 0.0; length = 11)
ω_out_grid = range(-0.5, 8.0; length = 851)
Γ_int = 0.6
Γ_fin = 0.1

println("Running MOAD rixs() (T_in = T_x, T_out = T_y) over $(length(ω_in_grid)) ω_in × $(length(ω_out_grid)) ω_out points …")
@time moad_result = rixs(H_final_sp, H_int_sp, basis_xas, T_x, T_y, ψg_in_xas;
                         ω_in_grid       = ω_in_grid,
                         ω_out_grid      = ω_out_grid,
                         Γ_intermediate  = Γ_int,
                         Γ_final         = Γ_fin,
                         Eg              = Eg,
                         krylovdim       = 200)
moad_intensity = -imag.(moad_result.tensor)    # (n_ω_in, n_ω_out), real

# ---------------------------------------------------------------------
# Read PyQuanty + Quanty references
# ---------------------------------------------------------------------
# PyQuanty RIXSxy.txt: 851 rows (ω_out), 1 + 2·11 = 23 cols.
#   col 0    : ω_out
#   col 1-2  : Re/Im at ω_in = -6.0
#   col 3-4  : Re/Im at ω_in = -5.4
#   ...
#   col 21-22: Re/Im at ω_in =  0.0
pq_raw = readdlm(joinpath(REF_DIR, "RIXSxy.txt"))
pq_ω_out = pq_raw[:, 1]
pq_intensity = Matrix{Float64}(undef, length(ω_in_grid), length(pq_ω_out))
for k in 1:length(ω_in_grid)
    pq_intensity[k, :] = -pq_raw[:, 2k + 1]   # -Im column for ω_in_idx k
end

# Quanty RIXSxy_Quanty.txt: 5 header lines + 851 rows. 11 spectra → cols
# 1=ω_out, 2/3=Re/Im[0], 4/5=Re/Im[1], …. Same flattening (one spectrum
# per ω_in). Verified by spot check vs PyQuanty.
q_raw = readdlm(joinpath(REF_DIR, "RIXSxy_Quanty.txt"); skipstart = 5)
q_ω_out = q_raw[:, 1]
q_intensity = Matrix{Float64}(undef, length(ω_in_grid), length(q_ω_out))
for k in 1:length(ω_in_grid)
    q_intensity[k, :] = -q_raw[:, 2k + 1]
end

# Sanity: MOAD vs PyQuanty residual at every ω_in.
println("\nResiduals (max|Δχ| / max|χ|) per ω_in:")
println(@sprintf("  %6s   %12s   %12s", "ω_in", "MOAD vs PQ", "Quanty vs PQ"))
for (k, ω_in) in enumerate(ω_in_grid)
    pk = maximum(abs, pq_intensity[k, :])
    pk == 0 && (pk = 1)
    rm = maximum(abs, moad_intensity[k, :] .- pq_intensity[k, :]) / pk
    rq = maximum(abs, q_intensity[k, :]    .- pq_intensity[k, :]) / pk
    println(@sprintf("  %+6.2f   %12.2e   %12.2e", ω_in, rm, rq))
end

# ---------------------------------------------------------------------
# Plot: line cuts at five ω_in values across the L_3 peak
# ---------------------------------------------------------------------
# L_3 sits around ω ≈ -4.5 eV; pick ω_in = -5.4, -4.8, -4.2, -3.6, -3.0
# to span the resonance.
const ω_in_picks = [-5.4, -4.8, -4.2, -3.6, -3.0]
const PLOT_OUT_WINDOW = (-0.5, 7.0)

gr()

panels = Plots.Plot[]
for ω_in_target in ω_in_picks
    k = argmin(abs.(ω_in_grid .- ω_in_target))
    ω_in_actual = ω_in_grid[k]
    keep = findall(ω -> PLOT_OUT_WINDOW[1] ≤ ω ≤ PLOT_OUT_WINDOW[2], ω_out_grid)

    p = plot(ω_out_grid[keep], pq_intensity[k, keep];
             label = "PyQuanty", lw = 2, lc = :black,
             title  = @sprintf("ω_in = %+.2f eV", ω_in_actual),
             xlabel = "ω_out (eV, energy loss above E_g)",
             ylabel = "−Im χ_xy(ω_in, ω_out)",
             framestyle = :box, legend = :topright,
             titlefont = font(10))
    plot!(p, ω_out_grid[keep], q_intensity[k, keep];
          label = "Quanty", lw = 1.5, lc = :red, linestyle = :dash)
    plot!(p, ω_out_grid[keep], moad_intensity[k, keep];
          label = "MOAD",   lw = 1.5, lc = :blue, linestyle = :dot)
    push!(panels, p)
end

# Layout: 5 panels stacked vertically (one column).
fig_lines = plot(panels...; layout = (length(ω_in_picks), 1),
                 size = (900, 1300), dpi = 150)
savefig(fig_lines, joinpath(HERE, "RIXS_xy_line_cuts.png"))
println("\nSaved: ", joinpath(HERE, "RIXS_xy_line_cuts.png"))

# ---------------------------------------------------------------------
# 2D color maps (one per code) for visual side-by-side
# ---------------------------------------------------------------------

keep = findall(ω -> PLOT_OUT_WINDOW[1] ≤ ω ≤ PLOT_OUT_WINDOW[2], ω_out_grid)
ω_out_p = ω_out_grid[keep]

# Use a shared color scale for visual comparability.
vmax = max(maximum(moad_intensity[:, keep]),
           maximum(pq_intensity[:, keep]),
           maximum(q_intensity[:, keep]))

p_pq = heatmap(ω_out_p, ω_in_grid, pq_intensity[:, keep];
               xlabel = "ω_out (eV)", ylabel = "ω_in (eV)",
               title  = "PyQuanty", clim = (0, vmax),
               colorbar = false, framestyle = :box)
p_q  = heatmap(ω_out_p, ω_in_grid, q_intensity[:, keep];
               xlabel = "ω_out (eV)", ylabel = "",
               title  = "Quanty", clim = (0, vmax),
               colorbar = false, framestyle = :box)
p_m  = heatmap(ω_out_p, ω_in_grid, moad_intensity[:, keep];
               xlabel = "ω_out (eV)", ylabel = "",
               title  = "MOAD", clim = (0, vmax),
               colorbar = true, framestyle = :box)

fig_2d = plot(p_pq, p_q, p_m; layout = (1, 3),
              size = (1500, 500), dpi = 150,
              plot_title = "NiO RIXS_xy(ω_in, ω_out) — three-code comparison")
savefig(fig_2d, joinpath(HERE, "RIXS_xy_2d.png"))
println("Saved: ", joinpath(HERE, "RIXS_xy_2d.png"))
println("\nAll RIXS plots generated.")
