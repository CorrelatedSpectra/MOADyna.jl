# =====================================================================
# NiO L_{2,3} XAS — apples-to-apples comparison plot (MOAD ↔ PyQuanty ↔ Quanty)
# =====================================================================
#
# Re-runs MOAD's Phase 7 NiO XAS calculation and compares the resulting
# spectrum tensor to the two reference outputs:
#
#   - PyQuanty: `XAS_lanczos_cont_frac.txt` (Lanczos cont-frac on the
#     QuSpin sparse Hamiltonian; col 1 = ω, cols 2-3 / 4-5 / 6-7 = Re/Im
#     of XAS_x / XAS_y / XAS_z for ψ_g[1]).
#
#   - Quanty: `XAS_Quanty.txt` (CreateSpectra; same conventions, but
#     with 9 spectra = 3 polarisations × 3 ψ_g; we plot only ψ_g[1]:
#     Re/Im pair 0 / 1 / 2 → cols 2-3 / 4-5 / 6-7).
#
# Output: three figures (XAS_x.png, XAS_y.png, XAS_z.png) in this
# directory. Each figure has a top panel with all three spectra
# overlaid (-Im of the complex tensor → physical absorption intensity)
# and a bottom panel showing residuals MOAD-PyQuanty and Quanty-PyQuanty.
#
# Run from the package root:
#   julia --project=docs/dev/validation/spectroscopy/nio_xas/plots \
#         docs/dev/validation/spectroscopy/nio_xas/plots/make_xas_comparison.jl

using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Printf
using MOAD
using MOAD: xas, eigen
using MOAD.Spectroscopy: polarise
using Plots

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

const HERE   = @__DIR__
const NIO    = dirname(HERE)
const OP_DIR  = joinpath(NIO, "operators")
const REF_DIR = joinpath(NIO, "reference")

# ---------------------------------------------------------------------
# Site / mode layout — must match Phase 7 test
# ---------------------------------------------------------------------

s_2p = FermionSite{6}(:p)
s_3d = FermionSite{10}(:d)
s_Ld = FermionSite{10}(:L)
h    = Hilbert(:p => s_2p, :d => s_3d, :L => s_Ld)
map_fn = i -> i < 6  ? (:p, i + 1) :
              i < 16 ? (:d, i - 5) :
                       (:L, i - 15)

# ---------------------------------------------------------------------
# Step 1: GS — match PyQuanty's loose basisGS
# ---------------------------------------------------------------------

basis_gs = EagerBasis(h,
    n_fermion([s_2p]) == 6,
    n_fermion([s_2p, s_3d, s_Ld]) == 24)
H_gs_op = read_quanty_operator(joinpath(OP_DIR, "Hamiltonian.txt"), h, map_fn)
H_gs_sp = assemble(compile(H_gs_op, basis_gs), basis_gs)

gs = eigen(H_gs_sp, basis_gs; n = 3, which = :SR)
Eg = gs.values[1]
ψg_in_gs = gs.vectors[:, 1]
println("MOAD Eg = $(round(Eg; digits = 6)) eV")

# ---------------------------------------------------------------------
# Step 2: XAS basis — covers (n_p=6) and (n_p=5) sectors; embed ψ_g
# ---------------------------------------------------------------------

basis_xas = EagerBasis(h,
    n_fermion([s_2p]) ∈ 5:6,
    n_fermion([s_2p, s_3d, s_Ld]) == 24)

ψg_in_xas = zeros(ComplexF64, length(basis_xas))
for i in 1:length(basis_xas)
    state = get_state(basis_xas, i)
    j = get_index(basis_gs, state)
    j > 0 && (ψg_in_xas[i] = ψg_in_gs[j])
end

H_xas_op = read_quanty_operator(joinpath(OP_DIR, "XASHamiltonian.txt"), h, map_fn)
H_xas_sp = assemble(compile(H_xas_op, basis_xas), basis_xas)

T_x = read_quanty_operator(joinpath(OP_DIR, "TXASx.txt"), h, map_fn)
T_y = read_quanty_operator(joinpath(OP_DIR, "TXASy.txt"), h, map_fn)
T_z = read_quanty_operator(joinpath(OP_DIR, "TXASz.txt"), h, map_fn)

# ---------------------------------------------------------------------
# Step 3: MOAD scalar XAS for x, y, z polarisations
# ---------------------------------------------------------------------

Γ  = 0.6
ωs = range(-15.0, 25.0; length = 801)

println("Running MOAD xas() for x / y / z polarisations …")
moad_x = xas(H_xas_sp, basis_xas, T_x, ψg_in_xas; ω_grid = ωs, Γ = Γ, Eg = Eg, krylovdim = 200)
moad_y = xas(H_xas_sp, basis_xas, T_y, ψg_in_xas; ω_grid = ωs, Γ = Γ, Eg = Eg, krylovdim = 200)
moad_z = xas(H_xas_sp, basis_xas, T_z, ψg_in_xas; ω_grid = ωs, Γ = Γ, Eg = Eg, krylovdim = 200)

moad_intensity = (
    x = -imag.(moad_x.tensor),
    y = -imag.(moad_y.tensor),
    z = -imag.(moad_z.tensor),
)

# ---------------------------------------------------------------------
# Step 4: read PyQuanty + Quanty references
# ---------------------------------------------------------------------

# PyQuanty: 7 columns, no header. ω, Re_x, Im_x, Re_y, Im_y, Re_z, Im_z.
pq = readdlm(joinpath(REF_DIR, "XAS_lanczos_cont_frac.txt"))
pq_ω = pq[:, 1]
pq_intensity = (
    x = -pq[:, 3],   # -Im of complex spectrum
    y = -pq[:, 5],
    z = -pq[:, 7],
)

# Quanty: 5 header lines (`#Spectra:` etc.) then 801 data rows. The data
# row layout is `ω + 18 cols (Re/Im for 9 spectra)`, where the spectra
# are flattened in **operator-outer, ψ-inner** order:
#   spec 0/1/2 = (T_x, ψ_g[1..3])
#   spec 3/4/5 = (T_y, ψ_g[1..3])
#   spec 6/7/8 = (T_z, ψ_g[1..3])
# So the (T_pol, ψ_g[1]) pair we want sits at spec indices 0, 3, 6 →
# Im column = 1 + 2·spec_idx + 2 = 3, 9, 15.
# Verified empirically: residuals to PyQuanty's x/y/z drop to 1e-13 at
# exactly these columns; off-mapping gives 20-30% residuals.
q = readdlm(joinpath(REF_DIR, "XAS_Quanty.txt"); skipstart = 5)
q_ω = q[:, 1]
q_intensity = (
    x = -q[:, 3],     # spec 0 = (T_x, ψ_g[1])
    y = -q[:, 9],     # spec 3 = (T_y, ψ_g[1])
    z = -q[:, 15],    # spec 6 = (T_z, ψ_g[1])
)

# ---------------------------------------------------------------------
# Step 5: plot — for each polarisation, overlay + residual panel
# ---------------------------------------------------------------------

# Use the L_{2,3} XAS interesting region for the plot window (cropped
# from the wider [-15, 25] computational grid). Window must capture
# BOTH L₃ and L₂ peaks: with ζ_2p = 11.51 eV (Quanty Lua) the j=1/2
# vs j=3/2 splitting is (3/2)·ζ_2p ≈ 17.3 eV, so L₃ around ω = −4.5
# eV pairs with L₂ around ω = +12.8 eV. Crop generously to [-7, 16].
const PLOT_WINDOW = (-7.0, 16.0)

function plot_one_polarisation(label::String, moad_y, pq_y, q_y, ω, savepath::AbstractString)
    keep = findall(ω -> PLOT_WINDOW[1] ≤ ω ≤ PLOT_WINDOW[2], ω)
    ωp = ω[keep]
    m  = moad_y[keep]
    p  = pq_y[keep]
    q  = q_y[keep]

    upper = plot(ωp, p; label = "PyQuanty", lw = 2, lc = :black,
                 xlabel = "", ylabel = "−Im χ$(label)$(label)(ω)",
                 title  = "NiO L₂,₃ XAS — polarisation $(label)$(label)",
                 framestyle = :box, legend = :topleft)
    plot!(upper, ωp, q; label = "Quanty",   lw = 1.5, lc = :red,    linestyle = :dash)
    plot!(upper, ωp, m; label = "MOAD",     lw = 1.5, lc = :blue,   linestyle = :dot)

    # Residuals (MOAD − PyQuanty, Quanty − PyQuanty).
    peak = maximum(abs, p)
    res_moad = (m .- p) ./ peak
    res_quan = (q .- p) ./ peak
    lower = plot(ωp, res_moad; label = "MOAD − PyQuanty", lw = 1.5, lc = :blue,
                 xlabel = "ω (eV, above E_g)",
                 ylabel = "(Δχ) / max|χ|",
                 framestyle = :box, legend = :topright)
    plot!(lower, ωp, res_quan; label = "Quanty − PyQuanty", lw = 1.5, lc = :red, linestyle = :dash)
    hline!(lower, [0.0]; label = nothing, lc = :black, lw = 0.5, alpha = 0.5)

    fig = plot(upper, lower; layout = grid(2, 1; heights = [0.65, 0.35]),
               size = (900, 600), dpi = 150)
    savefig(fig, savepath)
    println("Saved: $(savepath)")
    # Print residual statistics for the headline numbers.
    @printf("  max|MOAD − PyQuanty| / peak = %.2e\n", maximum(abs, res_moad))
    @printf("  max|Quanty − PyQuanty| / peak = %.2e\n", maximum(abs, res_quan))
end

# Backend: GR (default; works headless without X server).
gr()

println("\nGenerating plots …")
plot_one_polarisation("x", moad_intensity.x, pq_intensity.x, q_intensity.x, ωs,
                      joinpath(HERE, "XAS_x.png"))
plot_one_polarisation("y", moad_intensity.y, pq_intensity.y, q_intensity.y, ωs,
                      joinpath(HERE, "XAS_y.png"))
plot_one_polarisation("z", moad_intensity.z, pq_intensity.z, q_intensity.z, ωs,
                      joinpath(HERE, "XAS_z.png"))

# ---------------------------------------------------------------------
# Step 6: combined isotropic spectrum (sum over polarisations)
# ---------------------------------------------------------------------

moad_iso = moad_intensity.x .+ moad_intensity.y .+ moad_intensity.z
pq_iso   = pq_intensity.x   .+ pq_intensity.y   .+ pq_intensity.z
q_iso    = q_intensity.x    .+ q_intensity.y    .+ q_intensity.z

keep = findall(ω -> PLOT_WINDOW[1] ≤ ω ≤ PLOT_WINDOW[2], ωs)
ωp = ωs[keep]
upper = plot(ωp, pq_iso[keep]; label = "PyQuanty", lw = 2, lc = :black,
             xlabel = "", ylabel = "−Im (χ_xx + χ_yy + χ_zz)",
             title  = "NiO L₂,₃ XAS — isotropic (x + y + z)",
             framestyle = :box, legend = :topleft)
plot!(upper, ωp, q_iso[keep];   label = "Quanty", lw = 1.5, lc = :red,  linestyle = :dash)
plot!(upper, ωp, moad_iso[keep]; label = "MOAD",  lw = 1.5, lc = :blue, linestyle = :dot)

peak = maximum(abs, pq_iso[keep])
lower = plot(ωp, (moad_iso[keep] .- pq_iso[keep]) ./ peak;
             label = "MOAD − PyQuanty", lw = 1.5, lc = :blue,
             xlabel = "ω (eV, above E_g)", ylabel = "Δχ / max|χ|",
             framestyle = :box, legend = :topright)
plot!(lower, ωp, (q_iso[keep] .- pq_iso[keep]) ./ peak;
      label = "Quanty − PyQuanty", lw = 1.5, lc = :red, linestyle = :dash)
hline!(lower, [0.0]; label = nothing, lc = :black, lw = 0.5, alpha = 0.5)

fig = plot(upper, lower; layout = grid(2, 1; heights = [0.65, 0.35]),
           size = (900, 600), dpi = 150)
savefig(fig, joinpath(HERE, "XAS_isotropic.png"))
println("Saved: ", joinpath(HERE, "XAS_isotropic.png"))

println("\nAll plots generated. See $(HERE)")
