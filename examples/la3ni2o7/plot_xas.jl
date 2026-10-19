using DelimitedFiles
using Plots

function plot_xas(output)
    data = readdlm(joinpath(output, "xas.txt"))
    omega = data[:, 1]
    plane = (data[:, 2] + data[:, 3])/2
    normal = data[:, 4]
    p = plot(omega, plane; label="E ∥ ab", color="#332288", linewidth=2.2,
             xlabel="Relative photon energy (eV)", ylabel="Intensity (arb. units)",
             size=(800, 450), guidefontsize=14, tickfontsize=12, legendfontsize=11,
             legend=:outertop, legend_columns=2, foreground_color_legend=:transparent,
             framestyle=:box, tick_direction=:in, grid=false, margin=3Plots.mm,
             widen=false, xlims=(-10, 25), xticks=-10:5:25,
             ylims=(0, 0.10), yticks=0:0.02:0.10)
    plot!(p, omega, normal; label="E ∥ c", color="#44AA99", linewidth=2.2, linestyle=:dash)
    savefig(p, joinpath(output, "xas.svg"))
    savefig(p, joinpath(output, "xas.pdf"))
    return p
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_xas(isempty(ARGS) ? "la3ni2o7-results" : ARGS[1])
end
