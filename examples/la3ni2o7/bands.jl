# Run with the downloaded data directory and an optional output directory.
# julia --project --threads=4 examples/la3ni2o7/bands.jl la3ni2o7-wannier results
include("model.jl")
using Plots
using DelimitedFiles

"""Read this tetragonal calculation's WIEN2k k list in the Wannier reciprocal cell."""
function dft_path(datadir, model)
    cell = parse.(Float64, split(readlines(joinpath(datadir, "model.struct"))[4]))
    @assert length(cell) == 6 && cell[1] == cell[2] && cell[4:6] == [90, 90, 90]
    lengths = cell[1:3] .* MOADyna.Units.bohr_angstrom
    reciprocal = 2π * Diagonal(1 ./ lengths)
    coordinates = Vector{Float64}[]
    labels = Pair{Int,String}[]
    for line in eachline(joinpath(datadir, "dft.klist_band"))
        startswith(strip(line), "END") && break
        integers = [parse(Int, line[i:i+4]) for i in (11, 16, 21, 26)]
        push!(coordinates, integers[1:3] ./ integers[4])
        label = strip(line[1:10])
        isempty(label) || push!(labels, length(coordinates) => (label == "G" ? "Γ" : label))
    end
    cartesian = reciprocal * reduce(hcat, coordinates)
    points = model.lattice' * cartesian / (2π)
    distance = vcat(0.0, cumsum(norm.(eachcol(diff(cartesian; dims=2)))))
    return KPath(points, distance; labels), cartesian
end

function plot_bands(datadir, output)
    mkpath(output)
    model, _ = load_wannier(datadir)
    path, cartesian = dft_path(datadir, model)
    dft = read_wien2k_bands(joinpath(datadir, "dft.spaghetti_ene");
                            energy_unit=:eV, path, atol=2e-5)
    # The source prints Cartesian k coordinates in bohr⁻¹ to five decimals.
    @assert maximum(abs, cartesian .* MOADyna.Units.bohr_angstrom -
                         dft.metadata[:source_kpoints]) < 1e-5
    computed = bandstructure(model, path)
    wannier_path = read_wannier_path(joinpath(datadir, "model"); lattice=model.lattice)
    reference = read_wannier_bands(joinpath(datadir, "model"); path=wannier_path)
    check = bandstructure(model, wannier_path)
    discrepancy = maximum(abs, check.energies - reference.energies)
    @info "Interpolation check" maximum_difference_eV=discrepancy
    @assert discrepancy < 1e-4

    ticks = (path.distance[first.(path.labels)], last.(path.labels))
    p = plot(; size=(850, 500), framestyle=:box, tick_direction=:in,
             guidefontsize=14, tickfontsize=12, legendfontsize=11,
             legend=:outertop, legend_columns=3, foreground_color_legend=:transparent,
             margin=3Plots.mm, grid=false, widen=false)
    vline!(p, ticks[1]; label="", color=:gray85, linewidth=0.6)
    hline!(p, [0]; label="", color=:gray70, linewidth=0.7)
    plot!(p, dft; label="DFT", color="#999999", linewidth=0.9)
    plot!(p, computed; label="MOADyna", color="#44AA99", linewidth=1.5)
    # Sparse reference markers keep coincident curves legible (Paul Tol muted colours).
    sampled = 1:90:length(reference.path.distance)
    markers = BandStructure(KPath(nothing, reference.path.distance[sampled]),
                            reference.energies[:, sampled])
    plot!(p, markers; label="Wannier90", seriestype=:scatter, markershape=:circle,
          markersize=2.1, markercolor=:white, markerstrokecolor="#332288",
          markerstrokewidth=0.7)
    plot!(p; xlabel="", ylabel="Energy (eV)", xticks=ticks,
          xlims=(first(path.distance), last(path.distance)), ylims=(-8, 4), yticks=-8:2:4)
    savefig(p, joinpath(output, "bands.svg"))
    savefig(p, joinpath(output, "bands.pdf"))
    writedlm(joinpath(output, "bands.txt"), hcat(path.distance, computed.energies'))
    return p
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("supply the la3ni2o7-wannier data directory")
    plot_bands(ARGS[1], length(ARGS) >= 2 ? ARGS[2] : "la3ni2o7-results")
end
