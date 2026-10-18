using LinearAlgebra
using Printf

@testset "Wannier paths and reference bands" begin
    mktempdir() do dir
        seed = joinpath(dir, "path")
        win = """
            bands_num_points=2
            begin unit_cell_cart
            2 0 0
            0 3 0
            0 0 4
            end unit_cell_cart
            begin kpoint_path
            G 0 0 0 X 0.5 0 0
            X 0.5 0 0 M 0.5 0.5 0
            R 0 0 0.5 G 0 0 0
            end kpoint_path
            """
        write(seed * ".win", win)
        points = [0.0 0.25 0.5 0.5 0.0 0.0; 0.0 0.0 0.0 0.5 0.0 0.0;
                  0.0 0.0 0.0 0.0 0.5 0.0]
        distance = [0.0, π/4, π/2, 5π/6, 5π/6, 13π/12]
        labels = [1 => "G", 3 => "X", 4 => "M", 5 => "R", 6 => "G"]
        path = read_wannier_path(seed)
        @test path.points ≈ points
        @test path.distance ≈ distance
        @test path.breaks == [5]
        @test path.labels == labels
        energies = [-1.0 -0.75 -0.5 -0.5 -1.0 -1.0; 1.0 1.0 1.0 1.5 1.0 1.0]
        open(seed * "_band.dat", "w") do io
            for band in 1:2
                for j in 1:6
                    println(io, distance[j], " ", energies[band, j])
                end
                println(io)
            end
        end
        reference = read_wannier_bands(seed; path)
        @test reference.energies == energies
        @test reference.path === path
        @test read_wannier_bands(seed).path.points === nothing

        open(seed * "_band.kpt", "w") do io
            println(io, 6)
            for j in 1:6
                println(io, join((points[:, j]..., 1.0), " "))
            end
        end
        open(seed * "_band.labelinfo.dat", "w") do io
            for (i, label) in labels
                println(io, join((label, i, distance[i], points[:, i]...), " "))
            end
        end
        write(seed * ".win", replace(win, "bands_num_points=2" => "bands_num_points=5"))
        explicit = read_wannier_path(seed)
        @test explicit.points == points
        @test explicit.distance ≈ distance
        @test explicit.labels == labels
        @test explicit.breaks == [5]

        write(seed * "_band.kpt", "3\n0 0 0 1\n0.166667 0 0 1\n0.333333 0 0 1\n")
        length_k = 2π / 0.04 / 3
        open(seed * "_band.labelinfo.dat", "w") do io
            println(io, "G 1 0 0 0 0")
            @printf(io, "X 3 %.10f 0.3333333333 0 0\n", length_k)
        end
        printed = read_wannier_path(seed; lattice=Diagonal([0.04, 1.0, 1.0]))
        @test printed.points[1, end] == 0.333333
        @test printed.distance ≈ [0.0, length_k/2, length_k] atol=1e-10
        open(seed * "_band.dat", "w") do io
            for x in [0.0, length_k/2, length_k]
                @printf(io, "%16.7E %16.7E\n", x, 1.0)
            end
        end
        @test read_wannier_bands(seed; path=printed).energies == ones(1, 3)

        junction = joinpath(dir, "junction")
        write(junction * ".win", "bands_num_points=100\nbegin kpoint_path\n" *
              "G 0 0 0 X 0.5 0 0\nX 0.5005 0 0 Y 1 0 0\nend kpoint_path\n")
        joined = read_wannier_path(junction; lattice=Matrix{Float64}(I, 3, 3))
        @test joined.points[1, 102] ≈ 0.505495
        @test joined.distance[102] ≈ π + 2π*0.4995/100
    end
end

@testset "DFT reference bands" begin
    mktempdir() do dir
        file = joinpath(dir, "PBAND.dat")
        write(file, """
            #K-Path Energy dz2 tot
            # NKPTS & NBANDS: 3 2
            # Band-Index 7
            0.0 -1.0 0.3 0.5
            0.5 -0.8 0.4 0.6
            1.0 -0.6 0.5 0.7

            # Band-Index 9
            1.0 1.4 0.5 0.6
            0.5 1.2 0.6 0.7
            0.0 1.0 0.7 0.8
            """)
        write(joinpath(dir, "KLABELS"), "K-Label Coordinate\nGAMMA 0.000\nX 1.000\n\n* Tip\n")
        bands = read_vaspkit_bands(file)
        @test bands.energies == [-1.0 -0.8 -0.6; 1.0 1.2 1.4]
        @test bands.path.labels == [1 => "GAMMA", 3 => "X"]
        @test bands.path.points === nothing
        @test bands.metadata[:band_indices] == [7, 9]
        @test bands.metadata[:columns]["dz2"] == [0.3 0.4 0.5; 0.7 0.6 0.5]

        spaghetti = joinpath(dir, "case.spaghetti_ene")
        write(spaghetti, "bandindex: 4\n0 0 0 0.0 0.1\n0.5 0 0 0.5 0.2\n\n" *
                         "bandindex: 5\n0 0 0 0.0 0.3\n0.5 0 0 0.5 0.4\n")
        @test_throws ArgumentError read_wien2k_bands(spaghetti)
        ry = read_wien2k_bands(spaghetti; energy_unit=:Ry)
        @test ry.energies ≈ MOADyna.Units.convert_energy(1, :Ry => :eV) * [0.1 0.2; 0.3 0.4]
        @test ry.path.distance ≈ [0.0, 0.5/MOADyna.Units.bohr_angstrom]
        @test ry.path.points === nothing
        @test ry.metadata[:band_indices] == [4, 5]
        write(joinpath(dir, "case.insp"), """
            ### Figure configuration
            5.0 3.0
            10.0 15.0
            1.0 4
            1.0 1
            1.1 2 4
            ### Data configuration
            -5.0 5.0 2
            1 0.5
            1 999
            0 1 0.02
            """)
        ev = read_wien2k_bands(spaghetti)
        @test ev.energies == [0.1 0.2; 0.3 0.4]
        @test ev.metadata[:energy_reference] == :fermi
        spinfile = joinpath(dir, "case.spaghettiup_ene")
        cp(spaghetti, spinfile)
        @test read_wien2k_bands(spinfile).energies == ev.energies
        insp = read(joinpath(dir, "case.insp"), String)
        write(joinpath(dir, "case.insp"), replace(insp, "-5.0 5.0 2" => "-5.0 5.0 1", "1 0.5" => "0 999.0"))
        @test !haskey(read_wien2k_bands(spaghetti).metadata, :fermi_energy_eV)
        write(joinpath(dir, "case.insp"), replace(insp, "1 0.5" => "0 999.0"))
        @test_throws ArgumentError read_wien2k_bands(spaghetti)
    end
end
