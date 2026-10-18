using LinearAlgebra

function write_hr_fixture(seed, blocks, weights)
    n = size(last(first(blocks)), 1)
    open(seed * "_hr.dat", "w") do io
        println(io, "Synthetic Wannier Hamiltonian")
        println(io, n)
        println(io, length(blocks))
        println(io, join(weights, " "))
        for (R, h) in blocks, j in n:-1:1, i in n:-1:1
            z = h[i, j]
            println(io, join((R..., i, j, real(z), imag(z)), " "))
        end
    end
end

@testset "Wannier Hamiltonians" begin
    mktempdir() do dir
        seed = joinpath(dir, "model")
        h0 = ComplexF64[0.2 0.4im; -0.4im -0.3]
        t = ComplexF64[0.1im 0.3+0.2im; -0.5+0.1im 0.2]
        blocks = [(1, 0, 0) => 2t, (0, 0, 0) => h0, (-1, 0, 0) => 2t']
        write_hr_fixture(seed, blocks, [2, 1, 2])
        write(seed * ".win", """
            num_wann = 2
            use_ws_distance = false
            begin unit_cell_cart
            ang
            2.0 0.0 0.0
            0.7 3.0 0.0
            0.0 0.0 4.0
            end unit_cell_cart
            begin atoms_frac
            Ni 0.1 0.2 0.3
            O 0.5 0.0 0.0
            end atoms_frac
            """)
        k = [0.17, 0.23, 0.11]
        plain = read_wannier90(seed)
        @test bloch_hamiltonian(plain, k) ≈ h0 + cispi(2k[1])*t + cispi(-2k[1])*t'
        @test plain.lattice == [2.0 0.7 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]
        @test plain.metadata[:interpolation] == :plain
        @test plain.metadata[:atoms].labels == ["Ni", "O"]
        @test plain.metadata[:atoms].positions ≈ plain.lattice * [0.1 0.5; 0.2 0; 0.3 0]
        write(seed * ".win", replace(read(seed * ".win", String), "ang" => "bohr"))
        @test read_wannier90(seed).lattice ≈ 0.529177210903 * plain.lattice
        write(seed * ".win", replace(read(seed * ".win", String), "bohr" => "ang"))
        fractional_input = read(seed * ".win", String)
        write(seed * ".win", replace(fractional_input,
            "atoms_frac" => "atoms_cart", "Ni 0.1" => "bohr\nNi 0.1"))
        @test read_wannier90(seed).metadata[:atoms].positions ≈
              0.529177210903 * [0.1 0.5; 0.2 0; 0.3 0]
        write(seed * ".win", fractional_input)

        tbseed = joinpath(dir, "combined")
        write(tbseed * ".win", "use_ws_distance=false\n")
        open(tbseed * "_tb.dat", "w") do io
            println(io, "Synthetic combined Wannier output\n1 0 0\n0 2 0\n0 0 3\n2\n1\n1\n\n0 0 0")
            for j in 2:-1:1, i in 2:-1:1
                println(io, join((i, j, real(h0[i,j]), imag(h0[i,j])), " "))
            end
            println(io, "\n0 0 0")
            for j in 2:-1:1, i in 2:-1:1
                x, y = i == j == 2 ? (0.25, 1.0) : (0.0, 0.0)
                println(io, join((i, j, x, 0.0, y, 0.0, 0.0, 0.0), " "))
            end
        end
        combined = read_wannier90(tbseed * "_tb.dat")
        @test hopping(combined, (0, 0, 0)) == h0
        @test combined.lattice == Diagonal([1.0, 2.0, 3.0])
        @test combined.centres == [0.0 0.25; 0.0 0.5; 0.0 0.0]

        write(seed * ".win", replace(read(seed * ".win", String), "false" => "true"))
        open(seed * "_wsvec.dat", "w") do io
            println(io, "# use_ws_distance=.true.")
            for (R, _) in blocks, i in 1:2, j in 1:2
                println(io, join((R..., i, j), " "))
                shifts = R == (0, 0, 0) ? [(0, 0, 0)] : [(0, 0, 0), (-2R[1], 0, 0)]
                println(io, length(shifts))
                for shift in shifts
                    println(io, join(shift, " "))
                end
            end
        end
        shifted = read_wannier90(seed)
        expected = h0 + cospi(2k[1]) * (t + t')
        @test bloch_hamiltonian(shifted, k) ≈ expected
        @test shifted.metadata[:interpolation] == :wsvec

        expanded = joinpath(dir, "expanded")
        write_hr_fixture(expanded, [(0, 0, 0) => h0, (1, 0, 0) => (t+t')/2,
                                    (-1, 0, 0) => (t+t')/2], [1, 1, 1])
        write(expanded * "_wsvec.dat", "# use_ws_distance=.true. write_ndegen_applied=.true.\n99 0 0 1 1\n1\n0 0 0\n")
        @test bloch_hamiltonian(read_wannier90(expanded), k) ≈ expected
        @test read_wannier90(expanded).metadata[:interpolation] == :expanded

        geometric = joinpath(dir, "geometric")
        write_hr_fixture(geometric, [(0, 0, 0) => h0], [1])
        write(geometric * ".win", """
            num_wann=2
            use_ws_distance=true
            translate_home_cell=false
            mp_grid=1 1 1
            begin unit_cell_cart
            1 0 0
            10 0.4 0
            0 0 1
            end unit_cell_cart
            """)
        write(geometric * "_centres.xyz", "2\nWannier centres\nX 0 0 0\nX 4.9 0.196 0\n")
        recovered = read_wannier90(geometric)
        @test bloch_hamiltonian(recovered, k) ≈ [0.2 0.4im*cispi(-10k[1]); -0.4im*cispi(10k[1]) -0.3]
        @test recovered.centres ≈ [0.0 0.0; 0.0 0.49; 0.0 0.0] atol=1e-14
        @test recovered.metadata[:interpolation] == :reconstructed
        write(geometric * ".win", replace(read(geometric * ".win", String),
                                          "translate_home_cell=false" => "translate_home_cell=true"))
        @test_throws ArgumentError read_wannier90(geometric)
        write(geometric * ".wout", """
            Release: 3.1.0
            Final State
            WF centre and spread 1 ( 0.0, 0.0, 0.0 ) 1.0
            WF centre and spread 2 ( 4.9, 0.196, 0.0 ) 1.0
            """)
        @test bloch_hamiltonian(read_wannier90(geometric), k) ≈ bloch_hamiltonian(recovered, k)

        write(geometric * ".win", replace(read(geometric * ".win", String),
                                          "use_ws_distance=true" => "use_ws_distance=false"))
        open(geometric * "_wsvec.dat", "w") do io
            println(io, "# use_ws_distance=.false.")
            for i in 1:2, j in 1:2
                println(io, "0 0 0 $i $j\n1\n0 0 0")
            end
        end
        @test bloch_hamiltonian(read_wannier90(geometric; interpolation=:ws), k) ≈
              bloch_hamiltonian(recovered, k)
        write(geometric * "_wsvec.dat", "# use_ws_distance=foo\n")
        @test_throws ArgumentError read_wannier90(geometric)
        rm(geometric * "_wsvec.dat")
        open(geometric * ".win", "a") do io
            println(io, "bands_plot=true\nbands_plot_mode=cut")
        end
        @test read_wannier90(geometric).centres === nothing

        bare = joinpath(dir, "bare")
        write_hr_fixture(bare, [(0, 0, 0) => h0], [1])
        @test_throws ArgumentError read_wannier90(bare)
        @test hopping(read_wannier90(bare; interpolation=:plain), (0, 0, 0)) == h0
        write(bare * "_centres.xyz", "2\nUnknown wrapping\nX 0 0 0\nX 0.2 0 0\n")
        @test read_wannier90(bare; interpolation=:plain, lattice=Matrix{Float64}(I, 3, 3)).centres === nothing
    end
end
