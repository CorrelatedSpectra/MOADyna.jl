using Test, MOADyna, HDF5
using MOADyna.Algebra: FermionSite, Hilbert, n, n_fermion
using MOADyna.Bases: EagerBasis, basis_id
using MOADyna.ED: save_eigensystem, load_eigensystem

@testset "save/load_eigensystem — round-trip without basis_id" begin
    # Diagonal 4-mode Hamiltonian on the half-filled (n=2) sector.
    # Eigenvalues are sums of pairs from {1,2,3,4}; vectors are computational
    # basis states. Round-trip equality is exact (no rotation in eigvecs).
    s = FermionSite{4}(:s)
    h = Hilbert(:s => s)
    bas = EagerBasis(h, n_fermion(h) == 2)
    H = 1.0 * n(s, 1) + 2.0 * n(s, 2) + 3.0 * n(s, 3) + 4.0 * n(s, 4)
    sys = eigen(H, bas; n = length(bas))
    mktempdir() do dir
        path = joinpath(dir, "sys.h5")
        save_eigensystem(path, sys)
        loaded = load_eigensystem(path)
        @test loaded.values ≈ sys.values atol = 1e-12
        @test loaded.vectors ≈ sys.vectors atol = 1e-12
        @test loaded.basis_id === nothing
    end
end

@testset "save/load_eigensystem — basis_id round-trips" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    bas = EagerBasis(h)
    H = 1.0 * n(s, 1) - 2.0 * n(s, 2)
    sys = eigen(H, bas; n = length(bas))
    mktempdir() do dir
        # Real basis_id (UInt64 token from the basis itself).
        path = joinpath(dir, "sys.h5")
        save_eigensystem(path, sys; basis_id = basis_id(bas))
        loaded = load_eigensystem(path)
        @test loaded.basis_id == basis_id(bas)
        @test loaded.basis_id isa Unsigned

        # String basis_id is also accepted.
        path2 = joinpath(dir, "sys_str.h5")
        save_eigensystem(path2, sys; basis_id = "tag-xyz")
        loaded2 = load_eigensystem(path2)
        @test loaded2.basis_id == "tag-xyz"
    end
end

@testset "load_eigensystem — bogus version → ArgumentError" begin
    mktempdir() do dir
        path = joinpath(dir, "bad.h5")
        HDF5.h5open(path, "w") do f
            HDF5.attributes(f)["version"] = "999"
            f["values"]  = Float64[]
            f["vectors"] = ComplexF64[]
        end
        err = try
            load_eigensystem(path)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("999", err.msg)
        @test occursin("\"1\"", err.msg)
    end
end
