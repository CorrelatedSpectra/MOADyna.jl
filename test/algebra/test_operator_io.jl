using Test, MOADyna, HDF5
using MOADyna.Algebra: FermionSite, BosonSite, SpinSite, Hilbert,
                    c, cdag, n, b, bdag, Sx,
                    OperatorSum, OperatorTerm, LadderEntry

@testset "save/load_operator — round-trip" begin
    s1 = FermionSite{3}(:s1)
    s2 = FermionSite{2}(:s2)
    h  = Hilbert(:s1 => s1, :s2 => s2)
    op = 0.7 * cdag(s1, 1) * c(s2, 1) +
         (1.2 + 0.5im) * cdag(s1, 2) * cdag(s2, 1) * c(s2, 2) * c(s1, 3) +
         0.3 * n(s1, 1)
    mktempdir() do dir
        path = joinpath(dir, "op.h5")
        save_operator(path, op)
        s1_new = FermionSite{3}(:s1)
        s2_new = FermionSite{2}(:s2)
        h_new  = Hilbert(:s1 => s1_new, :s2 => s2_new)
        op_loaded = load_operator(path, h_new)
        @test op_loaded == op
    end
end

@testset "save/load_operator — identity term round-trips" begin
    s = FermionSite{2}(:s)
    h = Hilbert(:s => s)
    op = 2.0 + 0.5 * n(s, 1)        # has empty-chain (identity) term
    mktempdir() do dir
        path = joinpath(dir, "op.h5")
        save_operator(path, op)
        s_new = FermionSite{2}(:s)
        h_new = Hilbert(:s => s_new)
        op_loaded = load_operator(path, h_new)
        @test op_loaded == op
    end

    # Pure-scalar OperatorSum (only the identity term, no ladder ops).
    op_scalar = 2.5 * one(OperatorSum{ComplexF64})
    mktempdir() do dir
        path = joinpath(dir, "scalar.h5")
        save_operator(path, op_scalar)
        s_new = FermionSite{2}(:s)
        h_new = Hilbert(:s => s_new)
        @test load_operator(path, h_new) == op_scalar
    end

    # Truly-empty OperatorSum (zero terms) — pins the Vector{ComplexF64}(undef, 0) writer path.
    op_empty = OperatorSum{ComplexF64}()
    mktempdir() do dir
        path = joinpath(dir, "empty.h5")
        save_operator(path, op_empty)
        s_new = FermionSite{2}(:s)
        h_new = Hilbert(:s => s_new)
        @test load_operator(path, h_new) == op_empty
    end
end

@testset "save_operator — non-fermion sites rejected" begin
    s_f = FermionSite{2}(:f)
    s_b = BosonSite{3}(:bos)
    # Build a hand-crafted OperatorSum with a bosonic ladder entry to exercise
    # the v0.1 guard (the public API for BosonSite ladders is `b`/`bdag`).
    op = bdag(s_b) * b(s_b)
    mktempdir() do dir
        path = joinpath(dir, "op.h5")
        err = try
            save_operator(path, op)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("FermionSite", err.msg)
    end

    # Also reject a SpinSite ladder
    s_s = SpinSite{1//2}(:spin)
    op2 = Sx(s_s, 1//2)
    mktempdir() do dir
        path = joinpath(dir, "op.h5")
        err = try
            save_operator(path, op2)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("FermionSite", err.msg)
    end

    # A pure-fermion OperatorSum saves fine.
    op_ok = n(s_f, 1)
    mktempdir() do dir
        path = joinpath(dir, "op.h5")
        save_operator(path, op_ok)
        @test isfile(path)
    end
end

@testset "load_operator — bogus version → ArgumentError" begin
    mktempdir() do dir
        path = joinpath(dir, "bad.h5")
        HDF5.h5open(path, "w") do f
            HDF5.attributes(f)["version"] = "999"
            HDF5.create_group(f, "chains")
        end
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        err = try
            load_operator(path, h)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("999", err.msg)
        @test occursin("\"1\"", err.msg)
    end
end

@testset "load_operator — missing site name → KeyError" begin
    s = FermionSite{2}(:s_orig)
    h_orig = Hilbert(:s_orig => s)
    op = n(s, 1)
    mktempdir() do dir
        path = joinpath(dir, "op.h5")
        save_operator(path, op)
        s_other = FermionSite{2}(:s_other)
        h_other = Hilbert(:s_other => s_other)
        @test_throws KeyError load_operator(path, h_other)
    end
end
