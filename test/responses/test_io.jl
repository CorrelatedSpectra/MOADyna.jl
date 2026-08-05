# =====================================================================
# MOAD.Responses — HDF5 round-trip per-type tests
# =====================================================================
#
# Round-trip coverage for every concrete `AbstractResponse` type: write to a
# temp file, load back, and confirm (a) the loaded object is field-equal to the
# original (`_resp_equal`) and (b) `R(ω)` evaluates identically on a sample grid.
# A round-trip has one property — `load(save(x)) ≈ x` — so each case is one
# equality check plus the numerical R(ω) check, not a field-by-field litany.
#
# Also covers the version-1 → version-2 fallback (a hand-built file lacking the
# `/converged` dataset must load as `converged = true`).

@testset "Responses I/O — round-trip per type" begin
    using HDF5
    using MOAD: LanczosResponse, PoleResponse, GridResponse, GreensFunction
    using MOAD.Responses: save_response, load_response

    # --- builders: small but non-trivial responses of each concrete type ---

    function _make_lanczos_K1(; converged::Bool = true, T::Type = ComplexF64,
                              prefactor = one(T), sign::Int = 1)
        α = [Matrix{T}([1.0  0.5; 0.5  2.0])]
        β = Matrix{T}[]
        R = Matrix{T}([1.0  0.0; 0.0  1.0])
        return LanczosResponse(α, β, R, 0.0, 0.1, sign, prefactor, converged)
    end

    function _make_lanczos_K3(; T::Type = ComplexF64)
        α = [Matrix{T}([1.0 0.2; 0.2 1.5]),
             Matrix{T}([2.0 0.0; 0.0 2.5]),
             Matrix{T}([3.0 0.1; 0.1 3.5])]
        β = [Matrix{T}([0.3 0.0; 0.0 0.4]),
             Matrix{T}([0.5 0.0; 0.0 0.6])]
        R = Matrix{T}([1.0 0.0; 0.0 1.0])
        return LanczosResponse(α, β, R, -0.7, 0.05, 1, T(1.5), true)
    end

    function _make_pole(; T::Type = ComplexF64)
        a0       = zeros(T, 2, 2)
        poles    = Float64[1.0, 2.5, 4.0]
        residues = [Matrix{T}([0.5 0.1; 0.1 0.4]),
                    Matrix{T}([0.3 0.0; 0.0 0.2]),
                    Matrix{T}([0.1 0.05; 0.05 0.15])]
        return PoleResponse{T}(a0, poles, residues, -0.7, 0.05)
    end

    # GridResponse{T, N}: last data axis is ω. N=2 ⇒ (M, nω); N=3 ⇒ (M, M', nω).
    function _make_grid_2d(; T::Type = ComplexF64)
        ω    = collect(0.0:0.1:5.0)
        data = T.(reshape(0.1 .+ 0.01 .* (1:2*length(ω)), 2, length(ω)))
        return GridResponse{T, 2}(data, ω, -0.7, 0.05)
    end

    function _make_grid_3d(; T::Type = ComplexF64)
        ω    = collect(0.0:0.5:2.0)
        data = T.(reshape(0.1 .+ 0.01 .* (1:2*3*length(ω)), 2, 3, length(ω)))
        return GridResponse{T, 3}(data, ω, -0.7, 0.05)
    end

    _sample_ωs() = Float64[0.0, 0.5, 1.0, 1.5, 2.0]

    _eval_close(R, R_loaded, ωs) =
        maximum(ω -> maximum(abs.(R(ω) .- R_loaded(ω))), ωs)

    # --- field-by-field equality per concrete type (recurses into GF) ---
    _resp_equal(a, b) = false
    _resp_equal(a::LanczosResponse, b::LanczosResponse) =
        typeof(a) === typeof(b) && a.α == b.α && a.β == b.β && a.R == b.R &&
        a.Eg == b.Eg && a.Γ == b.Γ && a.sign == b.sign &&
        a.prefactor == b.prefactor && a.converged == b.converged
    _resp_equal(a::PoleResponse, b::PoleResponse) =
        typeof(a) === typeof(b) && a.a0 == b.a0 && a.poles == b.poles &&
        a.residues == b.residues && a.Eg == b.Eg && a.Γ == b.Γ
    _resp_equal(a::GridResponse, b::GridResponse) =
        typeof(a) === typeof(b) && a.data == b.data && a.ω == b.ω &&
        a.Eg == b.Eg && a.Γ == b.Γ
    function _resp_equal(a::GreensFunction, b::GreensFunction)
        chan_eq(x, y) = (x === nothing && y === nothing) ||
                        (x !== nothing && y !== nothing && _resp_equal(x, y))
        return chan_eq(a.addition, b.addition) && chan_eq(a.removal, b.removal)
    end

    # Save → load → assert field-equal + R(ω) identical. One helper, every type.
    function _roundtrip(orig; evalωs = _sample_ωs(), tol = 1e-12)
        path = tempname() * ".h5"
        try
            save_response(path, orig)
            loaded = load_response(path)
            @test _resp_equal(loaded, orig)
            @test _eval_close(orig, loaded, evalωs) < tol
            return loaded
        finally
            isfile(path) && rm(path)
        end
    end

    # --- LanczosResponse: K=1, K=3, sign=-1, Float32/ComplexF32 ---
    _roundtrip(_make_lanczos_K1(converged = true))
    _roundtrip(_make_lanczos_K1(converged = false))   # converged flag preserved
    _roundtrip(_make_lanczos_K3())
    _roundtrip(_make_lanczos_K1(sign = -1))           # sign=-1 (format v2)
    _roundtrip(_make_lanczos_K1(T = Float32); tol = 1e-5)
    _roundtrip(_make_lanczos_K1(T = ComplexF32, prefactor = ComplexF32(1, 0.5)); tol = 1e-5)

    # --- PoleResponse / GridResponse ---
    _roundtrip(_make_pole())
    _roundtrip(_make_grid_2d(); evalωs = _make_grid_2d().ω, tol = 1e-14)
    _roundtrip(_make_grid_3d(); evalωs = _make_grid_3d().ω, tol = 1e-14)

    # --- GreensFunction: each inner rep + nothing-slot cases ---
    let L_add = _make_lanczos_K1(converged = true),
        L_rem = _make_lanczos_K1(sign = -1, converged = false)
        _roundtrip(GreensFunction(L_add, L_rem))               # both channels
        _roundtrip(GreensFunction(L_add, nothing))             # addition-only
        _roundtrip(GreensFunction(nothing, L_rem))             # removal-only
    end
    _roundtrip(GreensFunction(_make_pole(), _make_pole()))
    _roundtrip(GreensFunction(_make_grid_2d(), _make_grid_2d());
               evalωs = _make_grid_2d().ω, tol = 1e-14)

    # --- v1 → v2 fallback: hand-built version-"1" file (no /converged) ---
    @testset "v1 fallback: missing /converged loads as converged = true" begin
        path = tempname() * ".h5"
        try
            HDF5.h5open(path, "w") do f
                HDF5.attributes(f)["version"]  = "1"
                HDF5.attributes(f)["type_tag"] = "LanczosResponse"
                HDF5.attributes(f)["eltype"]   = "ComplexF64"
                α_grp = HDF5.create_group(f, "alpha")
                α_grp["block_1"] = ComplexF64[1.0+0im 0.5+0im; 0.5+0im 2.0+0im]
                HDF5.create_group(f, "beta")           # empty (K=1)
                f["R"]            = ComplexF64[1.0+0im 0; 0 1.0+0im]
                f["Eg"]           = 0.0
                f["Gamma"]        = 0.1
                f["sign"]         = 1
                f["prefactor_re"] = 1.0
                f["prefactor_im"] = 0.0
            end
            L = load_response(path)
            @test L isa LanczosResponse && L.converged == true   # default for v1
            @test _resp_equal(L, _make_lanczos_K1(converged = true))
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "load_response rejects unrecognized version" begin
        path = tempname() * ".h5"
        try
            HDF5.h5open(path, "w") do f
                HDF5.attributes(f)["version"]  = "99"
                HDF5.attributes(f)["type_tag"] = "LanczosResponse"
            end
            @test_throws ArgumentError load_response(path)
        finally
            isfile(path) && rm(path)
        end
    end

    @test MOAD.Responses._RESPONSES_IO_FORMAT_VERSION == "2"
end
