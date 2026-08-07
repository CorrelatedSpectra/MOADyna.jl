# =====================================================================
# MOADyna.Responses — correlator(...) kernel correctness + API contract
# =====================================================================
#
# Tests:
#   1. Kernel — Heisenberg dimer dynamic structure factor.
#   2. Kernel — Hubbard dimer neutral density-density correlator.
#   3. API contract — all error paths.

using Test
using LinearAlgebra
using SparseArrays
import Logging
using MOADyna
using MOADyna: LanczosResponse, GridResponse, PoleResponse
using MOADyna.Responses: to_pole, to_grid

# ---------------------------------------------------------------------------
# Dense reference correlator: χ(ω) = ⟨ψ₀|O†(ω·I + Eg − H + iΓ/2)⁻¹ O|ψ₀⟩
# Matches the correlator.jl convention z = ω + Eg + iΓ/2.
# ---------------------------------------------------------------------------
function dense_correlator(H::Matrix, O::Matrix, ψ₀::Vector, Eg::Float64,
                          ωs::AbstractVector{<:Real}, Γ::Float64)
    x   = O * ψ₀
    out = Vector{ComplexF64}(undef, length(ωs))
    for (k, ω) in enumerate(ωs)
        z = ω + Eg + im * Γ / 2
        out[k] = dot(x, (z * I - H) \ x)
    end
    return out
end

_local_dot(a::NTuple{3, OperatorSum}, b::NTuple{3, OperatorSum}) =
    a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# Heisenberg dimer fixture, used across kernel + API tests.
function _heisenberg_dimer()
    s1 = SpinSite{1//2}(:s1)
    s2 = SpinSite{1//2}(:s2)
    h  = Hilbert(:s1 => s1, :s2 => s2)
    H_op  = _local_dot(S(s1), S(s2))     # J = 1
    basis = EagerBasis(h)
    H_sp  = assemble(compile(H_op, basis), basis)
    Sz1_op = Sz(s1, -1//2) + Sz(s1, 1//2)
    Sz1_sp = assemble(compile(Sz1_op, basis), basis)
    F  = eigen(Hermitian(Matrix{Float64}(H_sp)))
    k  = argmin(F.values)
    return (s1 = s1, s2 = s2, basis = basis, H_sp = H_sp,
            H_dense = Matrix{Float64}(H_sp), Sz1_op = Sz1_op,
            Sz1_mat = Matrix{Float64}(Sz1_sp), ψ₀ = F.vectors[:, k],
            Eg = F.values[k])
end

# =====================================================================
# 1. Kernel correctness — Heisenberg dimer dynamic structure factor
#
# Physical setup: H = S₁·S₂ (J=1), 4-dim Hilbert space.
# Singlet GS at Eg = −3/4. Operator O = S₁ᶻ. S₁ᶻ|s⟩ ≠ 0, so the response
# is non-trivial with one triplet pole at excitation energy +1.
# =====================================================================

@testset "correlator — Heisenberg dimer (kernel)" begin
    d   = _heisenberg_dimer()
    ops = [d.Sz1_op]                         # As === Bs
    Γ   = 0.05
    @test length(d.basis) == 4
    @test d.Eg ≈ -3/4 atol = 1e-12

    # Reference on a 100-point grid
    ωs  = range(-0.5, 2.0, length = 100)
    ref = dense_correlator(d.H_dense, d.Sz1_mat, d.ψ₀, d.Eg, collect(ωs), Γ)

    # form = :grid — direct comparison
    resp_grid = correlator(d.H_sp, d.basis, ops, ops;
                           channel = :neutral, form = :grid, ω = ωs, Γ = Γ)
    @test resp_grid isa GridResponse
    @test maximum(abs, resp_grid.data[1, 1, :] .- ref) < 1e-10

    # form = :lanczos — same numbers, evaluated point-by-point
    L = correlator(d.H_sp, d.basis, ops, ops;
                   channel = :neutral, form = :lanczos, Γ = Γ)
    @test L isa LanczosResponse
    @test maximum(abs, [L(ω)[1, 1] for ω in ωs] .- ref) < 1e-10

    # Explicit state + Eg agrees with the auto-ground-state mode
    resp_state = correlator(d.H_sp, d.basis, ops, ops;
                            state = d.ψ₀, Eg = d.Eg,
                            channel = :neutral, form = :grid, ω = ωs, Γ = Γ)
    @test maximum(abs, resp_state.data .- resp_grid.data) < 1e-10
end

# =====================================================================
# 2. Kernel correctness — Hubbard dimer density-density (:neutral)
#
# H = -t (c†₁↑c₂↑ + c†₁↓c₂↓ + h.c.) + U (n₁↑n₁↓ + n₂↑n₂↓), t=1, U=4.
# Half-filling Sz=0 sector (4 states). Operator: O = n₁↑ + n₁↓, number-
# conserving so the response stays in the same sector.
# =====================================================================

@testset "correlator — Hubbard dimer density-density (kernel)" begin
    s = FermionSite{4}(:s)                   # 1↑, 1↓, 2↑, 2↓
    h = Hilbert(:s => s)
    t, U = 1.0, 4.0
    H_hop = -t * (cdag(s, 1) * c(s, 3) + cdag(s, 2) * c(s, 4))
    H_op  = (H_hop + H_hop') + U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))
    basis = EagerBasis(h,
                       n_fermion(h) == 2,
                       WeightedParticleCount([s], [1, -1, 1, -1]) == 0)
    @test length(basis) == 4
    H_sp = assemble(compile(H_op, basis), basis)

    n1_op  = n(s, 1) + n(s, 2)
    ops    = [n1_op]
    Γ      = 0.05
    ωs     = range(-2.0, 6.0, length = 100)

    resp = correlator(H_sp, basis, ops, ops;
                      channel = :neutral, form = :grid, ω = ωs, Γ = Γ)
    @test resp isa GridResponse

    F   = eigen(Hermitian(Matrix{Float64}(H_sp)))
    k   = argmin(F.values)
    ψ₀  = F.vectors[:, k]
    Eg  = F.values[k]
    n1_mat = Matrix{Float64}(assemble(compile(n1_op, basis), basis))
    ref = dense_correlator(Matrix{Float64}(H_sp), n1_mat, ψ₀, Eg, collect(ωs), Γ)
    @test maximum(abs, resp.data[1, 1, :] .- ref) < 1e-10
end

# =====================================================================
# 3. API contract
# =====================================================================

@testset "correlator — API contract" begin
    d   = _heisenberg_dimer()
    ops = [d.Sz1_op]
    Γ   = 0.05
    ωs  = range(-1.0, 2.0, length = 10)

    @testset "autocorrelator: As === Bs required" begin
        # Two different OperatorSum objects (even if values are equal) fail ===
        op_a = Sz(d.s1, -1//2) + Sz(d.s1, 1//2)
        op_b = Sz(d.s1, -1//2)
        @test_throws ArgumentError correlator(d.H_sp, d.basis, [op_a], [op_b];
            channel = :neutral, form = :lanczos, Γ = Γ)
        op_c1 = Sz(d.s1, -1//2) + Sz(d.s1, 1//2)
        op_c2 = Sz(d.s1, -1//2) + Sz(d.s1, 1//2)   # value-equal, identity-distinct
        @test_throws ArgumentError correlator(d.H_sp, d.basis, [op_c1], [op_c2];
            channel = :neutral, form = :lanczos, Γ = Γ)
        # Same object: no error
        ops_same = [Sz(d.s1, -1//2) + Sz(d.s1, 1//2)]
        @test_nowarn correlator(d.H_sp, d.basis, ops_same, ops_same;
            channel = :neutral, form = :lanczos, Γ = Γ)
    end

    @testset "channel kwarg validation" begin
        for bad_ch in (:bogus,)
            @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
                channel = bad_ch, form = :lanczos, Γ = Γ)
        end
    end

    @testset "single-channel: explicit empty indices rejected" begin
        # Explicit empty addition_indices in :addition mode is a clean
        # ArgumentError, not a raw BoundsError from block_lanczos. Use
        # channel = :both if one side should be empty.
        @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
            channel = :addition, addition_indices = Int[],
            form = :lanczos, Γ = Γ)
        @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
            channel = :removal, removal_indices = Int[],
            form = :lanczos, Γ = Γ)
    end

    @testset "form kwarg validation + dispatch" begin
        # Bad form values raise
        for bad_form in (:bogus, :pole_residue)
            @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
                channel = :neutral, form = bad_form, Γ = Γ)
        end
        # form = :grid requires ω
        @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
            channel = :neutral, form = :grid, Γ = Γ)
        # Each valid form returns the expected concrete type
        @test correlator(d.H_sp, d.basis, ops, ops;
            channel = :neutral, form = :lanczos, Γ = Γ) isa LanczosResponse
        @test correlator(d.H_sp, d.basis, ops, ops;
            channel = :neutral, form = :pole, Γ = Γ) isa PoleResponse
        @test correlator(d.H_sp, d.basis, ops, ops;
            channel = :neutral, form = :grid, ω = ωs, Γ = Γ) isa GridResponse
    end

    @testset "source-input mode validation" begin
        # :ground_state with explicit Eg
        @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
            state = :ground_state, Eg = d.Eg,
            channel = :neutral, form = :lanczos, Γ = Γ)
        # state::Vector without Eg
        @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
            state = d.ψ₀, channel = :neutral, form = :lanczos, Γ = Γ)
        # source_block with non-nothing state (both :ground_state and Vector)
        src_block = reshape(complex(d.Sz1_mat * d.ψ₀), :, 1)
        @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
            source_block = src_block, Eg = d.Eg, state = :ground_state,
            channel = :neutral, form = :lanczos, Γ = Γ)
        @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
            source_block = src_block, Eg = d.Eg, state = d.ψ₀,
            channel = :neutral, form = :lanczos, Γ = Γ)
        # source_block without Eg
        @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
            source_block = src_block, state = nothing,
            channel = :neutral, form = :lanczos, Γ = Γ)
    end

    @testset "Bs empty: rejected in state modes, bypassed in source_block mode" begin
        empty_ops = OperatorSum{Float64}[]
        # :ground_state mode
        @test_throws ArgumentError correlator(d.H_sp, d.basis, empty_ops, empty_ops;
            channel = :neutral, form = :lanczos, Γ = Γ)
        # state::Vector mode
        @test_throws ArgumentError correlator(d.H_sp, d.basis, empty_ops, empty_ops;
            state = d.ψ₀, Eg = d.Eg, channel = :neutral, form = :lanczos, Γ = Γ)
        # source_block mode: Bs is bypassed, no error
        src_block = reshape(complex(d.Sz1_mat * d.ψ₀), :, 1)
        @test_nowarn correlator(d.H_sp, d.basis, empty_ops, empty_ops;
            source_block = src_block, Eg = d.Eg, state = nothing,
            channel = :neutral, form = :lanczos, Γ = Γ)
    end

    @testset "source_block mode result agrees with :ground_state mode" begin
        src_block = reshape(complex(d.Sz1_mat * d.ψ₀), :, 1)
        ωgrid = range(-0.5, 2.0, length = 40)
        empty_ops = OperatorSum{Float64}[]
        resp_c = correlator(d.H_sp, d.basis, empty_ops, empty_ops;
            source_block = src_block, Eg = d.Eg, state = nothing,
            channel = :neutral, form = :grid, ω = ωgrid, Γ = Γ)
        resp_a = correlator(d.H_sp, d.basis, ops, ops;
            channel = :neutral, form = :grid, ω = ωgrid, Γ = Γ)
        @test maximum(abs, resp_c.data .- resp_a.data) < 1e-10
    end
end

@testset "correlator — :both partition contract" begin
    # -----------------------------------------------------------------------
    # Minimal fixture: 2-mode spinless fermion with nearest-neighbour hopping.
    # Unrestricted basis spans N ∈ {0, 1, 2}, so both c and c† operators have
    # non-trivial images within the basis — required for the positive cases.
    # -----------------------------------------------------------------------
    _s = FermionSite{2}(:sf)
    _h = Hilbert(:sf => _s)
    _H_op = -(cdag(_s, 1) * c(_s, 2) + cdag(_s, 2) * c(_s, 1))
    _basis_both = EagerBasis(_h)          # 4 states: |0⟩, |↑⟩, |↓⟩, |↑↓⟩
    _H_sp_both  = assemble(compile(_H_op, _basis_both), _basis_both)
    # ops_all: indices 1,2 are removal (c); 3,4 are addition (c†)
    _ops_all = [c(_s, 1), c(_s, 2), cdag(_s, 1), cdag(_s, 2)]
    _Γ = 0.05

    @testset "both nothing → explicit hint" begin
        err = try
            correlator(_H_sp_both, _basis_both, _ops_all, _ops_all;
                       channel = :both, form = :lanczos, Γ = _Γ)
            nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("explicit", err.msg)
    end

    @testset "both empty → explicit hint" begin
        err = try
            correlator(_H_sp_both, _basis_both, _ops_all, _ops_all;
                       channel = :both, form = :lanczos, Γ = _Γ,
                       addition_indices = Int[], removal_indices = Int[])
            nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("explicit", err.msg)
    end

    @testset "duplicate addition_indices" begin
        err = try
            correlator(_H_sp_both, _basis_both, _ops_all, _ops_all;
                       channel = :both, form = :lanczos, Γ = _Γ,
                       addition_indices = [1, 1])
            nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("duplicate", err.msg)
    end

    @testset "out-of-bounds addition_indices" begin
        err = try
            correlator(_H_sp_both, _basis_both, _ops_all, _ops_all;
                       channel = :both, form = :lanczos, Γ = _Γ,
                       addition_indices = [99])
            nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("out-of-bounds", err.msg)
    end

    @testset "overlapping addition / removal indices" begin
        err = try
            correlator(_H_sp_both, _basis_both, _ops_all, _ops_all;
                       channel = :both, form = :lanczos, Γ = _Γ,
                       addition_indices = [1], removal_indices = [1])
            nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("overlap", err.msg)
    end

    @testset "unequal-length non-empty channels" begin
        err = try
            correlator(_H_sp_both, _basis_both, _ops_all, _ops_all;
                       channel = :both, form = :lanczos, Γ = _Γ,
                       addition_indices = [1, 2], removal_indices = [3])
            nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("unequal length", err.msg)
    end

    @testset "source_block rejected for :both" begin
        # source_block is single-channel only; :both must use state= mode
        src = reshape(ones(ComplexF64, length(_basis_both)), :, 1)
        err = try
            correlator(_H_sp_both, _basis_both, _ops_all, _ops_all;
                       channel = :both, form = :lanczos, Γ = _Γ,
                       source_block = src,
                       addition_indices = [3, 4], removal_indices = [1, 2])
            nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("single-channel only", err.msg)
    end

    @testset "length-with-one-empty: addition only (no length check)" begin
        # addition_indices populated, removal_indices defaulted to empty →
        # length check is skipped; only the addition sub-call runs.
        gf = correlator(_H_sp_both, _basis_both, _ops_all, _ops_all;
                        channel = :both, form = :lanczos, Γ = _Γ,
                        addition_indices = [3, 4])
        @test gf isa GreensFunction
        @test gf.addition isa LanczosResponse
        @test gf.removal === nothing
    end

    @testset "valid both-channel: addition [3,4] + removal [1,2]" begin
        # Covered in Task 10's Hubbard dimer fixture for a physically motivated
        # model; here we use the minimal 2-mode spinless fermion to confirm the
        # full :both path returns a properly shaped GreensFunction.
        gf = correlator(_H_sp_both, _basis_both, _ops_all, _ops_all;
                        channel = :both, form = :lanczos, Γ = _Γ,
                        addition_indices = [3, 4], removal_indices = [1, 2])
        @test gf isa GreensFunction
        @test gf.addition isa LanczosResponse
        @test gf.removal  isa LanczosResponse
        # Both channels have 2 operators → 2×2 output matrices
        @test size(gf.addition) == (2, 2)
        @test size(gf.removal)  == (2, 2)
    end
end

# =====================================================================
# 4. Channel return types and form dispatch
#
# Focusses on TYPE-LEVEL and FORM-DISPATCH assertions not covered by
# Task 7a's `:both` partition testset (which used form = :lanczos
# throughout and checked structural correctness of the GreensFunction,
# not the inner response types for :pole / :grid).
#
# Fixture: the same 2-mode spinless fermion from the `:both` partition
# testset (unrestricted EagerBasis, spans N ∈ {0,1,2}).
#   ops_all = [c₁, c₂, c†₁, c†₂]
#   GS is in the N=1 sector at Eg ≈ −1.
# =====================================================================

@testset "correlator — channel return types and form dispatch" begin
    # Re-use the 2-mode spinless fermion fixture (same as :both partition testset).
    _s = FermionSite{2}(:sf)
    _h = Hilbert(:sf => _s)
    _H_op = -(cdag(_s, 1) * c(_s, 2) + cdag(_s, 2) * c(_s, 1))
    _basis = EagerBasis(_h)           # 4 states: |0⟩, |1⟩, |2⟩, |12⟩
    _H_sp  = assemble(compile(_H_op, _basis), _basis)
    # ops_all: indices 1,2 → annihilation (removal); 3,4 → creation (addition)
    _ops   = [c(_s, 1), c(_s, 2), cdag(_s, 1), cdag(_s, 2)]
    _add   = [cdag(_s, 1), cdag(_s, 2)]   # stand-alone addition list
    _rem   = [c(_s, 1), c(_s, 2)]         # stand-alone removal list
    _Γ     = 0.05
    _ωs    = range(-2.0, 2.0, length = 10)

    # GS energy of the unrestricted basis: Eg ≈ −1 (bonding orbital).
    import LinearAlgebra: eigen, Hermitian
    _F  = eigen(Hermitian(Matrix{Float64}(_H_sp)))
    _k  = argmin(_F.values)
    _Eg = _F.values[_k]

    @testset ":addition returns LanczosResponse with sign = +1 and correct Eg" begin
        L = correlator(_H_sp, _basis, _add, _add;
                       channel = :addition, form = :lanczos, Γ = _Γ)
        @test L isa LanczosResponse
        @test L.sign == +1
        @test L.Eg ≈ _Eg atol = 1e-12
    end

    @testset ":removal returns LanczosResponse with sign = -1 and correct Eg" begin
        L = correlator(_H_sp, _basis, _rem, _rem;
                       channel = :removal, form = :lanczos, Γ = _Γ)
        @test L isa LanczosResponse
        @test L.sign == -1
        @test L.Eg ≈ _Eg atol = 1e-12
    end

    @testset ":both [3,4]/[1,2] returns GreensFunction with both channels" begin
        # addition_indices=[3,4] → cdag₁,cdag₂; removal_indices=[1,2] → c₁,c₂
        gf = correlator(_H_sp, _basis, _ops, _ops;
                        channel = :both, form = :lanczos, Γ = _Γ,
                        addition_indices = [3, 4], removal_indices = [1, 2])
        @test gf isa GreensFunction
        @test gf.addition isa LanczosResponse
        @test gf.removal  isa LanczosResponse
    end

    @testset ":both with removal_indices=[1] only → addition === nothing" begin
        # Only removal_indices supplied → addition field is nothing.
        # ops[1] = c(_s, 1) is an annihilation operator (semantically
        # correct for :removal; kernel does not inspect operator type).
        gf = correlator(_H_sp, _basis, _ops, _ops;
                        channel = :both, form = :lanczos, Γ = _Γ,
                        removal_indices = [1])
        @test gf isa GreensFunction
        @test gf.addition === nothing
        @test gf.removal isa LanczosResponse
    end

    @testset ":both with form = :pole returns GreensFunction{T, PoleResponse{T}}" begin
        gf = correlator(_H_sp, _basis, _ops, _ops;
                        channel = :both, form = :pole, Γ = _Γ,
                        addition_indices = [3, 4], removal_indices = [1, 2])
        T = eltype(gf)
        @test gf isa GreensFunction{T, PoleResponse{T}}
    end

    @testset ":both with form = :grid returns GreensFunction{T, GridResponse{T,N}}" begin
        # form = :grid yields a 3-D array (rows × cols × ω), so N = 3.
        gf = correlator(_H_sp, _basis, _ops, _ops;
                        channel = :both, form = :grid, ω = _ωs, Γ = _Γ,
                        addition_indices = [3, 4], removal_indices = [1, 2])
        T = eltype(gf)
        @test gf isa GreensFunction{T, GridResponse{T, 3}}
    end
end

# =====================================================================
# 5. Rank-zero rethrow with channel-specific hint
#
# Constructs scenarios where the initial Krylov block is all-zero:
#   :addition — apply cdag to the fully-filled N=2 state (Pauli exclusion
#               forces cdag|↑↓⟩ = 0 for both modes).
#   :removal  — apply c to the vacuum N=0 state (c|0⟩ = 0).
# In both cases block_lanczos sees a rank-zero starting block and raises
# an ArgumentError. The correlator catch-and-rethrow wraps this with a
# channel-specific hint string, tested here.
# =====================================================================

@testset "correlator — rank-zero rethrow with channel-specific hint" begin
    _s    = FermionSite{2}(:sf)
    _h    = Hilbert(:sf => _s)
    _H_op = -(cdag(_s, 1) * c(_s, 2) + cdag(_s, 2) * c(_s, 1))
    _basis = EagerBasis(_h)        # |0⟩, |1⟩, |2⟩, |12⟩ (indices 1..4)
    _H_sp  = assemble(compile(_H_op, _basis), _basis)
    _Γ     = 0.05

    # Fully-filled N=2 state: cdag on either mode gives 0 (Pauli exclusion).
    # H acts trivially on it (hopping has no image in N=3 space), so Eg = 0.
    _full_state = [0.0, 0.0, 0.0, 1.0]
    _Eg_full    = 0.0

    # Vacuum N=0 state: c on either mode gives 0.
    _vac_state  = [1.0, 0.0, 0.0, 0.0]
    _Eg_vac     = 0.0

    @testset ":addition with all-zero initial block rethrows with addition hint" begin
        add_ops = [cdag(_s, 1), cdag(_s, 2)]
        err = try
            correlator(_H_sp, _basis, add_ops, add_ops;
                       channel = :addition, form = :lanczos, Γ = _Γ,
                       state = _full_state, Eg = _Eg_full)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("correlator(:addition)", err.msg)
        @test occursin("rank zero", err.msg)
    end

    @testset ":removal with all-zero initial block rethrows with removal hint" begin
        rem_ops = [c(_s, 1), c(_s, 2)]
        err = try
            correlator(_H_sp, _basis, rem_ops, rem_ops;
                       channel = :removal, form = :lanczos, Γ = _Γ,
                       state = _vac_state, Eg = _Eg_vac)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("correlator(:removal)", err.msg)
        @test occursin("rank zero", err.msg)
    end
end

@testset "correlator — rank-zero initial block propagates as ArgumentError" begin
    # Sz_total on the singlet GS vanishes, so block_lanczos's rank-revealing
    # QR catches an all-zero starting block.
    s1, s2 = SpinSite{1//2}(:s1), SpinSite{1//2}(:s2)
    h     = Hilbert(:s1 => s1, :s2 => s2)
    basis = EagerBasis(h)
    H_sp  = assemble(compile(_local_dot(S(s1), S(s2)), basis), basis)
    Sz_tot = (Sz(s1, -1//2) + Sz(s1, 1//2)) + (Sz(s2, -1//2) + Sz(s2, 1//2))
    ops_zero = [Sz_tot]

    err = try
        correlator(H_sp, basis, ops_zero, ops_zero;
                   channel = :neutral, form = :lanczos, Γ = 0.05)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("rank zero", err.msg)
end

# =====================================================================
# 6. Finite-T validation gate (2d Task 1)
#
# Reuses the Heisenberg dimer fixture (_heisenberg_dimer) and its
# neutral Sz1 operator: H = S₁·S₂ (J=1), 4-dim space, channel=:neutral.
# All cases exercise the new gate inserted after the As===Bs block and
# BEFORE all channel dispatch (ordering guard included).
# =====================================================================

@testset "finite-T validation gate (2d)" begin
    d   = _heisenberg_dimer()
    ops = [d.Sz1_op]
    Γ   = 0.05

    # T < 0: T must be ≥ 0
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :pole, Γ = Γ, T = -1.0)

    # T ≥ 0 but channel ≠ :neutral: charged channels not yet supported
    err_addition = try
        correlator(d.H_sp, d.basis, ops, ops;
            channel = :addition, form = :pole, Γ = Γ, T = 1.0)
        nothing
    catch e; e end
    @test err_addition isa ArgumentError
    @test occursin("neutral", err_addition.msg)

    # T with channel = :both, ordering guard: must raise BEFORE :both dispatch
    err_both = try
        correlator(d.H_sp, d.basis, ops, ops;
            channel = :both, form = :pole, Γ = Γ, T = 1.0,
            addition_indices = [1])
        nothing
    catch e; e end
    @test err_both isa ArgumentError
    @test occursin("neutral", err_both.msg)

    # T with form = :lanczos: thermal sum has no single Lanczos representation
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :lanczos, Γ = Γ, T = 1.0)

    # T with explicit state (non-:ground_state): must use ensemble from spectrum
    explicit_state = d.ψ₀
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :pole, Γ = Γ, T = 1.0,
        state = explicit_state, Eg = d.Eg)

    # T with explicit Eg: thermal path computes Eg from the spectrum
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :pole, Γ = Γ, T = 1.0, Eg = 0.0)

    # T with source_block: incompatible (ensemble computed from H's spectrum)
    src_block = reshape(complex(d.Sz1_mat * d.ψ₀), :, 1)
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :pole, Γ = Γ, T = 1.0,
        source_block = src_block, state = nothing)

    # N_states without T (T=nothing): thermal-only kwarg guard
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :lanczos, Γ = Γ, N_states = 5)

    # degen_tol without T (T=nothing): thermal-only kwarg guard
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :lanczos, Γ = Γ, degen_tol = 1e-9)

    # T with form = :grid but ω omitted: must raise ArgumentError (not leak a
    # MethodError from to_grid) — thermal gate returns before the common ω check.
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :grid, Γ = Γ, T = 1.0)

    # T with non-positive degen_tol: would disable degeneracy detection / NaN weights.
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :pole, Γ = Γ, T = 1.0, degen_tol = 0.0)

    # T with N_states = true (Bool <: Integer): must be rejected, not treated as 1.
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :pole, Γ = Γ, T = 1.0, N_states = true)
end

# =====================================================================
# 7. Finite-T thermal eigenpairs + Boltzmann weights (2d Task 2)
#
# Heisenberg dimer: singlet GS at E = -3/4 (nondegenerate), triplet at
# E = +1/4 (3-fold degenerate). _thermal_states must:
#   - at T=0 return only the nondegenerate singlet with weight 1.0;
#   - at T>0 return 1/Z-normalised Boltzmann weights;
#   - reject N_states = 2 (splits the 3-fold triplet manifold).
# =====================================================================

@testset "_thermal_states (2d)" begin
    d   = _heisenberg_dimer()
    degen_tol = 1e-8

    # 1. T = 0: only the nondegenerate singlet survives, weight 1.0.
    states0, energies0, weights0, E0_0 =
        MOADyna.Responses._thermal_states(d.H_sp, d.basis, 0.0, nothing, degen_tol)
    @test length(states0) == 1
    @test weights0[1] ≈ 1.0 atol = 1e-12

    # 2. T > 0: weights == exp.(-β (E - E0)) / Z recomputed independently.
    T = 0.5
    β = 1 / T
    states_T, energies_T, weights_T, E0_T =
        MOADyna.Responses._thermal_states(d.H_sp, d.basis, T, nothing, degen_tol)
    raw = exp.(-β .* (energies_T .- E0_T))
    @test weights_T ≈ raw ./ sum(raw)

    # 3. N_states = 1 (singlet nondegenerate) does NOT raise; N_states = 2
    #    cuts the 3-fold triplet (states 2,3,4 degenerate) → raises.
    @test MOADyna.Responses._thermal_states(d.H_sp, d.basis, T, 1, degen_tol) isa Tuple
    @test_throws ArgumentError MOADyna.Responses._thermal_states(
        d.H_sp, d.basis, T, 2, degen_tol)
end

# =====================================================================
# 8. Finite-T thermal fold wiring (2d Task 3)
#
# At T = 0 with a nondegenerate ground state, the thermal trace collapses
# to the single-GS T=0 result. Compare the thermal pole response against
# the plain (T === nothing) neutral pole response on the Heisenberg dimer.
# =====================================================================

@testset "finite-T thermal fold wiring (2d)" begin
    d   = _heisenberg_dimer()
    ops = [d.Sz1_op]
    Γ   = 0.05

    P_thermal = correlator(d.H_sp, d.basis, ops, ops;
                           channel = :neutral, form = :pole, Γ = Γ, T = 0.0)
    P_t0      = correlator(d.H_sp, d.basis, ops, ops;
                           channel = :neutral, form = :pole, Γ = Γ)
    @test P_thermal isa PoleResponse
    for ω in (-0.5, 0.0, 1.0, 1.5)
        @test P_thermal(ω) ≈ P_t0(ω) atol = 1e-10
    end
end

# =====================================================================
# 9. Finite-T API contract (2d Task 5)
#
# Pins four API properties not covered by the earlier finite-T testsets:
#   P2  — T===nothing (default) is indistinguishable from omitting T.
#   RTC — finite-T returns PoleResponse or GridResponse, never GreensFunction.
#   P10 — small N_states triggers @warn; full spectrum is silent.
#   DT  — degen_tol is honoured: loosening it past the singlet-triplet gap
#          (ΔE = 1) makes N_states=1 split the manifold → ArgumentError.
# =====================================================================

@testset "finite-T API contract (2d)" begin
    d   = _heisenberg_dimer()
    ops = [d.Sz1_op]
    Γ   = 0.05
    ωs  = range(-1.0, 2.0, length = 20)

    # ------------------------------------------------------------------
    # P2 regression: T===nothing (default) ≡ omitting T entirely.
    # Both calls must take the same code path; assert poles, residues, Eg
    # and Γ match to ≤ 1e-14.
    # ------------------------------------------------------------------
    P_default = correlator(d.H_sp, d.basis, ops, ops;
                           channel = :neutral, form = :pole, Γ = Γ)
    P_nothing = correlator(d.H_sp, d.basis, ops, ops;
                           channel = :neutral, form = :pole, Γ = Γ, T = nothing)
    @test P_default.poles    == P_nothing.poles
    @test maximum(abs, hcat(P_default.residues...) .- hcat(P_nothing.residues...)) ≤ 1e-14
    @test P_default.Eg       == P_nothing.Eg
    @test P_default.Γ        == P_nothing.Γ

    # ------------------------------------------------------------------
    # Return-type contract: finite-T neutral → PoleResponse or GridResponse,
    # never GreensFunction.
    # ------------------------------------------------------------------
    P_T = correlator(d.H_sp, d.basis, ops, ops;
                     channel = :neutral, form = :pole, Γ = Γ, T = 1.0)
    @test P_T isa PoleResponse
    @test !(P_T isa GreensFunction)

    G_T = correlator(d.H_sp, d.basis, ops, ops;
                     channel = :neutral, form = :grid, ω = ωs, Γ = Γ, T = 1.0)
    @test G_T isa GridResponse
    @test !(G_T isa GreensFunction)

    # ------------------------------------------------------------------
    # P10 under-convergence warning:
    #   N_states=1, T=0.5: excluded triplet (ΔE=1) has weight exp(-2)≈0.135
    #   > 1e-3 → must warn.
    #   N_states=nothing (full spectrum): no excluded states → no warning.
    # ------------------------------------------------------------------
    @test_logs (:warn,) correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :pole, Γ = Γ, T = 0.5, N_states = 1)

    @test_logs min_level=Logging.Warn correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :pole, Γ = Γ, T = 0.5)

    # ------------------------------------------------------------------
    # degen_tol override: singlet-triplet gap = 1.0 (J=1).
    #   degen_tol=0.5 < 1.0 → gap > tol → N_states=1 does NOT split → no raise.
    #     Use T=0.01 so excluded weight exp(-100) ≪ 1e-3 → no P10 warning either.
    #   degen_tol=1.5 > 1.0 → gap < tol → N_states=1 appears to split → raises.
    # ------------------------------------------------------------------
    @test_nowarn correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :pole, Γ = Γ, T = 0.01,
        N_states = 1, degen_tol = 0.5)
    @test_throws ArgumentError correlator(d.H_sp, d.basis, ops, ops;
        channel = :neutral, form = :pole, Γ = Γ, T = 0.5,
        N_states = 1, degen_tol = 1.5)
end
