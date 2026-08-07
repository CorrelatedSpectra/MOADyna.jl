using Test
using LinearAlgebra: norm
using Random
using MOADyna
using MOADyna.Diagnostics: configuration_weights

@testset "configuration_weights — single basis state has weight 1.0 on its own bucket" begin
    m = ShellModel([:Ni_3d, :L_3d])
    bas = basis(m.hilbert, n_fermion(m.hilbert) == 18)
    @test length(bas) > 0

    # Pick the first basis state; compute its true (Ni_3d, L_3d) occupations
    # by reading the bit-packed state directly via the same EncodingMap the
    # function under test uses, then build psi as the unit vector e_1.
    enc = bas.encoding
    s = bas.states
    nw = bas.nwords
    function shell_pop(state_idx::Int, shell::Symbol)
        n_modes = length(MOADyna.range_of(m, shell))
        off = (state_idx - 1) * nw + 1
        c = 0
        for k in 1:n_modes
            e = enc.entries[(shell, (k,))]
            w = s[off + e.word_idx - 1]
            c += Int((w >> e.bit_offset) & UInt64(1))
        end
        return c
    end
    n_Ni = shell_pop(1, :Ni_3d)
    n_L  = shell_pop(1, :L_3d)
    @test n_Ni + n_L == 18    # sanity: total particle count is 18

    psi = zeros(ComplexF64, length(bas))
    psi[1] = 1.0
    weights = configuration_weights(psi, bas, m; group_by = (:Ni_3d, :L_3d))
    @test length(weights) == 1
    @test weights[1].first == (Ni_3d = n_Ni, L_3d = n_L)
    @test weights[1].second ≈ 1.0 atol = 1e-12
end

@testset "configuration_weights — buckets sum to ‖ψ‖² and are sorted" begin
    m = ShellModel([:Ni_3d, :L_3d])
    bas = basis(m.hilbert, n_fermion(m.hilbert) == 18)

    rng_state = Random.MersenneTwister(0xCAFE)
    psi = randn(rng_state, ComplexF64, length(bas))
    psi ./= norm(psi)

    weights_full   = configuration_weights(psi, bas, m; group_by = (:Ni_3d, :L_3d))
    weights_subset = configuration_weights(psi, bas, m; group_by = (:Ni_3d,))

    # Norm conservation: every basis state lands in exactly one bucket
    # regardless of grouping granularity.
    @test sum(p.second for p in weights_full)   ≈ 1.0 atol = 1e-12
    @test sum(p.second for p in weights_subset) ≈ 1.0 atol = 1e-12

    # Descending sort.
    @test issorted([p.second for p in weights_full];   rev = true)
    @test issorted([p.second for p in weights_subset]; rev = true)

    # Subset has at most (n_modes_of_Ni_3d + 1) buckets.
    @test length(weights_subset) ≤ length(MOADyna.range_of(m, :Ni_3d)) + 1

    # Coarsening (drop :L_3d) cannot increase bucket count.
    @test length(weights_subset) ≤ length(weights_full)

    # NamedTuple keys carry only the requested shells.
    @test all(keys(p.first) == (:Ni_3d,) for p in weights_subset)
    @test all(keys(p.first) == (:Ni_3d, :L_3d) for p in weights_full)
end

@testset "configuration_weights — argument validation" begin
    m = ShellModel([:Ni_3d, :L_3d])
    bas = basis(m.hilbert, n_fermion(m.hilbert) == 18)
    psi = zeros(ComplexF64, length(bas))
    psi[1] = 1.0

    @test_throws ArgumentError configuration_weights(psi, bas, m; group_by = ())
    @test_throws ArgumentError configuration_weights(psi, bas, m; group_by = (:Ni_3d, :Nope))
    @test_throws DimensionMismatch configuration_weights(zeros(ComplexF64, length(bas) + 1),
                                                         bas, m; group_by = (:Ni_3d,))
end
