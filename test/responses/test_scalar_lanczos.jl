using Test
using LinearAlgebra
using SparseArrays
using Random: MersenneTwister
using MOADyna
using MOADyna.Responses: block_lanczos, cf_block
using MOADyna.Spectroscopy: _block_resolvent_action

# Existing callers may provide matrix action without any vector interface.
struct ScalarLanczosMatrixOnlyAction{T}
    matrix::Matrix{T}
end
Base.size(H::ScalarLanczosMatrixOnlyAction, args...) = size(H.matrix, args...)
Base.:*(H::ScalarLanczosMatrixOnlyAction, X::AbstractMatrix) = H.matrix * X

function scalar_lanczos_allocations(H, X, k)
    @allocated block_lanczos(H, X; krylovdim=k, reorth=:none,
        retain_basis=false, tol=0., deflate_tol=1e-14, threaded=false)
end

@testset "Scalar Lanczos regressions" begin
    rng = MersenneTwister(23)
    A = randn(rng, ComplexF64, 10, 10)
    H = sparse((A + A') / 2)
    x = randn(rng, ComplexF64, 10)
    X = reshape(x, :, 1)
    z = 0.4 + 0.7im
    expected = dot(x, (z*I - Matrix(H)) \ x)
    L = block_lanczos(H, X; krylovdim=13, reorth=:full, retain_basis=true)
    @test (L.R' * cf_block(L.α, L.β, z) * L.R)[1, 1] ≈ expected atol=3e-10
    @test L.V_basis[1] * L.R ≈ X atol=3e-10

    # RIXS/FY use the first K vectors; the trailing vector remains separate.
    K = 4
    kept = block_lanczos(H, X; krylovdim=K, reorth=:full, retain_basis=true)
    @test (kept.n_iter, length(kept.α), length(kept.β), length(kept.V_basis)) ==
        (K, K, K, K+1)
    V = hcat(kept.V_basis[1:K]...)
    @test V' * V ≈ Matrix{ComplexF64}(I, K, K) atol=2e-11
    action = V * _block_resolvent_action(kept.α, kept.β, z, kept.R)
    @test vec(action) ≈ V * ((z*I - V'*H*V) \ (V'*x)) atol=2e-10
    first_vector = copy(kept.V_basis[1])
    kept.V_basis[end][1, 1] += 1
    @test kept.V_basis[1] == first_vector

    fallback = block_lanczos(ScalarLanczosMatrixOnlyAction(Matrix(H)), X;
        krylovdim=13, reorth=:full)
    @test (fallback.R' * cf_block(fallback.α, fallback.β, z) * fallback.R)[1, 1] ≈
        expected atol=3e-10

    @testset "Unretained allocation growth" begin
        N, short, long = 2048, 8, 80
        diagonal = Diagonal(collect(range(-2., 3.; length=N)))
        source = reshape(ones(N), :, 1)
        scalar_lanczos_allocations(diagonal, source, short)
        scalar_lanczos_allocations(diagonal, source, long)
        @test all(block_lanczos(diagonal, source; krylovdim=k, tol=0.,
            deflate_tol=1e-14, threaded=false).n_iter == k for k in (short, long))
        short_bytes = minimum(scalar_lanczos_allocations(diagonal, source, short) for _ in 1:3)
        long_bytes = minimum(scalar_lanczos_allocations(diagonal, source, long) for _ in 1:3)
        # Permit O(K) coefficient storage, reject an O(N*K) stream of vectors.
        @test long_bytes - short_bytes < 8*N*sizeof(Float64) + 2048*(long-short)
    end

    @testset "Projected scalar dynamics" begin
        site = FermionSite{3}(:scalar_restriction_site)
        b = basis(Hilbert(:scalar_restriction_site => site))
        allowed = [j for j in 1:length(b) if count_ones(get_state(b, j)[1]) == 1]
        outside = setdiff(collect(1:length(b)), allowed)
        restriction = n_fermion([site]) == 1
        A = randn(rng, ComplexF64, length(b), length(b))
        H = sparse((A + A') / 2)
        source = randn(rng, ComplexF64, length(b), 1)
        original = copy(source)
        L = block_lanczos(H, source; restrictions=restriction, basis=b,
            krylovdim=8, reorth=:full)
        restricted = source[allowed, 1]
        expected = dot(restricted, (z*I - Matrix(H[allowed, allowed])) \ restricted)
        @test (L.R' * cf_block(L.α, L.β, z) * L.R)[1, 1] ≈ expected atol=2e-10
        @test source == original
        annihilated = zeros(ComplexF64, length(b), 1)
        annihilated[first(outside), 1] = 1
        @test_throws ArgumentError block_lanczos(H, annihilated;
            restrictions=restriction, basis=b)
    end
end
