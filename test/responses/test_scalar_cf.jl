using LinearAlgebra
using MOADyna.Responses: cf_block

@testset "Scalar continued fractions" begin
    @testset "Complex couplings match a dense resolvent" begin
        for T in (Float32, Float64)
            alpha = [fill(T(a), 1, 1) for a in (-1.2, 0.3, 2.1, 3.4)]
            beta = [fill(Complex{T}(b), 1, 1) for b in (0.4+0.7im, -0.2+0.3im, 0.6-0.1im)]
            h = diagm(Complex{T}[a[1, 1] for a in alpha])
            for k in eachindex(beta)
                h[k+1, k] = beta[k][1, 1]
                h[k, k+1] = conj(beta[k][1, 1])
            end
            z = Complex{T}(0.8, 0.25)
            actual = cf_block(alpha, beta, z)
            @test size(actual) == (1, 1)
            @test eltype(actual) === Complex{T}
            @test actual[1, 1] ≈ inv(z*I-h)[1, 1] rtol=20eps(T)
            # A final beta points beyond the represented tridiagonal and is ignored.
            @test cf_block(alpha, [beta; [ones(Complex{T}, 2, 3)]], z) == actual
        end
        @test cf_block([fill(2.0, 1, 1)], Matrix{Float64}[], 1+0.5im)[1, 1] == inv(-1+0.5im)
    end

    @testset "Ragged blocks retain the matrix recurrence" begin
        alpha = [fill(-0.3+0im, 1, 1), ComplexF64[0.2 0.1im; -0.1im 1.0], fill(1.7+0im, 1, 1)]
        beta = [reshape(ComplexF64[0.4+0.2im, 0.1-0.3im], 2, 1), ComplexF64[0.2+0.1im 0.5-0.2im]]
        h = zeros(ComplexF64, 4, 4)
        h[1, 1] = alpha[1][1, 1]
        h[2:3, 2:3] = alpha[2]
        h[4, 4] = alpha[3][1, 1]
        h[2:3, 1:1] = beta[1]; h[1:1, 2:3] = beta[1]'
        h[4:4, 2:3] = beta[2]; h[2:3, 4:4] = beta[2]'
        z = 0.6+0.15im
        @test cf_block(alpha, beta, z)[1, 1] ≈ inv(z*I-h)[1, 1] rtol=1e-13
    end

    @testset "Scalar evaluation allocates independently of chain length" begin
        alpha = [fill(0.01k, 1, 1) for k in 1:400]
        beta = [fill(0.2+0.1im, 1, 1) for _ in 1:399]
        short_alpha, short_beta = alpha[1:4], beta[1:3]
        z = 0.7+0.3im
        cf_block(short_alpha, short_beta, z)
        cf_block(alpha, beta, z)
        short_bytes = @allocated cf_block(short_alpha, short_beta, z)
        long_bytes = @allocated cf_block(alpha, beta, z)
        # Allow allocator/runtime variation, but no temporary matrix per step.
        @test long_bytes <= short_bytes + 4096
    end
end
