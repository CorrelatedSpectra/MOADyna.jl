@testset "RIXS block resolvent action" begin
    using LinearAlgebra
    using MOADyna.Spectroscopy: _block_resolvent_action, _build_T_K

    # Exact dense references for complex Hermitian, ragged block systems.
    # The source occupies the first block; other blocks have zero RHS.
    cases = [
        ([ComplexF64[1 0.3im; -0.3im 2], reshape(ComplexF64[3], 1, 1)],
         [ComplexF64[0.4 0.2im]], ComplexF64[1 0.2 0; 0.1im 1 0.5]),
        ([reshape(ComplexF64[1], 1, 1), reshape(ComplexF64[2], 1, 1),
          reshape(ComplexF64[4], 1, 1)],
         [reshape(ComplexF64[0.5im], 1, 1), reshape(ComplexF64[0.3], 1, 1)],
         reshape(ComplexF64[1, 0.5im], 1, 2)),
        ([ComplexF64[1 0.2im; -0.2im 3]], Matrix{ComplexF64}[],
         ComplexF64[1 0; 0.2im 0.5]),
    ]
    errors = Float64[]
    residuals = Float64[]
    for (alpha, beta, R) in cases, z in (0.5 + 0.2im, 2.0 + 0.025im)
        T = _build_T_K(alpha, beta)
        source = zeros(ComplexF64, size(T, 1), size(R, 2))
        source[1:size(R, 1), :] .= R
        reference = (z*I - T) \ source
        # A final beta block may exist outside the represented Krylov T_K.
        for blocks in (beta, vcat(beta, [zeros(ComplexF64, 1, size(alpha[end], 1))]))
            W = _block_resolvent_action(alpha, blocks, z, R)
            push!(errors, norm(W-reference)/norm(reference))
            push!(residuals, norm((z*I-T)*W-source)/norm(source))
        end
    end
    @test maximum(errors) < 1e-12
    @test maximum(residuals) < 1e-12
end
