# test/gradients/test_resolvent_jvp.jl
#
# Build-step 3 of the v0.3 differentiable forward model (MOADyna.Gradients): the
# resolvent JVP at fixed source X and fixed reference E0. Response
# C(ω) = X† G(ω) X and its directional derivative dC = X† G dH G X, validated on
# a toy Hubbard dimer against (i) a dense resolvent and (ii) central finite
# differences. Reproducible (no RNG).

using MOADyna
using LinearAlgebra
using SparseArrays
using Test

const G = MOADyna.Gradients

@testset "Gradients build-step 3: resolvent JVP" begin
    # --- toy Hubbard dimer, half-filled Sz=0 sector (4-dim) -----------------
    L = 2
    sites   = [FermionSite{2}(Symbol("s$i")) for i in 1:L]   # modes 1,2 = ↑,↓
    hilbert = Hilbert(s.name => s for s in sites)
    hopop   = sum(c'(sites[i], σ) * c(sites[i + 1], σ) for i in 1:L-1, σ in (1, 2))
    Mhop_op = hopop + hopop'
    MU_op   = sum(n(sites[i], 1) * n(sites[i], 2) for i in 1:L)
    bb = EagerBasis(hilbert, n_fermion(hilbert) == L,
                    WeightedParticleCount(sites, repeat([1, -1], L)) == 0)

    asm(op) = assemble(compile(op, bb), bb)
    M_hop = asm(Mhop_op)
    M_U   = asm(MU_op)
    N = size(M_hop, 1)
    @test N == 4

    # H(θ) = t·M_hop + U·M_U
    model = G.AffineModel([M_hop, M_U],
                          G.AffineMap([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
                          [:t, :U])
    θ = [1.0, 4.0]

    # fixed source vector (complex, normalized) and fixed reference
    X  = ComplexF64[1.0, 0.5im, -0.3, 0.2];  X ./= norm(X)
    E0 = 0.0
    Γ  = 0.1
    ω_grid = [-2.0, 0.5, 3.0]

    Hθ = G.hamiltonian(model, θ)

    # === (a) C(ω) vs dense resolvent ========================================
    Cnum = G.response_C(model, θ, X, ω_grid; E0 = E0, Γ = Γ)
    for (i, ω) in enumerate(ω_grid)
        z = ω + E0 + im * Γ / 2
        Cden = dot(X, (z * I - Matrix(Hθ)) \ X)
        @test Cnum[i] ≈ Cden atol = 1e-9
    end

    # === (c) dC(ω) vs dense explicit X† G dH G X (strongest oracle) =========
    δ = 1e-5
    for k in 1:2
        ek = zeros(2); ek[k] = 1.0
        dCnum = G.response_jvp(model, θ, ek, X, ω_grid; E0 = E0, Γ = Γ)
        dHk = G.dhamiltonian(model, θ, ek)
        for (i, ω) in enumerate(ω_grid)
            z = ω + E0 + im * Γ / 2
            Gz = inv(z * I - Matrix(Hθ))
            dCden = dot(X, Gz * (Matrix(dHk) * (Gz * X)))
            @test dCnum[i] ≈ dCden atol = 1e-9
        end
        @test norm(dCnum) > 1e-3                      # (d) nonzero response
    end

    # === (b) dC(ω) vs central FD of C(ω; θ) =================================
    for k in 1:2
        ek = zeros(2); ek[k] = 1.0
        dCnum = G.response_jvp(model, θ, ek, X, ω_grid; E0 = E0, Γ = Γ)
        Cp = G.response_C(model, θ .+ δ .* ek, X, ω_grid; E0 = E0, Γ = Γ)
        Cm = G.response_C(model, θ .- δ .* ek, X, ω_grid; E0 = E0, Γ = Γ)
        dC_fd = (Cp .- Cm) ./ (2δ)
        @test maximum(abs, dCnum .- dC_fd) < 1e-6
    end

    # === input guards =======================================================
    @test_throws ArgumentError G.response_C(model, θ, X, ω_grid; E0 = E0, Γ = 0.0)
    @test_throws ArgumentError G.response_C(model, θ, X, [Inf]; E0 = E0, Γ = Γ)
    @test_throws DimensionMismatch G.response_C(model, θ, X[1:3], ω_grid; E0 = E0, Γ = Γ)
    @test_throws ArgumentError G.response_jvp(model, θ, [1.0, 0.0], X, ω_grid;
                                              E0 = E0, Γ = Γ, rtol = 0.0)
end
