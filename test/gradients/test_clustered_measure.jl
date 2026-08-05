# test/gradients/test_clustered_measure.jl
#
# Build-step 6b of the v0.3 differentiable forward model (MOAD.Gradients): the
# degeneracy-clustered spectral measure. Moments of S = −Im C/π over FROZEN energy
# windows. A controlled DIAGONAL final-state fixture (H_f(t) = diag(1+t, 1, 5, 6),
# E0 = 0) gives an analytic pole spectrum C(ω) = Σ_n |X_n|²/(z − ε_n) for independent
# cross-checks. Tests: window definition, singleton stability vs the analytic
# integral, gauge-invariance under a unitary on the final space, JVP-vs-FD over frozen
# windows (smooth even across a pole crossing), and that freeze_windows itself is the
# θ-dependent (non-smooth) map. Reproducible (no RNG).

using MOAD
using LinearAlgebra
using Test

const G = MOAD.Gradients

# Diagonal fixture: 1 parameter t moves the first final-state pole. H_g = diag(0,3)
# (E0 = 0, ψ0 = e1, non-degenerate). E = [I2; 0] (4×2 isometry). A = B = (T,) with the
# pole weights X = T·E·ψ0 = [0.5, 1, 1, 0.8].
function _fixture(; ω_grid = collect(range(0.0, 7.0; length = 141)))
    names = [:t]
    Dg = Matrix{ComplexF64}(Diagonal([0.0, 3.0]))
    g_model = G.AffineModel([Dg], G.AffineMap([1.0], zeros(1, 1)), names)

    M0f = Matrix{ComplexF64}(Diagonal([1.0, 1.0, 5.0, 6.0]))
    M1f = Matrix{ComplexF64}(Diagonal([1.0, 0.0, 0.0, 0.0]))
    f_model = G.AffineModel([M0f, M1f], G.AffineMap([1.0, 0.0], reshape([0.0, 1.0], 2, 1)),
                            names)

    Emb = ComplexF64[1 0; 0 1; 0 0; 0 0]                 # 4×2 isometry
    Xw  = ComplexF64[0.5, 1.0, 1.0, 0.8]
    T = zeros(ComplexF64, 4, 4); T[:, 1] = Xw; T[1, :] = conj(Xw)
    model = G.XASGradientModel(g_model, f_model, (T,), (T,), Emb, ω_grid; Γ = 0.3)
    return model, Xw
end

# Analytic intensity S(ω) = −(1/π) Im Σ_n |X_n|² / (ω + iΓ/2 − ε_n), ε from H_f(t).
function _analytic_S(ω_grid, Xw, εf, Γ)
    w = abs2.(Xw)
    return [(-1 / π) * imag(sum(w ./ (ω + im * Γ / 2 .- εf))) for ω in ω_grid]
end

@testset "Gradients build-step 6b: clustered spectral measure" begin
    model, Xw = _fixture()
    Γ = model.Γ
    δ = 1e-5

    # --- freeze_windows: structure + contiguity ---
    win = G.freeze_windows(model, [0.0]; cluster_tol = 0.2, pad = 0.22)
    @test length(win) == 3                                   # {1,1}, {5}, {6}
    @test all(w -> w.members == (first(w.members):last(w.members)), win)
    @test isapprox(win[1].center, 1.0; atol = 0.1)
    # a grid that doesn't reach the 5/6 clusters ⇒ a window catches no grid point
    model_short, _ = _fixture(ω_grid = collect(range(0.0, 3.0; length = 61)))
    @test_throws ArgumentError G.freeze_windows(model_short, [0.0]; cluster_tol = 0.2, pad = 0.22)
    # a window spanning only ONE grid point ⇒ trapezoid needs ≥2 (coarse Δ=1 grid)
    model_coarse, _ = _fixture(ω_grid = collect(0.0:1.0:7.0))
    @test_throws ArgumentError G.freeze_windows(model_coarse, [0.0]; cluster_tol = 0.2, pad = 0.3)

    # --- singleton stability: window 2 (isolated pole ε=5) vs an INDEPENDENT analytic
    # WINDOW-LOCAL trapezoid integral (composite trapezoid over the sampled extent
    # [ω[a], ω[b]] — NOT the global-grid weights, which would overcount the endpoints) ---
    clusters = G.spectral_clusters(model, [0.0], win)
    @test length(clusters) == 3
    εf0 = [1.0, 1.0, 5.0, 6.0]
    Sana = _analytic_S(model.ω_grid, Xw, εf0, Γ)
    ω = model.ω_grid
    mem = win[2].members; a, b = first(mem), last(mem)
    wloc = [i == a ? (ω[a+1] - ω[a]) / 2 :
            i == b ? (ω[b] - ω[b-1]) / 2 :
            (ω[i+1] - ω[i-1]) / 2 for i in mem]
    m0_ana = sum(wloc[k] * Sana[mem[k]] for k in eachindex(mem))
    @test isapprox(clusters[2].moment0[1, 1], m0_ana; atol = 1e-9)
    m1_ana = sum(wloc[k] * (ω[mem[k]] - win[2].center) * Sana[mem[k]] for k in eachindex(mem))
    @test isapprox(clusters[2].moment1[1, 1], m1_ana; atol = 1e-9)
    # window-local weights sum to the sampled extent ω[b]−ω[a] (a constant spectrum c
    # integrates to c·(ω[b]−ω[a])); the buggy global weights would give a larger span.
    @test isapprox(sum(wloc), ω[b] - ω[a]; atol = 1e-12)
    @test clusters[2].width == win[2].hi - win[2].lo
    @test clusters[2].members == win[2].members

    # --- gauge-invariance: a unitary W on the final space leaves C (hence the
    # moments) invariant; eigenvalues of H_f and the windows are unchanged ---
    M = ComplexF64[1 0.3im 0 0.2; -0.1 1 0.4im 0; 0 0.2 1 0.5im; 0.3 0 0.1 1]
    W = Matrix(qr(M).Q)
    @test norm(W' * W - I) < 1e-10
    f2 = G.AffineModel([W * Matrix{ComplexF64}(Diagonal([1.0,1.0,5.0,6.0])) * W',
                        W * Matrix{ComplexF64}(Diagonal([1.0,0.0,0.0,0.0])) * W'],
                       G.AffineMap([1.0, 0.0], reshape([0.0, 1.0], 2, 1)), [:t])
    Dg = Matrix{ComplexF64}(Diagonal([0.0, 3.0]))
    g_model = G.AffineModel([Dg], G.AffineMap([1.0], zeros(1, 1)), [:t])
    Xw2 = ComplexF64[0.5, 1.0, 1.0, 0.8]
    T = zeros(ComplexF64, 4, 4); T[:, 1] = Xw2; T[1, :] = conj(Xw2)
    Emb2 = W * ComplexF64[1 0; 0 1; 0 0; 0 0]
    model2 = G.XASGradientModel(g_model, f2, (W * T * W',), (W * T * W',), Emb2,
                                model.ω_grid; Γ = Γ)
    # integrate BOTH spectra over the SAME frozen windows `win`: this isolates the
    # gauge-invariance of the measure (C is unitarily invariant) from the unrelated
    # floating-point sensitivity of recomputing windows from perturbed eigenvalues.
    cl2 = G.spectral_clusters(model2, [0.0], win)
    @test length(cl2) == length(clusters)
    for c in 1:length(clusters)
        @test isapprox(cl2[c].moment0[1, 1], clusters[c].moment0[1, 1]; atol = 1e-9)
        @test isapprox(cl2[c].moment1[1, 1], clusters[c].moment1[1, 1]; atol = 1e-9)
    end

    # --- JVP vs central FD over frozen windows (smooth functional) ---
    dcl = G.spectral_clusters_jvp(model, [0.0], [1.0], win)
    cp = G.spectral_clusters(model, [δ], win)
    cm = G.spectral_clusters(model, [-δ], win)
    for c in 1:length(win)
        fd0 = (cp[c].moment0 - cm[c].moment0) ./ (2δ)
        fd1 = (cp[c].moment1 - cm[c].moment1) ./ (2δ)
        @test maximum(abs, dcl[c].dmoment0 - fd0) < 1e-6
        @test maximum(abs, dcl[c].dmoment1 - fd1) < 1e-6
    end

    # --- frozen windows stay smooth even when a pole crosses a window boundary ---
    # pole 1 (ε = 1+t) sits at 1.6 for t=0.6, OUTSIDE window 1's [0.78,1.22]; the FROZEN
    # functional is still a smooth fixed linear functional, so JVP matches FD.
    θx = [0.6]
    dclx = G.spectral_clusters_jvp(model, θx, [1.0], win)
    cpx = G.spectral_clusters(model, θx .+ δ, win)
    cmx = G.spectral_clusters(model, θx .- δ, win)
    for c in 1:length(win)
        fd0 = (cpx[c].moment0 - cmx[c].moment0) ./ (2δ)
        @test maximum(abs, dclx[c].dmoment0 - fd0) < 1e-6
    end
    # the non-smoothness lives in freeze_windows itself: recomputing windows at the
    # shifted θ changes the cluster structure (pole 1 splits off its own window).
    win_shift = G.freeze_windows(model, θx; cluster_tol = 0.2, pad = 0.22)
    @test length(win_shift) != length(win)                  # 4 windows vs 3 — membership moved
end
