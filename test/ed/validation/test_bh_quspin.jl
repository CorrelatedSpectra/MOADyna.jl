# =====================================================================
# Bose-Hubbard chain — QuSpin cross-comparison (boson layer)
# =====================================================================
#
# 4-site BH chain, OBC, unit filling (Nb = 4), Nmax = 3 per site,
# t = 1, U = 4, μ = 0. Reference data captured via QuSpin (see
# `docs/dev/validation/ed/bose_hubbard/scripts/quspin_bh.py`).
# Compares the first six eigenvalues and onsite ⟨n_i⟩ in the GS.
#
# Both sides do dense ED on the same particle-conserving product
# basis (no symmetry block reduction either side), so agreement to
# ≲ 1e−10 is expected and required.

@testset "Bose-Hubbard — QuSpin reference" begin
    using HDF5
    using LinearAlgebra: dot

    ref_file = joinpath(@__DIR__, "..", "..", "..",
                        "docs", "dev", "validation", "ed",
                        "bose_hubbard", "reference", "bh_reference.h5")
    @test isfile(ref_file)
    isfile(ref_file) || return

    L, Nb, Nmax, t_hop, U, μ, ref_E, ref_n = h5open(ref_file, "r") do fh
        a = HDF5.attributes(fh)
        (Int(read(a["L"])), Int(read(a["Nb"])), Int(read(a["Nmax"])),
         Float64(read(a["t"])), Float64(read(a["U"])),
         Float64(read(a["mu"])),
         Vector{Float64}(read(fh, "energies")),
         Vector{Float64}(read(fh, "n_expect")))
    end

    sites = [BosonSite{Nmax}(Symbol("p$i")) for i in 1:L]
    h = Hilbert(s.name => s for s in sites)

    H_hop = sum(bdag(sites[i]) * b(sites[i + 1]) for i in 1:L - 1)
    H_hop = -t_hop * (H_hop + H_hop')
    H_int = (U / 2) * sum(n_b(s) * n_b(s) - n_b(s) for s in sites)
    H_chem = -μ * sum(n_b(s) for s in sites)
    H = H_hop + H_int + H_chem

    bb = EagerBasis(h, n_boson(h) == Nb)

    E = eigen(H, bb; n = length(ref_E), dense_below = typemax(Int))
    @test maximum(abs, E.values .- ref_E) < 1e-10

    psi0 = E.vectors[:, 1]
    n_moad = [
        let op = assemble(compile(n_b(sites[i]), bb), bb)
            real(dot(psi0, op * psi0))
        end
        for i in 1:L
    ]
    @test maximum(abs, n_moad .- ref_n) < 1e-10
end
