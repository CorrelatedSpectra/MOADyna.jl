# =====================================================================
# Spinless Holstein dimer — QuSpin cross-comparison (mixed Fock layer)
# =====================================================================
#
# 2-site spinless Holstein, OBC, single fermion, one phonon mode per
# site (Nmax_ph = 10), t = 1, ω = 1, g = 0.7. Reference data captured
# via QuSpin's tensor_basis(boson, spinless_fermion); see
# `docs/dev/validation/ed/holstein/scripts/quspin_holstein.py`.
#
# Spinless rather than spinful Hubbard-Holstein: the Hubbard piece is
# already pinned by Plan 2 / NiO XAS validation. The new content here
# is fermion-boson mixing — captured fully in the spinless model and
# matching QuSpin's idiomatic example10 pattern.

@testset "Holstein — QuSpin reference" begin
    using HDF5
    using LinearAlgebra: dot

    ref_file = joinpath(@__DIR__, "..", "..", "..",
                        "docs", "dev", "validation", "ed",
                        "holstein", "reference", "holstein_reference.h5")
    @test isfile(ref_file)
    isfile(ref_file) || return

    L, Nf, Nmax_ph, t_hop, ω, g_eph,
    ref_E, ref_nph, ref_nf0 = h5open(ref_file, "r") do fh
        a = HDF5.attributes(fh)
        (Int(read(a["L"])), Int(read(a["Nf"])), Int(read(a["Nmax_ph"])),
         Float64(read(a["t"])), Float64(read(a["omega"])),
         Float64(read(a["g"])),
         Vector{Float64}(read(fh, "energies")),
         Float64(read(fh, "nph_total")),
         Float64(read(fh, "nf_site0")))
    end

    fermions = [FermionSite{1}(Symbol("e$i")) for i in 1:L]
    phonons  = [BosonSite{Nmax_ph}(Symbol("p$i")) for i in 1:L]
    h = Hilbert(s.name => s for s in vcat(fermions, phonons))

    H_hop = -t_hop * (cdag(fermions[1], 1) * c(fermions[2], 1) +
                      cdag(fermions[2], 1) * c(fermions[1], 1))
    H_ph  = ω * sum(n_b(p) for p in phonons)
    H_eph = g_eph * sum((bdag(phonons[i]) + b(phonons[i])) *
                        n(fermions[i], 1) for i in 1:L)
    H = H_hop + H_ph + H_eph

    bb = EagerBasis(h, n_fermion(h) == Nf)

    E = eigen(H, bb; n = length(ref_E), dense_below = typemax(Int))
    @test maximum(abs, E.values .- ref_E) < 1e-10

    psi0    = E.vectors[:, 1]
    nph_op  = assemble(compile(sum(n_b(p) for p in phonons), bb), bb)
    nf0_op  = assemble(compile(n(fermions[1], 1), bb), bb)
    @test real(dot(psi0, nph_op * psi0)) ≈ ref_nph atol = 1e-10
    @test real(dot(psi0, nf0_op * psi0)) ≈ ref_nf0 atol = 1e-10
end
