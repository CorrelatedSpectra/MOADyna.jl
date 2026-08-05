# =====================================================================
# MOAD.Responses — finite-T charged-channel Green's function (Item B)
# =====================================================================
#
# Finite-T single-particle GF: G(ω,T) = Σ_m ρ_m G_m(ω), summing the addition
# (m→N+1) and removal (m→N−1) Lehmann pieces over the thermally populated
# N-sector states |m⟩. The N-sector ensemble cannot be obtained by
# diagonalizing the N±1-spanning propagation basis (its global ground state may
# live in N±1), so the caller supplies it: `ensemble_states` (each an N-sector
# eigenstate embedded in the N±1 basis) + `ensemble_energies`.
#
# Validation: the wrapper must equal an INDEPENDENT Boltzmann sum of the
# (already-validated, T=0) per-state `:both` correlators, and must reduce to the
# T=0 single-GS GF as T→0. Half-filled Hubbard dimer, basis spanning N=1,2,3.

using Test
using LinearAlgebra
using SparseArrays
using MOAD
using MOAD: GridResponse, GreensFunction

@testset "Finite-T charged GF (Item B)" begin
    s = FermionSite{4}(:s)                       # 1↑,1↓,2↑,2↓
    h = Hilbert(:s => s)
    t, U = 1.0, 4.0
    H_hop = -t * (cdag(s, 1) * c(s, 3) + cdag(s, 2) * c(s, 4))
    H_op  = (H_hop + H_hop') + U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))
    # removal [1,2] = c_{1↑},c_{2↑};  addition [3,4] = c†_{1↑},c†_{2↑}
    ops = [c(s, 1), c(s, 3), cdag(s, 1), cdag(s, 3)]
    addidx, remidx = [3, 4], [1, 2]

    basis_full = EagerBasis(h, n_fermion(h) ∈ 1:3,
                            WeightedParticleCount([s], [1, -1, 1, -1]) ∈ -1:1)
    Nb = length(basis_full)
    H_full = assemble(compile(H_op, basis_full), basis_full)

    basis_N2 = EagerBasis(h, n_fermion(h) == 2,
                          WeightedParticleCount([s], [1, -1, 1, -1]) == 0)
    H_N2 = assemble(compile(H_op, basis_N2), basis_N2)
    F = eigen(Hermitian(Matrix(H_N2)))           # N=2, Sz=0 sector (4 states)

    # Embed an N=2 vector into basis_full (zero outside the N=2 block).
    function _embed(v)
        out = zeros(ComplexF64, Nb)
        for i in 1:length(basis_N2)
            j = get_index(basis_full, get_state(basis_N2, i))
            @assert j > 0
            out[j] = v[i]
        end
        return out
    end
    ens_states = [_embed(F.vectors[:, k]) for k in eachindex(F.values)]
    ens_E      = collect(Float64, F.values)

    Γ  = 0.05
    ωs = collect(range(-6.0, 6.0; length = 160))
    τ  = 1.5

    # GreensFunction(:both) grid → sum the addition + removal channel data.
    _bothdata(g::GreensFunction) = g.addition.data .+ g.removal.data

    # --- wrapper: finite-T :both on the supplied N=2 ensemble --------------
    gT = correlator(H_full, basis_full, ops, ops;
                    T = τ, ensemble_states = ens_states, ensemble_energies = ens_E,
                    channel = :both, addition_indices = addidx, removal_indices = remidx,
                    form = :grid, ω = ωs, Γ = Γ)
    # :both grid keeps both channels (consistent with the T=0 :both grid path).
    @test gT isa GreensFunction
    @test gT.addition isa GridResponse && gT.removal isa GridResponse

    # --- independent oracle: Boltzmann sum of per-state T=0 :both grids ----
    E0 = minimum(ens_E)
    ws = exp.(-(ens_E .- E0) ./ τ); ws ./= sum(ws)
    acc = zeros(ComplexF64, size(_bothdata(gT)))
    for k in eachindex(ens_states)
        gk = correlator(H_full, basis_full, ops, ops;
                        state = ens_states[k], Eg = ens_E[k],
                        channel = :both, addition_indices = addidx, removal_indices = remidx,
                        form = :grid, ω = ωs, Γ = Γ)
        acc .+= ws[k] .* _bothdata(gk)
    end
    @test maximum(abs, _bothdata(gT) .- acc) < 1e-9

    # --- T → 0 reduces to the T=0 single-GS :both GF -----------------------
    g0 = correlator(H_full, basis_full, ops, ops;
                    T = 0.0, ensemble_states = ens_states, ensemble_energies = ens_E,
                    channel = :both, addition_indices = addidx, removal_indices = remidx,
                    form = :grid, ω = ωs, Γ = Γ)
    ψ0  = ens_states[argmin(ens_E)]
    gGS = correlator(H_full, basis_full, ops, ops;
                     state = ψ0, Eg = E0,
                     channel = :both, addition_indices = addidx, removal_indices = remidx,
                     form = :grid, ω = ωs, Γ = Γ)
    @test maximum(abs, _bothdata(g0) .- _bothdata(gGS)) < 1e-9

    # --- single-channel finite-T (addition only) vs oracle -----------------
    ops_add = [cdag(s, 1), cdag(s, 3)]
    aT = correlator(H_full, basis_full, ops_add, ops_add;
                    T = τ, ensemble_states = ens_states, ensemble_energies = ens_E,
                    channel = :addition, form = :grid, ω = ωs, Γ = Γ)
    acc_a = zeros(ComplexF64, size(aT.data))
    for k in eachindex(ens_states)
        ak = correlator(H_full, basis_full, ops_add, ops_add;
                        state = ens_states[k], Eg = ens_E[k],
                        channel = :addition, form = :grid, ω = ωs, Γ = Γ)
        acc_a .+= ws[k] .* ak.data
    end
    @test maximum(abs, aT.data .- acc_a) < 1e-9

    # --- contract guards ---------------------------------------------------
    # charged finite-T WITHOUT an ensemble → still :neutral-only (auto path)
    err = try
        correlator(H_full, basis_full, ops, ops;
                   T = τ, channel = :both, addition_indices = addidx,
                   removal_indices = remidx, form = :grid, ω = ωs, Γ = Γ)
        nothing
    catch e; e end
    @test err isa ArgumentError && occursin("neutral", err.msg)

    # ensemble WITHOUT T → thermal-only kwargs require T
    @test_throws ArgumentError correlator(H_full, basis_full, ops, ops;
                   ensemble_states = ens_states, ensemble_energies = ens_E,
                   channel = :both, addition_indices = addidx, removal_indices = remidx,
                   form = :pole, Γ = Γ)

    # only one of the ensemble pair supplied → error
    @test_throws ArgumentError correlator(H_full, basis_full, ops, ops;
                   T = τ, ensemble_states = ens_states,
                   channel = :both, addition_indices = addidx, removal_indices = remidx,
                   form = :grid, ω = ωs, Γ = Γ)
end
