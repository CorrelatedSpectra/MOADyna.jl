# =====================================================================
# 2-site Hubbard dimer Green's function — end-to-end validation
# =====================================================================
#
# Headline validation fixture for Plan 2c (responses, `:addition` /
# `:removal` / `:both` channels). Compares the block-Lanczos /
# continued-fraction kernel exposed by `MOADyna.correlator(...)` against an
# independent dense-Lehmann reference built by direct diagonalisation
# (`eigen` on the dense (N±1)-sector Hamiltonians) at every grid point.
#
# System:
#
#   H = -t (c†_{1↑} c_{2↑} + c†_{1↓} c_{2↓} + h.c.)
#       + U (n_{1↑} n_{1↓} + n_{2↑} n_{2↓}),    t = 1, U = 4
#
# half-filling N = 2. Single `FermionSite{4}(:s)` with mode layout
#
#   mode 1 → site 1, spin ↑
#   mode 2 → site 1, spin ↓
#   mode 3 → site 2, spin ↑
#   mode 4 → site 2, spin ↓
#
# (matches the spin pairing in `H_hop`'s `cdag(s,1)*c(s,3)` and
# `cdag(s,2)*c(s,4)` terms — hopping is intra-spin.)
#
# Operators used in the `:both` Green's function:
#
#   ops              = [c(s,1), c(s,3), c†(s,1), c†(s,3)]   # spin-↑ only
#   removal_indices  = [1, 2]   # c_{1↑}, c_{2↑}
#   addition_indices = [3, 4]   # c†_{1↑}, c†_{2↑}
#
# Inter-site hopping induces non-zero off-diagonal elements G_{12}(ω)
# (site 1↑ ↔ site 2↑), so the 2×2 matrix is non-trivial — element-wise
# comparison catches transposes, phase flips, and off-diagonal sign errors.
#
# Subtle point — ground-state sector and `state =` mode:
#
# The "physical" GS for this problem lives in (N=2, Sz=0). However the
# combined basis used for the `:both` channels also covers (N=1) and
# (N=3) sectors so that c†|ψ₀⟩ and c|ψ₀⟩ stay in the basis. In that
# combined basis the global minimum of H is the N=1 bonding state at
# E = −t = −1, NOT the half-filled GS at E = U/2 − √(U²/4 + 4t²) ≈
# −0.828. We therefore compute ψ₀ (the half-filled GS) by restricting
# `eigen` to the N=2 sub-basis, embed it into the combined basis, and
# pass it to `correlator(...)` via the explicit `state =`, `Eg =` mode.
# Letting `state = :ground_state` here would silently pick up the wrong
# (N=1 or N=3) state.
#
# Tier 1 — Sector-machinery precondition:
#
#   - Combined basis covers (N=1, N=2, N=3). `correlator(...; channel=:both)`
#     returns a `GreensFunction` with both fields populated.
#   - Basis restricted to (N=2) only forces the rank-zero rethrow with the
#     channel-specific hint message — confirms the kernel's loud-failure
#     mode is wired to the validation fixture (Plan 2c Task 5/6).
#
# Tier 2 — Full complex G(ω) matrix vs. dense-Lehmann reference:
#
#   The reference is computed by diagonalising H in (N=1, N=2, N=3)
#   sectors with dense `eigen`, then summing the Lehmann formula
#
#     G_{αβ}^add(ω) = Σ_n ⟨ψ₀ | c_α   | n_{N+1}⟩⟨n_{N+1} | c_β†  | ψ₀⟩
#                          / (ω - (E_n^{N+1} - E_0^{N}) + iΓ/2)
#     G_{αβ}^rem(ω) = Σ_m ⟨ψ₀ | c_α†  | m_{N-1}⟩⟨m_{N-1} | c_β   | ψ₀⟩
#                          / (ω + (E_m^{N-1} - E_0^{N}) + iΓ/2)
#
#   element-wise. The dense-Lehmann path is COMPLETELY INDEPENDENT of the
#   block-Lanczos + continued-fraction path; agreement to 1e-10 on the
#   full complex 2×2 matrix is the unambiguous correctness check.
#
# Solver settings (pinned per Plan 2c §10 to make 1e-10 reachable):
#
#   krylovdim   = length(basis_full)    # exhaust the Krylov subspace
#   reorth      = :full                 # full reorthogonalisation
#   tol         = 1e-13                 # tight soft-stop
#   deflate_tol = 1e-13                 # tight rank-revealing-QR threshold
#   Γ           = 0.05                  # finite, off-axis evaluation well-conditioned

@testset "Hubbard dimer Green's function — :both end-to-end" begin
    using LinearAlgebra
    using SparseArrays

    # ---------------------------------------------------------------
    # Fixture: 4-mode FermionSite, dimer Hamiltonian, half-filling.
    # ---------------------------------------------------------------
    s = FermionSite{4}(:s)
    h = Hilbert(:s => s)
    t, U = 1.0, 4.0

    # Intra-spin hopping: (1↑↔2↑) is (mode 1 ↔ mode 3); (1↓↔2↓) is
    # (mode 2 ↔ mode 4).
    H_hop = -t * (cdag(s, 1) * c(s, 3) + cdag(s, 2) * c(s, 4))
    H_op  = (H_hop + H_hop') + U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))

    Γ  = 0.05
    ωs = collect(range(-4.0, 4.0; length = 200))

    # Spin-↑ operators (mode 1 = 1↑, mode 3 = 2↑)
    c_1up    = c(s, 1)
    c_2up    = c(s, 3)
    cdag_1up = cdag(s, 1)
    cdag_2up = cdag(s, 3)
    # ops layout: removal [1,2] = c_{1↑},c_{2↑};  addition [3,4] = c†_{1↑},c†_{2↑}
    ops = [c_1up, c_2up, cdag_1up, cdag_2up]

    # ---------------------------------------------------------------
    # Build the combined basis (N=1,2,3, Sz ∈ {-1/2, 0, +1/2}) and the
    # N=2 sub-basis. ψ₀ is the (N=2, Sz=0) GS, computed on the N=2
    # sub-basis, then embedded into the combined basis.
    # ---------------------------------------------------------------
    basis_full = EagerBasis(h,
                            n_fermion(h) ∈ 1:3,
                            WeightedParticleCount([s], [1, -1, 1, -1]) ∈ -1:1)
    Nb_full = length(basis_full)
    H_sp_full = assemble(compile(H_op, basis_full), basis_full)

    # N=2 sub-basis restricted to Sz=0 — the half-filled singlet GS
    # lives here. This is also exactly the (N=2) ∩ basis_full block, so
    # ψ₀_N2 embeds into basis_full without truncation.
    basis_N2 = EagerBasis(h,
                          n_fermion(h) == 2,
                          WeightedParticleCount([s], [1, -1, 1, -1]) == 0)
    H_sp_N2  = assemble(compile(H_op, basis_N2), basis_N2)
    # Closed-form GS energy at half-filling: U/2 - √((U/2)² + 4t²)
    F_N2 = eigen(Hermitian(Matrix(H_sp_N2)))
    Eg   = F_N2.values[1]
    ψ₀_N2 = F_N2.vectors[:, 1]
    @test isapprox(Eg, U / 2 - sqrt((U / 2)^2 + 4 * t^2); atol = 1e-12)

    # Embed ψ₀ (length Nb_N2) into basis_full (length Nb_full): zero on
    # every basis_full state that isn't in the N=2 sub-basis; on N=2
    # states, take the value from ψ₀_N2 at the corresponding index.
    ψ₀ = zeros(ComplexF64, Nb_full)
    for i in 1:length(basis_N2)
        st = get_state(basis_N2, i)
        j = get_index(basis_full, st)
        @assert j > 0  "N=2 basis state $i is not in basis_full"
        ψ₀[j] = ψ₀_N2[i]
    end
    @test sum(abs2, ψ₀) ≈ 1.0  atol = 1e-12

    # ---------------------------------------------------------------
    # Tier 1 — Sector-machinery precondition
    # ---------------------------------------------------------------
    @testset "Tier 1 — sector-spanning + rank-zero rethrow" begin
        # 1a. `state = ψ₀, Eg = Eg` mode — explicit half-filled GS.
        # Both sub-calls run, basis spans (N=1,2,3) so neither rank-zero
        # branch fires.
        gf = correlator(H_sp_full, basis_full, ops, ops;
                        channel = :both, form = :lanczos, Γ = Γ,
                        state = ψ₀, Eg = Eg,
                        addition_indices = [3, 4], removal_indices = [1, 2],
                        krylovdim = Nb_full, reorth = :full,
                        tol = 1e-13, deflate_tol = 1e-13)
        # Both sub-calls must complete and be wired into the right
        # channel with the right sign convention.
        @test gf isa GreensFunction
        @test gf.addition isa LanczosResponse && gf.addition.sign == +1
        @test gf.removal  isa LanczosResponse && gf.removal.sign  == -1

        # 1b. Basis restricted to (N=2) only. :addition needs (N+1=3) and
        # :removal needs (N−1=1); both should hit rank-zero with the
        # channel-specific hint (Plan 2c Task 5/6).
        # `As === Bs` is required (autocorrelator contract from 2b), so bind
        # the operator list once and pass it on both sides.
        ops_add_only = [cdag_1up, cdag_2up]
        err_add = try
            correlator(H_sp_N2, basis_N2, ops_add_only, ops_add_only;
                       channel = :addition, form = :lanczos, Γ = Γ,
                       state = ψ₀_N2, Eg = Eg)
            nothing
        catch e; e end
        @test err_add isa ArgumentError
        @test occursin("correlator(:addition)", err_add.msg)
        @test occursin("rank zero", err_add.msg)

        ops_rem_only = [c_1up, c_2up]
        err_rem = try
            correlator(H_sp_N2, basis_N2, ops_rem_only, ops_rem_only;
                       channel = :removal, form = :lanczos, Γ = Γ,
                       state = ψ₀_N2, Eg = Eg)
            nothing
        catch e; e end
        @test err_rem isa ArgumentError
        @test occursin("correlator(:removal)", err_rem.msg)
        @test occursin("rank zero", err_rem.msg)
    end

    # ---------------------------------------------------------------
    # Tier 2 — Full complex G(ω) matrix vs. dense-Lehmann reference
    # ---------------------------------------------------------------
    # Build the dense-Lehmann reference independently of the block-Lanczos
    # path. We work on `basis_full`: H is dense 12×12, and c_{1↑}, c_{2↑}
    # (and their adjoints) are 12×12 matrices. Diagonalising the full H
    # gives eigenstates in every N-sector; we extract the (N=1) and (N=3)
    # spectra by projecting eigenvectors onto the corresponding N-blocks.
    #
    @testset "Tier 2 — full 2×2 G(ω) matrix vs. dense Lehmann" begin
        H_dense = Matrix(H_sp_full)
        F = eigen(Hermitian(H_dense))

        # Compute N occupation on every basis_full state via the compiled
        # total-N operator; this is the diagonal matrix in basis_full.
        Nf_op = n(s, 1) + n(s, 2) + n(s, 3) + n(s, 4)
        Nf_sp = assemble(compile(Nf_op, basis_full), basis_full)
        # Nf is diagonal in basis_full (number-conserving basis), so the
        # diagonal entries are the integer occupations of each basis state.
        n_per_state = round.(Int, real.(diag(Matrix(Nf_sp))))

        # Build c_{1↑}, c_{2↑} as dense matrices on basis_full. Use
        # MOADyna's assemble path so the basis-encoding and fermion-sign
        # conventions match the kernel exactly.
        c1_mat  = Matrix(assemble(compile(c_1up,    basis_full), basis_full))
        c2_mat  = Matrix(assemble(compile(c_2up,    basis_full), basis_full))
        cd1_mat = Matrix(assemble(compile(cdag_1up, basis_full), basis_full))
        cd2_mat = Matrix(assemble(compile(cdag_2up, basis_full), basis_full))
        c_ops_α  = (c1_mat,  c2_mat)     # tuple, α ∈ {1,2}
        cd_ops_α = (cd1_mat, cd2_mat)    # α ∈ {1,2}

        # ---- Dense Lehmann reference ---------------------------------
        # Intermediate-state spectra: select F-eigenvectors that live in
        # the target N-sector. H is block-diagonal in N (conserved), so
        # every eigenvector lives in a single N-sector — partitioning by
        # > 99.9% weight is exact up to numerical conditioning.
        function _project_to_sector(target_n; weight_tol = 0.999)
            idx = Int[]
            for k in eachindex(F.values)
                v = F.vectors[:, k]
                w = 0.0
                for i in eachindex(v)
                    if n_per_state[i] == target_n
                        w += abs2(v[i])
                    end
                end
                if w > weight_tol
                    push!(idx, k)
                end
            end
            return F.values[idx], F.vectors[:, idx]
        end

        E_N1, V_N1 = _project_to_sector(1)
        E_N3, V_N3 = _project_to_sector(3)
        # Sector counts with the (Sz ∈ -1:1)-restricted basis:
        # N=1 → {|1↑⟩, |2↑⟩, |1↓⟩, |2↓⟩} = 4 states.
        # N=3 → particle-hole mirror, 4 states.
        @test length(E_N1) == 4
        @test length(E_N3) == 4

        # Pre-compute source vectors:
        #   add_src[β] = c_β† · ψ₀     (β=1,2 → 1↑, 2↑)   lives in (N+1)
        #   rem_src[β] = c_β  · ψ₀     (β=1,2)            lives in (N−1)
        add_src = (cd_ops_α[1] * ψ₀, cd_ops_α[2] * ψ₀)
        rem_src = (c_ops_α[1]  * ψ₀, c_ops_α[2]  * ψ₀)

        # Lehmann matrix elements:
        #   M_add_R[β, n] = ⟨n_{N+1} | c_β† | ψ₀⟩
        #   M_add_L[α, n] = ⟨ψ₀ | c_α | n_{N+1}⟩
        M_add_R = Matrix{ComplexF64}(undef, 2, length(E_N3))
        M_add_L = Matrix{ComplexF64}(undef, 2, length(E_N3))
        for n_idx in 1:length(E_N3), β in 1:2
            v_n = V_N3[:, n_idx]
            M_add_R[β, n_idx] = dot(v_n, add_src[β])
            M_add_L[β, n_idx] = dot(ψ₀, c_ops_α[β] * v_n)
        end
        M_rem_R = Matrix{ComplexF64}(undef, 2, length(E_N1))
        M_rem_L = Matrix{ComplexF64}(undef, 2, length(E_N1))
        for m_idx in 1:length(E_N1), β in 1:2
            v_m = V_N1[:, m_idx]
            M_rem_R[β, m_idx] = dot(v_m, rem_src[β])
            M_rem_L[β, m_idx] = dot(ψ₀, cd_ops_α[β] * v_m)
        end

        # Lehmann sum at one ω.
        function _lehmann_ref(ω::Real)
            G_add = zeros(ComplexF64, 2, 2)
            G_rem = zeros(ComplexF64, 2, 2)
            for n_idx in 1:length(E_N3)
                denom = ω - (E_N3[n_idx] - Eg) + im * Γ / 2
                for α in 1:2, β in 1:2
                    G_add[α, β] += M_add_L[α, n_idx] * M_add_R[β, n_idx] / denom
                end
            end
            for m_idx in 1:length(E_N1)
                denom = ω + (E_N1[m_idx] - Eg) + im * Γ / 2
                for α in 1:2, β in 1:2
                    G_rem[α, β] += M_rem_L[α, m_idx] * M_rem_R[β, m_idx] / denom
                end
            end
            return G_add, G_rem
        end

        # ---- Kernel side --------------------------------------------
        # `state = ψ₀, Eg = Eg` puts the kernel on the same physical
        # reference state as the dense-Lehmann reference. form=:lanczos
        # keeps both channels as LanczosResponse for the pole/residue
        # checks below; we materialise on the grid via to_grid(GF, ωs).
        gf = correlator(H_sp_full, basis_full, ops, ops;
                        channel = :both, form = :lanczos, Γ = Γ,
                        state = ψ₀, Eg = Eg,
                        addition_indices = [3, 4], removal_indices = [1, 2],
                        krylovdim = Nb_full, reorth = :full,
                        tol = 1e-13, deflate_tol = 1e-13)

        # Sanity: off-diagonal element of the addition channel is non-zero
        # (so the test actually exercises the off-diagonal Lehmann sum and
        # the (1,2) vs (2,1) comparison below is meaningful — without this
        # the element-wise max-abs test could pass on a transpose-broken
        # kernel with vanishing off-diagonals).
        G_add_ref_test, _ = _lehmann_ref(0.0)
        @test abs(G_add_ref_test[1, 2]) > 1e-3

        # ---- Element-wise full-matrix comparison on the grid --------
        gf_grid = to_grid(gf, ωs; Γ = Γ)
        max_err     = 0.0
        max_err_add = 0.0
        max_err_rem = 0.0
        for (k, ω) in enumerate(ωs)
            G_add_ref, G_rem_ref = _lehmann_ref(ω)
            G_ref = G_add_ref .+ G_rem_ref
            G_kernel = gf_grid.data[:, :, k]
            max_err = max(max_err, maximum(abs, G_kernel .- G_ref))
            # Per-channel: diagnoses signed-cancellation failures.
            G_add_k = gf.addition(ω)
            G_rem_k = gf.removal(ω)
            max_err_add = max(max_err_add, maximum(abs, G_add_k .- G_add_ref))
            max_err_rem = max(max_err_rem, maximum(abs, G_rem_k .- G_rem_ref))
        end
        @info "Hubbard dimer GF element-wise max |Δ|: total = $(max_err), " *
              "addition = $(max_err_add), removal = $(max_err_rem)"
        @test max_err     < 1e-10
        @test max_err_add < 1e-10
        @test max_err_rem < 1e-10

        # ---- Pole positions ----------------------------------------
        # to_pole gives the (N±1)-sector excitation energies under the
        # Plan 2c sign convention: poles_add = E_n^{N+1} − Eg;
        # poles_rem = −(E_m^{N−1} − Eg) = Eg − E_m^{N−1}.
        P_add = to_pole(gf.addition)
        P_rem = to_pole(gf.removal)

        # Reference excitation energies, filtered to states with non-zero
        # residue. Sz=±1/2 selection rule kills 2 of the 4 N=3 states
        # (those with Sz ≠ +1/2 have zero matrix element with c†_↑|ψ₀⟩);
        # same for N=1.
        function _ref_significant_poles(E_sector, M_L, M_R, sign; tol = 1e-12)
            ps = Float64[]
            for n_idx in eachindex(E_sector)
                R_ref_norm = 0.0
                for α in 1:2, β in 1:2
                    R_ref_norm += abs2(M_L[α, n_idx] * M_R[β, n_idx])
                end
                if sqrt(R_ref_norm) > tol
                    push!(ps, sign * (E_sector[n_idx] - Eg))
                end
            end
            return sort(ps)
        end
        ref_poles_add = _ref_significant_poles(E_N3, M_add_L, M_add_R, +1)
        ref_poles_rem = _ref_significant_poles(E_N1, M_rem_L, M_rem_R, -1)

        # block-Lanczos may return up to Nb_full raw poles; filter to those
        # whose residue has non-negligible Frobenius norm. With reorth=:full
        # and krylovdim=Nb_full the deflation typically prunes exactly the
        # zero-residue selection-rule states, but we filter defensively.
        function _filter_significant(P; tol = 1e-9)
            keep = [k for k in eachindex(P.poles) if norm(P.residues[k]) > tol]
            return sort(P.poles[keep])
        end
        # Selection rule cuts the N=3 sector down to its Sz=+1/2 sub-block,
        # which has 2 states (bonding/antibonding); same for N=1 with
        # Sz=-1/2. The kernel deflates the zero-residue components so its
        # significant-pole count must match the reference count.
        sig_poles_add = _filter_significant(P_add)
        sig_poles_rem = _filter_significant(P_rem)
        @test length(sig_poles_add) == length(ref_poles_add) == 2
        @test length(sig_poles_rem) == length(ref_poles_rem) == 2
        @test maximum(abs, sig_poles_add .- ref_poles_add) < 1e-10
        @test maximum(abs, sig_poles_rem .- ref_poles_rem) < 1e-10

        # ---- Residue matrices --------------------------------------
        # For each reference pole, the residue matrix equals the analytic
        # outer product M_L[:, n] * M_R[:, n]'.
        function _residue_at(P, target_pole; tol = 1e-8)
            best_k = 0
            best_d = Inf
            for k in eachindex(P.poles)
                d = abs(P.poles[k] - target_pole)
                if d < best_d
                    best_d = d
                    best_k = k
                end
            end
            @assert best_d < tol "no kernel pole within $tol of $target_pole"
            return P.residues[best_k]
        end

        max_res_err_add = 0.0
        for n_idx in 1:length(E_N3)
            ref_pole = E_N3[n_idx] - Eg
            R_ref = Matrix{ComplexF64}(undef, 2, 2)
            for α in 1:2, β in 1:2
                R_ref[α, β] = M_add_L[α, n_idx] * M_add_R[β, n_idx]
            end
            if norm(R_ref) < 1e-12
                continue  # vanishing reference residue (selection rule) — skip
            end
            R_kernel = _residue_at(P_add, ref_pole)
            max_res_err_add = max(max_res_err_add, norm(R_kernel .- R_ref))
        end
        max_res_err_rem = 0.0
        for m_idx in 1:length(E_N1)
            ref_pole = -(E_N1[m_idx] - Eg)
            R_ref = Matrix{ComplexF64}(undef, 2, 2)
            for α in 1:2, β in 1:2
                R_ref[α, β] = M_rem_L[α, m_idx] * M_rem_R[β, m_idx]
            end
            if norm(R_ref) < 1e-12
                continue
            end
            R_kernel = _residue_at(P_rem, ref_pole)
            max_res_err_rem = max(max_res_err_rem, norm(R_kernel .- R_ref))
        end
        @info "Hubbard dimer GF residue-matrix max Frobenius |Δ|: " *
              "addition = $(max_res_err_add), removal = $(max_res_err_rem)"
        @test max_res_err_add < 1e-10
        @test max_res_err_rem < 1e-10
    end
end
