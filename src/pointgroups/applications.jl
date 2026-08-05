# expand_clm (multiplicity-aware Akm pipeline), expand_clm_central
# (helper subcase), nparams, LiftedRep / lift (single-particle rotation
# matrices), classify (rep-theory-level: takes precomputed subspace
# characters or projector weights). The many-body Fock-space apply that
# consumes `lift` lives in `MOAD.Diagnostics` as
# `classify_state(ψ, basis, m, G)` (it needs the Shells/Bases layers,
# which load after PointGroups).

# ─── C^k_m matrix elements via Wigner-Eckart (8.2) ───────────────────

"""
    Bkm_matrix(ℓ::Int, k::Int, m::Int) -> Matrix{ComplexF64}

The (2ℓ+1)×(2ℓ+1) matrix `(B_{km})_{m₁,m₂} = ⟨ℓ m₁|C^k_m|ℓ m₂⟩` per
formula (8.2). Row index runs `m₁ = -ℓ..+ℓ`, column `m₂ = -ℓ..+ℓ`.
Selection rule `m₁ = m₂ + m` zeros all off-band entries.
"""
function Bkm_matrix(ℓ::Int, k::Int, m::Int)
    n = 2ℓ + 1
    B = zeros(ComplexF64, n, n)
    iseven(k) && 0 ≤ k ≤ 2ℓ || return B
    abs(m) ≤ k || return B
    pre = (2ℓ + 1) * Float64(wigner3j(Float64, ℓ, k, ℓ, 0, 0, 0))
    for (i, m1) in enumerate(-ℓ:ℓ), (j, m2) in enumerate(-ℓ:ℓ)
        m1 != m2 + m && continue
        sign = iseven(m1) ? 1.0 : -1.0
        w = Float64(wigner3j(Float64, ℓ, k, ℓ, -m1, m, m2))
        B[i, j] = sign * pre * w
    end
    return B
end

# Cached basis matrix M_ℓ = [vec(B_{km})] and its QR factorisation.
const _BKM_CACHE = Dict{Int, NamedTuple{(:keys, :M, :Q, :R), Tuple{Vector{Tuple{Int,Int}}, Matrix{ComplexF64}, Matrix{ComplexF64}, Matrix{ComplexF64}}}}()

function _bkm_basis(ℓ::Int)
    haskey(_BKM_CACHE, ℓ) && return _BKM_CACHE[ℓ]
    keys = Tuple{Int, Int}[]
    cols = Vector{Vector{ComplexF64}}()
    for k in 0:2:2ℓ, m in -k:k
        B = Bkm_matrix(ℓ, k, m)
        norm(B) < 1e-14 && continue
        push!(keys, (k, m))
        push!(cols, vec(B))
    end
    M = hcat(cols...)
    F = qr(M)
    Qmat = Matrix{ComplexF64}(F.Q)
    Rmat = Matrix{ComplexF64}(F.R)
    entry = (keys=keys, M=M, Q=Qmat, R=Rmat)
    _BKM_CACHE[ℓ] = entry
    return entry
end

# ─── Real tesseral basis matrix on V_ℓ ───────────────────────────────

# T columns (canonical real-tesseral order):
#   index 1   : T_{ℓ,0} = Y_{ℓ,0}                       (m=0)
#   index 2   : T_{ℓ,1c} = (Y_{ℓ,1}  + Y_{ℓ,-1}) / √2
#   index 3   : T_{ℓ,1s} = (Y_{ℓ,1}  - Y_{ℓ,-1}) / (i√2)
#   index 4   : T_{ℓ,2c} = (Y_{ℓ,2}  + Y_{ℓ,-2}) / √2
#   ...
#   index 2k  : T_{ℓ,kc}
#   index 2k+1: T_{ℓ,ks}
# Y_{ℓ, m} basis indexed by m = -ℓ..+ℓ (rows). Returns the unitary
# transform `T` and a tag list describing each column's `(|m|, c|s|0)`.
function _tesseral_transform(ℓ::Int)
    n = 2ℓ + 1
    T = zeros(ComplexF64, n, n)
    tags = Vector{Tuple{Int, Char}}(undef, n)   # (|m|, '0' | 'c' | 's')
    function row_idx(m)
        return m + ℓ + 1
    end
    # column 1: m=0
    T[row_idx(0), 1] = 1.0
    tags[1] = (0, '0')
    col = 2
    for m in 1:ℓ
        T[row_idx(m),  col] = 1 / sqrt(2)
        T[row_idx(-m), col] = 1 / sqrt(2)
        tags[col] = (m, 'c')
        col += 1
        T[row_idx(m),  col] = -1im / sqrt(2)
        T[row_idx(-m), col] = 1im / sqrt(2)
        tags[col] = (m, 's')
        col += 1
    end
    return T, tags
end

# ─── Canonical multiplicity-aware symmetry-adapted basis ─────────────

# For each Mulliken IR Γ with mult m_Γ: build `m_Γ * d_Γ` orthonormal
# columns of the symmetry-adapted basis U on V_ℓ, deterministically
# pinned by leading-tesseral pivoting + matrix-unit intertwiners +
# sign fix.
#
# Returns (U, ir_blocks) where ir_blocks is a vector of NamedTuples
# describing each Mulliken IR's slot:
#   (label, real_dim, m_Γ, col_range, complex_constituents)
function _symmetry_adapted_basis(G::PointGroup, ℓ::Int; experimental::Bool=false)
    n = 2ℓ + 1
    n_elems = length(G.elements)
    elem_to_class = Vector{Int}(undef, n_elems)
    for (k, C) in enumerate(G.classes), g in C
        elem_to_class[g] = k
    end
    # D^ℓ(g) per element (cached locally).
    Dl = [_wignerd_matrix(e.matrix, ℓ) for e in G.elements]
    # Tesseral basis (already deterministic).
    T_tess, _tags = _tesseral_transform(ℓ)
    cols = Vector{Vector{ComplexF64}}()
    blocks = Vector{NamedTuple{(:label, :real_dim, :m_Γ, :col_range,
                                :complex_constituents),
                                Tuple{Symbol, Int, Int, UnitRange{Int}, Vector{Symbol}}}}()
    occupied = falses(n)   # tracks which V_ℓ-rays are already used (for
                           # checks; not used to physically gate)

    for ir in G.irreps
        m_Γ = _mult_in_Dl(G, ir, ℓ, elem_to_class, Dl)
        m_Γ == 0 && continue

        # Build IR projector P_Γ on V_ℓ via sum-over-constituents.
        P_Γ = zeros(ComplexF64, n, n)
        for cid in ir.complex_constituents
            cir = G.complex_irreps[cid]
            P_γ = zeros(ComplexF64, n, n)
            for g in 1:n_elems
                P_γ .+= conj(cir.characters[elem_to_class[g]]) * Dl[g]
            end
            P_γ .*= cir.dim / n_elems
            P_Γ .+= P_γ
        end
        # Project tesseral basis through P_Γ.
        proj_tess = P_Γ * T_tess   # n × n; column j = P_Γ T_tess[:, j]

        # Step 2c — pivot to multiplicity copies.
        # Use rank-revealing QR with column pivoting; the rank should be
        # m_Γ * d_Γ. We pick m_Γ representative tesseral columns by
        # walking in canonical order and accepting columns that are
        # linearly independent of those already accepted.
        # (Implementation: column-pivoted QR returns a permutation; we
        # iterate over the permuted order, accepting one tesseral seed
        # per multiplicity copy — defined as a column whose contribution
        # span is not absorbed by previously-accepted seeds + their orbit.)
        seeds = Vector{Vector{ComplexF64}}()
        seed_indices = Int[]
        for tcol in 1:n
            v = proj_tess[:, tcol]
            norm(v) < 1e-9 && continue
            # Reject if v lies in the orbit-span of already-accepted seeds.
            if !isempty(seeds)
                # Build orbit basis from seeds: span of {D^ℓ(g) seed_k} for all g, k.
                orbit_cols = Vector{Vector{ComplexF64}}()
                for s in seeds
                    for g in 1:n_elems
                        push!(orbit_cols, Dl[g] * s)
                    end
                end
                Q_orbit = _orthonormalise(orbit_cols)
                v_perp = v - Q_orbit * (Q_orbit' * v)
                norm(v_perp) < 1e-7 && continue
                v = v_perp
            end
            push!(seeds, v / norm(v))
            push!(seed_indices, tcol)
            length(seeds) == m_Γ && break
        end
        length(seeds) == m_Γ || error(
            "canonical-basis pivot: only found $(length(seeds)) of $m_Γ " *
            "multiplicity-copy seeds for IR $(ir.label) at ℓ=$ℓ")

        # Step 2d — within-copy IR-basis adaptation via matrix-unit
        # intertwiners. Branch on real-type vs folded complex-type.
        is_folded = length(ir.complex_constituents) == 2
        copy_bases = Vector{Matrix{ComplexF64}}()   # one (n × d_Γ) matrix per copy
        for seed in seeds
            if !is_folded
                # Real-type: matrix units on the single constituent γ.
                cir = G.complex_irreps[ir.complex_constituents[1]]
                d_γ = cir.dim
                # Find IR-row index b* of seed.
                b_star = _find_seed_row(seed, cir, Dl, elem_to_class, n_elems)
                # Generate full IR-row basis.
                basis = zeros(ComplexF64, n, d_γ)
                for a in 1:d_γ
                    P_ab = _matrix_unit_intertwiner(cir, a, b_star, Dl, elem_to_class, n_elems)
                    v = P_ab * seed
                    nrm = norm(v)
                    nrm < 1e-9 && error(
                        "matrix-unit intertwiner P^$(cir.id)_{$a,$b_star} produced " *
                        "zero vector on seed for IR $(ir.label)")
                    basis[:, a] = v / nrm
                end
                # Sign fix: largest-component-positive in the tesseral frame.
                _signfix_basis!(basis, T_tess)
                push!(copy_bases, basis)
            else
                # Folded complex-type: per-constituent matrix-unit then
                # cos/sine combination. Cyclic-family irreps have d_γ = 1,
                # so this reduces to:
                #   |Γ, c⟩ = (|γ⟩ + |γ̄⟩) / √2
                #   |Γ, s⟩ = (|γ⟩ - |γ̄⟩) / (i√2)
                cid_γ, cid_γ̄ = ir.complex_constituents
                cir_γ  = G.complex_irreps[cid_γ]
                cir_γ̄ = G.complex_irreps[cid_γ̄]
                cir_γ.dim == 1 && cir_γ̄.dim == 1 || error(
                    "folded-complex branch only supports d_γ=1 cyclic-family " *
                    "constituents (got d_γ=$(cir_γ.dim))")
                P_γ_seed  = _scalar_irrep_projector(cir_γ,  Dl, elem_to_class, n_elems)
                P_γ̄_seed = _scalar_irrep_projector(cir_γ̄, Dl, elem_to_class, n_elems)
                vγ  = P_γ_seed  * seed
                vγ̄ = P_γ̄_seed * seed
                nγ  = norm(vγ)
                nγ̄ = norm(vγ̄)
                # Normalise; outer normalisation absorbs the 1/√2 factors.
                if nγ > 1e-9
                    vγ ./= nγ
                end
                if nγ̄ > 1e-9
                    vγ̄ ./= nγ̄
                end
                vc = (vγ + vγ̄) / sqrt(2)
                vs = (vγ - vγ̄) / (1im * sqrt(2))
                basis = zeros(ComplexF64, n, 2)
                if norm(vc) > 1e-9
                    basis[:, 1] = vc / norm(vc)
                end
                if norm(vs) > 1e-9
                    basis[:, 2] = vs / norm(vs)
                end
                _signfix_basis!(basis, T_tess)
                push!(copy_bases, basis)
            end
        end

        # Step 2e — multiplicity-frame alignment for real-type IRs with m_Γ > 1.
        # Each copy's basis spans a d_Γ-dimensional subspace. Different copies may
        # occupy subspaces whose real-tesseral D^ℓ subblocks realise the SAME IR
        # with opposite orientations (e.g. C4 sends T_{3,1c}→+T_{3,1s} for E^(1)
        # but T_{3,3c}→−T_{3,3s} for E^(2)). To align all copies to the same
        # real-tesseral convention, we rotate the multiplicity space by an
        # m_Γ×m_Γ orthogonal matrix W. The convention below is MOAD's internal
        # multiplicity-frame, anchored to a B_{2,0}-zero-coupling rule on the
        # last copy and a positivity rule on the first. It coincides with the
        # multiplicity convention exercised by the Quanty Akm regression suite
        # for the cells the suite covers; it has not been independently anchored
        # to a sourced explicit-irrep-matrix library (Koster / Bilbao / libmsym).
        # Sourced explicit-irrep-matrix anchoring (Koster / Bilbao /
        # libmsym) is a planned future extension.
        #
        # Convention:
        #   b_k = Re(⟨basis_k | B_{2,0} | basis_k⟩) / d_Γ  (per-copy A_{2,0} coupling)
        #   W is chosen so that:
        #     - W[end, :] (last copy) has ZERO B_{2,0} coupling: Σ_k W[m_Γ,k]² b_k = 0
        #     - W[1, :] (first copy) has all positive entries (canonical orientation)
        #   For m_Γ=2: W[2,:] = [√(−b₂/(b₁−b₂)), √(b₁/(b₁−b₂))] and
        #              W[1,:] = the orthogonal complement with both entries positive.
        #
        # This is triggered only for real-type (non-folded) IRs. Folded complex-type
        # IRs (cyclic d_γ=1) do not have this issue and are left unchanged.
        if m_Γ > 1 && !is_folded
            d_Γ = length(copy_bases) > 0 ? size(copy_bases[1], 2) : 0
            # Alignment is only meaningful for multi-dimensional IRs (d_Γ > 1).
            # For 1D IRs (d_Γ = 1) with multiplicity > 1, the copies represent
            # genuinely distinct physical states (e.g. D2h Ag: z² vs x²-y²) that
            # Quanty already separates without mixing — no alignment needed.
            if d_Γ > 1
                # Compute per-copy B_{2,0} coupling.
                B20 = Bkm_matrix(ℓ, 2, 0)
                b_vec = zeros(Float64, m_Γ)
                for k in 1:m_Γ
                    bk_sum = ComplexF64(0)
                    for a in 1:d_Γ
                        bk_sum += copy_bases[k][:, a]' * B20 * copy_bases[k][:, a]
                    end
                    b_vec[k] = real(bk_sum) / d_Γ
                end

                # Build W: m_Γ × m_Γ orthogonal matrix.
                # For m_Γ ≥ 3 with d_Γ > 1, the alignment is not yet
                # implemented; `experimental=true` falls back to identity
                # (MOAD's internal gauge — possibly convention-drifting).
                W = (m_Γ ≥ 3 && experimental) ?
                    Matrix{Float64}(I, m_Γ, m_Γ) :
                    _multiplicity_frame_W(b_vec)

                # Apply W to each IR-row r: new_copy_k_r = Σ_j W[k,j] * old_copy_j_r
                new_copy_bases = [zeros(ComplexF64, n, d_Γ) for _ in 1:m_Γ]
                for k in 1:m_Γ, r in 1:d_Γ
                    for j in 1:m_Γ
                        new_copy_bases[k][:, r] .+= W[k, j] .* copy_bases[j][:, r]
                    end
                end
                copy_bases = new_copy_bases
            end
        end

        # Step 2f — concatenate: each copy contributes d_Γ columns.
        col_start = length(cols) + 1
        for basis in copy_bases
            for c in eachcol(basis)
                push!(cols, ComplexF64.(c))
            end
        end
        col_end = length(cols)
        push!(blocks, (label=ir.label, real_dim=ir.real_dim, m_Γ=m_Γ,
                       col_range=col_start:col_end,
                       complex_constituents=ir.complex_constituents))
    end

    U = hcat(cols...)
    return U, blocks
end

function _orthonormalise(vecs::Vector{Vector{ComplexF64}})
    isempty(vecs) && return zeros(ComplexF64, 0, 0)
    M = hcat(vecs...)
    F = svd(M)
    rank_check = count(s -> s > 1e-9, F.S)
    return F.U[:, 1:rank_check]
end

function _mult_in_Dl(G::PointGroup, ir::IRrep, ℓ::Int,
                     elem_to_class::Vector{Int}, Dl::Vector{Matrix{ComplexF64}})
    n_elems = length(G.elements)
    χ_Dl = ComplexF64[tr(Dl[g]) for g in 1:n_elems]
    # multiplicity from any one constituent (asserted equal across pair)
    cir = G.complex_irreps[ir.complex_constituents[1]]
    s = ComplexF64(0)
    for g in 1:n_elems
        s += conj(cir.characters[elem_to_class[g]]) * χ_Dl[g]
    end
    return round(Int, real(s / n_elems))
end

function _matrix_unit_intertwiner(cir::ComplexIR, a::Int, b::Int,
                                   Dl::Vector{Matrix{ComplexF64}},
                                   elem_to_class::Vector{Int}, n_elems::Int)
    # P^γ_{ab} = (d_γ / |G|) Σ_g ρ_γ(g)*_{ab} D^ℓ(g)
    n = size(Dl[1], 1)
    P = zeros(ComplexF64, n, n)
    for g in 1:n_elems
        P .+= conj(cir.matrices[g][a, b]) * Dl[g]
    end
    P .*= cir.dim / n_elems
    return P
end

function _scalar_irrep_projector(cir::ComplexIR, Dl::Vector{Matrix{ComplexF64}},
                                  elem_to_class::Vector{Int}, n_elems::Int)
    # 1×1 case: P = (1/|G|) Σ_g χ_γ(g)* D^ℓ(g)
    n = size(Dl[1], 1)
    P = zeros(ComplexF64, n, n)
    for g in 1:n_elems
        P .+= conj(cir.characters[elem_to_class[g]]) * Dl[g]
    end
    P ./= n_elems
    return P
end

function _find_seed_row(seed::Vector{ComplexF64}, cir::ComplexIR,
                        Dl::Vector{Matrix{ComplexF64}},
                        elem_to_class::Vector{Int}, n_elems::Int)
    d_γ = cir.dim
    best_b = 1
    best_norm = -1.0
    for b in 1:d_γ
        P_bb = _matrix_unit_intertwiner(cir, b, b, Dl, elem_to_class, n_elems)
        v = P_bb * seed
        nrm = norm(v)
        if nrm > best_norm
            best_norm = nrm
            best_b = b
        end
    end
    return best_b
end

function _signfix_basis!(basis::Matrix{ComplexF64}, T_tess::Matrix{ComplexF64})
    # Express each column in the tesseral basis: c_tess = T_tess^† column.
    # First nonzero component (in canonical tesseral order) must be
    # positive real. Multiply column by appropriate phase.
    for col in 1:size(basis, 2)
        c_tess = T_tess' * basis[:, col]
        for i in 1:length(c_tess)
            abs(c_tess[i]) < 1e-9 && continue
            phase = c_tess[i] / abs(c_tess[i])
            basis[:, col] .*= conj(phase)
            break
        end
    end
end

"""
    _multiplicity_frame_W(b_vec::Vector{Float64}) -> Matrix{Float64}

Compute the m_Γ × m_Γ orthogonal multiplicity-frame alignment matrix W for a
real-type IR with multiplicity m_Γ = length(b_vec), where `b_vec[k]` is the
per-copy B_{2,0} coupling:

    b_k = Re(⟨basis_k | B_{2,0} | basis_k⟩) / d_Γ

Convention (canonical B_{2,0}-zero alignment):
- W[end, :] has ZERO B_{2,0} coupling: `Σ_k W[m_Γ, k]² × b_k = 0`.
- W[1, :] has all positive entries (canonical orientation).
- For m_Γ = 1 (trivial case): returns the 1×1 identity.
- For m_Γ = 2: the unique (up to sign of individual columns) solution.
- For m_Γ ≥ 3 with d_Γ > 1: not yet implemented — throws `ArgumentError`.
  `expand_clm(...; experimental=true)` falls back to the identity
  placeholder for these cells.

When all b_k are equal (indeterminate case) or the zero-coupling condition
has no real solution (all b_k same sign), W defaults to the identity.
"""
function _multiplicity_frame_W(b_vec::Vector{Float64})
    m = length(b_vec)
    m == 1 && return ones(Float64, 1, 1)

    b_min, b_max = minimum(b_vec), maximum(b_vec)
    # Indeterminate: all couplings equal or no zero-coupling solution exists
    # (i.e. all b_k same sign).
    if abs(b_max - b_min) < 1e-10 || b_min * b_max >= 0.0
        return Matrix{Float64}(I, m, m)
    end

    if m == 2
        b1, b2 = b_vec
        denom = b1 - b2
        # W[2,:] = [α, β] with b1 α² + b2 β² = 0 and α² + β² = 1.
        # α² = -b2/denom, β² = b1/denom  (both positive when b1 > 0 > b2)
        α = sqrt(max(0.0, -b2 / denom))
        β = sqrt(max(0.0,  b1 / denom))
        # W[1,:] = [-β, α] or [β, α] — choose both positive for canonical form.
        # If b1 > b2: α > β, so W[1,:] = [β, α] gives row-1 = [smaller, larger] > 0.
        #             W[2,:] = [α, -β] (orthogonal complement with sign fix).
        # Verify: W[1,1]=β>0, W[1,2]=α>0 ✓; W[2,:] orthogonal to W[1,:].
        W = Float64[β  α;
                    α -β]
        return W
    end

    # m ≥ 3 with d_Γ > 1: alignment not yet implemented. Reachable for
    # higher-shell low-symmetry combinations (e.g. C3v ℓ=4 has E with
    # m_Γ=3, d_Γ=2). Refuse loudly rather than return an arbitrary gauge.
    # Pass `experimental=true` to `expand_clm` to fall back to the identity
    # placeholder (no canonical alignment).
    throw(ArgumentError(
        "multiplicity-frame alignment for m_Γ = $m with d_Γ ≥ 2 is not " *
        "yet implemented (reached for example by C3v ℓ=4 E with m_Γ=3, " *
        "d_Γ=2). The current pipeline cannot place the m_Γ × m_Γ " *
        "multiplicity block in a stable canonical orientation without " *
        "a sourced explicit-irrep-matrix table. Pass `experimental=true` " *
        "to `expand_clm` to accept MOAD's internal (possibly " *
        "convention-drifting) basis for this cell."))
end

# ─── nparams ─────────────────────────────────────────────────────────

"""
    nparams(G::PointGroup, ℓ::Int) -> Int

Return the length of the flat `params` vector accepted by
`expand_clm(G, ℓ, params)`. This equals the total number of independent
real crystal-field parameters needed to specify an arbitrary
`G`-symmetric operator on an angular-momentum shell of rank `ℓ`.

Computed as `Σ_Γ dim(CF-reachable_Γ)` — the dimension of each irrep's
CF-reachable Hermitian basis (`_cf_block_basis`) summed over the
irreps Γ occurring in `D^ℓ↓G`. For `⊗I` irreps this equals
`m_Γ(m_Γ+1)/2` (the historical real-symmetric count); for `⊗J`
(trigonal/pentagonal/no-symmetry) irreps the basis additionally carries
an imaginary direction. Asserts the per-block sum equals the global CF
rank (`_cf_rank`), erroring if shared `A_km` directions would be
double-counted across blocks.
"""
function nparams(G::PointGroup, ℓ::Int)
    U, blocks = _symmetry_adapted_basis(G, ℓ)
    n = sum(length(_cf_block_basis(G, ℓ, blk, U)) for blk in blocks; init=0)
    n == _cf_rank(G, ℓ) || error(
        "nparams: per-block CF dims ($n) ≠ global CF rank ($(_cf_rank(G, ℓ))) for " *
        ":$(G.name) ℓ=$ℓ — shared Akm directions are double-counted across blocks; " *
        "this group needs a global CF-coordinate parametrization (not yet implemented).")
    return n
end

# ─── expand_clm + expand_clm_central ─────────────────────────────────

# Strict-mode gate: every IR must carry a `:reference` provenance label.
function _expand_clm_strict_gate(G::PointGroup, experimental::Bool)
    if !experimental && !all(ir.provenance === :reference for ir in G.irreps)
        throw(ArgumentError(
            "expand_clm: group :$(G.name) has IR labels with provenance " *
            "$(unique(ir.provenance for ir in G.irreps)); production mode requires " *
            "every IR to carry :reference provenance. Pick a group from " *
            "`reference_label_groups()` or pass `experimental=true`."))
    end
end

# Assemble V_shell by placing each per-IR Hermitian full isotypic block (in
# `ir_blocks` order) into its `col_range` of the block-diagonal V_sym, then
# solve for the Akm.
function _expand_clm_core(G::PointGroup, ℓ::Int, U::AbstractMatrix, ir_blocks,
                          Hblocks::AbstractVector{<:AbstractMatrix})
    n = 2ℓ + 1
    V_sym = zeros(ComplexF64, n, n)
    for (blk, Hblk) in zip(ir_blocks, Hblocks)
        V_sym[blk.col_range, blk.col_range] = Hblk     # full isotypic block
    end
    V_shell = U * V_sym * U'
    @assert norm(V_shell - V_shell') / max(1.0, norm(V_shell)) < 1e-7 "expand_clm: V_shell not Hermitian — basis-pin bug"
    V_shell = (V_shell + V_shell') / 2
    return _solve_Akm(V_shell, ℓ)
end

"""
    expand_clm(G::PointGroup, ℓ::Int, blocks::AbstractVector{<:AbstractMatrix};
               experimental::Bool=false)

Multiplicity-aware crystal-field expansion. `blocks[i]` is the Hermitian
block `H_Γ` for the i-th occurring IR (in `subduce(G, ℓ)` order), supplied
either as a compact `m_Γ × m_Γ` matrix (interpreted as `H_Γ ⊗ I_{d_Γ}`) or
as the full `(m_Γ·d_Γ) × (m_Γ·d_Γ)` isotypic block. Each block must lie in the
IR's **CF-reachable subspace** (the projection of the multiplicative crystal
field onto that block); non-CF blocks are rejected. Returns a vector of
`(k=k, m=m, coeff=A_km)` named tuples for the non-zero `A_{km}`.

The full isotypic form is needed to express row-endomorphism couplings (`⊗J`,
e.g. the trigonal `A_{4,±3}` of D3d) that a compact `⊗I` block cannot reach.
Results are at MOAD's single canonical orientation (the hand-typed generators);
multi-setting support is a separate (spgrep/libmsym) effort. The
[`expand_clm(G, ℓ, params::AbstractVector{<:Real})`](@ref) overload takes real
coordinates on each IR's CF-reachable Hermitian basis — for real `⊗I` IRs these
are the historical diagonal-then-symmetric-off-diagonal entries (byte-identical
to the pre-fix encoding), for trigonal/pentagonal and no-symmetry IRs they
include the imaginary couplings.

# The `experimental` keyword

By default (`experimental=false`), `expand_clm` is in **strict mode**
and only accepts cells where the output convention is anchored:

- Every IR of `G` must carry `:reference` provenance — i.e., the group
  has a curated Mulliken-label table (`has_reference_labels(G.name)`),
  so the IR ordering of `blocks` matches the published convention.
- Every multiplicity-frame block must be in a stable canonical
  orientation. Currently this means `m_Γ ≤ 2` for any IR with `d_Γ > 1`
  (the alignment for `m_Γ ≥ 3` is not yet implemented).

`experimental=true` relaxes both gates:

- Auto-named groups (`:computed_auto` provenance) become acceptable;
  the IR ordering is MOAD's deterministic but possibly
  convention-drifting choice.
- The `m_Γ ≥ 3` multiplicity-frame alignment falls back to identity;
  the `m_Γ × m_Γ` block is in MOAD's internal basis without external
  anchoring.

Use `experimental=true` only when you accept that the output may not
match an external code's convention.
"""
function expand_clm(G::PointGroup, ℓ::Int, blocks::AbstractVector{<:AbstractMatrix};
                    experimental::Bool=false)
    _expand_clm_strict_gate(G, experimental)
    sub = subduce(G, ℓ)
    length(blocks) == length(sub) || throw(ArgumentError(
        "expand_clm: $(length(blocks)) blocks supplied but D^$ℓ↓$(G.name) decomposes " *
        "into $(length(sub)) IRs (one block per IR in `subduce` order): " *
        join(["$lbl⊕$m" for (lbl, m) in sub], " + ")))
    U, ir_blocks = _symmetry_adapted_basis(G, ℓ; experimental=experimental)
    basis_of = Dict(blk.label => _cf_block_basis(G, ℓ, blk, U) for blk in ir_blocks)
    dim_of   = Dict(blk.label => (blk.m_Γ, blk.real_dim) for blk in ir_blocks)
    # Accept a compact mΓ×mΓ block (interpreted as H⊗I_dΓ) OR the full
    # (mΓ·dΓ)×(mΓ·dΓ) isotypic block; then enforce CF-membership. The full form
    # is needed to express row-endomorphism (⊗J, e.g. trigonal) couplings.
    full_of = Dict{Symbol, Matrix{ComplexF64}}()
    for (i, (lbl, _)) in enumerate(sub)
        mq, dq = dim_of[lbl]
        B = ComplexF64.(blocks[i])
        if size(B) == (mq, mq)
            Hf = kron(B, Matrix{ComplexF64}(I, dq, dq))
        elseif size(B) == (mq * dq, mq * dq)
            Hf = Matrix{ComplexF64}(B)
        else
            throw(ArgumentError(
                "expand_clm: block $i for IR :$lbl must be $(mq)×$(mq) (compact, ⊗I) " *
                "or $(mq*dq)×$(mq*dq) (full isotypic block), got $(size(B))"))
        end
        norm(Hf - Hf') < 1e-9 || throw(ArgumentError(
            "expand_clm: block $i for IR :$lbl is not Hermitian"))
        _, resid = _project_onto_basis(Hf, basis_of[lbl])
        resid < _CF_RESID_TOL || throw(ArgumentError(
            "expand_clm: block for IR :$lbl is not in the CF-reachable subspace " *
            "(rel residual $resid). Its structure cannot be produced by any " *
            "multiplicative crystal field of D^$ℓ↓$(G.name) at this orientation."))
        full_of[lbl] = Hf
    end
    Hblocks = [full_of[blk.label] for blk in ir_blocks]
    return _expand_clm_core(G, ℓ, U, ir_blocks, Hblocks)
end

# Flat-vector overload: real coordinates on each IR block's CF-reachable
# Hermitian basis (`_cf_block_basis`), concatenated in subduce() IR order. For
# real-coupling IRs these are the historical diagonal-then-symmetric-off-diagonal
# entries; for trigonal/pentagonal and no-symmetry IRs they include the imaginary
# (e.g. A_{4,±3}) couplings the former real-symmetric encoding could not represent.
function expand_clm(G::PointGroup, ℓ::Int, params::AbstractVector{<:Real};
                    experimental::Bool=false)
    _expand_clm_strict_gate(G, experimental)
    sub = subduce(G, ℓ)
    U, ir_blocks = _symmetry_adapted_basis(G, ℓ; experimental=experimental)
    basis_of = Dict(blk.label => _cf_block_basis(G, ℓ, blk, U) for blk in ir_blocks)
    expected = sum(length(basis_of[lbl]) for (lbl, _) in sub; init=0)
    length(params) == expected || throw(ArgumentError(
        "expand_clm: expected $expected real CF coordinates for D^$ℓ↓$(G.name) " *
        "(IRs: $(["$lbl×$m" for (lbl, m) in sub])); got $(length(params))"))
    H_of = Dict{Symbol, Matrix{ComplexF64}}()
    idx = 1
    for (lbl, _) in sub
        b = basis_of[lbl]
        sz = isempty(b) ? 0 : size(b[1], 1)
        H = zeros(ComplexF64, sz, sz)
        for P in b
            H .+= params[idx] .* P
            idx += 1
        end
        H_of[lbl] = H
    end
    Hblocks = [H_of[blk.label] for blk in ir_blocks]
    return _expand_clm_core(G, ℓ, U, ir_blocks, Hblocks)
end

"""
    expand_clm_central(G::PointGroup, ℓ::Int, energies::AbstractVector{<:Real};
                       experimental::Bool=false)

Crystal-field expansion for the "central" (scalar-multiple-of-identity)
case: assign one energy level `εᵢ` per distinct irrep in `D^ℓ↓G`,
with no off-diagonal mixing between multiplicity copies.

`energies[i]` is the level energy for the i-th irrep in `subduce(G, ℓ)`
order. Each irrep block is set to `H_Γ = εᵢ I_{m_Γ}`, so all `m_Γ`
multiplicity copies of irrep Γ are degenerate. The function delegates to
`expand_clm` and returns a vector of `(k, m, coeff)` named tuples giving
the non-zero `A_{km}` crystal-field coefficients.

This is the special case of `expand_clm` in which the `m_Γ × m_Γ` blocks
are diagonal and constant; use `expand_clm` directly when off-diagonal or
non-degenerate multiplicity-space entries are needed.
"""
function expand_clm_central(G::PointGroup, ℓ::Int, energies::AbstractVector{<:Real};
                              experimental::Bool=false)
    sub = subduce(G, ℓ)
    length(energies) == length(sub) || throw(ArgumentError(
        "expand_clm_central: expected $(length(sub)) energies (one per IR " *
        "in $(["$lbl×$m" for (lbl, m) in sub])), got $(length(energies))"))
    blocks = [Matrix{Float64}(energies[i] * I, m_Γ, m_Γ)
              for (i, (_, m_Γ)) in enumerate(sub)]
    return expand_clm(G, ℓ, blocks; experimental=experimental)
end

# ─── A_km least-squares solve via cached B_km QR ─────────────────────

function _solve_Akm(V_shell::AbstractMatrix, ℓ::Int)
    cache = _bkm_basis(ℓ)
    v = vec(V_shell)
    A = cache.R \ (cache.Q' * v)
    res = norm(cache.M * A - v) / max(1.0, norm(v))
    res < _CF_RESID_TOL || error(
        "_solve_Akm: V is not in the multiplicative-CF subspace (rel residual = $res). " *
        "Either the input operator is not a crystal field, or a basis-pin bug produced a " *
        "non-CF V_shell.")
    out = NamedTuple{(:k, :m, :coeff), Tuple{Int, Int, ComplexF64}}[]
    for (i, (k, m)) in enumerate(cache.keys)
        abs(A[i]) < 1e-12 && continue
        push!(out, (k=k, m=m, coeff=A[i]))
    end
    return out
end

# ─── CF-reachable block basis ────────────────────────────────────────
#
# The multiplicative crystal field is V = Σ A_km C^k_m. Its Hermitian
# generators (using (C^k_m)† = (-1)^m C^k_{-m}) are, for even k in 0..2ℓ:
#   m = 0 :  B_{k0}
#   m > 0 :  H^c_{km} = B_{km} + (-1)^m B_{k,-m}
#            H^s_{km} = i(B_{km} - (-1)^m B_{k,-m})
# Group-symmetrising these and restricting to each irrep block's multiplicity
# space gives the CF-reachable Hermitian subspace per block — the true set of
# crystal-field degrees of freedom, including imaginary (trigonal) couplings
# that the former real-symmetric parametrisation could not represent.

# Named tolerances for the CF-reachable basis machinery (one place).
const _CF_RESID_TOL = 1e-8    # CF-membership / _solve_Akm relative residual
const _CF_RANK_TOL  = 1e-9    # rank / linear-independence counting

function _hermitian_cf_generators(ℓ::Int)
    gens = Matrix{ComplexF64}[]
    for k in 0:2:2ℓ
        B0 = Bkm_matrix(ℓ, k, 0)
        norm(B0) > 1e-12 && push!(gens, B0)
        for m in 1:k
            Bp = Bkm_matrix(ℓ, k, m)
            Bm = Bkm_matrix(ℓ, k, -m)
            (norm(Bp) < 1e-12 && norm(Bm) < 1e-12) && continue
            s = iseven(m) ? 1.0 : -1.0
            push!(gens, Bp + s * Bm)        # H^c (Hermitian)
            push!(gens, im * (Bp - s * Bm)) # H^s (Hermitian)
        end
    end
    return gens
end

# Orthonormalise Hermitian matrices as a REAL inner-product space
# (⟨A,B⟩ = Re tr(A'B)) by DETERMINISTIC modified Gram-Schmidt in the given
# input order (no column pivoting ⇒ reproducible basis across calls), with a
# sign fix (largest-magnitude real coordinate made positive). These coordinates
# are a public flat contract, so determinism here is required, not cosmetic.
function _orthonormal_hermitian_basis(mats::AbstractVector{<:AbstractMatrix})
    isempty(mats) && return Matrix{ComplexF64}[]
    d = size(mats[1], 1)
    tovec(A) = vcat(real(vec(Matrix{ComplexF64}(A))), imag(vec(Matrix{ComplexF64}(A))))
    fromvec(v) = begin
        R = reshape(ComplexF64.(v[1:d^2]), d, d) .+ im .* reshape(ComplexF64.(v[d^2+1:end]), d, d)
        (R + R') / 2
    end
    qs = Vector{Float64}[]                       # accepted orthonormal real coord vectors
    basis = Matrix{ComplexF64}[]
    for A in mats
        v = tovec(A)
        for q in qs
            v -= (q' * v) .* q                   # modified Gram-Schmidt, fixed order
        end
        norm(v) < _CF_RANK_TOL && continue       # linearly dependent ⇒ skip
        v ./= norm(v)
        k = argmax(abs.(v)); v[k] < 0 && (v .= -v)   # deterministic sign
        push!(qs, v)
        push!(basis, fromvec(v))
    end
    return basis
end

# Full-block matrix-unit ⊗ I_dΓ basis: the historical real-symmetric multiplicity
# units E_ij (diagonal first, then symmetric off-diagonal) promoted to the
# (mΓ·dΓ)-dim isotypic block via ⊗ I_dΓ. Coordinates on this basis equal the
# historical mΓ×mΓ matrix entries (⊗I embedding), preserving the pre-fix encoding.
function _matrix_unit_kron_basis(mΓ::Int, dΓ::Int)
    Id = Matrix{ComplexF64}(I, dΓ, dΓ)
    basis = Matrix{ComplexF64}[]
    for i in 1:mΓ
        E = zeros(ComplexF64, mΓ, mΓ); E[i, i] = 1; push!(basis, kron(E, Id))
    end
    for i in 1:mΓ-1, j in i+1:mΓ
        E = zeros(ComplexF64, mΓ, mΓ); E[i, j] = 1; E[j, i] = 1; push!(basis, kron(E, Id))
    end
    return basis
end

# Rank of a set of Hermitian matrices as a real subspace (Re/Im coordinates).
_herm_realcols(mats) = hcat([vcat(real(vec(Matrix{ComplexF64}(A))),
                                  imag(vec(Matrix{ComplexF64}(A)))) for A in mats]...)
_herm_rank(mats) = isempty(mats) ? 0 : rank(_herm_realcols(mats); atol=_CF_RANK_TOL)

# True iff Hermitian-operator sets A and B span the same real subspace:
# equal ranks AND mutual projection containment (the strong ⊗I-span detector).
function _cf_span_equals(A::AbstractVector, B::AbstractVector)
    _herm_rank(A) == _herm_rank(B) || return false
    all(_project_onto_basis(a, B)[2] < _CF_RESID_TOL for a in A) &&
    all(_project_onto_basis(b, A)[2] < _CF_RESID_TOL for b in B)
end

# CF-reachable Hermitian basis for one irrep block of (G, ℓ), as operators on the
# FULL isotypic block (`blk.col_range`, size mΓ·dΓ). No H⊗I assumption: the block
# may carry ⊗I and row-endomorphism (⊗J) pieces, handled uniformly. Built directly
# from the group-symmetrised C^k_m restrictions, so it is the gauge-safe invariant.
function _cf_block_basis(G::PointGroup, ℓ::Int, blk, U::AbstractMatrix)
    Dl = [_wignerd_matrix(e.matrix, ℓ) for e in G.elements]
    nG = length(G.elements)
    cr = blk.col_range; mΓ = blk.m_Γ; dΓ = blk.real_dim
    raw = Matrix{ComplexF64}[]
    for H in _hermitian_cf_generators(ℓ)
        S = sum(Dl[g] * H * Dl[g]' for g in 1:nG) / nG     # G-invariant Hermitian
        Hs = U' * S * U
        Q = Matrix{ComplexF64}(Hs[cr, cr]); Q = (Q + Q') / 2
        push!(raw, Q)
    end
    # Backward-compat: if the CF span equals the historical matrix-unit⊗I span,
    # return that basis (byte-identical encoding for real ⊗I groups).
    units = _matrix_unit_kron_basis(mΓ, dΓ)
    _cf_span_equals(raw, units) && return units
    return _orthonormal_hermitian_basis(raw)
end

# Least-squares coordinates of a Hermitian matrix H on a Hermitian basis,
# plus the relative residual ‖H - Σ c_b P_b‖ / max(1, ‖H‖).
function _project_onto_basis(H::AbstractMatrix, basis::AbstractVector{<:AbstractMatrix})
    isempty(basis) && return (Float64[], norm(H) / max(1.0, norm(H)))
    tov(A) = vcat(real(vec(Matrix{ComplexF64}(A))), imag(vec(Matrix{ComplexF64}(A))))
    M = hcat([tov(P) for P in basis]...)
    h = tov(H)
    c = M \ h
    return (c, norm(M * c - h) / max(1.0, norm(h)))
end

# Global CF dimension on V_ℓ for group G (rank of the G-invariant Hermitian
# multiplicative-CF space) — the true number of independent CF DOF.
function _cf_rank(G::PointGroup, ℓ::Int)
    Dl = [_wignerd_matrix(e.matrix, ℓ) for e in G.elements]
    nG = length(G.elements)
    cols = Vector{Float64}[]
    for H in _hermitian_cf_generators(ℓ)
        S = sum(Dl[g] * H * Dl[g]' for g in 1:nG) / nG
        push!(cols, vcat(real(vec(S)), imag(vec(S))))
    end
    return rank(hcat(cols...); atol=_CF_RANK_TOL)
end

# ─── LiftedRep + lift (single-particle rotation primitive) ───────────

struct LiftedRep
    group::PointGroup
    Dl_cache::Dict{Tuple{Int, Int}, Matrix{ComplexF64}}   # (element_idx, ℓ) → D^ℓ
    mode_action::Dict{Tuple{Int, Int}, Matrix{ComplexF64}} # (element_idx, shell_idx) → action
    shell_ℓ::Vector{Int}
end

"""
    lift(G::PointGroup, shell_ls::AbstractVector{<:Integer}) -> LiftedRep

Build a `LiftedRep` whose `mode_action[(g_idx, s_idx)]` is the
`(2ℓ_s+1)×(2ℓ_s+1)` matrix realising element `g_idx`'s action on shell
`s_idx`'s creation operators (spin-up and spin-down get the same
matrix). The full Fock-space apply (`(rep::LiftedRep)(g_idx, ψ)`) that
the chapter §10.3 specifies needs Layer-1 single-particle rotation
wiring; not yet exposed here.
"""
function lift(G::PointGroup, shell_ls::AbstractVector{<:Integer})
    Dl_cache = Dict{Tuple{Int, Int}, Matrix{ComplexF64}}()
    mode_action = Dict{Tuple{Int, Int}, Matrix{ComplexF64}}()
    n_elems = length(G.elements)
    for s_idx in eachindex(shell_ls)
        ℓ = Int(shell_ls[s_idx])
        for g_idx in 1:n_elems
            D = get!(Dl_cache, (g_idx, ℓ)) do
                _wignerd_matrix(G.elements[g_idx].matrix, ℓ)
            end
            mode_action[(g_idx, s_idx)] = D
        end
    end
    return LiftedRep(G, Dl_cache, mode_action, collect(Int, shell_ls))
end

# ─── classify (rep-theory level) ─────────────────────────────────────

"""
    classify_subspace(χ_S::AbstractVector{<:Number}, G::PointGroup; tol=1e-6)

Decompose a `G`-invariant subspace into Mulliken irreps from its
subspace character `χ_S(g) = Tr_S U(g)` (one entry per element of `G`,
in `G.elements` order). Returns a vector of `:label => integer-mult`
pairs. Asserts `χ_S` is class-constant up to `tol`.

Multiplicity of complex-type folded `E_j` is computed via constituent
(per §10.4 (10.3)–(10.4)) and the conjugate-pair equality is asserted.
"""
function classify_subspace(χ_S::AbstractVector{<:Number}, G::PointGroup; tol=1e-6)
    n_elems = length(G.elements)
    length(χ_S) == n_elems || throw(ArgumentError(
        "classify_subspace: χ_S must have length |G| = $n_elems, got $(length(χ_S))"))
    elem_to_class = Vector{Int}(undef, n_elems)
    for (k, C) in enumerate(G.classes), g in C
        elem_to_class[g] = k
    end
    # Class-constant check
    for C in G.classes
        χ0 = χ_S[C[1]]
        for g in C
            abs(χ_S[g] - χ0) < tol || @warn(
                "classify_subspace: χ_S not class-constant on class $(C); " *
                "subspace may not be G-invariant")
        end
    end
    out = Pair{Symbol, Int}[]
    for ir in G.irreps
        ms = Int[]
        for cid in ir.complex_constituents
            cir = G.complex_irreps[cid]
            s = ComplexF64(0)
            for g in 1:n_elems
                s += conj(cir.characters[elem_to_class[g]]) * χ_S[g]
            end
            m = s / n_elems
            mi = round(Int, real(m))
            abs(m - mi) < 1e-4 || error(
                "classify_subspace: complex multiplicity for $(ir.label) not " *
                "integer: $m")
            push!(ms, mi)
        end
        if length(ms) == 2
            ms[1] == ms[2] || error(
                "classify_subspace: folded constituent multiplicities differ " *
                "for $(ir.label): $ms")
        end
        m_Γ = ms[1]
        m_Γ > 0 && push!(out, ir.label => m_Γ)
    end
    return out
end

"""
    classify_state(weights::AbstractVector{<:Number}, G::PointGroup; tol=1e-6)

Given precomputed `⟨ψ|U(g)|ψ⟩` values (one per element in `G.elements`
order), return projector weights `w_Γ = ⟨ψ|P_Γ|ψ⟩` per Mulliken IR
(per §10.4 (10.1), summed over **all** elements, complex constituents
folded). Returns a `(weights::Vector{Pair{Symbol, Float64}},
dominant_IR::Symbol, dominant_weight::Float64)` named tuple.
"""
function classify_state(matrix_elements::AbstractVector{<:Number}, G::PointGroup;
                        tol::Real=1e-6)
    n_elems = length(G.elements)
    length(matrix_elements) == n_elems || throw(ArgumentError(
        "classify_state: matrix_elements must have length |G| = $n_elems"))
    elem_to_class = Vector{Int}(undef, n_elems)
    for (k, C) in enumerate(G.classes), g in C
        elem_to_class[g] = k
    end
    out = Pair{Symbol, Float64}[]
    best_label = G.irreps[1].label
    best_w = -Inf
    for ir in G.irreps
        w = 0.0 + 0im
        for cid in ir.complex_constituents
            cir = G.complex_irreps[cid]
            s = ComplexF64(0)
            for g in 1:n_elems
                s += conj(cir.characters[elem_to_class[g]]) * matrix_elements[g]
            end
            w += s * cir.dim / n_elems
        end
        wr = real(w)
        push!(out, ir.label => wr)
        if wr > best_w
            best_w = wr
            best_label = ir.label
        end
    end
    return (weights=out, dominant_IR=best_label, dominant_weight=best_w)
end
