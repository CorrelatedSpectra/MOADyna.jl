using Test
using MOADyna
using MOADyna.Algebra: OperatorSum
using MOADyna.Shells: ShellModel, dipole, _ckq_coeff, _ckm_contract, site_of, _orbital_mode_pair
using MOADyna.PointGroups: Bkm_matrix
using LinearAlgebra: tr

# Sub-phase 2h, Stage 0 — convention-oracle tests for the raw rank-k
# Wigner-Eckart primitives. Locks the C^k_q matrix-element convention
# against trusted in-tree code (`Bkm_matrix`, the same-shell oracle, and
# `dipole`, the k=1 cross-shell oracle) and the Quanty Q-component
# combinations, BEFORE any public 2h wrapper is written.
#
# (The nIXS q̂-conjugation lock (item b) and the spin-dipole T-component
# lock (item d) require the `SphericalHarmonicC` angular primitive and the
# spin coupling respectively — both built in their own stages (B, D) — so
# those locks land as the first executable test there, not here.)

@testset "multipole (2h Stage 0 — convention oracle)" begin

    @testset "(a) _ckq_coeff matches Bkm_matrix (same-shell trusted oracle)" begin
        for ℓ in (1, 2, 3)
            for k in 0:2:(2ℓ), q in -k:k
                B = Bkm_matrix(ℓ, k, q)
                for m_a in -ℓ:ℓ
                    m_b = m_a + q
                    (m_b < -ℓ || m_b > ℓ) && continue
                    ref = B[m_b + ℓ + 1, m_a + ℓ + 1]
                    @test imag(ref) ≈ 0 atol = 1e-12
                    @test _ckq_coeff(ℓ, ℓ, k, q, m_a) ≈ real(ref) atol = 1e-12
                end
            end
        end
    end

    @testset "(a) parity / triangle / range vanish" begin
        @test _ckq_coeff(2, 2, 1, 0, 0) == 0.0   # same-shell odd k (parity)
        @test _ckq_coeff(2, 1, 2, 0, 0) == 0.0   # p→d k=2: ℓa+ℓb+k = 5 odd
        @test _ckq_coeff(2, 1, 4, 0, 0) == 0.0   # p→d k=4 > ℓa+ℓb = 3 (triangle)
        @test _ckq_coeff(2, 1, 1, 0, 0) != 0.0   # p→d k=1 allowed (dipole)
        @test _ckq_coeff(2, 1, 3, 0, 0) != 0.0   # p→d k=3 allowed (octupole)
        @test _ckq_coeff(1, 1, 2, 2, 1) == 0.0   # m_b = 3 out of range for ℓ_b = 1
        @test _ckq_coeff(2, 1, 1, 2, 0) == 0.0   # |q| = 2 > k = 1
    end

    @testset "(a) _ckm_contract reproduces dipole at k=1 (cross-shell oracle)" begin
        # Build the three spherical T^1_q via the raw contraction, assemble
        # Cartesian exactly as dipole.jl does, and compare to dipole().
        m = ShellModel([:Ni_2p, :Ni_3d])
        Tm1 = _ckm_contract(m, :Ni_2p, :Ni_3d, [(1, -1, 1.0 + 0im)])
        T0  = _ckm_contract(m, :Ni_2p, :Ni_3d, [(1,  0, 1.0 + 0im)])
        Tp1 = _ckm_contract(m, :Ni_2p, :Ni_3d, [(1,  1, 1.0 + 0im)])
        s = 1 / sqrt(2)
        Tx = s * (Tm1 - Tp1)
        Ty = (im * s) * (Tm1 + Tp1)
        Tz = T0
        Td = dipole(m, :Ni_2p => :Ni_3d)
        @test isempty(chop(Tx - Td[1]; tol = 1e-12))
        @test isempty(chop(Ty - Td[2]; tol = 1e-12))
        @test isempty(chop(Tz - Td[3]; tol = 1e-12))
    end

    @testset "(a) raw contraction does NOT Hermitize (absorption ≠ emission)" begin
        m = ShellModel([:Ni_2p, :Ni_3d])
        T0 = _ckm_contract(m, :Ni_2p, :Ni_3d, [(1, 0, 1.0 + 0im)])
        @test !isempty(T0)
        @test !isempty(chop(T0 - T0'; tol = 1e-12))
    end

    @testset "(c) charge-quadrupole Q components (vs Quanty Qxx/Qyy/Qzz)" begin
        # Verified Quanty combinations (StandardOperators.cpp:3315-3376):
        #   Qxx = −1·C²₀ + √1.5(C²₋₂ + C²₊₂)
        #   Qyy = −1·C²₀ − √1.5(C²₋₂ + C²₊₂)
        #   Qzz = 2·C²₀
        # Dimensionless, spin-summed, raw. Trace Qxx+Qyy+Qzz = 0; Hermitian.
        m = ShellModel([:Ni_3d])
        r = sqrt(1.5)
        Qxx = _ckm_contract(m, :Ni_3d, :Ni_3d,
                            [(2, 0, -1.0 + 0im), (2, -2, r + 0im), (2, 2, r + 0im)])
        Qyy = _ckm_contract(m, :Ni_3d, :Ni_3d,
                            [(2, 0, -1.0 + 0im), (2, -2, -r + 0im), (2, 2, -r + 0im)])
        Qzz = _ckm_contract(m, :Ni_3d, :Ni_3d, [(2, 0, 2.0 + 0im)])
        @test !isempty(Qzz)
        @test isempty(chop(Qxx + Qyy + Qzz; tol = 1e-12))   # traceless
        @test isempty(chop(Qxx - Qxx'; tol = 1e-12))        # Hermitian
        @test isempty(chop(Qyy - Qyy'; tol = 1e-12))
        @test isempty(chop(Qzz - Qzz'; tol = 1e-12))
    end

end

@testset "multipole (2h Stage A — engine + dipole refactor + E2)" begin

    @testset "engine returns 2k+1 spherical components" begin
        m = ShellModel([:Ni_1s, :Ni_3d])
        T2 = multipole(m, :Ni_1s => :Ni_3d, 2)
        @test length(T2) == 5
        @test all(t -> t isa OperatorSum, T2)
        @test all(!isempty, T2)
    end

    @testset "selection-rule guards (triangle + parity)" begin
        md = ShellModel([:Ni_3d])
        @test_throws ArgumentError multipole(md, :Ni_3d => :Ni_3d, 1)   # same-shell odd k
        @test_throws ArgumentError multipole(md, :Ni_3d => :Ni_3d, 5)   # k > 2ℓ
        @test_throws ArgumentError multipole(md, :Ni_3d => :Ni_3d, -1)  # k < 0
        mpd = ShellModel([:Ni_2p, :Ni_3d])
        @test_throws ArgumentError multipole(mpd, :Ni_2p => :Ni_3d, 2)  # p→d k=2 parity
        @test_throws ArgumentError multipole(mpd, :Ni_2p => :Ni_3d, 5)  # triangle
        @test length(multipole(mpd, :Ni_2p => :Ni_3d, 1)) == 3          # E1 allowed
        @test length(multipole(mpd, :Ni_2p => :Ni_3d, 3)) == 7          # E3 allowed
        @test length(multipole(md, :Ni_3d => :Ni_3d, 2)) == 5           # same-shell E2 allowed
    end

    @testset "dipole == k=1 multipole, both directions (refactor regression)" begin
        # Cartesian assembly from multipole(...,1) reproduces dipole() for
        # BOTH p→d (absorption) and d→p (emission), bit-for-bit. (The
        # external Quanty-fixture oracle for p→d lives in test_dipole.jl.)
        for shells in (:Ni_2p => :Ni_3d, :Ni_3d => :Ni_2p)
            m = ShellModel([:Ni_2p, :Ni_3d])
            Tm1, T0, Tp1 = multipole(m, shells, 1)
            s = 1 / sqrt(2)
            Tx = s * (Tm1 - Tp1)
            Ty = (im * s) * (Tm1 + Tp1)
            Tz = T0
            Td = dipole(m, shells)
            @test isempty(chop(Tx - Td[1]; tol = 1e-12))
            @test isempty(chop(Ty - Td[2]; tol = 1e-12))
            @test isempty(chop(Tz - Td[3]; tol = 1e-12))
        end
    end

    @testset "tesseral helper reproduces dipole Cartesian at k=1" begin
        m = ShellModel([:Ni_2p, :Ni_3d])
        comps = multipole(m, :Ni_2p => :Ni_3d, 1)
        tess, tags = MOADyna.Shells._real_tesseral_components(comps, 1)
        @test tags == [(0, '0'), (1, 'c'), (1, 's')]   # = [Tz, Tx, Ty]
        Td = dipole(m, :Ni_2p => :Ni_3d)
        @test isempty(chop(tess[1] - Td[3]; tol = 1e-12))   # (0,'0') = Tz
        @test isempty(chop(tess[2] - Td[1]; tol = 1e-12))   # (1,'c') = Tx
        @test isempty(chop(tess[3] - Td[2]; tol = 1e-12))   # (1,'s') = Ty
    end

    @testset "E2 quadrupole_transition — shape, radial scaling, selection" begin
        # 1s → 3d K pre-edge quadrupole: ℓ_a=0, ℓ_b=2, k=2 allowed.
        m = ShellModel([:Ni_1s, :Ni_3d])
        Q = quadrupole_transition(m, :Ni_1s => :Ni_3d)
        @test length(Q) == 5
        @test all(t -> t isa OperatorSum, Q)
        @test all(!isempty, Q)
        Q2 = quadrupole_transition(m, :Ni_1s => :Ni_3d; radial = 2.0)
        @test isempty(chop(Q2[1] - 2.0 * Q[1]; tol = 1e-12))   # linear in radial
        # E2 rejects an E1-only (Δℓ=1, odd-parity) transition.
        mpd = ShellModel([:Ni_2p, :Ni_3d])
        @test_throws ArgumentError quadrupole_transition(mpd, :Ni_2p => :Ni_3d)
    end

end

@testset "nIXS (2h Stage B — scattering op + radial integrals + conjugation lock)" begin

    SH = MOADyna.Shells._spherical_harmonic_C
    PL = MOADyna.Shells._assoc_legendre
    JB = MOADyna.Shells._spherical_bessel_j

    cosγ(θ1, φ1, θ2, φ2) = sin(θ1) * sin(θ2) * cos(φ1 - φ2) + cos(θ1) * cos(θ2)

    @testset "(b) CONJUGATION LOCK — addition theorem Σ_m C_m(q̂)* C_m(r̂) = P_k(q̂·r̂)" begin
        dirs = [(0.7, 1.1, 2.0, 0.3), (1.3, -0.4, 0.9, 2.7), (0.0, 0.0, 1.2, 0.8)]
        for k in (1, 2, 3, 4), (θq, φq, θr, φr) in dirs
            conj_sum = sum(conj(SH(k, mq, θq, φq)) * SH(k, mq, θr, φr) for mq in -k:k)
            Pk = PL(k, 0, cosγ(θq, φq, θr, φr))
            @test imag(conj_sum) ≈ 0 atol = 1e-12      # real (as P_k must be)
            @test real(conj_sum) ≈ Pk atol = 1e-12      # the lock
        end
        # The NON-conjugate sum is NOT the addition theorem (proves the
        # conjugate convention is the correct one).
        k = 2
        θq, φq, θr, φr = 0.7, 1.1, 2.0, 0.3
        noconj = sum(SH(k, mq, θq, φq) * SH(k, mq, θr, φr) for mq in -k:k)
        @test !isapprox(real(noconj), PL(k, 0, cosγ(θq, φq, θr, φr)); atol = 1e-6)
    end

    @testset "spherical harmonic C basics" begin
        for k in 0:4
            @test SH(k, 0, 0.0, 0.0) ≈ 1.0 atol = 1e-12      # C^k_0(0,0) = 1
            for mm in 1:k
                # C_{-m} = (−1)^m conj(C_m)
                c = SH(k, mm, 0.9, 0.4)
                @test SH(k, -mm, 0.9, 0.4) ≈ (-1)^mm * conj(c) atol = 1e-12
            end
        end
    end

    @testset "spherical Bessel j_k closed forms" begin
        for x in (0.3, 1.7, 5.0, 9.2)
            @test JB(0, x) ≈ sin(x) / x atol = 1e-12
            @test JB(1, x) ≈ sin(x) / x^2 - cos(x) / x atol = 1e-12
            @test JB(2, x) ≈ (3 / x^2 - 1) * sin(x) / x - 3 * cos(x) / x^2 atol = 1e-10
        end
        @test JB(0, 0.0) == 1.0
        @test JB(3, 0.0) == 0.0
    end

    @testset "spherical Bessel small-x (upward-recurrence-bug regression)" begin
        # j_k(x) → x^k / (2k+1)!! as x→0. The old all-x upward recurrence was
        # catastrophically wrong here (e.g. j₄(1e-3) gave 9.55e-3 vs ~1.06e-15;
        # j₆(1e-3) gave 9.46e5). The small-x series branch fixes it.
        df = Dict(3 => 105.0, 4 => 945.0, 6 => 135135.0)   # (2k+1)!!
        for k in (3, 4, 6), x in (1e-3, 1e-2, 0.1)
            lead = x^k / df[k]
            jk = JB(k, x)
            @test isfinite(jk)
            @test jk > 0
            @test jk ≈ lead rtol = max(x^2, 1e-9)
        end
        @test JB(4, 1e-3) ≈ 1e-12 / 945 rtol = 1e-4
        @test JB(6, 1e-3) < 1e-20                          # not 9.46e5
        # series and recurrence agree in their overlap (x ≳ k, both valid).
        @test MOADyna.Shells._bessel_small_x_series(4, 6.0) ≈ JB(4, 6.0) rtol = 1e-10
        @test MOADyna.Shells._bessel_small_x_series(3, 5.0) ≈ JB(3, 5.0) rtol = 1e-10
    end

    @testset "radial_integral analytic checks" begin
        r = collect(range(0.0, 1.0; length = 4001))
        ones_ = ones(length(r))
        # ∫₀¹ r² dr = 1/3 (physical weight, R=1); ∫₀¹ 1 dr = 1 (reduced).
        @test radial_integral(ones_, ones_, r, 0; kind = :power, weight = :physical) ≈ 1 / 3 atol = 1e-6
        @test radial_integral(ones_, ones_, r, 0; kind = :power, weight = :reduced) ≈ 1.0 atol = 1e-9
        # ∫₀¹ r² · r² dr = 1/5 (k=2 power, physical).
        @test radial_integral(ones_, ones_, r, 2; kind = :power, weight = :physical) ≈ 1 / 5 atol = 1e-6
        # q→0: j_0(0)=1 ⇒ bessel reduces to ∫ R_a R_b dr (reduced).
        @test radial_integral(ones_, ones_, r, 0; kind = :bessel, q = 0.0, weight = :reduced) ≈ 1.0 atol = 1e-9
        # grid validation
        @test_throws ArgumentError radial_integral([1.0], [1.0], [0.0], 0; kind = :power)
        @test_throws ArgumentError radial_integral([1.0, 1.0], [1.0, 1.0], [1.0, 0.0], 0; kind = :power)
        @test_throws ArgumentError radial_integral([1.0, 1.0], [1.0, 1.0], [0.0, Inf], 0; kind = :power)
    end

    @testset "_nixs_allowed_ranks (triangle + parity)" begin
        @test MOADyna.Shells._nixs_allowed_ranks(2, 2) == [0, 2, 4]   # d→d
        @test MOADyna.Shells._nixs_allowed_ranks(1, 2) == [1, 3]      # p→d
        @test MOADyna.Shells._nixs_allowed_ranks(0, 2) == [2]         # s→d (pure E2)
    end

    @testset "nixs operator — shape, selection, q̂ ∥ ẑ" begin
        m = ShellModel([:Ni_3d])
        Rj = Dict(0 => 1.0, 2 => 0.5, 4 => 0.2)
        T = nixs(m, :Ni_3d => :Ni_3d; theta = 0.0, phi = 0.0, radial_integrals = Rj)
        @test T isa OperatorSum
        @test !isempty(T)
        # disallowed rank rejected
        @test_throws ArgumentError nixs(m, :Ni_3d => :Ni_3d;
            theta = 0.0, phi = 0.0, radial_integrals = Dict(1 => 1.0))
        @test_throws ArgumentError nixs(m, :Ni_3d => :Ni_3d;
            theta = 0.0, phi = 0.0, radial_integrals = Dict{Int,Float64}())
        # q̂ ∥ ẑ: only m=0 survives (C^k_m(0,0)=0 for m≠0), so the operator
        # equals the m=0-only build.
        only0 = OperatorSum{ComplexF64}()
        for (k, Rjk) in sort(collect(Rj))
            pref = (im^k) * (2k + 1) * Rjk * conj(SH(k, 0, 0.0, 0.0))
            only0 += MOADyna.Shells._ckm_contract(m, :Ni_3d, :Ni_3d, [(k, 0, pref)])
        end
        @test isempty(chop(T - only0; tol = 1e-12))
    end

    @testset "Quanty NiO 3d nIXS radial moments (skip if reference missing)" begin
        # Quanty Tutorials/20_NiO_Crystal_Field/28_NIXS_dd.lua reads the
        # reduced radial function u_3d(r) from NiO_Radial/RnlNi_Atomic_Hartree_Fock
        # (cols: r 1S 2S 2P 3S 3P 3D) and prints the k=0,2,4 strength ratio at
        # q = 4.5 / a₀. We reproduce Rj_k with `radial_integral(weight=:reduced)`.
        quanty_root = get(ENV, "MOADYNA_QUANTY_ROOT", "")
        path = joinpath(quanty_root,
                        "Tutorials/20_NiO_Crystal_Field/NiO_Radial",
                        "RnlNi_Atomic_Hartree_Fock")
        if isempty(quanty_root) || !isfile(path)
            @info "Skipping Quanty nIXS radial check — set MOADYNA_QUANTY_ROOT to a Quanty checkout to enable."
        else
            rs = Float64[]; u3d = Float64[]
            for ln in readlines(path)[2:end]
                isempty(strip(ln)) && continue
                cols = split(ln)
                push!(rs, parse(Float64, cols[1]))
                push!(u3d, parse(Float64, cols[7]))
            end
            q = 4.5
            Rj0 = radial_integral(u3d, u3d, rs, 0; kind = :bessel, q = q, weight = :reduced)
            Rj2 = radial_integral(u3d, u3d, rs, 2; kind = :bessel, q = q, weight = :reduced)
            Rj4 = radial_integral(u3d, u3d, rs, 4; kind = :bessel, q = q, weight = :reduced)
            # u_3d is normalized ∫u² dr = 1, so each Rj_k ∈ (0,1). At this
            # high q the k=2 channel DOMINATES (the quadrupolar d-d nIXS that
            # 28_NIXS_dd.lua is built to show) — it is NOT a monotonic falloff.
            @test all(0 .< (Rj0, Rj2, Rj4) .< 1)
            @test all(isfinite, (Rj0, Rj2, Rj4))
            @test Rj2 > Rj0 && Rj2 > Rj4          # k=2 dominant at q=4.5/a₀
            @info "Quanty NiO 3d nIXS moments at q=$q/a₀" Rj0 Rj2 Rj4 ratio = (Rj2 / Rj0, Rj4 / Rj0)
        end
    end

end

@testset "charge-quadrupole Q (2h Stage C)" begin

    @testset "shape, ℓ-guard, radial scaling" begin
        m = ShellModel([:Ni_3d])
        Q = quadrupole(m, :Ni_3d)
        @test length(Q) == 6
        @test all(t -> t isa OperatorSum, Q)
        # s-shell has no rank-2 moment
        ms = ShellModel([:Ni_1s])
        @test_throws ArgumentError quadrupole(ms, :Ni_1s)
        # radial is a linear dimensionless scale
        Q2 = quadrupole(m, :Ni_3d; radial = 3.0)
        for i in 1:6
            @test isempty(chop(Q2[i] - 3.0 * Q[i]; tol = 1e-12))
        end
    end

    @testset "Hermitian + traceless (physical observable)" begin
        m = ShellModel([:Ni_3d])
        Qxx, Qyy, Qzz, Qxy, Qxz, Qyz = quadrupole(m, :Ni_3d)
        for Q in (Qxx, Qyy, Qzz, Qxy, Qxz, Qyz)
            @test isempty(chop(Q - Q'; tol = 1e-12))     # each component Hermitian
        end
        @test isempty(chop(Qxx + Qyy + Qzz; tol = 1e-12))  # traceless
    end

    @testset "component coefficients vs verified Quanty C² combinations" begin
        # Tie each Cartesian component to the multipole engine's spherical
        # C²_q (comps[q+3] for q=-2..2), with the coefficients read directly
        # from Quanty CreateOperatorQ* (StandardOperators.cpp:3315-3786).
        m = ShellModel([:Ni_3d])
        comps = multipole(m, :Ni_3d => :Ni_3d, 2)   # [C²₋₂,C²₋₁,C²₀,C²₁,C²₂]
        C(q) = comps[q + 3]
        r = sqrt(1.5)
        Qxx, Qyy, Qzz, Qxy, Qxz, Qyz = quadrupole(m, :Ni_3d)
        @test isempty(chop(Qzz - 2 * C(0); tol = 1e-12))
        @test isempty(chop(Qxx - (-C(0) + r * (C(-2) + C(2))); tol = 1e-12))
        @test isempty(chop(Qyy - (-C(0) - r * (C(-2) + C(2))); tol = 1e-12))
        @test isempty(chop(Qxy - (im * r * (C(-2) - C(2))); tol = 1e-12))
        @test isempty(chop(Qxz - (r * (C(-1) - C(1))); tol = 1e-12))
        @test isempty(chop(Qyz - (im * r * (C(-1) + C(1))); tol = 1e-12))
    end

end

@testset "spin-dipole T (2h Stage D)" begin

    @testset "shape + ℓ-guard" begin
        m = ShellModel([:Ni_3d])
        T = spin_dipole_T(m, :Ni_3d)
        @test length(T) == 3
        @test all(t -> t isa OperatorSum, T)
        @test all(!isempty, T)
        @test_throws ArgumentError spin_dipole_T(ShellModel([:Ni_1s]), :Ni_1s)
    end

    @testset "each component Hermitian (physical observable)" begin
        m = ShellModel([:Ni_3d])
        for Tα in spin_dipole_T(m, :Ni_3d)
            @test isempty(chop(Tα - Tα'; tol = 1e-12))
        end
    end

    @testset "single-particle matrix: Hermitian + traceless" begin
        # Independent numeric check: assemble each component in the
        # single-electron sector of the d-shell (10 states). T is built from
        # the traceless C² tensor ⊗ spin, so Tr T_α = 0; each is Hermitian.
        m = ShellModel([:Ni_3d])
        h = m.hilbert
        bas = EagerBasis(h, n_fermion(h) == 1)
        @test length(bas) == 10
        for Tα in spin_dipole_T(m, :Ni_3d)
            M = Matrix(assemble(compile(Tα, bas), bas))
            @test M ≈ M' atol = 1e-12
            @test abs(tr(M)) < 1e-12
        end
    end

    @testset "direct coefficient lock vs Quanty T table" begin
        # Independent, table-driven transcription of Quanty CreateOperatorT{x,y,z}
        # (StandardOperators.cpp:2600-2886), structurally different from the
        # imperative implementation, to freeze the per-(C²_q, spin) coefficients
        # against sign/coefficient swaps. Each entry: (q, c†-spin, c-spin, f(v))
        # with v = ⟨ml+q|C²_q|ml⟩, c† on orbital ml+q, c on orbital ml.
        ell = 2
        m = ShellModel([:Ni_3d])
        site = site_of(m, :Ni_3d)
        up(ml) = _orbital_mode_pair(ell, ml)[2]
        dn(ml) = _orbital_mode_pair(ell, ml)[1]
        cval(q, ml) = _ckq_coeff(ell, ell, 2, q, ml)
        s6 = sqrt(6.0); s32 = sqrt(1.5)
        inr(ml) = -ell <= ml <= ell

        Tz_tab = [(0, :u, :u, v -> -v), (0, :d, :d, v -> v),
                  (-1, :u, :d, v -> -0.5 * s6 * v), (1, :d, :u, v -> 0.5 * s6 * v)]
        Tx_tab = [(1, :u, :u, v -> 0.5 * s32 * v), (1, :d, :d, v -> -0.5 * s32 * v),
                  (0, :u, :d, v -> 0.5 * v), (0, :d, :u, v -> 0.5 * v),
                  (2, :d, :u, v -> -s32 * v),
                  (-1, :u, :u, v -> -0.5 * s32 * v), (-1, :d, :d, v -> 0.5 * s32 * v),
                  (-2, :u, :d, v -> -s32 * v)]
        Ty_tab = [(1, :u, :u, v -> -0.5 * s32 * v * im), (1, :d, :d, v -> 0.5 * s32 * v * im),
                  (0, :u, :d, v -> -0.5 * v * im), (0, :d, :u, v -> 0.5 * v * im),
                  (2, :d, :u, v -> s32 * v * im),
                  (-1, :u, :u, v -> -0.5 * s32 * v * im), (-1, :d, :d, v -> 0.5 * s32 * v * im),
                  (-2, :u, :d, v -> -s32 * v * im)]

        build(tab) = begin
            out = OperatorSum{ComplexF64}()
            for ml in -ell:ell, (q, sc, sa, f) in tab
                inr(ml + q) || continue
                v = cval(q, ml)
                iszero(v) && continue
                cr = sc === :u ? up(ml + q) : dn(ml + q)
                an = sa === :u ? up(ml) : dn(ml)
                out += ComplexF64(f(v)) * (cdag(site, cr) * c(site, an))
            end
            out
        end

        Tx, Ty, Tz = spin_dipole_T(m, :Ni_3d)
        @test isempty(chop(Tx - build(Tx_tab); tol = 1e-12))
        @test isempty(chop(Ty - build(Ty_tab); tol = 1e-12))
        @test isempty(chop(Tz - build(Tz_tab); tol = 1e-12))
    end

end
