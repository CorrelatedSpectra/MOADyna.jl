# Quanty Akm closed-form oracle fixture.
#
# Transcribed verbatim from:
#   Quanty/source/quanty/Core/BasicMath/BasicMath_StandardFunctions.cpp
#   function PotentialExpandedOnC(char const *s, uint32_t const l,
#                                  double const *EigenEnergies,
#                                  HarmonicsExpansion * const Akm)
#
# Each entry is a NamedTuple:
#   group        :: Symbol   — MOADyna group name (matches MOADyna :Symbol convention)
#   ell          :: Int      — angular momentum ℓ
#   source_lines :: String   — "first-last" lines in BasicMath_StandardFunctions.cpp
#   param_names  :: Vector{Symbol}
#       Quanty's EigenEnergies[] labels, reordered to match MOADyna's subduce() IR
#       ordering and parameter-packing convention.  Verified numerically: for
#       each fixture entry the MOADyna output equals the Quanty formula when
#       params[i] is fed as EigenEnergies[corresponding_Quanty_index].
#   entries :: Vector{NamedTuple}
#       Each (k, m, coeffs) means A_{k,m} = dot(coeffs, params).
#   broken  :: Bool
#       true when MOADyna's multiplicity-frame convention differs from Quanty's
#       (expected to be resolved by future libmsym-based convention alignment).
#
# Groups NOT in this fixture:
#   Low-symmetry groups C1, Ci, Cs (ℓ > 0): intentionally excluded.
#     Quanty's PotentialExpandedOnClm for C1/Ci/Cs always emits N=1 (only the
#     trace-shift A_{0,0} = E[0]) regardless of ℓ, because those groups have a
#     single orbit with no crystal-field splitting.  MOADyna performs the full
#     multiplicity-aware crystal-field expansion (nparams grows as (2ℓ+1)² for
#     C1).  These are different APIs, not different gauges — comparing them would
#     be misleading and is therefore omitted.  The C1/Ci ℓ=0 entries (A_{0,0}
#     trace-shift only) are kept because both codes agree there.
#   Quanty stubs (printf "not yet implemented" then return 1):
#     Cs, C2, C3, C4, C5, C6, C2v, C3v, C5v, C6v, C2h, C3h, C4h, C5h, C6h,
#     D2, D3, D4, D5, D6, D5h, D6h, D2d, D4d, D5d, D6d, I, Ih.
#   Quanty D3d ell=2: dispatches to D3dA/B/C/D/E which emit complex Akm
#     (QComplex=1, non-zero Im on the trigonal A_{4,±3}). MOADyna now emits the same
#     complex Akm (CF-reachable full-block expansion) and is validated bit-exact
#     against the C2//100 variant (D3dB/C, pure-imaginary trigonal) in
#     test_group.jl, "D3d ℓ=2 == Quanty D3dB/C ... via solver". ell=3 is a Quanty
#     "not implemented" stub, so no Quanty oracle exists there.
#   Quanty D2h ell=3: has "not implemented" guard before the case block.
#   Quanty D3h ell=3: dispatches to D3hx or D3hy depending on setting;
#     both are transcribed here under :D3hx and :D3hy; :D3h itself errors.
#   T, Th groups: Quanty implements them but MOADyna does not yet have reference
#     labels for :T or :Th — skipped by the test loop automatically.

const QUANTY_AKM_FIXTURES = [

    # ─────────────────────────────────────────────────────────────────────
    # C1 — trivial group, single orbit.  All ℓ: only A_{0,0} = E[0].
    # source_lines 1098-1130
    # ─────────────────────────────────────────────────────────────────────
    (group=:C1, ell=0, source_lines="1098-1130",
     param_names=[:A],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    # ─────────────────────────────────────────────────────────────────────
    # Ci — same structure as C1 (inversion adds gerade/ungerade but
    # ell=0,2 → Ag, ell=1,3 → Au; in Quanty the output is always
    # N=1, A_{0,0} = E[0] regardless of ℓ).
    # source_lines 1164-1193
    # ─────────────────────────────────────────────────────────────────────
    (group=:Ci, ell=0, source_lines="1164-1193",
     param_names=[:Ag],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    # ─────────────────────────────────────────────────────────────────────
    # C4v — ℓ=0,1,2 mult-free; ℓ=3 has multiplicity-2 E (Me mixing).
    # source_lines 1748-1865
    # ─────────────────────────────────────────────────────────────────────
    (group=:C4v, ell=0, source_lines="1795-1800",
     # subduce(:C4v,0) = [:A1=>1], nparams=1
     param_names=[:Ea1],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:C4v, ell=1, source_lines="1802-1810",
     # subduce(:C4v,1) = [:A1=>1, :E=>1], nparams=2
     # Quanty: E[0]=Ea1, E[1]=Ee  (A1 first, E second → matches MOADyna order)
     param_names=[:Ea1, :Ee],
     entries=[
         (k=0, m=0, coeffs=[1/3,    2/3]),       # (Ea1+2*Ee)/3
         (k=2, m=0, coeffs=[5/3,   -5/3]),       # (Ea1-Ee)*5/3
     ],
     broken=false),

    (group=:C4v, ell=2, source_lines="1812-1828",
     # subduce(:C4v,2) = [:A1=>1, :B1=>1, :B2=>1, :E=>1], nparams=4
     # Quanty: E[0]=Ea1, E[1]=Eb1, E[2]=Eb2, E[3]=Ee  — same as MOADyna order
     param_names=[:Ea1, :Eb1, :Eb2, :Ee],
     entries=[
         (k=0, m= 0, coeffs=[1/5,  1/5,  1/5,  2/5]),   # (Ea1+Eb1+Eb2+2*Ee)/5
         (k=2, m= 0, coeffs=[1.0, -1.0, -1.0,  1.0]),   # (Ea1-Eb1-Eb2+Ee)
         (k=4, m= 0, coeffs=[6*0.3, 0.3, 0.3, -8*0.3]), # (6*Ea1+Eb1+Eb2-8*Ee)*0.3
         (k=4, m= 4, coeffs=[0.0, 1.5*sqrt(0.7), -1.5*sqrt(0.7), 0.0]),  # (Eb1-Eb2)*1.5*sqrt(0.7)
         (k=4, m=-4, coeffs=[0.0, 1.5*sqrt(0.7), -1.5*sqrt(0.7), 0.0]),  # same
     ],
     broken=false),

    (group=:C4v, ell=3, source_lines="1831-1862",
     # subduce(:C4v,3) = [:A1=>1, :B1=>1, :B2=>1, :E=>2], nparams=6
     # Packing: Ea1(1×1), Eb1(1×1), Eb2(1×1), E-block(2×2)=[Ee1,Ee2,Me]
     # Quanty: EigenEnergies = [Ea1, Eb1, Eb2, Ee1, Ee2, Me]
     # Multiplicity-frame fixed: B_{2,0}-decoupling aligns Ee1/Ee2 to Quanty convention.
     param_names=[:Ea1, :Eb1, :Eb2, :Ee1, :Ee2, :Me],
     entries=[
         (k=0, m= 0, coeffs=[1/7, 1/7, 1/7, 2/7, 2/7, 0.0]),
         (k=2, m= 0, coeffs=[5/7, 0.0, 0.0, -5/7, 0.0, 5*sqrt(15)/7]),
         (k=4, m= 0, coeffs=[12*3/28, -14*3/28, -14*3/28, 9*3/28, 7*3/28, -2*sqrt(15)*3/28]),
         (k=4, m= 4, coeffs=[0.0, 10*3/(4*sqrt(70)), -10*3/(4*sqrt(70)), 15*3/(4*sqrt(70)), -15*3/(4*sqrt(70)), 2*sqrt(15)*3/(4*sqrt(70))]),
         (k=4, m=-4, coeffs=[0.0, 10*3/(4*sqrt(70)), -10*3/(4*sqrt(70)), 15*3/(4*sqrt(70)), -15*3/(4*sqrt(70)), 2*sqrt(15)*3/(4*sqrt(70))]),
         (k=6, m= 0, coeffs=[40*13/280, 12*13/280, 12*13/280, -25*13/280, -39*13/280, -14*sqrt(15)*13/280]),
         (k=6, m= 4, coeffs=[0.0, 12*13/(40*sqrt(14)), -12*13/(40*sqrt(14)), -15*13/(40*sqrt(14)), 15*13/(40*sqrt(14)), -2*sqrt(15)*13/(40*sqrt(14))]),
         (k=6, m=-4, coeffs=[0.0, 12*13/(40*sqrt(14)), -12*13/(40*sqrt(14)), -15*13/(40*sqrt(14)), 15*13/(40*sqrt(14)), -2*sqrt(15)*13/(40*sqrt(14))]),
     ],
     broken=false),

    # ─────────────────────────────────────────────────────────────────────
    # D2h — ℓ=0,1 mult-free; ℓ=2 has 2×2 Ag block (Eagmix mixing).
    # D2h ℓ=3: Quanty has "not implemented" guard → excluded.
    # source_lines 2815-2931
    # ─────────────────────────────────────────────────────────────────────
    (group=:D2h, ell=0, source_lines="2864-2869",
     # subduce(:D2h,0) = [:Ag=>1], nparams=1
     param_names=[:Ag],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:D2h, ell=1, source_lines="2871-2884",
     # subduce(:D2h,1) = [:B1u=>1, :B2u=>1, :B3u=>1], nparams=3
     # Quanty: E[0]=B1u (z-axis), E[1]=B2u, E[2]=B3u
     # MOADyna subduce order matches Quanty: [B1u, B2u, B3u]
     param_names=[:Eb1u, :Eb2u, :Eb3u],
     entries=[
         (k=0, m= 0, coeffs=[1/3,           1/3,           1/3]),
         (k=2, m= 0, coeffs=[2*5/6,         -5/6,          -5/6]),
         (k=2, m= 2, coeffs=[0.0,  -5/(2*sqrt(6)),  5/(2*sqrt(6))]),
         (k=2, m=-2, coeffs=[0.0,  -5/(2*sqrt(6)),  5/(2*sqrt(6))]),
     ],
     broken=false),

    (group=:D2h, ell=2, source_lines="2887-2921",
     # subduce(:D2h,2) = [:Ag=>2, :B1g=>1, :B2g=>1, :B3g=>1], nparams=6
     # Packing: Ag(2×2)=[Eagz2, Eagx2y2, Eagmix], then B1g, B2g, B3g.
     # Numerical probe confirmed: MOADyna Ag[1,1]=Eagz2, Ag[2,2]=Eagx2y2,
     # Ag[1,2]=Ag[2,1]=Eagmix.
     # Quanty: E[0]=Eagx2y2, E[1]=Eagz2, E[2]=Eagmix, E[3]=Eb1g, E[4]=Eb2g, E[5]=Eb3g
     # MOADyna params: [Eagz2, Eagx2y2, Eagmix, Eb1g, Eb2g, Eb3g]
     # (Eagz2 and Eagx2y2 are SWAPPED relative to Quanty's E[0],E[1] ordering)
     param_names=[:Eagz2, :Eagx2y2, :Eagmix, :Eb1gxy, :Eb2gxz, :Eb3gyz],
     entries=[
         # A_{0,0} = 0.2*(Eagx2y2+Eagz2+Eb1gxy+Eb2gxz+Eb3gyz)
         # in MOADyna order [Eagz2, Eagx2y2, Eagmix, Eb1g, Eb2g, Eb3g]:
         (k=0, m= 0, coeffs=[0.2, 0.2, 0.0, 0.2, 0.2, 0.2]),
         # A_{2,0} = 0.5*(-2*Eagx2y2+2*Eagz2-2*Eb1gxy+Eb2gxz+Eb3gyz)
         (k=2, m= 0, coeffs=[0.5*2, 0.5*(-2), 0.0, 0.5*(-2), 0.5, 0.5]),
         # A_{4,0} = 0.3*(Eagx2y2+6*Eagz2+Eb1gxy-4*Eb2gxz-4*Eb3gyz)
         (k=4, m= 0, coeffs=[0.3*6, 0.3, 0.0, 0.3, -0.3*4, -0.3*4]),
         # A_{2,2} = 0.25*(sqrt(6)*(Eb2gxz-Eb3gyz)-4*sqrt(2)*Eagmix)
         (k=2, m= 2, coeffs=[0.0, 0.0, -4*0.25*sqrt(2), 0.0, 0.25*sqrt(6), -0.25*sqrt(6)]),
         (k=2, m=-2, coeffs=[0.0, 0.0, -4*0.25*sqrt(2), 0.0, 0.25*sqrt(6), -0.25*sqrt(6)]),
         # A_{4,2} = sqrt(0.9)*(Eb2gxz-Eb3gyz+sqrt(3)*Eagmix)
         (k=4, m= 2, coeffs=[0.0, 0.0, sqrt(0.9)*sqrt(3), 0.0, sqrt(0.9), -sqrt(0.9)]),
         (k=4, m=-2, coeffs=[0.0, 0.0, sqrt(0.9)*sqrt(3), 0.0, sqrt(0.9), -sqrt(0.9)]),
         # A_{4,4} = 1.5*sqrt(0.7)*(Eagx2y2-Eb1gxy)
         (k=4, m= 4, coeffs=[0.0, 1.5*sqrt(0.7), 0.0, -1.5*sqrt(0.7), 0.0, 0.0]),
         (k=4, m=-4, coeffs=[0.0, 1.5*sqrt(0.7), 0.0, -1.5*sqrt(0.7), 0.0, 0.0]),
     ],
     broken=false),

    # ─────────────────────────────────────────────────────────────────────
    # D3hx — ℓ=0..3 implemented.
    # D3h in Quanty dispatches to D3hx for ℓ=0,1,2; for ℓ=3 requires
    # explicit D3hx or D3hy call.  MOADyna's :D3h matches D3hx conventions
    # for all ℓ (verified numerically).
    # source_lines 2933-3032 (D3hx), 3034-3103 (D3hy)
    # ─────────────────────────────────────────────────────────────────────
    (group=:D3h, ell=0, source_lines="2980-2985",
     # subduce(:D3h,0) = [:A1prime=>1], nparams=1
     param_names=[:A1prime],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:D3h, ell=1, source_lines="2987-2994",
     # subduce(:D3h,1) = [:Eprime=>1, :A2dprime=>1], nparams=2
     # Quanty D3hx: E[0]=Eprime (in-plane), E[1]=A2dprime (vertical)
     # MOADyna order matches: [Eprime, A2dprime]
     param_names=[:Eprime, :A2dprime],
     entries=[
         (k=0, m=0, coeffs=[2/3,   1/3]),   # (2*Eprime+A2dprime)/3
         (k=2, m=0, coeffs=[-5/3,  5/3]),   # (-Eprime+A2dprime)*5/3
     ],
     broken=false),

    (group=:D3h, ell=2, source_lines="2997-3007",
     # subduce(:D3h,2) = [:A1prime=>1, :Eprime=>1, :Edprime=>1], nparams=3
     # Quanty D3hx: E[0]=A1prime, E[1]=Eprime, E[2]=Edprime
     # MOADyna order: [A1prime, Eprime, Edprime]
     param_names=[:A1prime, :Eprime, :Edprime],
     entries=[
         (k=0, m=0, coeffs=[1/5, 2/5, 2/5]),    # (A1prime+2*Eprime+2*Edprime)/5
         (k=2, m=0, coeffs=[1.0, -2.0, 1.0]),   # (A1prime-2*Eprime+Edprime)
         (k=4, m=0, coeffs=[3*3/5, 3/5, -4*3/5]), # (3*A1prime+Eprime-4*Edprime)*3/5
     ],
     broken=false),

    (group=:D3h, ell=3, source_lines="3010-3030",
     # subduce(:D3h,3) = [:A1prime=>1,:A2prime=>1,:Eprime=>1,:A2dprime=>1,:Edprime=>1], nparams=5
     # Quanty D3hx: E[0]=A1prime, E[1]=A2prime, E[2]=Eprime, E[3]=A2dprime, E[4]=Edprime
     # MOADyna order verified numerically: same ordering
     param_names=[:A1prime, :A2prime, :Eprime, :A2dprime, :Edprime],
     entries=[
         (k=0, m= 0, coeffs=[1/7, 1/7, 2/7, 1/7, 2/7]),
         (k=2, m= 0, coeffs=[(-5)*5/28, (-5)*5/28, 6*5/28, 4*5/28, 0.0]),
         (k=4, m= 0, coeffs=[3*3/14, 3*3/14, 2*3/14, 6*3/14, -14*3/14]),
         (k=6, m= 0, coeffs=[(-1)*13/140, (-1)*13/140, (-30)*13/140, 20*13/140, 12*13/140]),
         (k=6, m= 6, coeffs=[(13/20)*sqrt(33/7), -(13/20)*sqrt(33/7), 0.0, 0.0, 0.0]),
         (k=6, m=-6, coeffs=[(13/20)*sqrt(33/7), -(13/20)*sqrt(33/7), 0.0, 0.0, 0.0]),
     ],
     broken=false),

    # ─────────────────────────────────────────────────────────────────────
    # D4h — ℓ=0,1,2 mult-free; ℓ=3 has multiplicity-2 Eu (Me mixing).
    # source_lines 3129-3240
    # ─────────────────────────────────────────────────────────────────────
    (group=:D4h, ell=0, source_lines="3176-3181",
     param_names=[:A1g],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:D4h, ell=1, source_lines="3183-3190",
     # subduce(:D4h,1) = [:A2u=>1, :Eu=>1], nparams=2
     # Quanty: E[0]=A2u (axial), E[1]=Eu (planar)
     param_names=[:Ea2u, :Eeu],
     entries=[
         (k=0, m=0, coeffs=[1/3,   2/3]),
         (k=2, m=0, coeffs=[5/3,  -5/3]),
     ],
     broken=false),

    (group=:D4h, ell=2, source_lines="3193-3209",
     # subduce(:D4h,2) = [:A1g=>1, :B1g=>1, :B2g=>1, :Eg=>1], nparams=4
     # Quanty: E[0]=A1g, E[1]=B1g, E[2]=B2g, E[3]=Eg
     # MOADyna order: [A1g, B1g, B2g, Eg] — matches
     param_names=[:Ea1g, :Eb1g, :Eb2g, :Eeg],
     entries=[
         (k=0, m= 0, coeffs=[1/5, 1/5, 1/5, 2/5]),
         (k=2, m= 0, coeffs=[1.0, -1.0, -1.0, 1.0]),
         (k=4, m= 0, coeffs=[6*0.3, 0.3, 0.3, -8*0.3]),
         # A_{4,-4} = 0.15*sqrt(70)*(B1g-B2g)
         (k=4, m=-4, coeffs=[0.0, 0.15*sqrt(70.0), -0.15*sqrt(70.0), 0.0]),
         (k=4, m= 4, coeffs=[0.0, 0.15*sqrt(70.0), -0.15*sqrt(70.0), 0.0]),
     ],
     broken=false),

    (group=:D4h, ell=3, source_lines="3212-3238",
     # subduce(:D4h,3) = [:A2u=>1, :B1u=>1, :B2u=>1, :Eu=>2], nparams=6
     # Packing: Ea2u, Eb1u, Eb2u, Eu(2×2)=[Eu1,Eu2,Me]
     # Multiplicity-frame fixed: same B_{2,0}-decoupling as C4v ℓ=3.
     # Note: D4h B1u/B2u have opposite sign convention for k=4,6 m=±4 terms
     # relative to C4v B1/B2 — confirmed by MOADyna output matching Quanty.
     param_names=[:Ea2u, :Eb1u, :Eb2u, :Eu1, :Eu2, :Me],
     entries=[
         (k=0, m= 0, coeffs=[1/7, 1/7, 1/7, 2/7, 2/7, 0.0]),
         (k=2, m= 0, coeffs=[5/7, 0.0, 0.0, -5/7, 0.0, 5*sqrt(15.0)/7]),
         (k=4, m= 0, coeffs=[12*3/28, -14*3/28, -14*3/28, 9*3/28, 7*3/28, -2*sqrt(15.0)*3/28]),
         (k=4, m= 4, coeffs=[0.0, -10*3/(4*sqrt(70.0)), 10*3/(4*sqrt(70.0)), 15*3/(4*sqrt(70.0)), -15*3/(4*sqrt(70.0)), 2*sqrt(15.0)*3/(4*sqrt(70.0))]),
         (k=4, m=-4, coeffs=[0.0, -10*3/(4*sqrt(70.0)), 10*3/(4*sqrt(70.0)), 15*3/(4*sqrt(70.0)), -15*3/(4*sqrt(70.0)), 2*sqrt(15.0)*3/(4*sqrt(70.0))]),
         (k=6, m= 0, coeffs=[40*13/280, 12*13/280, 12*13/280, -25*13/280, -39*13/280, -14*sqrt(15.0)*13/280]),
         (k=6, m= 4, coeffs=[0.0, -12*13/(40*sqrt(14.0)), 12*13/(40*sqrt(14.0)), -15*13/(40*sqrt(14.0)), 15*13/(40*sqrt(14.0)), -2*sqrt(15.0)*13/(40*sqrt(14.0))]),
         (k=6, m=-4, coeffs=[0.0, -12*13/(40*sqrt(14.0)), 12*13/(40*sqrt(14.0)), -15*13/(40*sqrt(14.0)), 15*13/(40*sqrt(14.0)), -2*sqrt(15.0)*13/(40*sqrt(14.0))]),
     ],
     broken=false),

    # ─────────────────────────────────────────────────────────────────────
    # D3d — Quanty ℓ=0,1 are mult-free and non-complex.
    #        ℓ=2: Quanty dispatches to D3dA/B/C/D/E with QComplex=1 (complex Akm).
    #        ℓ=3: Quanty has "not implemented" guard.
    #        Only ℓ=0,1 are transcribed here.
    # source_lines 3479-3543
    # ─────────────────────────────────────────────────────────────────────
    (group=:D3d, ell=0, source_lines="3526-3531",
     param_names=[:A1g],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:D3d, ell=1, source_lines="3533-3541",
     # subduce(:D3d,1) = [:A2u=>1, :Eu=>1], nparams=2
     # Quanty: E[0]=A2u (axial), E[1]=Eu (planar)
     param_names=[:Ea2u, :Eeu],
     entries=[
         (k=0, m=0, coeffs=[1/3,   2/3]),
         (k=2, m=0, coeffs=[5/3,  -5/3]),
     ],
     broken=false),

    # ─────────────────────────────────────────────────────────────────────
    # O — same closed forms as Oh (no inversion label distinction in the
    # crystal-field formulas; different IR labels but same numerics).
    # source_lines 4460-4562
    # ─────────────────────────────────────────────────────────────────────
    (group=:O, ell=0, source_lines="4507-4512",
     param_names=[:A1],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:O, ell=1, source_lines="4514-4518",
     # subduce(:O,1) = [:T1=>1], nparams=1
     param_names=[:ET1],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:O, ell=2, source_lines="4521-4534",
     # subduce(:O,2) = [:E=>1, :T2=>1], nparams=2
     # Quanty: E[0]=E, E[1]=T2 — same as MOADyna order
     param_names=[:EE, :ET2],
     entries=[
         (k=0, m= 0, coeffs=[0.4,            0.6]),
         (k=4, m= 0, coeffs=[2.1,           -2.1]),
         (k=4, m=-4, coeffs=[1.5*sqrt(0.7), -1.5*sqrt(0.7)]),
         (k=4, m= 4, coeffs=[1.5*sqrt(0.7), -1.5*sqrt(0.7)]),
     ],
     broken=false),

    (group=:O, ell=3, source_lines="4537-4560",
     # subduce(:O,3) = [:A2=>1, :T1=>1, :T2=>1], nparams=3
     # Quanty: E[0]=A2, E[1]=T1, E[2]=T2 — verified matches MOADyna param order
     param_names=[:EA2, :ET1, :ET2],
     entries=[
         (k=0, m= 0, coeffs=[(1/7),     (3/7),        (3/7)]),
         (k=4, m= 0, coeffs=[(3/4)*(-2), (3/4)*3,      (3/4)*(-1)]),
         (k=4, m=-4, coeffs=[(3/4)*sqrt(5/14)*(-2), (3/4)*sqrt(5/14)*3,  (3/4)*sqrt(5/14)*(-1)]),
         (k=4, m= 4, coeffs=[(3/4)*sqrt(5/14)*(-2), (3/4)*sqrt(5/14)*3,  (3/4)*sqrt(5/14)*(-1)]),
         (k=6, m= 0, coeffs=[(39/280)*4,   (39/280)*5,    (39/280)*(-9)]),
         (k=6, m=-4, coeffs=[-(39/(40*sqrt(14)))*4, -(39/(40*sqrt(14)))*5, -(39/(40*sqrt(14)))*(-9)]),
         (k=6, m= 4, coeffs=[-(39/(40*sqrt(14)))*4, -(39/(40*sqrt(14)))*5, -(39/(40*sqrt(14)))*(-9)]),
     ],
     broken=false),

    # ─────────────────────────────────────────────────────────────────────
    # Oh — same closed forms as O.
    # source_lines 4564-4666
    # ─────────────────────────────────────────────────────────────────────
    (group=:Oh, ell=0, source_lines="4611-4616",
     param_names=[:A1g],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:Oh, ell=1, source_lines="4618-4622",
     # subduce(:Oh,1) = [:T1u=>1], nparams=1
     param_names=[:ET1u],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:Oh, ell=2, source_lines="4625-4638",
     # subduce(:Oh,2) = [:Eg=>1, :T2g=>1], nparams=2
     # Quanty: E[0]=Eg, E[1]=T2g — matches MOADyna order
     param_names=[:EEg, :ET2g],
     entries=[
         (k=0, m= 0, coeffs=[0.4,             0.6]),
         (k=4, m= 0, coeffs=[2.1,            -2.1]),
         (k=4, m=-4, coeffs=[1.5*sqrt(0.7),  -1.5*sqrt(0.7)]),
         (k=4, m= 4, coeffs=[1.5*sqrt(0.7),  -1.5*sqrt(0.7)]),
     ],
     broken=false),

    (group=:Oh, ell=3, source_lines="4641-4664",
     # subduce(:Oh,3) = [:A2u=>1, :T1u=>1, :T2u=>1], nparams=3
     # Quanty: E[0]=A2u, E[1]=T1u, E[2]=T2u — matches MOADyna subduce order
     param_names=[:EA2u, :ET1u, :ET2u],
     entries=[
         (k=0, m= 0, coeffs=[(1/7),          (3/7),         (3/7)]),
         (k=4, m= 0, coeffs=[(3/4)*(-2),     (3/4)*3,       (3/4)*(-1)]),
         (k=4, m=-4, coeffs=[sqrt(5/14)*(3/4)*(-2), sqrt(5/14)*(3/4)*3, sqrt(5/14)*(3/4)*(-1)]),
         (k=4, m= 4, coeffs=[sqrt(5/14)*(3/4)*(-2), sqrt(5/14)*(3/4)*3, sqrt(5/14)*(3/4)*(-1)]),
         (k=6, m= 0, coeffs=[(39/280)*4,     (39/280)*5,    (39/280)*(-9)]),
         (k=6, m=-4, coeffs=[-(39/(40*sqrt(14)))*4, -(39/(40*sqrt(14)))*5, -(39/(40*sqrt(14)))*(-9)]),
         (k=6, m= 4, coeffs=[-(39/(40*sqrt(14)))*4, -(39/(40*sqrt(14)))*5, -(39/(40*sqrt(14)))*(-9)]),
     ],
     broken=false),

    # ─────────────────────────────────────────────────────────────────────
    # Td — ℓ=0,1,2 mult-free; ℓ=3 has T1 and T2 whose basis ordering
    # differs from Quanty (subduce gives [:A1,:T1,:T2] but MOADyna's T1/T2
    # subspaces are NOT the same as Quanty's for ℓ=3).
    # source_lines 4238-4342
    # ─────────────────────────────────────────────────────────────────────
    (group=:Td, ell=0, source_lines="4285-4290",
     param_names=[:A1],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:Td, ell=1, source_lines="4292-4297",
     # subduce(:Td,1) = [:T2=>1], nparams=1
     param_names=[:ET2],
     entries=[
         (k=0, m=0, coeffs=[1.0]),
     ],
     broken=false),

    (group=:Td, ell=2, source_lines="4299-4312",
     # subduce(:Td,2) = [:E=>1, :T2=>1], nparams=2
     # Quanty: E[0]=E, E[1]=T2 — matches MOADyna param order
     param_names=[:EE, :ET2],
     entries=[
         (k=0, m= 0, coeffs=[0.4,             0.6]),
         (k=4, m= 0, coeffs=[2.1,            -2.1]),
         (k=4, m=-4, coeffs=[1.5*sqrt(0.7),  -1.5*sqrt(0.7)]),
         (k=4, m= 4, coeffs=[1.5*sqrt(0.7),  -1.5*sqrt(0.7)]),
     ],
     broken=false),

    (group=:Td, ell=3, source_lines="4315-4340",
     # subduce(:Td,3) = [:A1=>1, :T1=>1, :T2=>1], nparams=3
     # param_names in MOADyna canonical order: [EA1, ET1_MOADyna, ET2_MOADyna]
     #
     # Td ℓ=3: Quanty's Td source labels T1 ↔ T2 OPPOSITE from the
     # Bilbao/MOADyna convention.  Standard: T1 has χ(σd) = -1 (pseudo-vector);
     # T2 has χ(σd) = +1 (polar vector).  Quanty's Td source uses the opposite
     # labeling (their "T1" is the polar-vector subspace and their "T2" is the
     # pseudo-vector subspace).
     # Empirically verified: MOADyna's T1 (pseudo-vector, χ(σd)=-1) produces A_{4,0}=-0.75
     # which equals Quanty's "T2" coefficient; MOADyna's T2 (polar-vector, χ(σd)=+1) produces
     # A_{4,0}=+2.25 which equals Quanty's "T1" coefficient.
     # The fixture below accounts for the swap by reading Quanty's "T1 column" as MOADyna's T2
     # energy input, and vice versa.  This is a documented labeling difference, not a bug
     # in either code.  Coefficients are expressed in MOADyna's canonical ordering [EA1, ET1, ET2].
     param_names=[:EA1, :ET1, :ET2],
     entries=[
         (k=0, m= 0, coeffs=[(1/7),                      (3/7),                      (3/7)]),
         (k=4, m= 0, coeffs=[(3/4)*(-2),                 (3/4)*(-1),                 (3/4)*3]),
         (k=4, m=-4, coeffs=[sqrt(5/14)*(3/4)*(-2),      sqrt(5/14)*(3/4)*(-1),      sqrt(5/14)*(3/4)*3]),
         (k=4, m= 4, coeffs=[sqrt(5/14)*(3/4)*(-2),      sqrt(5/14)*(3/4)*(-1),      sqrt(5/14)*(3/4)*3]),
         (k=6, m= 0, coeffs=[(39/280)*4,                 (39/280)*(-9),              (39/280)*5]),
         (k=6, m=-4, coeffs=[-(39/(40*sqrt(14)))*4,      -(39/(40*sqrt(14)))*(-9),   -(39/(40*sqrt(14)))*5]),
         (k=6, m= 4, coeffs=[-(39/(40*sqrt(14)))*4,      -(39/(40*sqrt(14)))*(-9),   -(39/(40*sqrt(14)))*5]),
     ],
     broken=false),   # T1/T2 swap vs Quanty source is intentional — see comment above

]  # end QUANTY_AKM_FIXTURES
