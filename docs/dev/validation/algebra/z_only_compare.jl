# Build the z_only Hamiltonian in MOAD and compare term-by-term against
# the Quanty dump in z_only_quanty_output.txt.
#
# Conventions:
#   - MOAD modes are 1..10 (1-indexed); Quanty 0..9 (0-indexed)
#   - MOAD canonical chain order is ascending; Quanty descending
#   - Mode mapping (MOAD 1-indexed → Quanty 0-indexed): m → m-1

using MOAD

# Same parameters as the Quanty script
const U          = 6.0
const Delta      = 0.0
const dE_apin    = 0.20
const dE_apout   = 1.14
const t_sigma_in = 1.52
const t_sigma_out= 0.99

const nd  = 2.0
const nL  = 6.0
const dC  = U
const eL = (-0.0 - nd * Delta + nd * dC) / (nd + nL)    # = 1.5
const ed = eL + Delta - dC                              # = -4.5

# ----- Build Hamiltonian -----
# Single FermionSite{10} representing all 10 spin-orbitals.
# Quanty mode i ↔ MOAD label i+1.
s = FermionSite{10}(:s)

# On-site Ni (modes 1-4 in MOAD ↔ 0-3 in Quanty)
H_Ni = ed * sum(n(s, m) for m in 1:4)

# On-site L: eL on all 6 ligands + dE_apin on a0 + dE_apout on a1, a2
H_L  = eL * sum(n(s, m) for m in 5:10) +
       dE_apin  * sum(n(s, m) for m in 5:6) +
       dE_apout * sum(n(s, m) for m in 7:10)

# Coulomb on each Ni z (n_up * n_dn)
H_U  = U * (n(s, 1) * n(s, 2) + n(s, 3) * n(s, 4))

# Hopping (forward + backward = h.c.)
function hop(i_set, j_set, t)
    return t * sum(cdag(s, i) * c(s, j) + cdag(s, j) * c(s, i)
                   for (i, j) in zip(i_set, j_set))
end

H_kin = (
      hop([1, 2], [7, 8],  t_sigma_out)        # z1 -- a1, +t_so
    - hop([1, 2], [5, 6],  t_sigma_in)         # z1 -- a0, -t_si
    + hop([3, 4], [5, 6],  t_sigma_in)         # z2 -- a0, +t_si
    - hop([3, 4], [9, 10], t_sigma_out)        # z2 -- a2, -t_so
)

H = H_Ni + H_L + H_U + H_kin

# ----- Parse Quanty output -----
# Format:
#   C 0 A 6 |  9.90000000000000E-01     (length-2 entry)
#   C 1 C 0 A 1 A 0 | -4.00000000000000E+00   (length-4 entry)
function parse_quanty_dump(filename)
    terms = Vector{Tuple{Vector{Tuple{Symbol,Int}}, Float64}}()
    open(filename, "r") do io
        for line in eachline(io)
            line = strip(line)
            # Match lines that look like ladder chains
            m = match(r"^([CA0-9 ]+)\|\s*([-+0-9.eE]+)$", line)
            if m === nothing
                continue
            end
            chain_part = strip(m.captures[1])
            coef_str   = m.captures[2]
            coef = parse(Float64, coef_str)
            tokens = split(chain_part)
            chain = Tuple{Symbol,Int}[]
            i = 1
            while i <= length(tokens)
                t = tokens[i]
                if t == "C"
                    push!(chain, (:cdag, parse(Int, tokens[i+1])))
                elseif t == "A"
                    push!(chain, (:c, parse(Int, tokens[i+1])))
                end
                i += 2
            end
            isempty(chain) && continue
            push!(terms, (chain, coef))
        end
    end
    return terms
end

# Convert a MOAD OperatorTerm to Quanty's convention (0-indexed, descending)
function moad_to_quanty(term::MOAD.Algebra.OperatorTerm)
    chain_q = Tuple{Symbol,Int}[]
    # MOAD chain is ascending; Quanty wants descending. Same number of swaps
    # for cdag's vs c's individually, so the sign cancels (for normal-ordered chains
    # which is what we have post-canonicalization).
    cdags = Tuple{Symbol,Int}[]
    cs    = Tuple{Symbol,Int}[]
    for entry in term.chain
        idx = entry.label[1] - 1   # 1-indexed → 0-indexed
        if entry.kind === :cdag
            push!(cdags, (:cdag, idx))
        elseif entry.kind === :c
            push!(cs, (:c, idx))
        else
            error("unexpected kind in fermion-only validation: $(entry.kind)")
        end
    end
    # reverse to get descending
    return (vcat(reverse(cdags), reverse(cs)), term.coefficient)
end

# Build look-up tables from both
quanty_terms = parse_quanty_dump(joinpath(@__DIR__, "z_only_quanty_output.txt"))
moad_terms = [moad_to_quanty(t) for t in H]

println("Quanty parsed: ", length(quanty_terms), " terms")
println("MOAD    has:   ", length(moad_terms),   " terms")
println()

# Index by chain for fast lookup
quanty_dict = Dict(c => v for (c, v) in quanty_terms)
moad_dict   = Dict(c => v for (c, v) in moad_terms)

# Compare (wrapped to avoid top-level scope issues)
function compare_terms(quanty_terms, moad_terms, quanty_dict, moad_dict)
    matched = 0
    mismatched_coef = Tuple{Vector{Tuple{Symbol,Int}}, Float64, Float64}[]
    quanty_only = Vector{Tuple{Symbol,Int}}[]
    moad_only   = Vector{Tuple{Symbol,Int}}[]

    for (chain, qcoef) in quanty_terms
        if haskey(moad_dict, chain)
            mcoef = moad_dict[chain]
            if isapprox(mcoef, qcoef; atol=1e-10)
                matched += 1
            else
                push!(mismatched_coef, (chain, qcoef, mcoef))
            end
        else
            push!(quanty_only, chain)
        end
    end
    for (chain, mcoef) in moad_terms
        haskey(quanty_dict, chain) || push!(moad_only, chain)
    end
    return matched, mismatched_coef, quanty_only, moad_only
end

matched, mismatched_coef, quanty_only, moad_only =
    compare_terms(quanty_terms, moad_terms, quanty_dict, moad_dict)

println("=== Comparison ===")
println("Matched (chain & coef): ", matched, " / ", length(quanty_terms))
println("Mismatched coefs:       ", length(mismatched_coef))
println("Only in Quanty:         ", length(quanty_only))
println("Only in MOAD:           ", length(moad_only))

if !isempty(mismatched_coef)
    println()
    println("Mismatched coefficients:")
    for (chain, qc, mc) in mismatched_coef
        println("  $chain  Quanty=$qc  MOAD=$mc  diff=$(abs(qc-mc))")
    end
end

if !isempty(quanty_only)
    println()
    println("Terms only in Quanty:")
    for c in quanty_only
        println("  $c  coef=$(quanty_dict[c])")
    end
end

if !isempty(moad_only)
    println()
    println("Terms only in MOAD:")
    for c in moad_only
        println("  $c  coef=$(moad_dict[c])")
    end
end

if matched == length(quanty_terms) == length(moad_terms)
    println()
    println("✓ ALL TERMS MATCH (chain content and coefficients).")
else
    println()
    println("✗ Mismatch — see above.")
end
