# =====================================================================
# Spin-S Heisenberg chain — same code for any S
# =====================================================================
#
# Demonstrates the SpinSite{S} type and the S(site) bundle for Heisenberg
# dot products. Run with different S to see the same H expression cover
# spin-1/2, spin-1, spin-3/2, etc.
#
#   H = J Σ_<ij> S_i · S_j

using MOADyna

# Heisenberg dot product on the (Sx, Sy, Sz) bundle returned by S(site).
# This will eventually live in MOADyna; for now define locally.
⋅(a::NTuple{3, OperatorSum}, b::NTuple{3, OperatorSum}) =
    a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

const L = 4
const J = 1.0

function build_heisenberg(S_value)
    sites = [SpinSite{S_value}(Symbol("s$i")) for i = 1:L]
    hilbert = Hilbert(s.name => s for s in sites)
    H = J * sum(S(sites[i]) ⋅ S(sites[i+1]) for i = 1:L-1)
    return H, hilbert
end

for S_value in (1//2, 1, 3//2)
    H, hilbert = build_heisenberg(S_value)
    println("=" ^ 60)
    println("Heisenberg chain L=$L, S=$S_value")
    println("Number of terms: ", length(H))
    if S_value == 1//2
        println(H)   # show full structure for the simplest case
    end
end
