# =====================================================================
# Hubbard-Holstein — electron-phonon coupling on a chain
# =====================================================================
#
# Each lattice point hosts (a) a spinful electron site (FermionSite{2})
# and (b) a phonon site (BosonSite{Nmax}). The electron and phonon at the
# same lattice point are coupled by g (b† + b) n_e.
#
#   H = H_e (Hubbard) +  ω Σ_i n^p_i  +  g Σ_i (b†_i + b_i) (n^e_i↑ + n^e_i↓)

using MOAD

const L = 3
const Nmax_phonon = 4

const t_hop = 1.0
const U = 4.0
const ω = 1.0     # phonon frequency
const g = 0.5     # e-ph coupling

# Two parallel arrays of named sites — combine into one Hilbert
electrons = [FermionSite{2}(Symbol("e$i")) for i = 1:L]
phonons   = [BosonSite{Nmax_phonon}(Symbol("p$i")) for i = 1:L]
hilbert = Hilbert(s.name => s for s in vcat(electrons, phonons))

# Electron part — same as Hubbard
const UP = 1
const DN = 2
H_hop = sum(c'(electrons[i], σ) * c(electrons[i+1], σ) for i = 1:L-1, σ in (UP, DN))
H_hop = -t_hop * (H_hop + H_hop')
H_U = U * sum(n(electrons[i], UP) * n(electrons[i], DN) for i = 1:L)

# Phonon kinetic
H_ph = ω * sum(n_b(p) for p in phonons)

# Electron-phonon coupling
H_eph = g * sum((bdag(phonons[i]) + b(phonons[i])) *
                (n(electrons[i], UP) + n(electrons[i], DN)) for i = 1:L)

H = H_hop + H_U + H_ph + H_eph

println("Hubbard-Holstein L=$L, Nmax_ph=$Nmax_phonon, t=$t_hop, U=$U, ω=$ω, g=$g")
println("Number of distinct terms: ", length(H))
println("First several terms:")
for (i, term) in enumerate(H)
    i > 8 && break
    println("  ", term)
end
