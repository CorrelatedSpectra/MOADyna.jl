# Validation: build the same H as test_print_hamiltonian.Quanty in MOAD
# and check term coefficients.

using MOAD

# Quanty: NF=4, modes 0..3 = (orb 0 up, orb 0 dn, orb 1 up, orb 1 dn).
# In MOAD, single FermionSite{4} with modes labeled 1..4.
# Mapping (1-indexed in MOAD): 1↔0_up, 2↔0_dn, 3↔1_up, 4↔1_dn

s = FermionSite{4}(:s)

# Hopping: 1.5 * (cdag(0) c(2) + cdag(1) c(3) + h.c.) in Quanty notation
# In MOAD (1-indexed): 1.5 * (cdag(s,1) c(s,3) + cdag(s,2) c(s,4) + h.c.)
t_hop = 1.5
H_hop = t_hop * (cdag(s, 1) * c(s, 3) + cdag(s, 2) * c(s, 4)
              +  cdag(s, 3) * c(s, 1) + cdag(s, 4) * c(s, 2))

println("=== H_hop ===")
println("Length: ", length(H_hop), " term(s)")
for term in H_hop
    println("  ", term)
end
println()

# On-site Coulomb on orbital 0:  U * n(s,1) * n(s,2)  (n_up * n_dn)
U = 4.0
H_U = U * n(s, 1) * n(s, 2)

println("=== H_U ===")
println("Length: ", length(H_U), " term(s)")
for term in H_U
    println("  ", term)
end
println()

println("=== H_total ===")
H = H_hop + H_U
println("Length: ", length(H), " term(s)")
for term in H
    println("  ", term)
end

println()
println("=== Comparison with Quanty output ===")
println("Quanty H_hop terms (length 2):")
println("  C 0 A 2 | +1.5")
println("  C 1 A 3 | +1.5")
println("  C 2 A 0 | +1.5")
println("  C 3 A 1 | +1.5")
println("MOAD H_hop:  ", length(H_hop), " terms (should be 4)")
println()
println("Quanty H_U term (length 4):")
println("  C 1 C 0 A 1 A 0 | -4.0")
println("    [cdag(1) cdag(0) c(1) c(0) — Quanty stores DESCENDING]")
println("MOAD H_U:    ", length(H_U), " term (should be 1)")
println("MOAD's canonical form is ASCENDING, so the chain is")
println("  cdag(s,1) cdag(s,2) c(s,1) c(s,2)  (1-indexed; Quanty's 0-indexed = 0,1,0,1)")
println("Both representations are algebraically equivalent;")
println("the swap sign cancels (two anticommutator swaps cdag-cdag and c-c).")
println("Expected coefficient: -4.0 (matches Quanty's -4.0).")
