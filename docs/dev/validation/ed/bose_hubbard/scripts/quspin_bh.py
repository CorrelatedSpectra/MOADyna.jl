"""
QuSpin reference for Bose-Hubbard validation in MOADyna.

Generates `bh_reference.h5` (committed alongside this script under
`reference/`). The test under `test/ed/validation/test_bh_quspin.jl`
loads the fixture and compares against MOADyna.

Hamiltonian (matches MOADyna `examples/03_bose_hubbard.jl`):

    H = -t Σ_{<ij>} (b†_i b_j + h.c.)
       + (U/2) Σ_i n_i (n_i - 1)
       - μ Σ_i n_i

Boundary: open. No symmetry block projection (kblock / pblock
disabled) so the QuSpin basis spans the same Fock sector as MOADyna's
EagerBasis(h, n_boson(h) == Nb).

Re-run with:
    conda run -n quspin python quspin_bh.py
"""
import os

import numpy as np
import h5py
from quspin.basis import boson_basis_1d
from quspin.operators import hamiltonian


L      = 4
Nb     = 4         # total bosons (unit filling)
Nmax   = 3         # cutoff per site
t_hop  = 1.0
U      = 4.0
mu     = 0.0

basis = boson_basis_1d(L, Nb=Nb, sps=Nmax + 1)

# OBC site-coupling lists.
hop      = [[-t_hop,         i, i + 1] for i in range(L - 1)]
nn_list  = [[0.5 * U,        i, i]     for i in range(L)]
n_list   = [[-mu - 0.5 * U,  i]        for i in range(L)]

# (U/2) n(n-1) ≡ (U/2) n^2 - (U/2) n; the linear piece is folded into
# the chemical-potential term. Identical operator either way.
static = [
    ["+-", hop],
    ["-+", hop],
    ["nn", nn_list],
    ["n",  n_list],
]

H = hamiltonian(static, [], basis=basis, dtype=np.float64,
                check_symm=False, check_pcon=False, check_herm=True)

E, V = H.eigh()
psi0 = V[:, 0]

# Site-resolved <n_i> in the GS.
n_expect = np.empty(L, dtype=np.float64)
for i in range(L):
    n_op = hamiltonian([["n", [[1.0, i]]]], [], basis=basis,
                       dtype=np.float64,
                       check_symm=False, check_pcon=False,
                       check_herm=True)
    n_expect[i] = float(psi0.conj() @ (n_op.dot(psi0)))

ref_path = os.path.join(os.path.dirname(__file__),
                        "..", "reference", "bh_reference.h5")
ref_path = os.path.abspath(ref_path)

with h5py.File(ref_path, "w") as fh:
    fh.attrs["L"]    = L
    fh.attrs["Nb"]   = Nb
    fh.attrs["Nmax"] = Nmax
    fh.attrs["t"]    = t_hop
    fh.attrs["U"]    = U
    fh.attrs["mu"]   = mu
    fh["energies"]   = E[:6]
    fh["n_expect"]   = n_expect

print(f"L={L} Nb={Nb} Nmax={Nmax} t={t_hop} U={U} mu={mu}")
print(f"  basis dim   : {basis.Ns}")
print(f"  GS energy   : {E[0]:.12f}")
print(f"  Gap         : {E[1] - E[0]:.12f}")
print(f"  <n_i>       : {n_expect}")
print(f"  -> wrote {ref_path}")
