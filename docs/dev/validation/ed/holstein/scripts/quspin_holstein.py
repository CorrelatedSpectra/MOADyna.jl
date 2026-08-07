"""
QuSpin reference for spinless Holstein validation in MOADyna.

Generates `holstein_reference.h5` (committed under `reference/`).
The test `test/ed/validation/test_holstein_quspin.jl` loads it and
compares to MOADyna. Spinless because QuSpin's tensor_basis pairs
`boson_basis_1d` with `spinless_fermion_basis_1d` cleanly via the `|`
separator (cf. `ref/QuSpin/examples/scripts/example10.py`); the
spinful Hubbard piece is independently validated by Plan 2 / NiO XAS.

Hamiltonian (2-site OBC, single fermion, one local Einstein mode per
site):

    H = -t (c†_0 c_1 + c†_1 c_0)
       + ω Σ_i n^b_i
       + g Σ_i (b†_i + b_i) n^f_i

The fermion-boson coupling is split into two QuSpin static terms
(`+|n` weight g, `-|n` weight g); their sum realises (b†+b) n_f.

Re-run with:
    conda run -n quspin python quspin_holstein.py
"""
import os

import numpy as np
import h5py
from quspin.basis import (
    boson_basis_1d,
    spinless_fermion_basis_1d,
    tensor_basis,
)
from quspin.operators import hamiltonian


L       = 2
Nf      = 1
Nmax_ph = 10        # phonon cutoff per site; over-converged at g/ω=0.7
t_hop   = 1.0
omega   = 1.0
g_eph   = 0.7

basis_b = boson_basis_1d(L, sps=Nmax_ph + 1)              # no Nb constraint
basis_f = spinless_fermion_basis_1d(L, Nf=Nf)
basis   = tensor_basis(basis_b, basis_f)                  # (boson | fermion)

hop_f_pos = [[-t_hop, 0, 1]]
hop_f_neg = [[+t_hop, 0, 1]]
omega_list = [[omega, i]    for i in range(L)]
eph_list   = [[g_eph, i, i] for i in range(L)]

static = [
    ["n|",   omega_list],   # ω · n^b on the boson half
    ["|+-",  hop_f_pos],    # -t c†_0 c_1
    ["|-+",  hop_f_neg],    # +t c_0 c†_1  ≡ -t c†_1 c_0
    ["+|n",  eph_list],     # +g · b†_i · n^f_i
    ["-|n",  eph_list],     # +g · b_i  · n^f_i
]

H = hamiltonian(static, [], basis=basis, dtype=np.float64,
                check_symm=False, check_pcon=False, check_herm=True)

E, V = H.eigh()
psi0 = V[:, 0]

# Total phonon number <Σ n^b>.
nph_op = hamiltonian(
    [["n|", [[1.0, i] for i in range(L)]]], [], basis=basis,
    dtype=np.float64,
    check_symm=False, check_pcon=False, check_herm=True,
)
nph_total = float(psi0.conj() @ (nph_op.dot(psi0)))

# <n^f_0> — fraction of the single fermion sitting on site 0.
nf0_op = hamiltonian(
    [["|n", [[1.0, 0]]]], [], basis=basis,
    dtype=np.float64,
    check_symm=False, check_pcon=False, check_herm=True,
)
nf0 = float(psi0.conj() @ (nf0_op.dot(psi0)))

ref_path = os.path.join(os.path.dirname(__file__),
                        "..", "reference", "holstein_reference.h5")
ref_path = os.path.abspath(ref_path)

with h5py.File(ref_path, "w") as fh:
    fh.attrs["L"]       = L
    fh.attrs["Nf"]      = Nf
    fh.attrs["Nmax_ph"] = Nmax_ph
    fh.attrs["t"]       = t_hop
    fh.attrs["omega"]   = omega
    fh.attrs["g"]       = g_eph
    fh["energies"]      = E[:6]
    fh["nph_total"]     = nph_total
    fh["nf_site0"]      = nf0

print(f"L={L} Nf={Nf} Nmax_ph={Nmax_ph} "
      f"t={t_hop} omega={omega} g={g_eph}")
print(f"  basis dim    : {basis.Ns}")
print(f"  GS energy    : {E[0]:.12f}")
print(f"  Gap          : {E[1] - E[0]:.12f}")
print(f"  <n_ph_total> : {nph_total:.12f}")
print(f"  <n_f site 0> : {nf0:.12f}")
print(f"  -> wrote {ref_path}")
