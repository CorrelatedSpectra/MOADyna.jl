from quspin.operators import hamiltonian,quantum_operator
import numpy as np
import sys,os,time
import subprocess
from datetime import timedelta

# Add the qu2anty module to the path
sys.path.append(r"../../")
from pyquanty.interfaces import *
from pyquanty.basis import *
from pyquanty.spec import *

# Number of sites and particles as defined in Quanty
nd = 8
N  = 6 + 10 + 10
Np = 6 + 10 + nd

# Define the basis suited for Quanty Hamiltonians
op_args=np.array([],dtype=np.uint64)
op_dict = dict(op=op, op_args=op_args)

next_state_args=np.array([],dtype=np.uint64) # compulsory, even if empty
count_particles_args=np.array([N],dtype=np.int64)
n_sectors=1
pcon_dict = dict(Np=Np,
                 next_state=next_state,
                 next_state_args=next_state_args,
                 get_Ns_pcon=get_Ns_pcon,
                 get_s0_pcon=get_s0_pcon,
                 count_particles=count_particles,
                 count_particles_args=count_particles_args,
                 n_sectors=n_sectors)
# Define anti-commuting bits -- fermion signs on the integer bits (not sites!) that represent a fermion degree of freedom
# fermion signs are counted w.r.t. the shift operator <<
noncommuting_bits = [(np.arange(N),-1)]

# define masks corresponding to different orbitals
mask_p  = string2mask('111111 0000000000 0000000000')
mask_d  = string2mask('000000 1111111111 0000000000')
mask_L  = string2mask('000000 0000000000 1111111111')
mask_v  = string2mask('000000 1111111111 1111111111')
mask_All= string2mask('111111 1111111111 1111111111')

# Restrictions:
# 1. we require core to be filled or has one hole for the GS and excited states
# 2. we require total number to be exactly Np
# Note that the basis is generated from large to small numbers, so the first states are with filled cores (relevant for GS)
pcs_args=np.array([2, mask_p, 5, 6, mask_All, Np, Np],dtype=np.uint64)
pcs=(pre_check_state,pcs_args)
basis = user_basis(np.uint64,
                   N, op_dict,
                   allowed_ops=set("+-nI"),
                   sps=2,
                   pcon_dict=pcon_dict,
                   noncommuting_bits=noncommuting_bits,
                   pre_check_state=pcs)
pcs_args=np.array([2, mask_p, 6, 6, mask_All, Np, Np],dtype=np.uint64)
pcs=(pre_check_state,pcs_args)
basisGS = user_basis(np.uint64,
                   N, op_dict,
                   allowed_ops=set("+-nI"),
                   sps=2,
                   pcon_dict=pcon_dict,
                   noncommuting_bits=noncommuting_bits,
                   pre_check_state=pcs)
pcs_args=np.array([2, mask_p, 5, 5, mask_All, Np, Np],dtype=np.uint64)
pcs=(pre_check_state,pcs_args)
basisXAS = user_basis(np.uint64,
                   N, op_dict,
                   allowed_ops=set("+-nI"),
                   sps=2,
                   pcon_dict=pcon_dict,
                   noncommuting_bits=noncommuting_bits,
                   pre_check_state=pcs)
print("There are %d states in the basis." % (basis.Ns))
print("          %d for GS" % (basisGS.Ns))
print("          %d for XAS" % (basisXAS.Ns))

## Generating Quanty operator/Hamiltonian for benchmarking purposes
subprocess.run(["Quanty", "generate_quanty_operators.lua"], stdout=subprocess.DEVNULL)
op_dir = "./operators/"

no_checks=dict(check_symm=False,check_pcon=False,check_herm=False)
def constructObservable(flname, basisopt):
  strObs = readQuantyOperators(op_dir+flname)
  lstObs = convertOpStrsSpinless(strObs,shift=0)
  return hamiltonian(lstObs,[],basis=basisopt,dtype=np.complex128,**no_checks)

##
## GS calculation
##
## Load operators from Quanty output
print("Constructing Hamiltonian ...", flush=True)
H = constructObservable("Hamiltonian.txt", basisGS)

print("Constructing Observables ...", flush=True)
OppSsqr = constructObservable("OppSsqr.txt", basisGS)
OppLsqr = constructObservable("OppLsqr.txt", basisGS)
OppJsqr = constructObservable("OppJsqr.txt", basisGS)
OppSz_3d = constructObservable("OppSz_3d.txt", basisGS)
OppLz_3d = constructObservable("OppLz_3d.txt", basisGS)
Oppldots_3d = constructObservable("Oppldots_3d.txt", basisGS)
OppF2_3d = constructObservable("OppF2_3d.txt", basisGS)
OppF4_3d = constructObservable("OppF4_3d.txt", basisGS)
OppNeg_3d = constructObservable("OppNeg_3d.txt", basisGS)
OppNt2g_3d = constructObservable("OppNt2g_3d.txt", basisGS)
OppNeg_Ld = constructObservable("OppNeg_Ld.txt", basisGS)
OppNt2g_Ld = constructObservable("OppNt2g_Ld.txt", basisGS)
OppN_3d = constructObservable("OppN_3d.txt", basisGS)

obs_dict = dict(E=H, Ssqr=OppSsqr, Lsqr=OppLsqr, Jsqr=OppJsqr, Sz_3d=OppSz_3d, Lz_3d=OppLz_3d, ldots_3d=Oppldots_3d, F2_3d=OppF2_3d, F4_3d=OppF4_3d, Neg_3d=OppNeg_3d, Nt2g_3d=OppNt2g_3d, Neg_Ld=OppNeg_Ld, Nt2g_Ld=OppNt2g_Ld, N_3d=OppN_3d)

start = time.time()

nvals = 3
print("Computing targeted states ...", flush=True)
evals, efuns = H.eigsh(k=nvals, which='SA', return_eigenvectors=True)
idx = evals.argsort()
evals = evals[idx]
efuns = efuns[:,idx]


elevels = np.zeros(shape=(nvals, len(obs_dict)))
for i, (name, obs) in enumerate(obs_dict.items()):
  elevels[:,i] = obs.expt_value(efuns)
#np.savetxt("exp_values.txt",elevels,header=str(obs_dict.keys()))
E0 = elevels[0,0]
print(f"GS energy: {E0:8.4f}")

print("  #    <E>      <S^2>    <L^2>    <J^2>    <S_z^3d> <L_z^3d> <l.s>    <F[2]>   <F[4]>   <Neg^3d> <Nt2g^3d><Neg^Ld> <Nt2g^Ld><N^3d>")
for i in range(len(elevels)):
  print(f"{i:3d}", end=" ",flush=True)
  for j in range(len(elevels[0])-1):
    print(f"{elevels[i][j]:8.4f}", end=" ",flush=True)
  print(f"{elevels[i][len(elevels[0])-1]:8.4f}", flush=True)

end = time.time()
print(f"\n==============================")
print(f" GS time cost: {timedelta(seconds=end - start)}")
print(f"==============================\n")
start = end

##
## XAS and RIXS calculation
##
print("Constructing XAS Hamiltonian ...", flush=True)
HXAS = constructObservable("XASHamiltonian.txt", basisXAS)

print("Constructing XAS operators ...", flush=True)
TXASx = constructObservable("TXASx.txt", basis)
TXASy = constructObservable("TXASy.txt", basis)
TXASz = constructObservable("TXASz.txt", basis)

# Go back to the large Hilbert space to find the states T|psi>
psi0 = np.append(efuns[:,0], np.zeros(basisXAS.Ns))
psix = TXASx.dot(psi0,time=0,check=True)
psix = psix[basisGS.Ns:]
psiy = TXASy.dot(psi0,time=0,check=True)
psiy = psiy[basisGS.Ns:]
psiz = TXASz.dot(psi0,time=0,check=True)
psiz = psiz[basisGS.Ns:]

# Define the parameters for the XAS calculation once
xas_params = SpectrumParams(emin=-15, emax=25, ne=801, eta=0.3, ntri=100)

# --- Compute the XAS spectra ---

# Method 1: bi-conjugate gradient
calXAS_bicg = True
if calXAS_bicg:
    print("Computing XAS spectra with bi-conjugate gradient ...", flush=True)
    XASx, _ = CreateSpectra(psi=psix, H=HXAS, E0=E0, params=xas_params, method="bicg")
    XASy, _ = CreateSpectra(psi=psiy, H=HXAS, E0=E0, params=xas_params, method="bicg")
    XASz, _ = CreateSpectra(psi=psiz, H=HXAS, E0=E0, params=xas_params, method="bicg")
    XAS = np.column_stack((XASx[:, 0], XASx[:, 1], XASy[:, 1], XASz[:, 1]))
    save_spec("XAS_bicg.txt", XAS)

    end = time.time()
    print(f"\n==============================")
    print(f" XAS (bicg) time cost: {timedelta(seconds=end - start)}")
    print(f"==============================\n")
    start = end

# Method 2: Lanczos + continued fraction
calXAS_cfrac = True
if calXAS_cfrac:
    print("Computing XAS spectra with Lanczos (continued fraction) ...", flush=True)
    XASx, _ = CreateSpectra(psi=psix, H=HXAS, E0=E0, params=xas_params, method="lanczos_cont_frac")
    XASy, _ = CreateSpectra(psi=psiy, H=HXAS, E0=E0, params=xas_params, method="lanczos_cont_frac")
    XASz, _ = CreateSpectra(psi=psiz, H=HXAS, E0=E0, params=xas_params, method="lanczos_cont_frac")
    XAS = np.column_stack((XASx[:, 0], XASx[:, 1], XASy[:, 1], XASz[:, 1]))
    save_spec("XAS_lanczos_cont_frac.txt", XAS)

    end = time.time()
    print(f"\n==============================")
    print(f" XAS (Lanczos cont-frac) time cost: {timedelta(seconds=end - start)}")
    print(f"==============================\n")
    start = end

# Method 3: Lanczos + Lehmann representation
calXAS_lehmann = True
if calXAS_lehmann:
    print("Computing XAS spectra with Lanczos (Lehmann rep) ...", flush=True)
    XASx, _ = CreateSpectra(psi=psix, H=HXAS, E0=E0, params=xas_params, method="lanczos_lehmann")
    XASy, _ = CreateSpectra(psi=psiy, H=HXAS, E0=E0, params=xas_params, method="lanczos_lehmann")
    XASz, _ = CreateSpectra(psi=psiz, H=HXAS, E0=E0, params=xas_params, method="lanczos_lehmann")
    XAS = np.column_stack((XASx[:, 0], XASx[:, 1], XASy[:, 1], XASz[:, 1]))
    save_spec("XAS_lanczos_lehmann.txt", XAS)

    end = time.time()
    print(f"\n==============================")
    print(f" XAS (Lanczos Lehmann) time cost: {timedelta(seconds=end - start)}")
    print(f"==============================\n")
    start = end

# --- Compute the RIXS spectra ---
calRIXS = True
if calRIXS:
    psi0 = efuns[:, 0]

    # Define parameters for incident and emitted energy axes
    inc_params = SpectrumParams(emin=-6.0, emax=0.0, ne=11, eta=0.3)
    emit_params = SpectrumParams(emin=-0.5, emax=8.0, ne=851, eta=0.05, ntri=100)

    print("Computing RIXS spectra ...", flush=True)
    RIXSxx, polesxx = CreateResonantSpectra(
        psi=psi0, T1=TXASx, T2=TXASy.getH(copy=True), H1=HXAS, H2=H, E0=E0,
        inc_params=inc_params,
        emit_params=emit_params,
        method="lanczos",
        outfile="RIXSxy.txt"
    )
    np.savetxt("RIXSxy_poles.txt", polesxx)

    end = time.time()
    print(f"\n==============================")
    print(f" RIXS time cost: {timedelta(seconds=end - start)}")
    print(f"==============================\n")
    start = end

# --- Get excitation states for a fixed incident energy ---
psi0 = efuns[:, 0]
E, wts, V = CreateRIXSExcitations(
    psi=psi0, T1=TXASx, T2=TXASy.getH(copy=True), H1=HXAS, H2=H, E0=E0,
    ntri=100, ein=-4.0, eta=0.3
)
for i in range(len(E)):
    print(E[i], H.expt_value(V[:, i]).real)