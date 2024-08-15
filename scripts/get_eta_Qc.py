import numpy as np

import matplotlib.pyplot as plt

import pytest

from ase.io import read

from ml_spectroscopy import calculators as spec

# Quaternions for 
# 7 Li
QLi = np.abs(-0.04003) #  Units: 1barn ; 1barn=10^-28 m^2
ILi=3/2 # spin, parity=-1
# 47 Ti
QTi = np.abs(0.30210)
ITi = 5/2 # parity = -1 
# 17 O
QO = np.abs(-0.02562) # barn
IO = 5/2 # parity = +1 

Quadrupoles = {
 "Li": QLi,
 "O": QO,
 "Ti": QTi,
 }

spins = {
 "Li": ILi,
 "O": ITi,
 "Ti": IO,
 }


at = read("example-dataset_all_22_56c426.xyz")
idx=0
efg = at.arrays["ref_efgs"][idx].reshape((3,3))
element=at.symbols[idx]
print("element:", element)

assert pytest.approx(efg.trace(), abs=1e-8) == 0

Cq, eta = spec.getcq(efg.flatten(), species=element)
print(Cq, eta)


#ats = read("example-dataset_all_22_56c426.xyz", ":"),
dataset = "Test"
ats = read("/work/home/egg/data/lto_efgs_from_ange/LTO42-ratttle-pristine-fracscale_validate_1312_b73de8.xyz", ":")
#ats += read("/work/home/egg/data/lto_efgs_from_ange/LTO42-ratttle-pristine-fracscale_train_1312_b73de8.xyz", ":")
# dataset = "Train"


efgs_by_element = {
    "Li": [],
    "Ti": [],
    "O": [],
        }

traces = {
    "Li": [],
    "Ti": [],
    "O": [],
    }

for at in ats:
    symbols = at.symbols
    efgs = at.arrays["efg"]
    for sym, efg in zip(symbols, efgs):
        efg = efg.reshape((3, 3))
        efgs_by_element[sym].append(efg)
        traces[sym].append(efg.trace())

for element, tr in traces.items():
    print(f"max {element}: {np.max(tr)}")

for element, efgs in efgs_by_element.items():
    print(f"number {element}: {len(efgs)}")

all_cqs = {}
all_etas = {}

for element, efg_list in efgs_by_element.items():

    Cqs = []
    etas = []
    for efg in efg_list:
        Cq, eta = spec.getcq(efg.flatten(), element)
        Cqs.append(Cq)
        etas.append(eta)

    all_cqs[element] = Cqs
    all_etas[element] = etas 


    plt.figure()
    plt.hist(np.abs(Cqs), bins=40)
    ax = plt.gca()
    ax.set_xlabel(f"CQ {element} / MHz")
    ax.set_ylabel("Counts")
    plt.title(f"dataset: {dataset}")
    plt.savefig(f"/work/home/egg/mounted/EFGs/Cq_distribution.{element}.png", dpi=300)


    plt.figure()
    plt.hist(etas, bins=40)
    ax = plt.gca()
    ax.set_xlabel(f"eta {element}")
    ax.set_ylabel("Counts")
    plt.title(f"dataset: {dataset}")
    plt.savefig(f"/work/home/egg/mounted/EFGs/eta_distribution.{element}.png", dpi=300)

x=0
y=1
z=2

element = "Li"
LiDFT_evals = np.array([spec._get_haeberlen_eigs(efg) for efg in  efgs_by_element[element]])
LiDFT_Vzzs = LiDFT_evals[:,z]

LiDFT_Vzz_mean = np.mean(LiDFT_Vzzs)
LiDFT_Vzz_std = np.std(LiDFT_Vzzs)


# or normalized Vz
tilde_Cqs = (LiDFT_Vzzs - LiDFT_Vzz_mean)/LiDFT_Vzz_std
tilde_etas = (LiDFT_evals[:,x] - LiDFT_evals[:,y])/LiDFT_Vzz_mean


plt.figure()
plt.hist(tilde_Cqs, bins=20)
ax = plt.gca()
ax.set_xlabel(f"tilde_CQs {element}")
ax.set_ylabel("Counts")
plt.title(f"dataset: {dataset}")
plt.savefig(f"/work/home/egg/mounted/EFGs/tilde_Cqs_distribution.{element}.png", dpi=300)

plt.figure()
plt.hist(tilde_etas, bins=20)
ax = plt.gca()
ax.set_xlabel(f"tilde_etas {element}")
ax.set_ylabel("Counts")
plt.title(f"dataset: {dataset}")
plt.savefig(f"/work/home/egg/mounted/EFGs/tilde_etas_distribution.{element}.png", dpi=300)



# get angles

LiDFT_evecs_zz = np.array([spec._get_haeberlen_eig_vecs(efg)[2] for efg in  efgs_by_element[element]])
thetas = np.array([np.arccos(Vzz_evec[2]) for Vzz_evec in LiDFT_evecs_zz])
phis = np.array([np.arctan(Vzz_evec[1]/Vzz_evec[0])  if np.abs(Vzz_evec[0]) > 0 else 0 for Vzz_evec in LiDFT_evecs_zz])


plt.figure()
plt.hist(thetas, bins=20)
ax = plt.gca()
ax.set_xlabel(f"theta {element}")
ax.set_ylabel("Counts")
plt.title(f"dataset: {dataset}")
plt.savefig(f"/work/home/egg/mounted/EFGs/thetas_distribution.{element}.png", dpi=300)

plt.figure()
plt.hist(phis, bins=20)
ax = plt.gca()
ax.set_xlabel(f"phis {element}")
ax.set_ylabel("Counts")
plt.title(f"dataset: {dataset}")
plt.savefig(f"/work/home/egg/mounted/EFGs/phis_distribution.{element}.png", dpi=300)


LiDFT_evecs = np.array([spec._get_haeberlen_eig_vecs(efg) for efg in  efgs_by_element[element]])
quaternions = np.array([spec.calc_quaternion(evecs) for evecs in LiDFT_evecs])
print(f'quaternions: {quaternions.shape}')


def get_omega_q(Cq, eta, spin, theta, phi):
    
#     omega_q = 3/(2 * spin * (2 * spin - 1))
#     omega_q *= Cq/2
#     omega_q *= (3 * np.cos(theta)**2 - 1 - eta * np.sin(theta)**2 * np.cos(2 * phi))

    eterm2 = 3 * np.cos(theta) ** 2 - 1 
    eterm3 = -1 * (np.sin(theta)**2 * np.cos( 2 * phi) )
    spin_bit = 3/(2 * spin * (2 * spin - 1))

    omega_q = np.abs(spin_bit * 0.5 * Cq * ( eterm2 + eta * eterm3))
                
    return omega_q

element = "Li"
spin = spins[element]

Li_Cqs = all_cqs[element]
Li_etas = all_etas[element]

assert len(thetas) == len(phis)
assert len(Li_Cqs)  == len(phis)
assert len(Li_etas) == len(phis)

omegas_q  = []
for Cq, eta, theta, phi in zip(Li_Cqs, Li_etas, thetas, phis):
    
    omega_q = get_omega_q(
            Cq=Cq, 
            eta=eta,
            spin=spin,
            theta=theta,
            phi=phi)

    omegas_q.append(omega_q)
omegas_q = np.array(omegas_q)
omegas_q *= 1e3

plt.figure()
plt.hist(omegas_q, bins=20)
ax = plt.gca()
ax.set_xlabel(f"omegas_q {element}, kHz")
ax.set_ylabel("Counts")
plt.title(f"dataset: {dataset}")
plt.savefig(f"/work/home/egg/mounted/EFGs/omegas_q_distribution.{element}.png", dpi=300)







