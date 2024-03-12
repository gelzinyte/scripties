import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

from scipy.constants import epsilon_0, physical_constants, speed_of_light

from ase.io import read, write

from vibes.structure.convert import to_Atoms

import util
import util.phonopy
import util.aims
import util.geometry


def get_polarization(aims_output_file, regex_patt):

    with open(aims_output_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Cartesian Polarization" in line:
                re_match = regex_patt.search(line)
                polarization = np.array([float(mm) for mm in re_match.groups()])
                return polarization
    raise RuntimeError("No cartesian polarization found")


def get_born_charges():

    out_born_charges = []

    # Units of C/m^2
    cart_pol_regex = re.compile("(?:Cartesian Polarization)\s+(-?[\d.E-]+)\s+(-?[\d.E-]+)\s+(-?[\d.E-]+)")

    homedir = Path("2.born_effective_charges")
    workdir = homedir / "workdir"
    aims_output_file_name = 'aims.out'

    atoms = read(homedir / 'geometry.in')
    non_equivalent_atoms = np.arange(len(atoms))
    # from  vibes info geometry geometry.in -vv

    displacement_magnitude = 0.02 # Angstrom

    volume = atoms.get_volume()
    print(f'Volume of the cell is: {volume} AA^3')

    for idx, atom in enumerate(non_equivalent_atoms):

        entry = []
        for cart_idx, cart in enumerate(["x", "y", "z"]):

            atom_symbol = atoms.get_chemical_symbols()[atom]
            aims_output_file_path = workdir / f"{idx}.{atom_symbol}.{cart}" / aims_output_file_name

            polarization = get_polarization(aims_output_file_path, regex_patt = cart_pol_regex)
            # from aims tutorial, in multiples of elementary charge:
            # born_charges = (volume / 1.602) * (polarization/displacement_magnitude) * 0.1 # why 0.1?
            born_charges = volume * polarization / displacement_magnitude  # units: 10^-20 C 

            #print(f'Z* in x, y, z for atom {idx} {atom_symbol} displaced in {cart} is {born_charges}')
            entry.append([float(bec) for bec in born_charges])
        out_born_charges.append(entry)

    out_born_charges = np.array(out_born_charges)
    # to match the shape of array of reference methods
    out_born_charges = np.moveaxis(out_born_charges, 1, 2)
    print(f"born charges shape is {out_born_charges.shape} n_atoms x 3_polarization_directions x 3_cart_at_displacements")
    return out_born_charges

phonon, labels = util.phonopy.phonon_and_labels(
    trajectory = "4.phonopy_3x3x3_output/trajectory.son",
    eps=0.002,
    q_mesh = [5,5,5]
)

# -----------
# aside - get irrep labels 
# -----------

phonon.set_irreps(q=[0,0,0])
irrep_labels = phonon.irreps._get_ir_labels()
if len(irrep_labels)!=30:
    warnings.warn(f"Got {len(irrep_labels)} labels, not 30 as expected. Overwriting manually.")
    irrep_labels = ['Bu', "Bu", "Au", 'Ag', 'Bg', 'Bg', 'Au', 'Ag', 'Ag', 'Bu', 'Bu', 'Bu', 'Au', 'Ag', 'Ag', 'Bu', 'Bg', 'Ag', 'Bu', 'Au', 'Ag', 'Bg', 'Bu', 'Ag', 'Bg', 'Ag', 'Au', 'Bu', 'Bu', 'Ag']

relevant_mode_idcs = np.array([idx for idx, label in enumerate(irrep_labels) if label in ["Bu", "Au"]])
# skip the first three Acoustic modes
relevant_mode_idcs = relevant_mode_idcs[3:]
print(f"Have {len(relevant_mode_idcs)} relevant modes")

# -----------
# get and format normal modes 
# -----------

phonon.run_mesh(
    mesh=[5,5,5],
    with_eigenvectors=True)
mesh_dict = phonon.get_mesh_dict()
assert np.all(mesh_dict["qpoints"][0] == np.zeros(3))

# eigenvectors of dyn matrix
# different eigenvectors as columns, so 
# shape - [n_at x 3] x n_eigenvectors, 
# 30 x 30, eigenvectors as columns
gamma_evecs = mesh_dict["eigenvectors"][0]
# transpose so different eigenvectors are enumerated by the first axis (rows)
gamma_evecs = gamma_evecs.transpose()
# reshape so each eigenvector has the shape of n_at x 3D
gamma_evecs = gamma_evecs.reshape((30, 10, 3))   # [30,10,3]

# For each eigenvector, convert from primitive to conventional 
prim_to_conv_perm_mx = util.geometry.array_perm_duplicate_mx()  # [20,10]
# for each eigenvector the transformation is 
# prim_to_conv_perm_mx @ eigenvector
# So to rewrite for everyone
conv_evecs = np.einsum("ij,hjk->hik", prim_to_conv_perm_mx, gamma_evecs)


# -----------
# visualise the modes
# -----------

# get atoms
at_prim = to_Atoms(phonon.primitive)
at_conv = util.geometry.primitive_to_conventional_at(at_prim)


# write displacements 
ats_out = []
#for ref_at, evecs in [(at_prim, gamma_evecs), (at_conv, conv_evecs)]:
for ref_at, evecs in [(at_conv, conv_evecs)]:
    for mode_idx, nmode in enumerate(evecs):
        at = ref_at.copy()
        at.info[f"mode"] = mode_idx
        at.info[f"symmetry"] = irrep_labels[mode_idx]
        at.arrays["Displacement"] = np.real(nmode)
        ats_out.append(at)

write("/u/egg/mounted/nmode_displacements.xyz", ats_out)


# -----------
# high-frequency dielectric constant 
# -----------

out_fn = Path("5.dielectric_for_primitive/workdir/aims.out")
if not out_fn.exists():
    write("5.dielectric_for_primitive/geometry.in", at_conv, format="aims")
high_freq_eps = util.aims.read_dielectric(out_fn)
print('high frequency_eps:\n', high_freq_eps)


# -----------
# get masses, born charges, etc. Work in prmitive cell 
# -----------

# shape n_atoms x 3 polarization directions x 3 displacement directions
born_charges = get_born_charges()
# reorder into conventional order  
born_charges = np.einsum("ij,jkl->ikl", prim_to_conv_perm_mx, born_charges)


masses = at_conv.get_masses()
# copy into correct shape:
# first promote to 3D array of shape (1, 20, 1), then 
# repeat it (30, 1, 3) times along the respective axes. 
masses = np.tile(masses.reshape(1, 20, 1), (30, 1, 3))

# -----------
# implement equation 9 
# -----------

# shape n_evecs x n_atoms x 3
eigen_displacements = conv_evecs / np.sqrt(masses)

# i - number of atoms 
# "jk,k" is the matrix vector multiplication
# units: 10^-20 C / sqrt(amu)
# !!!! here only real??
S = np.einsum('ijk,lik->lj', born_charges, eigen_displacements)
assert S.shape == (30, 3)
# outer products of S for each mode
numerator = np.einsum("ai,aj->aij", S, np.conjugate(S))[relevant_mode_idcs]

gamma_frequencies = mesh_dict["frequencies"][0] # THz
gamma_frequencies *= util.THz_to_inv_cm  # invesse cm 
gamma_frequencies = gamma_frequencies[relevant_mode_idcs]

volume = at.get_volume()

amu = physical_constants["atomic mass constant"][0] # kg
my_units_to_SI = 1e-14/speed_of_light**2/amu

def epsilon_for_omega(omega, gamma_frequencies, numerator, volume, units_factor, gamma):
    # with scattering 
    denominator = gamma_frequencies ** 2 - omega ** 2 - omega * gamma * 1j 
    # cast into correct shape again
    denominator = np.tile(denominator.reshape(12, 1, 1), (1, 3, 3))
    epsilon_contribution = np.sum(np.divide(numerator, denominator), axis=0)/volume
    epsilon_contribution = epsilon_contribution * units_factor / epsilon_0 # relative epsilon
    return epsilon_contribution


omega_range = np.arange(0, 800, 2)
broadening = 10 # 
eps_for_omega = np.array([epsilon_for_omega(omega, gamma_frequencies, numerator, volume, my_units_to_SI, broadening) for omega in omega_range])
print(type(eps_for_omega))

# ------------------
# print and plot
# ------------------

print("frequencies, cm-1")
for idx, (freq, irrep_id) in enumerate(zip(gamma_frequencies, relevant_mode_idcs)):
    print(f"{idx}. {freq:.2f}, {irrep_labels[irrep_id]}")

min_y_r = np.min(np.real(eps_for_omega))
max_y_r = np.max(np.real(eps_for_omega))
min_y_r *= 1.1
max_y_r *= 1.1
min_y_i = np.min(np.imag(eps_for_omega))
max_y_i = np.max(np.imag(eps_for_omega))
min_y_i *= 1.1
max_y_i *= 1.1


vline_colors = ["tab:red" if irrep_labels[irrep_id] == "Au" else "tab:blue" for irrep_id in relevant_mode_idcs]

fig = plt.figure(figsize=(12,12), constrained_layout=False)
outer_grid = fig.add_gridspec(3, 3, wspace=0.4, hspace=0.4)
#gs = GridSpec(3, 3, wspace=0.4, hspace=0.3)

axes = []
for id1, xyz1 in enumerate(["x", "y", "z"]):
    for id2, xyz2 in enumerate(["x", "y", "z"]):

        # lazily pick colors by symmetry
        if xyz1 == "y" and xyz2 == "y":
            color= "tab:red"
        elif xyz1 == "y" or xyz2 == "y":
            color="gray"
        else:
            color="tab:blue"

        inner_grid = outer_grid[id1, id2].subgridspec(2, 1, wspace=0, hspace=0)
        axs = inner_grid.subplots(sharex=True)
        ax_r = axs[0]
        ax_i = axs[1]

        ys_r = eps_for_omega[...,id1,id2].real
        ys_i = eps_for_omega[...,id1,id2].imag

        ax_r.plot(omega_range, ys_r, color=color)
        ax_i.plot(omega_range, ys_i, color=color)

        ax_r.vlines(gamma_frequencies, min_y_r, max_y_r, alpha=0.1, color=vline_colors)
        ax_i.vlines(gamma_frequencies, min_y_i, max_y_i, alpha=0.1, color=vline_colors)


        ax_r.set_title(f"{xyz1}{xyz2}")
        ax_i.set_xlabel("IR mode frequency, cm$^{-1}$")
        #ax_r.set_ylabel("Real(epsilon), relative to epsilon0")
        #ax_i.set_ylabel("Imag(epsilon), relative to epsilon0")

        ax_r.set_ylim((min_y_r, max_y_r))
        if xyz1 != xyz2:
            ax_i.set_ylim((min_y_i, max_y_i))
        else:
            ax_i.set_ylim(top=max_y_i)


plt.savefig("/u/egg/mounted/epsilon.png", dpi=300)



