import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import util

from scipy.constants import epsilon_0



base = Path(util.__file__).parent / "data"

# data
mode_data = pd.read_csv(base / "schubert_paper.table_II.csv")
xyz_data = pd.read_csv(base / "schubert_paper.table_IV.csv", index_col="epsilon")
fcc_volume_density_param_N = 3.5e18 # cm^-3, page 15

mode_data["angle"] = np.radians(mode_data["angle"])

def get_rho(omega, A, f_TO, broadening):
    return A**2 / (f_TO**2 - omega**2 - omega * broadening * 1j)


def get_schubert(omega_range, epsilon_inf=True):

    warnings.warn("No FCC contribution to schubert permittivity")

    Bu = mode_data.loc[mode_data["symmetry"] == "Bu"]
    Au = mode_data.loc[mode_data["symmetry"] == "Au"]

    eps_xx =  np.array([np.sum(get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"]) * np.cos(Bu["angle"])**2) for omega in omega_range])

    eps_yy =  np.array([np.sum(get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"]) * np.sin(Bu["angle"])**2) for omega in omega_range])

    eps_xy =  np.array([np.sum(get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"]) * np.cos(Bu["angle"]) * np.sin(Bu["angle"])) for omega in omega_range])

    eps_zz =  np.array([np.sum(get_rho(omega, Au["A"], Au["freq"], Au["scatter"])) for omega in omega_range])


    if epsilon_inf:
        eps_xx += xyz_data["high_freq"]["xx"]
        eps_yy += xyz_data["high_freq"]["yy"]
        eps_zz += xyz_data["high_freq"]["zz"]
        eps_xy += xyz_data["high_freq"]["xy"]


    for idx, (A, freq, angle) in enumerate(zip(Bu["A"], Bu["freq"], Bu["angle"])):
        print(f"{idx}. A {freq:0f} cm-1 alpha {np.rad2deg(angle):.0f} deg freq {freq:.0f} cm^-1")


    data = {
        "xx": eps_xx,
        "yy": eps_yy,
        "xy": eps_xy,
        "zz": eps_zz}


    return data


def get_born_along_displacements(gamma_evecs, masses, born_charges):
    """
    arguments: 

    - gamma_evecs: 
        shape: len_masses*3 x len_masses x 3
        units: n/a
    - masses:
        shape: n/a
        units: kg
    - born_charges: 
        shape: len_masses x 3 x 3
        units: Coulomb

    """

    evec_shape = gamma_evecs.shape
    n_evecs = evec_shape[0]
    n_atoms = evec_shape[1]
    assert evec_shape[2] == 3
    assert n_atoms == len(masses)
    assert n_evecs == n_atoms * 3

    masses = np.tile(masses.reshape(1, n_atoms, 1), (n_evecs, 1, 3))

    # shape n_evecs x n_atoms x 3                                                                           
    eigen_displacements = gamma_evecs / np.sqrt(masses) # 1/sqrt(kg)                                         
    assert eigen_displacements.shape == (n_evecs, n_atoms, 3)

    # i - number of atoms                                                                                   
    # "jk,k" is the matrix vector multiplication                      
    S = np.einsum('ijk,lik->lj', born_charges, eigen_displacements) # C/sqrt(kg)
    assert S.shape == (evec_shape[0], 3)

    return S


def epsilon_for_omega(omega, gamma_frequencies, numerator, volume, gamma, broadening_type):

    # broadening prop to frequency
    if broadening_type == "proportional":
        denominator = gamma_frequencies ** 2 - omega ** 2 - gamma_frequencies**2 * omega * gamma * 1j    # THz^2
    elif broadening_type == "individual":
        assert gamma.shape == gamma_frequencies.shape
        real_part = gamma_frequencies ** 2 - omega ** 2
        imag_part = 1j * omega * gamma
        denominator = real_part - imag_part
    elif broadening_type == "constant":
        denominator = gamma_frequencies ** 2 - omega ** 2 - omega * gamma * 1j    # THz^2
    else:
        raise RuntimeError(f"gamma is of type {type(gamma)} neither float nor numpy array.")

    denominator *= (2 * np.pi) ** 2

    # cast into correct shape to later go with numerator
    denominator = np.tile(denominator.reshape(len(gamma_frequencies), 1, 1), (1, 3, 3)) # THz^2

    # axis=0 to sum over the [12] modes
    #                                       C^2/kg     THz^2                   m^3      THz^2 -> Hz^2
    epsilon_contribution = np.sum(np.divide(numerator, denominator), axis=0) / volume * 1e-24
    epsilon_contribution = epsilon_contribution / epsilon_0 # relative epsilon
    return epsilon_contribution


def get_numerator(S):
    # outer products of S for each mode
    numerator = np.einsum("ai,aj->aij", S, np.conjugate(S)) # C^2/kg
    return numerator


