import math
from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import util

from scipy.constants import epsilon_0
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from util.plot.ga2o3 import prepare_axes


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

def get_schubert_per_mode_epsilons_xy(A, omega_phonon, angle_alpha, scattering_gamma, eps_infty, omega_range):
    """angle in radians"""

    
    epsilon_for_omega = np.array([get_rho(omega, A, omega_phonon, scattering_gamma) for omega in omega_range])
    return epsilon_for_omega + eps_infty_contribution

def get_schubert_per_mode_epsilons_y(A, omega_phonon, scattering_gamma, eps_inf_zz, omega_range):
    return np.array([get_rho(omega, A, omega_phonon, scattering_gamma) for omega in omega_range])


def get_schubert_normalised_coupling_strengths():

    warnings.warn("No FCC contribution to schubert permittivity")

    Bu = mode_data.loc[mode_data["symmetry"] == "Bu"]
    Au = mode_data.loc[mode_data["symmetry"] == "Au"]


#     Bu_epsilons = get_schubert_per_mode_epsilons_xy(
#         A = Bu["A"], 
#         omega_phonon = Bu["freq"],
#         angle_alpha = Bu["angle"],
#         scattering_gamma = Bu["scatter"],
#         eps_inf_xx = 
# 

    #eps_xx =  np.array([np.sum(get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"]) * np.cos(Bu["angle"])**2) for omega in omega_range])







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
    #assert n_evecs == n_atoms * 3

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
    """
    arguments:
    - omega: a single frequency (THz) at which to compute the permittivity 
    - gamma_frequencies: phonon frequencies (THz)
    - numerator:  C^2/kg
    - volume: m^3
    - gamma: broadening, THz-adjacent units (depends on broadening_type)
    - broadening_type: proportionsl, individual or constant

    """

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

def diagonalise_mx(mx):
    evals, evecs = np.linalg.eig(mx)
    D = np.eye(len(evals)) * evals
    P = evecs
    A = mx 
    assert np.allclose(D, np.linalg.inv(P) @ A @ P)
    return np.linalg.inv(P) @ mx @ P


def get_direction_of_response(ph_numerator, threshold=1e-18):

    conv = "xyz"

    # check where we expect to see non-zero
    zeroed_numerator = deepcopy(ph_numerator)
    zeroed_numerator[np.abs(zeroed_numerator) <threshold] = 0
    nz = np.nonzero(zeroed_numerator)

    if len(nz[0]) == 0:
        # no response in permittivity
        return None, None, None

    if len(nz[0]) == len(nz[1]) and len(nz[0]) == 1:
        id1 = nz[0][0]
        id2 = nz[1][0]
        response_in = conv[id1] + conv[id2]
        return response_in, id1, id2

    response = []
    non_zero_ids = []
    for id1, id2 in zip(nz[0], nz[1]):
        vv1 = conv[id1]
        vv2 = conv[id2]
        if vv1 not in response:                
            response.append(vv1)
        if vv2 not in response:                
            response.append(vv2)
        if (id1, id2) not in non_zero_ids:
            non_zero_ids.append((id1, id2))
    response_in = "".join(response)

    return response_in, None, None


def get_normalised_coupling_strentghts(omega_range,  gamma_frequencies, numerator, volume, gamma, broadening_type, epsilon_inf, threshold=1e-14):

#     out_dir =  Path(
#     "/u/egg/mounted/high-throughput-permittivity/ionic_permittivities_from_jarvis_diagonalized_epsilon/tmp"
#         )
#     out_dir.mkdir(exist_ok=True)

    etas_dict = {"etas":[], "omega_phonon":[], "omega_eps0":[], "response_direction":[], "phonon_idx":[]} 

    for phonon_idx, phonon_freq in enumerate(gamma_frequencies):

        ph_numerator = numerator[phonon_idx]

        assert np.all(ph_numerator.imag == 0)
        ph_numerator = ph_numerator.real

        if np.all(np.abs(ph_numerator) < threshold) :
            continue

        orig_response_in, _, _ = get_direction_of_response(ph_numerator, threshold=threshold)
        
        ph_numerator = np.array([diagonalise_mx(ph_numerator)])
        _, id1, id2 = get_direction_of_response(ph_numerator[0])

        
        one_ph_eps =np.array(
            [
                epsilon_for_omega(
                    omega=omega,
                    gamma_frequencies=np.array([phonon_freq]),   # compute for this phonon only
                    numerator=ph_numerator,
                    volume=volume,
                    gamma=gamma,
                    broadening_type=broadening_type,
                )
                for omega in omega_range 
            ]
        )
        one_ph_eps += epsilon_inf
# 
#         out_fn = out_dir / f"phonon_{phonon_idx}.{phonon_freq*util.THz_to_inv_cm:.0f}.png"
#         axs = prepare_axes(omega_range)
#         omega_range_plt = omega_range * util.THz_to_inv_cm
#         axs["xx"]["real"].plot(omega_range_plt, one_ph_eps[:,0,0].real)
#         axs["xx"]["imag"].plot(omega_range_plt, one_ph_eps[:,0,0].imag)
#         axs["yy"]["real"].plot(omega_range_plt, one_ph_eps[:,1,1].real)
#         axs["yy"]["imag"].plot(omega_range_plt, one_ph_eps[:,1,1].imag)
#         axs["zz"]["real"].plot(omega_range_plt, one_ph_eps[:,2,2].real)
#         axs["zz"]["imag"].plot(omega_range_plt, one_ph_eps[:,2,2].imag)
#         axs["xy"]["real"].plot(omega_range_plt, one_ph_eps[:,0,1].real)
#         axs["xy"]["imag"].plot(omega_range_plt, one_ph_eps[:,0,1].imag)
# 
#         plt.savefig(out_fn)
# 

        freq_eps_0 = get_zero_crossing(one_ph_eps, omega_range, id1, id2, threshold)

        if freq_eps_0 == "na":
            eta = "na"
        else:
            eta = get_normalized_coupling_strength(omega_ph=phonon_freq, omega_zero=freq_eps_0)
            freq_eps_0 *= util.THz_to_inv_cm


        etas_dict["etas"].append(eta)
        etas_dict["omega_phonon"].append(phonon_freq * util.THz_to_inv_cm)
        etas_dict["omega_eps0"].append(freq_eps_0 )
        etas_dict["response_direction"].append(orig_response_in)
        etas_dict["phonon_idx"].append(phonon_idx)

    etas_df = pd.DataFrame(etas_dict, index=etas_dict["phonon_idx"])
    etas_df = etas_df.sort_values(by="omega_phonon", ascending=True)
    etas_df.reset_index(drop=True, inplace=True)

    etas_df["etas"] = etas_df["etas"].map(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    etas_df["omega_phonon"] = etas_df["omega_phonon"].map(lambda x: f"{x:.0f}")
    etas_df["omega_eps0"] = etas_df["omega_eps0"].map(lambda x: f"{x:.0f}" if isinstance(x, float) else x)


    column_renames = {
        "etas": "$\eta_\sigma$",
        "omega_phonon": "$\omega_\sigma$ [cm$^{-1}$]",
        "omega_eps0": "$\omega_{\sigma, \\varepsilon=0}$ [cm$^{-1}$]",
        "response_direction": "original component",
        "phonon_idx": "phonon no.",
    }
    etas_df = etas_df.rename(columns=column_renames)
         
    return etas_df 
                

def get_zero_crossing(one_ph_eps, omega_range, id1, id2, threshold):

    assert len(one_ph_eps) == len(omega_range)

    # mask small values to be zero
    one_ph_eps[np.abs(one_ph_eps) < threshold] = 0

    # where crossess zero?
    zero_crossings = np.where(np.diff(np.sign(one_ph_eps[:,id1,id2].real)))[0]
    if len(zero_crossings) == 0:
        freq_eps_0 = "na"
    elif len(zero_crossings)!=2:
        raise RuntimeError(f"got {len(zero_crossings)} zero crossings for phonon {phonon_idx} - extend omega range?")
    else:
        freq_eps_0 = omega_range[zero_crossings[1]] 
    
    return freq_eps_0
            

def get_normalized_coupling_strength(omega_ph, omega_zero):
    return np.sqrt(omega_zero ** 2 - omega_ph ** 2) / omega_ph


