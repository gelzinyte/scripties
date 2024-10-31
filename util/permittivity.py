import math
import warnings
from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import util

import bisect

from scipy.constants import epsilon_0
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from util.plot.ga2o3 import prepare_axes


base = Path(util.__file__).parent / "data"

# data
mode_data = pd.read_csv(base / "schubert_paper.table_II.csv")
xyz_data = pd.read_csv(base / "schubert_paper.table_IV.csv", index_col="epsilon")
fcc_volume_density_param_N = 3.5e18  # cm^-3, page 15

# radians
mode_data["angle"] = np.radians(mode_data["angle"])

eta_col_names_for_print = {
            "omega_sigma_root1": r"$\omega_{\sigma, \textrm{root 1}}$",
            "omega_eps_0_root2": r"$\omega_{\sigma, \varepsilon=0}$",
            "eta": r"$\eta_\sigma$",
            "phonon_freq": r"$\omega_{\sigma, \textrm{phonon}}$",
            "S_mag": r"$\textrm{len}(\mathbf{S})$",
            "const_gamma": r"$\gamma_\textrm{const}$",
            "prop_gamma": r"$\gamma_\textrm{prop}$",
        }


def get_rho(omega, A, f_TO, broadening):
    return A**2 / (f_TO**2 - omega**2 - omega * broadening * 1j)


def get_schubert(omega_range, epsilon_inf=True):

    warnings.warn("No FCC contribution to schubert permittivity")

    Bu = mode_data.loc[mode_data["symmetry"] == "Bu"]
    Au = mode_data.loc[mode_data["symmetry"] == "Au"]

    eps_xx = np.array(
        [
            np.sum(
                get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"])
                * np.cos(Bu["angle"]) ** 2
            )
            for omega in omega_range
        ]
    )

    eps_yy = np.array(
        [
            np.sum(
                get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"])
                * np.sin(Bu["angle"]) ** 2
            )
            for omega in omega_range
        ]
    )

    eps_xy = np.array(
        [
            np.sum(
                get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"])
                * np.cos(Bu["angle"])
                * np.sin(Bu["angle"])
            )
            for omega in omega_range
        ]
    )

    eps_zz = np.array(
        [
            np.sum(get_rho(omega, Au["A"], Au["freq"], Au["scatter"]))
            for omega in omega_range
        ]
    )

    if epsilon_inf:
        eps_xx += xyz_data["high_freq"]["xx"]
        eps_yy += xyz_data["high_freq"]["yy"]
        eps_zz += xyz_data["high_freq"]["zz"]
        eps_xy += xyz_data["high_freq"]["xy"]

    for idx, (A, freq, angle) in enumerate(zip(Bu["A"], Bu["freq"], Bu["angle"])):
        print(
            f"{idx}. A {freq:0f} cm-1 alpha {np.rad2deg(angle):.0f} deg freq {freq:.0f} cm^-1"
        )

    data = {"xx": eps_xx, "yy": eps_yy, "xy": eps_xy, "zz": eps_zz}

    return data


def get_schubert_individual_lorentz_oscillators(which, omega_range):

    assert which == "z"

    Bu = mode_data.loc[mode_data["symmetry"] == "Bu"]
    Au = mode_data.loc[mode_data["symmetry"] == "Au"]

    eps_zz = np.array(
        [get_rho(omega, Au["A"], Au["freq"], Au["scatter"]) for omega in omega_range]
    )

    eps_zz += xyz_data["high_freq"]["zz"]

    data = {"zz": eps_zz}

    return data


def find_first_above_threshold(numbers, threshold):
    index = bisect.bisect_right(numbers, threshold)
    return index if index < len(numbers) else -1


def get_schubert_mode_eps_zero(which, eps_infty, alpha, A, freq, scattering_gamma):
    """alpha in radians"""

    assert which in ["xy", "z"]

    if which == "z":
        eps_const = -eps_infty[2][2]
    elif which == "xy":
        top = eps_infty[1][0] ** 2 - eps_infty[0][0] * eps_infty[1][1]
        bottom = eps_infty[0][0] * np.sin(alpha) ** 2
        bottom += eps_infty[1][1] * np.cos(alpha) ** 2
        bottom -= 2 * eps_infty[0][1] * np.sin(alpha) * np.cos(alpha)
        eps_const = top / bottom
    else:
        raise ValueError(f'"which" should be either "xy" or "z", but got {which}')

    # quadratic equation
    QA = eps_const
    QB = scattering_gamma**2 * eps_const - 2 * freq**2 * eps_const + A**2
    QC = eps_const * freq**4 - A**2 * freq**2

    return solve_quadratic(A=QA, B=QB, C=QC)



def get_lorentz_roots(eps_infty, phonon_freq, S, V, scattering_gamma):

    assert np.all(S.imag == 0)
    S = S.real

    const = np.array([get_eps_infty_S_constant(eps_infty, s_row, V) for s_row in S])

    A = const 
    B = const * (scattering_gamma**2 - 2* phonon_freq**2) + 1
    C = const * phonon_freq**4 - phonon_freq**2

    return solve_quadratic(A=A, B=B, C=C)


def get_eps_infty_S_constant(eps_infty, S, V):

    # make notation easier
    e = eps_infty
    x = 0
    y = 1
    z = 2

    top = (
        e[x][x] * e[y][y] * e[z][z]
        + 2 * e[x][y] * e[y][z] * e[z][x]
        - e[x][x] * e[y][z] ** 2
        - e[y][y] * e[z][x] ** 2
        - e[z][z] * e[x][z] ** 2
    )

    bottom = (
        e[x][x] * e[y][y] * S[z] ** 2
        + e[y][y] * e[z][z] * S[x] ** 2
        + e[z][z] * e[x][x] * S[y] ** 2
        - e[x][y] ** 2 * S[z] ** 2
        - e[y][z] ** 2 * S[x] ** 2
        - e[z][x] ** 2 * S[y] ** 2
        - 2 * e[x][x] * e[y][z] * S[y] * S[z]
        - 2 * e[y][y] * e[z][x] * S[z] * S[x]
        - 2 * e[z][z] * e[x][y] * S[x] * S[y]
        + 2 * e[x][y] * e[y][z] * S[x] * S[z]
        + 2 * e[y][z] * e[z][x] * S[y] * S[x]
        + 2 * e[z][x] * e[x][z] * S[x] * S[z]
    )


    other_constants = epsilon_0 * (2 * np.pi)**2 * V * 1e24


    return -top / bottom * other_constants


def solve_quadratic(A, B, C):

    D = B**2 - 4 * A * C

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root1 = (+1 * B - np.sqrt(D)) / (2 * A)
        root2 = (+1 * B + np.sqrt(D)) / (2 * A)

    omega_0_1 = np.sqrt(np.abs(root1))
    omega_0_2 = np.sqrt(np.abs(root2))

    return omega_0_1, omega_0_2


def get_schubert_normalised_coupling_strengths():

    eps_infty = np.zeros((3, 3))
    eps_infty[0][0] = xyz_data["high_freq"]["xx"]
    eps_infty[0][1] = xyz_data["high_freq"]["xy"]
    eps_infty[1][0] = xyz_data["high_freq"]["xy"]
    eps_infty[1][1] = xyz_data["high_freq"]["yy"]
    eps_infty[2][2] = xyz_data["high_freq"]["zz"]

    Bu = mode_data.loc[mode_data["symmetry"] == "Bu"]
    Au = mode_data.loc[mode_data["symmetry"] == "Au"]

    z_cross_1, z_cross_2 = get_schubert_mode_eps_zero(
        "z", eps_infty, Au["angle"], Au["A"], Au["freq"], Au["scatter"]
    )
    xy_cross_1, xy_cross_2 = get_schubert_mode_eps_zero(
        "xy", eps_infty, Bu["angle"], Bu["A"], Bu["freq"], Bu["scatter"]
    )

    df_z = pd.DataFrame(
        {
            "omega_sigma_root1": z_cross_1,
            "omega_eps_0_root2": z_cross_2,
            "axis": ["z"] * len(z_cross_1),
            "phonon_freq": Au["freq"],
        }
    )
    df_xy = pd.DataFrame(
        {
            "omega_sigma_root1": xy_cross_1,
            "omega_eps_0_root2": xy_cross_2,
            "axis": ["xy"] * len(xy_cross_1),
            "phonon_freq": Bu["freq"]
        }
    )

    df_z["Schubert k"] = mode_data.iloc[df_z.index]["k"]
    df_xy["Schubert k"] = mode_data.iloc[df_xy.index]["k"]
    df = pd.concat([df_xy, df_z])
    df =  compute_coupling_strengths(df)


    new_column_order = [
        "Schubert k",
        "eta",
        "omega_sigma_root1",
        "omega_eps_0_root2",
        "phonon_freq",
        "axis",
    ]
    df = df[new_column_order]

    return df


def compute_coupling_strengths(df):

    df["eta"] = get_normalized_coupling_strength(
        omega_ph=df["omega_sigma_root1"], 
        omega_zero=df["omega_eps_0_root2"]
    )

    return df


def get_normalized_coupling_strength(omega_ph, omega_zero):
    return np.sqrt(omega_zero**2 - omega_ph**2) / omega_ph


def compute_dft_normalised_coupling_strengths(
    eps_infty, phonon_freq, S, volume, scattering_gamma, broadening_type, format_for_print=True
):
    calc_const_broadening = get_broadening(
        gamma_frequencies=phonon_freq, 
        gamma=scattering_gamma, 
        broadening_type=broadening_type)

    roots1, roots2 = get_lorentz_roots(
        eps_infty=eps_infty,
        phonon_freq=phonon_freq,
        S=S,
        V=volume,
        scattering_gamma=calc_const_broadening,
    )

    S_magnitudes = np.linalg.norm(S, axis=1)

    df_dict = {
        "omega_sigma_root1": roots1,
        "omega_eps_0_root2": roots2,
        "phonon_freq": phonon_freq,
        "S_mag": S_magnitudes,
        "const_gamma": calc_const_broadening,
    }

    if broadening_type == "proportional":
        df_dict["prop_gamma"] = [scattering_gamma] * len(phonon_freq)

    df = pd.DataFrame(df_dict)
    df = compute_coupling_strengths(df)


    df["direction"] = get_direction_of_response(S)
    df["omega_sigma_root1"] = df["omega_sigma_root1"] * util.THz_to_inv_cm
    df["omega_eps_0_root2"] = df["omega_eps_0_root2"] * util.THz_to_inv_cm
    df["phonon_freq"] = df["phonon_freq"] * util.THz_to_inv_cm

    ref_new_column_order = [
        "direction",
        "eta",
        "omega_sigma_root1",
        "omega_eps_0_root2",
        "phonon_freq",
        "S_mag",
        "const_gamma",
        "prop_gamma",
    ]
    new_column_order = [col for col in ref_new_column_order if col in df.columns]

    df = df[new_column_order]
    df = df.sort_values(by="phonon_freq", ignore_index=True)


    if format_for_print:
        df.fillna("", inplace=True)
        df["omega_sigma_root1"] = df["omega_sigma_root1"].map(lambda x: "" if x=="" else f"{x:.1f}" )
        df["omega_eps_0_root2"] = df["omega_eps_0_root2"].map(lambda x: "" if x=="" else f"{x:.1f}")
        df["phonon_freq"] = df["phonon_freq"].map(lambda x: "" if x=="" else f"{x:.1f}")
        df["S_mag"] = df["S_mag"].map(lambda x: f"{x:.1e}")
        df["const_gamma"] = df["const_gamma"].map(lambda x: f"{x:.1e}")
        df["prop_gamma"] = df["prop_gamma"].map(lambda x: f"{x:.1e}")
        df["eta"] = df["eta"].map(lambda x: "" if x=="" else f"{x:.2f}" )

        df = df.rename(columns=eta_col_names_for_print)

    return df        


def get_broadening(gamma_frequencies, gamma, broadening_type):

    # broadening prop to frequency
    if broadening_type == "proportional":
        return gamma_frequencies**2 * gamma  # THz
    elif broadening_type == "individual":
        assert gamma.shape == gamma_frequencies.shape
        return gamma
    elif broadening_type == "constant":
        return gamma  # THz
    else:
        raise RuntimeError(
            f"gamma is of type {type(gamma)} neither float nor numpy array."
        )


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
    # assert n_evecs == n_atoms * 3

    masses = np.tile(masses.reshape(1, n_atoms, 1), (n_evecs, 1, 3))

    # shape n_evecs x n_atoms x 3
    eigen_displacements = gamma_evecs / np.sqrt(masses)  # 1/sqrt(kg)
    assert eigen_displacements.shape == (n_evecs, n_atoms, 3)

    # i - number of atoms
    # "jk,k" is the matrix vector multiplication
    S = np.einsum("ijk,lik->lj", born_charges, eigen_displacements)  # C/sqrt(kg)
    assert S.shape == (evec_shape[0], 3)

    return S


def epsilon_for_omega(
    omega, gamma_frequencies, numerator, volume, gamma, broadening_type
):
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
        denominator = (
            gamma_frequencies**2 - omega**2 - gamma_frequencies**2 * omega * gamma * 1j
        )  # THz^2
    elif broadening_type == "individual":
        assert gamma.shape == gamma_frequencies.shape
        real_part = gamma_frequencies**2 - omega**2
        imag_part = 1j * omega * gamma
        denominator = real_part - imag_part
    elif broadening_type == "constant":
        denominator = gamma_frequencies**2 - omega**2 - omega * gamma * 1j  # THz^2
    else:
        raise RuntimeError(
            f"gamma is of type {type(gamma)} neither float nor numpy array."
        )

    denominator *= (2 * np.pi) ** 2

    # cast into correct shape to later go with numerator
    denominator = np.tile(
        denominator.reshape(len(gamma_frequencies), 1, 1), (1, 3, 3)
    )  # THz^2

    # axis=0 to sum over the [12] modes
    #                                       C^2/kg     THz^2                   m^3      THz^2 -> Hz^2
    epsilon_contribution = (
        np.sum(np.divide(numerator, denominator), axis=0) / volume * 1e-24
    )
    epsilon_contribution = epsilon_contribution / epsilon_0  # relative epsilon
    
    return epsilon_contribution


def get_numerator(S):
    # outer products of S for each mode
    numerator = np.einsum("ai,aj->aij", S, np.conjugate(S))  # C^2/kg
    return numerator


def get_direction_of_response(S, threshold=1e-7):

    conv = "xyz"

    S = S.copy()
    assert np.all(S.imag==0)
    S = S.real

    # check where we expect to see non-zero
    S[np.abs(S) < threshold] = 0
    nz = np.nonzero(S)

        # partition in a more useful shape
    dd = {}
    for id1, id2 in zip(nz[0], nz[1]):
        if id1 not in dd:
            dd[id1] = ""
        dd[id1] += conv[id2]

    out = [dd[idx] if idx in dd else np.nan for idx in range(S.shape[0])]

    return  out 


