import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import util

base = Path(util.__file__).parent / "data"

# data
mode_data = pd.read_csv(base / "schubert_paper.table_II.csv")
xyz_data = pd.read_csv(base / "schubert_paper.table_IV.csv", index_col="epsilon")
fcc_volume_density_param_N = 3.5e18 # cm^-3, page 15

mode_data["angle"] = np.radians(mode_data["angle"])

def get_rho(omega, A, f_TO, broadening):
    return A / (f_TO**2 - omega**2 - omega * broadening * 1j)


def get_schubert(omega_range):

    warnings.warn("No FCC contribution to schubert permittivity")

    Bu = mode_data.loc[mode_data["symmetry"] == "Bu"]
    Au = mode_data.loc[mode_data["symmetry"] == "Au"]

    # xx
    eps_xx = xyz_data["high_freq"]["xx"] \
        + np.array([np.sum(get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"]) * np.cos(Bu["angle"])**2) for omega in omega_range])

    eps_yy = xyz_data["high_freq"]["yy"] \
        + np.array([np.sum(get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"]) * np.sin(Bu["angle"])**2) for omega in omega_range])

    eps_xy = xyz_data["high_freq"]["xy"] \
        + np.array([np.sum(get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"]) * np.cos(Bu["angle"]) * np.sin(Bu["angle"])) for omega in omega_range])

    eps_zz = xyz_data["high_freq"]["zz"] \
        + np.array([np.sum(get_rho(omega, Au["A"], Au["freq"], Au["scatter"])) for omega in omega_range])


#     eps_xx = np.array([np.sum(get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"]) * np.cos(Bu["angle"])**2) for omega in omega_range])
# 
#     eps_yy = np.array([np.sum(get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"]) * np.sin(Bu["angle"])**2) for omega in omega_range])
# 
#     eps_xy = np.array([np.sum(get_rho(omega, Bu["A"], Bu["freq"], Bu["scatter"]) * np.cos(Bu["angle"]) * np.sin(Bu["angle"])) for omega in omega_range])
# 
#     eps_zz = np.array([np.sum(get_rho(omega, Au["A"], Au["freq"], Au["scatter"])) for omega in omega_range])
# 


    data = {
        "xx": eps_xx,
        "yy": eps_yy,
        "xy": eps_xy,
        "zz": eps_zz}

    return data

