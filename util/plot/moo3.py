import numpy as np
from copy import deepcopy
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import util

import util.permittivity


columns = ["main axis", "mode index", "wTO", "wLO", "gamma"]


exp_data = [
    ["x", 1, 506.7, 534.3, 49.1],
    ["x", 2, 821.4, 963.0, 6.0],
    ["x", 3, 998.7, 999.2, 0.35],
    ["y", 1, 544.6, 850.1, 9.5],
    ["z", 1, 956.7, 1006.9, 1.5]]
exp_df = pd.DataFrame(exp_data, columns=columns)


dft_data = [
        ["x", 1, 449, 467, 8.3],
        ["x", 2, 769, 947, 3.7],
        ["x", 3, 1016, 1018, 0.4],
        ["y", 1, 505, 820, 12],
        ["z", 2, 765, 772, 3.7],
        ["z", 1, 976, 1027, 0.4]]

dft_df = pd.DataFrame(dft_data, columns=columns)


def prepare_axes(omega_range,  figsize=(8,8)):

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    outer_grid = fig.add_gridspec(1, 3, wspace=0.4, hspace=0.4)

    axes_dd = {
        "xx":{},
        "yy":{},
        "zz":{},
        "xy":{}
    }

    for id1, xyz1 in enumerate(["x", "y", "z"]):
        for id2, xyz2 in enumerate(["x", "y", "z"]):

            title = xyz1 + xyz2
            if title not in ["xx", "yy",  "zz"]:
                continue

            inner_grid = outer_grid[id1].subgridspec(2, 1, wspace=0, hspace=0)
            axs = inner_grid.subplots(sharex=True)
            ax_r = axs[0]
            ax_i = axs[1]

            axes_dd[title]["real"] = ax_r
            axes_dd[title]["imag"] = ax_i

            ax_r.hlines(y=0, xmin=omega_range[0]*util.THz_to_inv_cm, xmax=omega_range[-1]*util.THz_to_inv_cm, color='lightgray', zorder=0)

            ax_r.set_ylim((-320, 320))
            ax_i.set_ylim((0, 600))

            ax_r.set_title(title)
            ax_i.set_xlabel(r"$\omega$, cm$^{-1}$")
            ax_r.set_ylabel(r"Re($\varepsilon$)")
            ax_i.set_ylabel(r"Im($\varepsilon$)")

    return axes_dd



def paper(axes_dd, omega_range, which="experimental", plot_kwargs=None):
    """
        Omega range in inv cm
    """
    if plot_kwargs is None:
        plot_kwargs = {"color":"k", "ls":"-", "lw":1}


    epsilon_infinity_contribution=True

    if which == "experimental":
        paper_eps =  get_permittivity_exp(omega_range)
        df = exp_df
    elif which == "paper_DFT":
        paper_eps = get_permittivity_dft(omega_range)
        df = dft_df

    for cart in "xyz":

        title = cart + cart 

        ax_r = axes_dd[title]["real"]
        ax_i = axes_dd[title]["imag"]


        ys = paper_eps[cart]


        vline_kwargs = deepcopy(plot_kwargs) 
        _ = vline_kwargs.pop("label", None)

        match = {
            "x": 2,
            "y": 1, 
            "z":1}

        sub_df = df[df["main axis"] == cart] 
        for idx, row in sub_df.iterrows():
            if match[cart]!=row["mode index"]:
                continue
            for ax in [ax_r, ax_i]:
                ax.vlines(row["wLO"], 0, 1, transform=ax.get_xaxis_transform(), **vline_kwargs)


        ax_r.plot(omega_range, ys.real, **plot_kwargs)
        ax_i.plot(omega_range, ys.imag, **plot_kwargs)


        #tt = ax_r.text(omega_range[0], ys[0].real+20, f'{ys[0].real:.1f}',)
        #plt.suptitle("Paper $\\varepsilon_r$, no FCC", y=0.93)


def permitivity_for_omega(omega, eps_inf, df_relevant_modes):


    perm = eps_inf 

    for idx, row in df_relevant_modes.iterrows():
        
        conserved_bit = -1 * omega**2 - 1j * row["gamma"] * omega

        numerator = row["wLO"] ** 2 + conserved_bit
        denominator = row["wTO"] ** 2 + conserved_bit

        perm *= numerator / denominator

    return perm

def get_permittivity_exp(omega_range):
    
    all_eps_inf = {
        "x": 5.78,
        "y": 6.07,
        "z": 4.47}

    df = exp_df
    
    all_values = {}

    for cart in "xyz":

        df_subsection = df[df["main axis"] == cart] 
        eps_inf = all_eps_inf[cart]

        perm_values = np.array([permitivity_for_omega(omega, eps_inf, df_subsection) for omega in omega_range])

        all_values[cart] = perm_values

    return all_values


def get_permittivity_dft(omega_range):

    df = dft_df
    
    all_eps_inf = {
        "x": 5.86,
        "y": 6.59,
        "z": 4.47}

    all_values = {}

    for cart in "xyz":

        df_subsection = df[df["main axis"] == cart] 
        eps_inf = all_eps_inf[cart]

        perm_values = np.array([permitivity_for_omega(omega, eps_inf, df_subsection) for omega in omega_range])

        all_values[cart] = perm_values

    return all_values

        




    




