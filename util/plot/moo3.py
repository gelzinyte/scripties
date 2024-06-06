import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import util

import util.permittivity



def prepare_axes(omega_range, mmy=min_max_y, figsize=(8,8)):

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

            inner_grid = outer_grid[grid_id_1, grid_id_2].subgridspec(2, 1, wspace=0, hspace=0)
            axs = inner_grid.subplots(sharex=True)
            ax_r = axs[0]
            ax_i = axs[1]

            axes_dd[title]["real"] = ax_r
            axes_dd[title]["imag"] = ax_i

            ax_r.hlines(y=0, xmin=omega_range[0]*util.THz_to_inv_cm, xmax=omega_range[-1]*util.THz_to_inv_cm, color='lightgray', zorder=0)

            ax_r.set_ylim((-320, 320))
            ax_i.set_ylim((0, 600))

            ax_r.set_title(title)
            ax_i.set_xlabel("IR mode frequency, cm$^{-1}$")
            ax_r.set_ylabel("Real")
            ax_i.set_ylabel("Imag")

    return axes_dd




def schubert(axes_dd, omega_range, epsilon_infinity_contribution=True, plot_kwargs=None):
    """
        Omega range in inv cm
    """
    if plot_kwargs is None:
        plot_kwargs = {"color":"k", "ls":"-", "lw":1}


    epsilon_infinity_contribution=True

    grid_map = {
        "xx": (0, 0),
        "yy": (0, 1),
        "xy": (1, 0),
        "zz": (1, 1)
    }

    schubert_eps = util.permittivity.get_schubert(omega_range, epsilon_inf=epsilon_infinity_contribution)

    # relabel axis
    my_schubert_eps = deepcopy(schubert_eps)
    my_schubert_eps["yx"] = schubert_eps["xy"]

    all_vals = np.concatenate([vals for key, vals in my_schubert_eps.items()])

    conv = "xyz"

    for id1, id2 in [(0,0), (1,1), (2,2), (0,1)]:


        title = conv[id1]+conv[id2]

        ax_r = axes_dd[title]["real"]
        ax_i = axes_dd[title]["imag"]


        ys = my_schubert_eps[title]


        ax_r.plot(omega_range, ys.real, **plot_kwargs)
        ax_i.plot(omega_range, ys.imag, **plot_kwargs)


        tt = ax_r.text(omega_range[0], ys[0].real+20, f'{ys[0].real:.1f}',)

        plt.suptitle("Schubert $\\varepsilon_r$, no FCC", y=0.93)

    return tt


