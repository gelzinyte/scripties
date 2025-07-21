import numpy as np
from copy import deepcopy
from random import randint

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

from ase.visualize.plot import plot_atoms  as ase_plot_atoms
from ase.data.colors import jmol_colors

import util

import util.permittivity


def prepare_axes_with_atoms(omega_range, figsize=(12,12), units="cm$^{-1}$"):

    axes_dd = {}

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    # grid for atoms + permittivity
    #super_out_grid = fig.add_gridspec(1, 2, width_ratios=[1, 2])

    # set up three grids
    #ax_atoms = fig.add_subplot(super_out_grid[0])

    # for epsilon structures
    #outer_grid = super_out_grid[1].subgridspec(2, 3, wspace=0.4, hspace=0.4)
    outer_grid = fig.add_gridspec(3, 3, wspace=0.4, hspace=0.4)

    grid_map = {
        "xx": (0,0),
        "yy": (0,1),
        "zz": (0,2),
        "xy":(1,0),
        "yz":(1,1),
        "zx":(1,2)}

    for title, grid_pos in grid_map.items():

        grid_id_1 = grid_pos[0]
        grid_id_2 = grid_pos[1]

        inner_grid = outer_grid[grid_id_1, grid_id_2].subgridspec(2, 1, wspace=0, hspace=0)
        axs = inner_grid.subplots(sharex=True)
        ax_r = axs[0]
        ax_i = axs[1]

        axes_dd[title] = {}

        axes_dd[title]["real"] = ax_r
        axes_dd[title]["imag"] = ax_i

        ax_r.hlines(y=0, xmin=omega_range[0], xmax=omega_range[-1], color='lightgray', zorder=0)

        ax_r.set_title(title)
        ax_i.set_xlabel(f"IR mode frequency, {units}")
        ax_r.set_ylabel("Real")
        ax_i.set_ylabel("Imag")
        ax_r.grid(color="lightgray")
        ax_i.grid(color="lightgray")

    ax_atoms = [fig.add_subplot(outer_grid[2,idx]) for idx in range(3)]

    return axes_dd, ax_atoms



def plot_atoms(at, ax_atoms):

    for idx in range(3):
        rr = np.array([randint(0, 90) for _ in range(3)])
        ase_plot_atoms(at, ax_atoms[idx], radii=0.3, rotation=(f'{rr[0]}x,{rr[1]}y,{rr[2]}z'))


    # follow PlottingVariables from ase
    symbols = at.get_chemical_symbols()
    numbers = at.get_atomic_numbers()
    ncolors = len(jmol_colors)
    colors = jmol_colors[numbers.clip(max=ncolors-1)]

    done_symbols = []
    handles = []
    for idx, number in enumerate(numbers):
        symbol = symbols[idx]
        if symbol not in done_symbols:
            patch = mpatches.Patch(color=colors[idx], label=symbol)
            handles.append(patch)
            done_symbols.append(symbol)

    ax_atoms[0].legend(handles=handles)



