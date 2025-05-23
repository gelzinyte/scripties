import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.spatial.transform import Rotation as R


polariton_type_colors = {
    "dielectric": "white",
    "elliptical": "#bababa",
    "I_out-of-plane": "#f59a97",
    "I_in-plane": "#abdef9",
    "II_out-of-plane": "#fdec7e",
    "II_in-plane": "#bdddb8",
}


polariton_print_labels = {
    "dielectric": "dielectric",
    "elliptical": "ellptical",
    "I_out-of-plane": r"Type I$_\bot$",
    "I_in-plane": r"Type I$_\parallel$",
    "II_out-of-plane": r"Type II$_\bot$",
    "II_in-plane": r"Type II$_\parallel$",
}



rifs = {"r": np.real, "i": np.imag}



def get_gamma_diag_eps_via_evecs(eps):
    evals, evecs = np.linalg.eig(eps.real)
    D = np.eye(len(evals)) * evals
    P = evecs
    A = eps.real
    assert np.allclose(D, np.linalg.inv(P) @ A @ P)
    gamma_from_evec = np.arccos(P[0][0]) * 180 / np.pi
    gamma_from_evec = gamma_from_evec if P[1][0] > 0 else gamma_from_evec * -1


    rotated_eps = np.linalg.inv(P) @ eps @ P

    # z and either x or y axes got swapped 
    if np.isclose(P[2][2], 0, atol=1e-2):
        # either [0][2] (around y axis) or [1][2] (around x axis) element is non zero +-sin(theta) 
        # from the other side - 
        # either [0][2] (around x axis) or [1][2] (around y axis) is zero
        if np.isclose(P[0][2], 0, atol=1e-3):
            assert np.abs(P[0][2]) < np.abs(P[1][2])
            rotate_around = "x"
        elif np.isclose(P[1][2], 0, atol=1e-3):
            assert np.abs(P[1][2]) < np.abs(P[0][2])
            rotate_around = "y"

        rotated_eps = rotate_mx(rotated_eps, 90, rot_axis=rotate_around)



    return gamma_from_evec, rotated_eps, P


def into_regions(section_indices, eps_array):
    eps_ranges = []
    for indices in section_indices:
        eps_ranges.append(eps_array[indices])
    return eps_ranges


def rot_eps_around_z(eps, angle):
    return rotate_mx(eps, angle, rot_axis="z")


def rotate_mx(eps, angle, rot_axis):
    rot = R.from_euler(rot_axis, angle, degrees=True).as_matrix()
    rotated = np.linalg.inv(rot) @ eps @ rot
    return rotated


def get_discontinuities(arr, threshold=30):
    diffs = np.diff(arr)
    disconts = np.where(np.abs(diffs) > threshold)[0]
    return disconts


def make_continuous(arr):

    arr = arr.copy()

    disc_locs = get_discontinuities(arr)

    if len(disc_locs) == 0:
        disc_locs = []

    else:

        disc_loc = disc_locs[0]
        signed_factor = get_shift_sign_multiplicity(arr, disc_locs)
        arr[: disc_loc + 1] += signed_factor * 90

        if len(disc_locs) > 1:
            assert len(disc_locs) < 10
            arr, _ = make_continuous(arr)

    return arr, disc_locs


def shift_closer_to_0(arr, shift_by=90):

    lower = np.min([-shift_by / 2, shift_by / 2])
    upper = np.max([-shift_by / 2, shift_by / 2])

    mult = round(arr[0] / shift_by)
    arrays = [arr, arr + shift_by, arr - shift_by, arr - mult * shift_by]
    scores = np.array([get_penalty(aa, lower=lower, upper=upper) for aa in arrays])
    return arrays[np.argmin(scores)]


def get_penalty(arr, lower=-45, upper=+45):
    return sum(1 for x in arr if x < lower or x > upper)


def get_shift_sign_multiplicity(arr, disc_locs):
    """
    determine whether to shift up or down
    by comparing values in the middle
    pre- and post discontinuity
    """
    pred_id = disc_locs[0]
    postd_id = disc_locs[0] + 1

    multiplicity = round((arr[postd_id] - arr[pred_id]) / 90)

    return multiplicity


def smooth_gammas(gammas_ranges, omega_ranges):

    shifted_gammas_out = []
    jump_regions = []

    for idx, (gamma_region, omega_region) in enumerate(
        zip(gammas_ranges, omega_ranges)
    ):
        # print(idx, "-" * 30)
        adjusted_gammas, jump_locs = make_continuous(gamma_region)
        jump_regions += [omega_region[x] for x in jump_locs if x is not None]

        adjusted_gammas = shift_closer_to_0(adjusted_gammas)
        shifted_gammas_out.append(adjusted_gammas)

    shifted_gammas_out = smooth_90_jumps(shifted_gammas_out)

    return shifted_gammas_out, jump_regions


def get_gamma_regions(gammas, phonon_freqs, omega_range_inv_cm, gamma_no_plot_tol):
    """ partition gammas at TO freqs and cut out `gamma_no_plot_tol` on either side"""

    phonon_freqs = np.array([freq for freq in phonon_freqs  if freq > omega_range_inv_cm[0] and freq < omega_range_inv_cm[-1]])
    phonon_freqs = np.sort(phonon_freqs)


    split_freqs = [
        x for ff in phonon_freqs for x in (ff - gamma_no_plot_tol, ff + gamma_no_plot_tol)
    ]
    split_indices = [(np.abs(omega_range_inv_cm - ff)).argmin() for ff in split_freqs]
    omega_range_indices = np.arange(len(omega_range_inv_cm))
    indices_for_gamma_sections = np.split(omega_range_indices, split_indices)

    gammas_out = []
    omegas_out = []
    indices_for_sections = []

    for idx, indices_to_plot in enumerate(indices_for_gamma_sections):
        if idx % 2 == 1:
            continue

        if len(indices_to_plot) == 0:
            continue

        gammas_out.append(gammas[indices_to_plot])
        omegas_out.append(omega_range_inv_cm[indices_to_plot])
        indices_for_sections.append(indices_to_plot)

    return omegas_out, gammas_out, indices_for_sections


def plot_gammas(ax, omegas_ranges, gammas_ranges, plot_kwargs):

    for idx, (omegas, gammas) in enumerate(zip(omegas_ranges, gammas_ranges)):

        label = plot_kwargs.pop("label", None) if idx == 0 else None
        ax.plot(
            omegas,
            gammas,
            label=label,
            **plot_kwargs,
        )


def get_eps_type(eps):

    if len(eps.shape) == 0:
        return "na"

    xx = -1 if eps[0][0].real < 0 else 1
    yy = -1 if eps[1][1].real < 0 else 1
    zz = -1 if eps[2][2].real < 0 else 1

    if xx > 0 and yy > 0 and zz > 0:
        return "dielectric"

    elif xx < 0 and yy < 0 and zz < 0:
        return "elliptical"

    if xx * yy * zz == -1:
        polariton_type = "I"
    else:
        polariton_type = "II"

    if zz < 0:
        if polariton_type == "I":
            direction = r"out-of-plane"
        elif polariton_type == "II":
            direction = r"in-plane"
    elif zz > 0:
        if polariton_type == "I":
            direction = r"in-plane"
        elif polariton_type == "II":
            direction = r"out-of-plane"

    return polariton_type + "_" + direction


def get_label_regions(arr):

    regions = {}
    for label in set(arr):
        regions[label] = []
    current_label = arr[0]
    start_idx = 0

    for idx, label in enumerate(arr):
        if label != current_label:

            regions[current_label].append((start_idx, idx - 1))
            current_label = label
            start_idx = idx
    regions[current_label].append((start_idx, idx - 1))
    return regions


def post_gamma_axs(ax, omega_range_inv_cm, TO_freqs):

    # ax.set_ylim(-100, 100)
    TO_freqs = [freq for freq in TO_freqs if freq > omega_range_inv_cm[0]]
    rmin, rmax = ax.get_ylim()
    ax.vlines(TO_freqs, rmin, rmax, lw=0.5, color="k", label=r"$\omega_{TO}$")

    ax.hlines(
        y=0,
        xmin=omega_range_inv_cm[0],
        xmax=omega_range_inv_cm[-1],
        color="k",
        lw=0.5,
        zorder=0,
    )

    ax.set_xlim(omega_range_inv_cm[0], omega_range_inv_cm[-1])
    ax.minorticks_on()
    ax.set_ylabel(r"$\gamma(\omega)$")


def setup_axes_compare(axs_specs, labels):

    side = 2.5
    col_width = side * 3
    row_height = side

    num_cols = len(labels)
    num_rows = 1 + len(axs_specs)
    width = col_width * num_cols
    height = row_height * num_rows

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(ncols=num_cols, nrows=num_rows, wspace=0.15, hspace=0)

    all_eps_axes = {}
    all_gamma_axes = {}

    for label_idx, label in enumerate(labels):

        if label_idx == 0:
            sharey = None
        else:
            sharey = all_gamma_axes[labels[0]]

        ax_gamma = fig.add_subplot(gs[0, label_idx], sharey=sharey)
        all_gamma_axes[label] = ax_gamma
        all_eps_axes[label] = {}

        for idx, axsp in enumerate(axs_specs):

            ax_label = "".join(str(sp) for sp in axsp)

            if label_idx == 0:
                sharey = None
            else:
                sharey = all_eps_axes[labels[0]][ax_label]

            ax = fig.add_subplot(gs[idx + 1, label_idx], sharex=ax_gamma, sharey=sharey)
            all_eps_axes[label][ax_label] = ax

    return all_eps_axes, all_gamma_axes


def setup_axes_analyse(axs_specs, make_gamma_ax=True, make_atoms_ax=True, side=2.5, width_ratio=3):

    col_width = side * width_ratio
    row_height = side

    num_cols = 1
    num_rows =len(axs_specs)
    if make_atoms_ax:
        num_rows += 1
    if make_gamma_ax:
        num_rows += 1
    width = col_width * num_cols
    height = row_height * num_rows

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(ncols=num_cols, nrows=num_rows, wspace=0, hspace=0)

    all_axes = {}
    start_ax_idx = 0
    
    if make_gamma_ax:
        ax_gamma = fig.add_subplot(gs[1, 0])
        start_ax_idx += 1
    else:
        ax_gamma = None

    sharex=ax_gamma
    
    if make_atoms_ax:
        at_grid = gs[0, 0].subgridspec(1, 3, wspace=0, hspace=0)
        axs_at = at_grid.subplots()
        start_ax_idx += 1
    else:
        axs_at = None

    ref_share_y = None

    for idx, axsp in enumerate(axs_specs):


        if axsp[2] == "i":
            sharey = None
        elif ref_share_y is None:
            sharey = None

        ax_label = "".join(str(sp) for sp in axsp)

        ax = fig.add_subplot(gs[idx + start_ax_idx, 0], sharex=sharex, sharey=sharey)

        if sharex is None:
            sharex = ax

        all_axes[ax_label] = ax

        if axsp[2] == "r" and ref_share_y is None:
            ref_share_y = ax

        # Remove the spines
    if make_atoms_ax:
        for ax in axs_at:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            # Remove the ticks
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])


    return ax_gamma, all_axes, axs_at 



def post_eps_ax(ax, sp, min_max_y=None, axmin=None, axmax=None):

    conv = "xyz"
    t = conv[sp[0]] + conv[sp[1]]

    if sp[2] == "r":
        if t == "xx":
            ax.set_ylabel(r"$\mathrm{Re}(\varepsilon_{xx})$")
        elif t == "yy":
            ax.set_ylabel(r"$\mathrm{Re}(\varepsilon_{yy})$")
        elif t == "zz":
            ax.set_ylabel(r"$\mathrm{Re}(\varepsilon_{zz})$")
        elif t == "xy" or t== "yx":
            ax.set_ylabel(r"$\mathrm{Re}(\varepsilon_{xy})$")
        elif t == "yz" or t== "zy":
            ax.set_ylabel(r"$\mathrm{Re}(\varepsilon_{yz})$")
        elif t == "xz" or t== "zx":
            ax.set_ylabel(r"$\mathrm{Re}(\varepsilon_{xz})$")


    if sp[2] == "i":
        ax.set_ylabel(r"$\mathcal{I}(\varepsilon_{t})$")

    if min_max_y is not None:
        axmin = min_max_y[t][sp[2]]["min"]
        axmax = min_max_y[t][sp[2]]["max"]

    if axmin is not None and axmax is not None:
        ax.set_xlim(axmin, axmax)

        ax.hlines(
            y=0,
            xmin=axmin,
            xmax=axmax,
            color="k",
            lw=1,
            zorder=0,
        )
    ax.minorticks_on()


def smooth_90_jumps(gamma_ranges):

    for idx in range(len(gamma_ranges) - 1):

        jjump = gamma_ranges[idx + 1][0] - gamma_ranges[idx][-1]

        if np.isclose(np.abs(jjump), 90, atol=40):
            sign = -1 if jjump > 0 else 1
            gamma_ranges[idx + 1] += sign * 90

    gamma_ranges = [shift_closer_to_0(gamma, shift_by=180) for gamma in gamma_ranges]

    return gamma_ranges


def smooth_epsilons(diag_raw_epsilons, orig_gamma_regions, tidy_gamma_regions):

    smooth_epsilons_out = []
    for raw_eps_reg, orig_gamma_reg, tidy_gamma_reg in zip(
        diag_raw_epsilons, orig_gamma_regions, tidy_gamma_regions
    ):

        changes = tidy_gamma_reg - orig_gamma_reg

        assert len(changes) == len(raw_eps_reg)
        rotated_eps = np.array(
            [
                rotate_mx(eps, angle, rot_axis="z")
                for eps, angle in zip(raw_eps_reg, changes)
            ]
        )

        smooth_epsilons_out.append(rotated_eps)

    return smooth_epsilons_out


def color_polariton_regions(
    axs_dict, tidy_eps_ranges, polariton_type_colors, omega_ranges,polariton_print_labels, legend_loc="best", color=True
):

    pol_type_labels = [
        [get_eps_type(eps) for eps in eps_range] for eps_range in tidy_eps_ranges
    ]

    pol_region_bounds = [get_label_regions(pol_types) for pol_types in pol_type_labels]

    if color:
        first_ax_label = list(axs_dict.keys())[0]
        for idx, (ax_label, ax) in enumerate(axs_dict.items()):
            if ax_label[2] == "i" and first_ax_label is None: 
                first_ax_label = ax_label
            for pol_type_dd, omega_range in zip(pol_region_bounds, omega_ranges):
                for pol_type_label, stretch_bounds in pol_type_dd.items():
                    for bp in stretch_bounds:
                        ax.axvspan(
                            omega_range[bp[0]],
                            omega_range[bp[1]],
                            color=polariton_type_colors[pol_type_label],
                            zorder=-1,
                        )

        if legend_loc is not None:
            handles = [mpatches.Patch(color=color, label=polariton_print_labels[label]) for label, color in polariton_type_colors.items()]
            axs_dict[first_ax_label].legend(handles=handles,loc=legend_loc) 

    return get_polariton_region_widths(omega_ranges, pol_region_bounds)

def get_polariton_region_widths(omega_ranges, pol_region_bounds):

    scores_dict = {}

    for region_bounds, omega_range in zip(pol_region_bounds, omega_ranges):
        for label, bounds in region_bounds.items():

            if label not in scores_dict:
                scores_dict[label] = 0

            scores_dict[label] += np.sum([omega_range[bb[1]] - omega_range[bb[0]] for bb in bounds])

    total_pol_region = np.sum([val for key, val in scores_dict.items() if key!="dielectric"])
    scores_dict["total_accounted_for"] =  np.sum([val for key, val in scores_dict.items()])
    scores_dict["total_polariton"] = total_pol_region

    return scores_dict



def plot_epsilons(axs_specs, axs_eps, eps_for_omega_tidy, omega_ranges, phonon_freqs, plot_kwargs=None, vlines=True):

    min_oo = np.min([np.min(oo) for oo in omega_ranges])
    max_oo = np.max([np.max(oo) for oo in omega_ranges])
    xticks = np.arange(round(min_oo, -2)+100, max_oo, 100)

    if plot_kwargs is None:
        plot_kwargs = {"color":"k", "ls":"-"}
    if "label" not in plot_kwargs:
        plot_kwargs["label"] = None

    ccols = {
        0: "tab:red",
        1: "tab:blue",
        2: "tab:green",
    }


    for ax_idx, (sp, ax) in enumerate(zip(axs_specs, axs_eps.values())):


        if "color" in plot_kwargs and plot_kwargs["color"] == "custom":
            plot_kwargs["color"] = ccols[ax_idx]

        conv = "xyz"
        t = conv[sp[0]] + conv[sp[1]]

        for idx, (eps_range, omega_range) in enumerate(
            zip(eps_for_omega_tidy, omega_ranges)
        ):
            ys = rifs[sp[2]](eps_range[:, sp[0], sp[1]])
            if sp[0] != sp[1] and sp[2] == "i":
                ys = np.abs(ys)
            ax.plot(omega_range, ys, **plot_kwargs)

        min_max_y = {t:{sp[2]:{"min":min_oo, "max":max_oo}}}
        post_eps_ax(ax, sp, min_max_y=min_max_y)

        rmin, rmax = ax.get_ylim()
        if vlines:
            ax.vlines(phonon_freqs, rmin, rmax, lw=0.5, color="k", label=r"$\omega_{TO}$")
        ax.set_ylim(rmin, rmax)

        if ax_idx == len(axs_specs) - 1:
            xlabel = ax.set_xlabel(r"$\omega$, cm$^{-1}$")
            xlabel.set_zorder(2)

            ax.tick_params(axis="both", which="both", zorder=2)
            for spine in ax.spines.values():
                spine.set_zorder(2)
        
            ax.set_xticks(xticks)

