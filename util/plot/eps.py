import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


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


def get_gamma_regions(gammas, TO_freqs, omega_range_inv_cm, gamma_no_plot_tol):

    split_freqs = [
        x for ff in TO_freqs for x in (ff - gamma_no_plot_tol, ff + gamma_no_plot_tol)
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
            direction = "out-of-plane"
        elif polariton_type == "II":
            direction = "in-plane"
    elif zz > 0:
        if polariton_type == "I":
            direction = "in-plane"
        elif polariton_type == "II":
            direction = "out-of-plane"

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


def setup_axes(axs_specs, labels):

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


def post_eps_ax(ax, sp, min_max_y):

    conv = "xyz"
    t = conv[sp[0]] + conv[sp[1]]

    if sp[2] == "r":
        ax.set_ylabel(f"{t} Real")
    if sp[2] == "i":
        ax.set_ylabel(f"{t} Imag")

    axmin = min_max_y[t][sp[2]]["min"]
    axmax = min_max_y[t][sp[2]]["max"]

    ax.set_ylim(axmin, axmax)

    xmin, xmax = ax.get_xlim()
    ax.hlines(
        y=0,
        xmin=xmin,
        xmax=xmax,
        color="k",
        lw=1,
        zorder=0,
    )


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
    axs_dict, tidy_eps_ranges, polariton_type_colors, omega_ranges
):

    pol_type_labels = [
        [get_eps_type(eps) for eps in eps_range] for eps_range in tidy_eps_ranges
    ]

    pol_region_bounds = [get_label_regions(pol_types) for pol_types in pol_type_labels]

    for ax in axs_dict.values():
        for pol_type_dd, omega_range in zip(pol_region_bounds, omega_ranges):
            for pol_type_label, stretch_bounds in pol_type_dd.items():
                for bp in stretch_bounds:
                    ax.axvspan(
                        omega_range[bp[0]],
                        omega_range[bp[1]],
                        color=polariton_type_colors[pol_type_label],
                        zorder=-1,
                    )
