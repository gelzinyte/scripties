import numpy as np
from copy import deepcopy
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
from adjustText import adjust_text

from scipy.constants import epsilon_0, physical_constants, speed_of_light
from scipy.constants import elementary_charge
from scipy.spatial.transform import Rotation as R

from ase.io import read, write
from ase import Atoms

from vibes.structure.convert import to_Atoms
from vibes.phonopy import postprocess

import util
import util.phonopy
import util.aims
import util.geometry
import util.permittivity
import util.born
import util.material_specific_data
import util.visualise
import util.plot.ga2o3
from util.permittivity import epsilon_for_omega


def get_ga2o3_eps_vs_omega(
    phonon_ats_fn,
    phonopy_yaml,
    born_file=None,
    nac_direction=None,
    broadening=None,
    omega_range=None,
    born_charges=None,
):

    print(f"born file:", born_file)

    if born_file is None:
        assert nac_direction is None
    else:
        assert nac_direction is not None

    phonon = util.phonopy.postprocess_wfl(
        evaled_ats_fn=phonon_ats_fn,
        phonopy_yaml_fn=phonopy_yaml,
        born_charges_file=born_file,
    )

    R = util.material_specific_data.ga2o3_R
    irrep_labels, relevant_mode_idcs = (
        util.material_specific_data.get_ga2o3_irreps_and_relevant_modes(phonon)
    )

    relevant_mode_idcs = np.array([idx for idx, label in enumerate(irrep_labels)])

    phonon.run_qpoints(
        q_points=np.array([[0, 0, 0]]),
        with_eigenvectors=True,
        with_dynamical_matrices=False,
        nac_q_direction=nac_direction,
    )

    qpoint_dict = phonon.get_qpoints_dict()
    evecs = util.material_specific_data.get_conventional_ga2o3_gamma_eigenvectors(
        qpoint_dict
    )
    freqs = qpoint_dict["frequencies"]
    assert freqs.shape[0] == 1
    freqs = freqs[0]

    at_prim = to_Atoms(phonon.primitive)
    at_conv = util.geometry.primitive_to_conventional_at(at_prim)

    masses = (
        at_conv.get_masses() * physical_constants["atomic mass constant"][0]
    )  # units: kg

    S = util.permittivity.get_born_along_displacements(
        gamma_evecs=evecs, masses=masses, born_charges=born_charges
    )
    S = np.einsum("ij,vj->vi", R, S)

    numerator = util.permittivity.get_numerator(S)[relevant_mode_idcs]
    relevant_freqs = freqs[relevant_mode_idcs]

    # 20-atom conventional unit cell
    volume = at_conv.get_volume() * 1e-30  # m^3

    # get omega, diagonalize and label (?) it.
    eps_for_omega = np.array(
        [
            epsilon_for_omega(
                omega,
                relevant_freqs,
                numerator,
                volume,
                broadening,
                broadening_type="proportional",
            )
            for omega in omega_range
        ]
    )  # C^2 s^2 / (kg m^3)

    # add high-freq contribution
    eps_for_omega[:, 0, 0] += high_freq_eps_from_aims["xx"]
    eps_for_omega[:, 1, 1] += high_freq_eps_from_aims["yy"]
    eps_for_omega[:, 2, 2] += high_freq_eps_from_aims["zz"]

    return eps_for_omega, freqs, irrep_labels


def get_gamma_omega_evecs(eps):
    # if np.iscomplexobj(eps):
    #    eps = eps.real
    evals, evecs = np.linalg.eig(eps.real)
    D = np.eye(len(evals)) * evals
    P = evecs
    A = eps.real
    assert np.allclose(D, np.linalg.inv(P) @ A @ P)
    gamma_from_evec = np.arccos(P[0][0]) * 180 / np.pi
    gamma_from_evec = gamma_from_evec if P[1][0] > 0 else 90 - gamma_from_evec

    rotated_eps = np.linalg.inv(P) @ eps @ P

    return gamma_from_evec, rotated_eps, P


def get_gamma_omega_from_2x2_evecs(eps):
    eps_2x2 = eps[0:2, 0:2]
    return get_gamma_omega_evecs(eps_2x2)


def get_gamma_omega_eqn_2(eps):
    inside = 2 * np.real(eps[0][1]) / (np.real(eps[0][0]) - np.real(eps[1][1]))
    in_rads = 0.5 * np.arctan(inside)
    in_degs = in_rads * 180 / np.pi

    rot = R.from_euler("z", in_degs, degrees=True).as_matrix()
    rotated_eps = np.linalg.inv(rot) @ eps @ rot

    return in_degs, rotated_eps


def into_regions(section_indices, eps_array):
    eps_ranges = []
    for indices in section_indices:
        eps_ranges.append(eps_array[indices])
    return eps_ranges


def rotate_by_gamma(sections_indices, gamma_ranges, eps_array):

    eps_ranges = []

    for gamma_range, section_locs in zip(gamma_ranges, sections_indices):

        rot_epss = []
        for gamma, loc in zip(gamma_range, section_locs):
            rotated = rot_eps_around_z(eps_array[loc], gamma)
            rot_epss.append(rotated)
        eps_ranges.append(np.array(rot_epss))
    return eps_ranges


def rot_eps_around_z(eps, angle):
    rot = R.from_euler("z", angle, degrees=True).as_matrix()
    # rotated =  np.linalg.inv(rot) @ eps @ rot
    rotated = np.linalg.inv(rot) @ eps @ rot
    return rotated


def get_high_freq_eps_from_aims():

    R = util.material_specific_data.ga2o3_R
    high_freq_eps = util.aims.read_dielectric(
        "5.dielectric_for_primitive/workdir/aims.out"
    )
    high_freq_eps = R @ high_freq_eps @ R.T
    high_freq_eps_from_aims = {}
    high_freq_eps_from_aims["xx"] = high_freq_eps[0][0]
    high_freq_eps_from_aims["yy"] = high_freq_eps[1][1]
    high_freq_eps_from_aims["zz"] = high_freq_eps[2][2]

    print("high freq eps:\n", high_freq_eps)

    return high_freq_eps_from_aims


def get_born_charges():
    prim_to_conv_perm_mx = util.geometry.array_perm_duplicate_mx()
    born_charges = util.born.get_born_charges(
        homedir="2.born_effective_charges", displacement_magnitude=0.02
    )[0]
    born_charges = np.einsum(
        "ij,jkl->ikl", prim_to_conv_perm_mx, born_charges
    )  # reoder

    return born_charges


def get_schubert_bu_au_freqs():
    # pick out relevant frequencies to plot
    base = Path(util.__file__).parent / "data"
    mode_data = pd.read_csv(base / "schubert_paper.table_II.csv")
    Bu = mode_data.loc[mode_data["symmetry"] == "Bu"]
    Bu_freqs_schubert = np.sort(np.array(Bu["freq"]))
    Au = mode_data.loc[mode_data["symmetry"] == "Au"]
    Au_freqs_schubert = np.sort(np.array(Au["freq"]))

    return Bu_freqs_schubert, Au_freqs_schubert


def get_schubert_eps_in_array(omega_range):
    schubert_eps = util.permittivity.get_schubert(
        omega_range * util.THz_to_inv_cm, epsilon_inf=True
    )

    schubert_eps_arr = np.zeros((len(omega_range), 3, 3), dtype=complex)
    for idx in range(len(omega_range)):
        xx = schubert_eps["xx"][idx]
        xy = schubert_eps["xy"][idx]
        yy = schubert_eps["yy"][idx]
        zz = schubert_eps["zz"][idx]
        schubert_eps_arr[idx][0][0] = xx
        schubert_eps_arr[idx][1][0] = xy
        schubert_eps_arr[idx][0][1] = xy
        schubert_eps_arr[idx][1][1] = yy
        schubert_eps_arr[idx][2][2] = zz
    return schubert_eps_arr


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
            assert len(disc_locs) < 3
            arr, _ = make_continuous(arr)

    return arr, disc_locs


def shift_closer_to_0(arr, shift_by=90):

    lower = np.min([-shift_by/2, shift_by/2])
    upper = np.max([-shift_by/2, shift_by/2])

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
    pred_id = disc_locs[0] - 2
    postd_id = disc_locs[0] + 2

    multiplicity = round((arr[postd_id] - arr[pred_id]) / 90)

    return multiplicity


def smooth_gammas(gammas_ranges, omega_ranges):

    shifted_gammas_out = []
    jump_regions = []

    for idx, (gamma_region, omega_region) in enumerate(
        zip(gammas_ranges, omega_ranges)
    ):
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


def get_DFT_Bu_Au_indices(irrep_labels):
    DFT_Bu_indices = np.array(
        [idx for idx, label in enumerate(irrep_labels) if label in ["Bu"] and idx > 2]
    )

    DFT_Au_indices = np.array(
        [idx for idx, label in enumerate(irrep_labels) if label in ["Au"] and idx > 2]
    )
    return DFT_Bu_indices, DFT_Au_indices


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


def post_gamma_axs(ax, omega_range, TO_freqs):

    ax.set_ylim(-100, 100)
    TO_freqs = [freq for freq in TO_freqs if freq > omega_range_inv_cm[0]]
    ax.vlines(TO_freqs, -90, 90, lw=0.5, color="k", label=r"$\omega_{TO}$")

    for ang in np.arange(-45, 45+1, 45):
        ax.hlines(
            y=ang,
            xmin=omega_range[0] * util.THz_to_inv_cm,
            xmax=omega_range[-1] * util.THz_to_inv_cm,
            color="darkgreen",
            lw=1,
            zorder=0,
        )


def setup_axes(axs_specs):

    side = 2
    col_width = side * 3
    row_height = side
    num_cols = 1
    num_rows = 1 + len(axs_specs)
    width = col_width * num_cols
    height = row_height * num_rows

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(ncols=num_cols, nrows=num_rows, wspace=0, hspace=0)

    all_axes = {}
    ax_gamma = fig.add_subplot(gs[0, 0])
    # all_axes["gamma"] = ax_gamma

    for idx, axsp in enumerate(axs_specs):
        ax = fig.add_subplot(gs[idx + 1, 0], sharex=ax_gamma)
        all_axes["".join(str(sp) for sp in axsp)] = ax

    return all_axes, ax_gamma


def post_eps_ax(ax, sp):

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

        if np.isclose(np.abs(jjump), 90, atol=20):
            sign = -1 if jjump > 0 else 1
            gamma_ranges[idx + 1] += sign * 90

    gamma_ranges = [shift_closer_to_0(gamma, shift_by=180) for gamma in gamma_ranges]

    return gamma_ranges


def smooth_epsilons(diag_raw_epsilons, orig_gamma_regions, tidy_gamma_regions):

    smooth_epsilons_out = []
    for raw_eps_reg, orig_gamma_reg, tidy_gamma_reg in zip(diag_raw_epsilons, orig_gamma_regions, tidy_gamma_regions):

        smooth_eps = []
        changes = orig_gamma_reg - tidy_gamma_reg

        import pdb; pdb.set_trace()



min_max_y = {
    "xx": {
        "r": {"min": -100, "max": 100},
        "i": {"min": 0, "max": 100},
    },
    "xy": {
        "r": {"min": -100, "max": 100},
        "i": {"min": 0, "max": 2},
    },
}
min_max_y["yy"] = min_max_y["xx"]
min_max_y["zz"] = min_max_y["xx"]


# --------------------
# get stuff
# --------------------

high_freq_eps_from_aims = get_high_freq_eps_from_aims()
born_charges = get_born_charges()
Bu_freqs_schubert, Au_freqs_schubert = get_schubert_bu_au_freqs()

# for proportional to frequency
broadening = 0.03 / util.THz_to_inv_cm
# cm-1                  cm^-1 -> THz
# omega_range_inv_cm = np.arange(0, 800, 1)
omega_range_inv_cm = np.arange(180, 800, 0.2)
omega_range = omega_range_inv_cm / util.THz_to_inv_cm  # THz
gamma_no_plot_tol = 1  # cm-1

formula = "Ga2O3"
lw = 1
conv = "xyz"

ats_file = "6.phonopy_with_different_masses/3.O16/4.wfl/supercell3_in.aims.xyz"
phonopy_yaml_fn = "6.phonopy_with_different_masses/3.O16/4.wfl/supercell3_phonon.yml"
born_file = "3.create_BORN_file/BORN"


eps_for_omega, dft_phonon_freqs, irrep_labels = get_ga2o3_eps_vs_omega(
    phonon_ats_fn=ats_file,
    phonopy_yaml=phonopy_yaml_fn,
    broadening=broadening,
    omega_range=omega_range,
    born_charges=born_charges,
)
DFT_Bu_indices, DFT_Au_indices = get_DFT_Bu_Au_indices(irrep_labels)


schubert_eps_array = get_schubert_eps_in_array(omega_range)


# ------------------------
# plot stuff
# ------------------------

polariton_type_colors = {
    "dielectric": "white",
    "elliptical": "#bababa",
    "I_out-of-plane": "#f59a97",
    "I_in-plane": "#abdef9",
    "II_out-of-plane": "#fdec7e",
    "II_in-plane": "#bdddb8",
}


rifs = {"r": np.real, "i": np.imag}

# axs_specs = [(0, 0, "r"), (1, 1, "r"), (0, 1, "i"), (2, 2, "r")]
axs_specs = [(0, 0, "r"), (1, 1, "r")]
eps_axs, ax_gamma = setup_axes(axs_specs)

# -------------------------------------
# plot gammas
# -------------------------------------

# from eqns
gammas = np.array([get_gamma_omega_eqn_2(eps)[0] for eps in schubert_eps_array])
omega_ranges_schs, gamma_ranges_schs, indices_for_sections_schs = get_gamma_regions(
    gammas=gammas,
    TO_freqs=Bu_freqs_schubert,
    omega_range_inv_cm=omega_range_inv_cm,
    gamma_no_plot_tol=gamma_no_plot_tol,
)
plot_gammas(
    ax=ax_gamma,
    omegas_ranges=omega_ranges_schs,
    gammas_ranges=gamma_ranges_schs,
    plot_kwargs={
        "label": "Eqn2",
        "color": "tab:blue",
        "ls": ":",
    },
)

shifted_gammas_eqn2, jump_locs_eqn2 = smooth_gammas(
    gamma_ranges_schs, omega_ranges_schs
)
plot_gammas(
    ax=ax_gamma,
    omegas_ranges=omega_ranges_schs,
    gammas_ranges=shifted_gammas_eqn2,
    plot_kwargs={
        "label": "Eqn2_tidy",
        "color": "k",
        "ls": "-",
    },
)


# from _evecs
gammas = np.array([get_gamma_omega_evecs(eps)[0] for eps in schubert_eps_array])
omega_ranges_schs_evecs, gamma_ranges_schs_evecs, indices_for_sections_schs_evecs = (
    get_gamma_regions(
        gammas=gammas,
        TO_freqs=Bu_freqs_schubert,
        omega_range_inv_cm=omega_range_inv_cm,
        gamma_no_plot_tol=gamma_no_plot_tol,
    )
)

plot_gammas(
    ax=ax_gamma,
    omegas_ranges=omega_ranges_schs_evecs,
    gammas_ranges=gamma_ranges_schs_evecs,
    plot_kwargs={
        "label": "from_evecs",
        "color": "tab:orange",
        "ls": ":",
        "zorder": 5,
    },
)
shifted_gammas_evec, jump_locs_evec = smooth_gammas(
    gamma_ranges_schs_evecs, omega_ranges_schs_evecs
)
plot_gammas(
    ax=ax_gamma,
    omegas_ranges=omega_ranges_schs_evecs,
    gammas_ranges=shifted_gammas_evec,
    plot_kwargs={
        "label": "from_evecs_tidy",
        "color": "tab:red",
        "ls": "--",
    },
)

post_gamma_axs(ax_gamma, omega_range, Bu_freqs_schubert)

# ------------------------------------------------
# plot diagonalized epsilons
# ---------------------------------------------


# omega_ranges_schs, gamma_ranges_schs, indices_for_sections_schs = get_gamma_regions(
eps_for_omega_rotated_sch_tidy = rotate_by_gamma(
    sections_indices=indices_for_sections_schs,
    gamma_ranges=shifted_gammas_eqn2,
    eps_array=schubert_eps_array,
)

# omega_ranges_schs_evecs, gamma_ranges_schs_evecs, indices_for_sections_schs_evecs = get_gamma_regions(
eps_for_omega_rotated_sch_evecs = np.array(
    [get_gamma_omega_evecs(eps)[1] for eps in schubert_eps_array]
)
eps_for_omega_rotated_sch_evecs = into_regions(
    indices_for_sections_schs_evecs, eps_for_omega_rotated_sch_evecs
)

eps_for_omega_rotated_evecs_tidy = smooth_epsilons(
    diag_raw_epsilons = eps_for_omega_rotated_sch_evecs,
    orig_gamma_regions = gamma_ranges_schs_evecs, 
    tidy_gamma_regions = shifted_gammas_evec,
)

 
 
 
for sp, ax in zip(axs_specs, eps_axs.values()):

    # ------------------------------------
    # schubert - rotated as in paper
    # ------------------------------------

    for idx, (eps_range, omega_range_sch) in enumerate(
        zip(eps_for_omega_rotated_sch_tidy, omega_ranges_schs)
    ):
        label = "rotated_eqn_2_tidy" if idx == 0 else None

        ys = rifs[sp[2]](eps_range[:, sp[0], sp[1]])
        if sp == (0, 1, "i"):
            ys = np.abs(ys)
        ax.plot(omega_range_sch, ys, color="k", label=label)

    # ------------------------------------
    # schubert - rotated via evecs
    # ------------------------------------

    for idx, (eps_range, omega_range) in enumerate(
        zip(eps_for_omega_rotated_sch_evecs, omega_ranges_schs_evecs)
    ):
        label = "rotated_evecs_naive" if idx == 0 else None
        ys = rifs[sp[2]](eps_range[:, sp[0], sp[1]])
        if sp == (0, 1, "i"):
            ys = np.abs(ys)
        ax.plot(omega_range, ys, color="tab:orange", label=label, ls=":")

    rmin, rmax = ax.get_ylim()
    ax.vlines(Bu_freqs_schubert, rmin, rmax, lw=0.5, color="k", label=r"$\omega_{TO}$")

    post_eps_ax(ax, sp)


colour_bands = False
if colour_bands:

    # label the regions
    pol_type_labels = [
        [get_eps_type(eps) for eps in eps_range]
        for eps_range in eps_for_omega_rotated_sch_tidy
    ]

    pol_region_bounds = [get_label_regions(pol_types) for pol_types in pol_type_labels]

    for ax_label, ax in all_axes.items():

        if ax_label == "gamma":
            continue

        for pol_type_dd, omega_range in zip(pol_region_bounds, omega_ranges_schs):

            for pol_type_label, stretch_bounds in pol_type_dd.items():

                for bp in stretch_bounds:

                    ax.axvspan(
                        omega_range[bp[0]],
                        omega_range[bp[1]],
                        color=polariton_type_colors[pol_type_label],
                        zorder=-1,
                    )


ax_gamma.set_xlim(omega_range_inv_cm[0], omega_range_inv_cm[-1])
ax_gamma.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# all_axes["zzr"].set_xlabel("frequency, cm-1")
# all_axes["xxr"].legend()
ax_gamma.set_ylabel("gamma(omega), degrees")
# ax_gamma.set_ylim((-90, 90))
ax_gamma.minorticks_on()
ax_gamma.set_title("Angle from components")
plt.tight_layout()
plt.savefig(
    f"/raven/u/egg/mounted/diagonalized_eps/ga2o3_gammas.png",
    dpi=300,
    bbox_inches="tight",
)

"""
    ys_r = eps_for_omega_rotated_dft[..., ec[0], ec[1]].real
    ys_i = eps_for_omega_rotated_dft[..., ec[0], ec[1]].imag

    # ax_r.plot(omega_range_inv_cm, ys_r, lw=lw, color="tab:blue")
    # ax_i.plot(
    #    omega_range_inv_cm,
    #    ys_i,
    #    lw=lw,
    #    label="DFT",
    #    color="tab:blue",
    # )
    TO_freqs = dft_phonon_freqs[DFT_Bu_indices] * util.THz_to_inv_cm
    TO_freqs = [f for f in TO_freqs if f > omega_range_inv_cm[0]]


"""
