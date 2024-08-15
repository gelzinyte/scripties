import numpy as np
from pytest import approx

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib import gridspec

from ml_spectroscopy import calculators as spec

spins = {
    "Li": 3 / 2,
    "O": 5 / 2,
    "Ti": 5 / 2,
}

x = 0
y = 1
z = 2


def extract_efgs(ats, efg_label):

    efgs_by_element = {}

    for at in ats:

        elements = at.symbols
        efgs = at.arrays[efg_label].reshape((-1, 3, 3))

        for element, efg in zip(elements, efgs):

            if element not in efgs_by_element:
                efgs_by_element[element] = []

            assert approx(np.trace(efg), abs=1e-7) == 0

            efgs_by_element[element].append(efg)

    for key, vals in efgs_by_element.items():
        efgs_by_element[key] = np.array(vals)

    return efgs_by_element


def get_props(efgs, element):

    data = {
        "Cqs": [],
        "etas": [],
        "omegas_q": [],
    }

    evals = np.array([spec._get_haeberlen_eigs(efg) for efg in efgs])

    Vzzs = evals[:, z]
    Vzz_mean = np.mean(Vzzs)
    Vzz_std = np.std(Vzzs)

    tilde_Cqs = (Vzzs - Vzz_mean) / Vzz_std
    tilde_etas = (evals[:, x] - evals[:, y]) / Vzz_mean

    data["tilde_Cqs"] = tilde_Cqs
    data["tilde_etas"] = tilde_etas

    # get angles
    evecs = np.array([spec._get_haeberlen_eig_vecs(efg) for efg in efgs])
    # take zz evector and its z component
    thetas = np.array([np.arccos(evec[z][z]) for evec in evecs])
    phis = np.array(
        [
            np.arctan(evec[z][y] / evec[z][x]) if np.abs(evec[z][x]) > 0 else 0
            for evec in evecs
        ]
    )

    data["thetas"] = thetas
    data["phis"] = phis

    quaternions = np.array([spec.calc_quaternion(evecs) for evecs in evecs])
    data["quaternions"] = quaternions

    for efg in efgs:

        Cq, eta = spec.getcq(efg.flatten(), species=element)

        data["Cqs"].append(Cq)
        data["etas"].append(eta)

    data["Cqs"] = np.array(data["Cqs"])
    data["etas"] = np.array(data["etas"])

    omegas_q = np.array(
        [
            get_omega_q(Cq=Cq, eta=eta, spin=spins[element], theta=theta, phi=phi)
            for Cq, eta, theta, phi in zip(
                data["Cqs"], data["etas"], data["thetas"], data["phis"]
            )
        ]
    )

    data["omegas_q"] = omegas_q * 1e3  # kHz

    exp_length = len(efgs)
    for key, vals in data.items():
        assert len(vals) == exp_length

    return data


def get_omega_q(Cq, eta, spin, theta, phi):

    eterm2 = 3 * np.cos(theta) ** 2 - 1
    eterm3 = -1 * (np.sin(theta) ** 2 * np.cos(2 * phi))
    spin_bit = 3 / (2 * spin * (2 * spin - 1))

    omega_q = np.abs(spin_bit * 0.5 * Cq * (eterm2 + eta * eterm3))

    return omega_q


def prep_axis(ref_vals, pred_vals, figsize=(6, 9)):

    plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0, wspace=0.1)

    ax_hist = plt.subplot(gs[0])
    ax_par = plt.subplot(gs[1])

    rmse = get_rmse(ref_vals, pred_vals)
    mae = get_mae(ref_vals, pred_vals)
    pearr, _ = pearsonr(ref_vals, pred_vals)
    mpe = get_mpe(ref_vals, pred_vals)
    parity_label = (
        f"RMSE: {rmse:.2f}; MAE: {mae:.2f}; \nR: {pearr:.2f}; error: {mpe:.1f}%"
    )

    ax_par.scatter(ref_vals, pred_vals, label=parity_label, s=2, alpha=1)

    ax_hist.hist(pred_vals, bins=20, label="MACE")
    ax_hist.hist(ref_vals, histtype="step", bins=20, label="DFT", color="k")

    min_val = np.min(np.concatenate([ref_vals, pred_vals]))
    max_val = np.max(np.concatenate([ref_vals, pred_vals]))

    ax_par.plot([min_val, max_val], [min_val, max_val], color="k", ls="--")

    for ax in [ax_par, ax_hist]:
        ax.grid(color="lightgrey", zorder=0)
        ax.legend()

    return ax_hist, ax_par


def get_rmse(ref, pred):
    return np.sqrt(np.mean((ref - pred) ** 2))


def get_mae(ref, pred):
    return np.mean(np.abs(ref - pred))


def get_mpe(ref, pred):
    """mean percentage error"""

    return np.mean(100 * np.abs(ref - pred) / ref)
