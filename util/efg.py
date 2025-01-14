import numpy as np
import warnings
from pytest import approx

from scipy.stats import pearsonr
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from matplotlib import gridspec

from ml_spectroscopy import calculators as spec
from ml_spectroscopy.calculators import _get_quaternion_from_matrix as get_quat 


spins = {
    "Li": 3 / 2,
    "O": 5 / 2,
    "Ti": 5 / 2,
}

x = 0
y = 1
z = 2


def tidy_sign_convention(t):
    """ t - array with eigenvectors as columns """
    for idx in range(3):
        num_pve = np.sum([1 for x in t[:, idx] if x > 0])

        if num_pve < 2:
            t[:, idx] *= -1

    return t


def get_haeberlen_ordered_eigh_evecs_evals(t: np.ndarray):
    """Return Haeberlen convention eigenvectors (TAKEN FROM MATADOR-DB)
    EG: the same as Ange's code, but uses np.linalg.eigh instead of np.linalg.eig
    for eigenvector sign/direction stability.  
    Arguments:
        t: numpy array containing a NxN square tensor.
    Returns:
        np.ndarray: eigenvectors of input matrix
    """
    eig_vals, eig_vecs = np.linalg.eigh(t)
    eig_vals, eig_vecs = zip(
        *sorted(zip(eig_vals, eig_vecs), key=lambda eig: abs(eig[0] - np.trace(t) / 3))
    )

    eig_vecs = tidy_sign_convention(np.array(eig_vecs))

    return eig_vals, eig_vecs


def get_haeberlen_eigh_evecs(t: np.ndarray):
    """ for convenience return only evecs"""
    _, eig_vecs = get_haeberlen_ordered_eigh_evecs_evals(t)
    return eig_vecs

def get_haeberlen_eigh_evals(t: np.ndarray):
    """Return Haeberlen convention ordered eigenvalues of the passed tensor.
    ``|s_zz - s_iso| >= |s_xx - s_iso| >= |s_yy - s_iso|``
    Arguments:
        t: numpy array containing a NxN square tensor.
    Returns:
        np.ndarray: N-d numpy array containing the ordered eigenvalues.
    """
    eig_vals, _ = get_haeberlen_ordered_eigh_evecs_evals(t)

    new_eigvals = [eig_val for eig_val in eig_vals]

    # This will make Vxx Vyy the opposite sign of Vzz
    if eig_vals[0] > 0 or eig_vals[1] > 0:
        new_eigvals[0] = -1*eig_vals[0]
        new_eigvals[1] = -1*eig_vals[1]
    if eig_vals[2] < 0:
        new_eigvals[2] = eig_vals[2]*-1

    return new_eigvals


def extract_efgs(ats, efg_label):

    efgs_by_element = {}

    for at in ats:

        elements = at.symbols
        efgs = at.arrays[efg_label].reshape((-1, 3, 3))

        for element, efg in zip(elements, efgs):

            if element not in efgs_by_element:
                efgs_by_element[element] = []

            if np.abs(np.trace(efg)) > 1e-6:
                print("!"*30, np.abs(np.trace(efg)))
            #assert approx(np.trace(efg), abs=1e-7) == 0

            efgs_by_element[element].append(efg)

    for key, vals in efgs_by_element.items():
        efgs_by_element[key] = np.array(vals)

    return efgs_by_element

def extract_ir_entries(efg):
    m = efg

    scalar = m[0][0] + m[1][1] + m[2][2]

    v1 = m[2][1] - m[1][2]
    v2 = m[0][2] - m[2][0]
    v3 = m[1][0] - m[0][1]

    t1 = m[2][0] + m[0][2]
    t2 = m[1][0] + m[0][1]
    t3 = 2 * m[1][1] - m[0][0] - m[2][2]
    t4  = m[2][1] + m[1][2]
    t5 = m[2][2] - m[0][0]

    return scalar, np.array([v1, v2, v3]), np.array([t1, t2, t3, t4, t5])


def get_prop(efg, element):

    evals = get_haeberlen_eigh_evals(efg)
    evecs = get_haeberlen_eigh_evecs(efg)
    theta = np.arccos(evec[2])
    phi = np.arctan(evec[1] / evec[0]) if np.abs(evec[0]) > 0 else 0
    Cq, eta = spec.getcq(efg.flatten(), species=element)
    spin =  spins[element]
    omega_q = get_omega_q(Cq, eta, spin, theta, phi)
    data = {
        "Cq": Cq,
        "eta": eta,
        "theta": theta,
        "phi":phi,
        "omega_q":omega_q,
    }

    return data


def _get_tilde_Cqs_etas(evals):

    Vzzs = evals[:, z]
    Vzz_mean = np.mean(Vzzs)
    Vzz_std = np.std(Vzzs)


    tilde_Cqs = (Vzzs - Vzz_mean) / Vzz_std
    tilde_etas = (evals[:, x] - evals[:, y]) / Vzz_mean

    return tilde_Cqs, tilde_etas


def _get_thetas_phis(evecs):
    # take zz evector and its z component
    thetas = np.array([np.arccos(evec[z][z]) for evec in evecs])
    phis = np.array(
        [
            np.arctan(evec[z][y] / evec[z][x]) if np.abs(evec[z][x]) > 0 else 0
            for evec in evecs
        ]
    )

    return thetas, phis


def _get_omegas_q(data, element):
    omegas_q = np.array(
        [
            get_omega_q(Cq=Cq, eta=eta, spin=spins[element], theta=theta, phi=phi)
            for Cq, eta, theta, phi in zip(
                data["Cqs"], data["etas"], data["thetas"], data["phis"]
            )
        ]
    )
    return omegas_q

# 
# def get_non_ordered_orientation_props(efgs):
# 
#     quaternions = []
#     phis = []
#     thetas = []
#     for efg in efgs:
#         evals_my, evecs_my = np.linalg.eig(efg)
#         evecs_an = spec._get_haeberlen_eig_vecs(efg)
#         evals_an = spec._get_haeberlen_eigs(efg)
#         
#         D_my = np.eye(3) * evals_my
#         rot_my_efgs = np.linalg.inv(evecs_my) @ efg @ evecs_my
#         #assert np.allclose(D_my, )
# 
#         D_an = np.eye(3) * evals_an
#         rot_an_efgs = np.linalg.inv(evecs_an) @ efg @ evecs_an
#         #assert np.allclose(D_an, )
# 
#         #import pdb; pdb.set_trace()
# 
# 
#         quat_my =  spec.calc_quaternion(evecs_my)
#         quat_an =  spec.calc_quaternion(evecs_an)
# 
# 
#         theta, phi = _get_thetas_phis(np.array([evecs]))         
#         
#         rot = R.from_quat(quaternions[-1])
#         euler = rot.as_euler("xyz", degrees=True)
# 
#         import pdb; pdb.set_trace()
# 
#         phis.append(phi)
#         thetas.append(theta)
# 
#     phis = np.array(phis)
#     thetas = np.array(thetas)
#     quaternions = np.array(quaternions)
#     
#     return quaternions, phis, thetas
 

def get_phis_thetas_from_quaternions(quaternions):

    thetas = []
    phis = []

    return
    for quat in quaternions: 

        rot = R.from_quat(quat) 




def get_props_ref_pred(ref_efgs, pred_efgs, element):

    data_ref = {}
    data_pred = {}


    ref = [get_quat(ref_efg) for ref_efg in ref_efgs]
    ref_quats = np.array([rr[0] for rr in ref])
    ref_evecs = np.array([rr[1] for rr in ref])
    ref_evals = np.array([get_haeberlen_eigh_evals(efg) for efg in ref_efgs])

    data_ref["quaternions"] =  np.array(ref_quats)

    data_ref = get_single_dataset_properties(
        efgs=ref_efgs, 
        evals=ref_evals, 
        evecs=ref_evecs, 
        data=data_ref,
        element=element,
    )

    pred = [get_quat(pred_efg, dft_quat=ref_quat) for pred_efg, ref_quat in zip(pred_efgs, ref_quats)]
    pred_quats = np.array([pp[0] for pp in pred])
    pred_evecs = np.array([pp[1] for pp in pred])
    pred_evals = np.array([get_haeberlen_eigh_evals(efg) for efg in pred_efgs])


    data_pred["quaternions"] =  np.array(pred_quats)

    data_pred = get_single_dataset_properties(
        efgs=pred_efgs, 
        evals=pred_evals, 
        evecs=pred_evecs, 
        data=data_pred,
        element=element,
    )

    return data_ref, data_pred

def get_single_dataset_properties(efgs, evals, evecs, data, element):

    tilde_Cqs, tilde_etas = _get_tilde_Cqs_etas(evals)
    data["tilde_Cqs"] = tilde_Cqs
    data["tilde_etas"] = tilde_etas

    thetas, phis = _get_thetas_phis(evecs)
    data["thetas"] = thetas
    data["phis"] = phis

    for idx in range(3):
        data[f"evals_{idx}"] = np.array([val[idx] for val in evals])
    data["evecs"] = np.array(evecs)

    cq_eta_data = [spec.getcq(efg.flatten(), species=element) for efg in efgs]
    data["Cqs"] = np.array([dd[0] for dd in cq_eta_data])
    data["etas"] = np.array([dd[1] for dd in cq_eta_data])

    omegas_q = _get_omegas_q(data, element)
    data["omegas_q"] = omegas_q * 1e3  # kHz

    #data["efg"] = np.array([val[np.triu_indices(3)] for val in efgs])
    efg_L2_irs = np.array([extract_ir_entries(val)[2] for val in efgs])
    for idx in range(5):
        data[f"efg_L2_ir_{idx}"] = np.array([val[idx] for val in efg_L2_irs])

    data["determinants"] = np.array([np.linalg.det(efg) for efg in efgs])

    for i, j in [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]:
        data[f"efg_{i}{j}"] = np.array([efg[i][j] for efg in efgs])
    

    exp_length = len(efgs)
    for key, vals in data.items():
        assert len(vals) == exp_length


    return data


def get_props(efgs, element):

    #warnings.warn("This function is outdated", UserWarning)

    data = {
        "Cqs": [],
        "etas": [],
        "omegas_q": [],
    }

    # uses np.linalg.eig, which returns eigenvectors as columns 
    # eigenvectors[:,i] correspond to eigenvalues[i]
    evecs = np.array([get_haeberlen_eigh_evecs(efg) for efg in efgs])
    evals = np.array([get_haeberlen_eigh_evals(efg) for efg in efgs])

    tilde_Cqs, tilde_etas = _get_tilde_Cqs_etas(evals)
    data["tilde_Cqs"] = tilde_Cqs
    data["tilde_etas"] = tilde_etas

    # get angles
    thetas, phis = _get_thetas_phis(evecs)
    data["thetas"] = thetas
    data["phis"] = phis
    data["cos_phis"] = np.cos(phis)
    data["cos_2_phis"] = np.cos(2*phis)
    data["sin_phis"] = np.sin(phis)
    data["cos_thetas"] = np.cos(thetas)
    data["sin_thetas"] = np.sin(thetas)

    #quat, phis, thetas = get_non_ordered_orientation_props(efgs)
    #data["non_ord_quaternions"] = quat 
    #data["non_ord_phis"] = phis 
    #data["non_ord_thetas"] = thetas 


    # evals should be as columns
    #quaternions = np.array([spec.calc_quaternion(sub_evecs) for sub_evecs in evecs])
    #data["quaternions"] = quaternions

    #phis, theats = get_phis_thetas_from_quaternions(quaternions)
    #data["thetas_from_quats"] = thetas
    #data["phis_from_quats"] = phis

    for idx in range(3):
        data[f"evals_{idx}"] = np.array([val[idx] for val in evals])
    data["evecs"] = np.array(evecs)


    for efg in efgs:
        Cq, eta = spec.getcq(efg.flatten(), species=element)
        data["Cqs"].append(Cq)
        data["etas"].append(eta)

    data["Cqs"] = np.array(data["Cqs"])
    data["etas"] = np.array(data["etas"])

    omegas_q = _get_omegas_q(data, element)
    data["omegas_q"] = omegas_q * 1e3  # kHz

    #data["efg"] = np.array([val[np.triu_indices(3)] for val in efgs])
    efg_L2_irs = np.array([extract_ir_entries(val)[2] for val in efgs])
    for idx in range(5):
        data[f"efg_L2_ir_{idx}"] = np.array([val[idx] for val in efg_L2_irs])

    data["determinants"] = np.array([np.linalg.det(efg) for efg in efgs])

    for i, j in [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]:
        data[f"efg_{i}{j}"] = np.array([efg[i][j] for efg in efgs])
    

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


def prep_axis(figsize):

    plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0, wspace=0.1)

    ax_hist = plt.subplot(gs[0])
    ax_par = plt.subplot(gs[1])

    return ax_hist, ax_par

def compute_error_quaternion_metrics(ref_vals, pred_vals):
    dot_products = get_quaternion_dot_products(ref_vals, pred_vals)
    mean = dot_products.mean()
    percentile = np.percentile(dot_products, 10)

    metrics = {
        "mean" : float(mean),
        "90_percentile":float(percentile),
    }

    return metrics


def get_quaternion_dot_products(vals1, vals2):
    return np.abs(np.dot(vals1, vals2.transpose()).diagonal())


def plot_quaternions(ref_vals, pred_vals, ax, dataset_label, plot_kwargs):
    
    dot_products = get_quaternion_dot_products(ref_vals, pred_vals)

    mean = dot_products.mean()
    percentile = np.percentile(dot_products, 10)
    
    label = f"{dataset_label} mean(q.q): {mean:.2f}"
    bins = np.linspace(0.0, 1, 20)

    ax.hist(dot_products, bins=bins, label=label, **plot_kwargs)

    ls = plot_kwargs.get("ls", "-")
    color=plot_kwargs.get("color", "k")
    ax.axvline(percentile, label=f"90% count at {percentile:.2f}", ls=ls, color=color)
    
    ax.set_yscale("log")
    

def compute_error_metrics(ref_vals, pred_vals):

    assert len(ref_vals) == len(pred_vals)

    rmse = get_rmse(ref_vals, pred_vals)
    mae = get_mae(ref_vals, pred_vals)
    try:
        pearr, _ = pearsonr(ref_vals, pred_vals)
        pearr = float(pearr)
    except:
        pearr = 0.0

    mpe = get_mpe(ref_vals, pred_vals)

    error_metrics = {
        "rmse":float(rmse),
        "mae":float(mae),
        "mpe":float(mpe),
        "pearsonr":pearr,
    }
    return error_metrics


def plot(
    ref_vals,
    pred_vals,
    ax_hist,
    ax_par,
    dataset_label,
    scatter_kwargs,
    hist_kwargs_pred,
    hist_kwargs_ref,
    mode="all",
):

    error_metrics = compute_error_metrics(ref_vals=ref_vals, pred_vals=pred_vals)

    rmse = error_metrics["rmse"] 
    mae = error_metrics["mae"] 
    pearr = error_metrics["pearsonr"]
    mpe = error_metrics["mpe"]


    if mode=="all":
        parity_label = f"{dataset_label} RMSE: {rmse:.2g}; MAE: {mae:.2g}; \nR: {pearr:.2g}; rel. error: {mpe:.2g}%"
    elif mode == "r_mae":
        parity_label = f"MAE: {mae:.2g}; \nR: {pearr:.2g}"
        
    else: 
        parity_label=""


    ax_par.scatter(ref_vals, pred_vals, label=parity_label, **scatter_kwargs)

    min_val = np.min(np.concatenate([pred_vals, ref_vals]))
    max_val = np.max(np.concatenate([pred_vals, ref_vals]))
    bins =  np.linspace(min_val,max_val,20)

    if len(pred_vals.shape) == 1:
        ax_hist.hist(pred_vals, bins=bins, label=f"MACE {dataset_label}", **hist_kwargs_pred)
        # update train and test dft
        ax_hist.hist(
            ref_vals,
            #histtype="step",
            bins=bins,
            label=f"DFT {dataset_label}",
            **hist_kwargs_ref,
        )


def post_process_axis(ax_hist, ax_par):

    ymin, ymax = ax_par.get_ylim()
    xmin, xmax = ax_par.get_xlim()

    min_val = np.min([ymin, xmin])
    max_val = np.max([ymax, xmax])

    ax_par.plot([min_val, max_val], [min_val, max_val], color="k", ls="--")

    for ax in [ax_par, ax_hist]:
        ax.grid(color="lightgrey", zorder=-1)

    #ax_par.legend()

def get_rmse(ref, pred):
    return np.sqrt(np.mean((ref - pred) ** 2))


def get_mae(ref, pred):
    return np.mean(np.abs(ref - pred))


def get_mpe(ref, pred):
    """mean percentage error"""

    return np.mean(100 * np.abs((ref - pred) / ref))
