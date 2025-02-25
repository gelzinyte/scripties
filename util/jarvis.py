import shutil
from copy import deepcopy
import numpy as np
import warnings

import matplotlib.pyplot as plt

from scipy.constants import elementary_charge
from scipy.constants import physical_constants

from ase.io import read, write
from ase.io.vasp import read_vasp_out

from jarvis.io.vasp.outputs import Vasprun, Outcar

from pypdf import PdfWriter

from util.permittivity import epsilon_for_omega
from util.plot import prepare_axes_with_atoms as prepare_axes
import util.plot
import util.permittivity



def consolidate_plots(all_dims, all_crystals, wdir, out_dir): # consolidate into single file
    for dim in all_dims:
        for crys in all_crystals:

            wdir = out_dir / dim / crys
            if not wdir.exists():
                continue

            out_fn_base = f"{dim}.{crys}"

            # first move xyz file
            xyz_file_src = wdir / "structure.xyz"
            xyz_file_dest = out_dir / (out_fn_base + ".structures.xyz")
            shutil.copy(xyz_file_src, str(xyz_file_dest).replace(" ", "_"))

            # iterate over all individual stuctures and make pdf
            pdf_out_fn = out_dir / (out_fn_base + ".permittivities.pdf")
            merger = PdfWriter()
            for fn in wdir.iterdir():
                if str(fn).endswith(".pdf"):
                    merger.append(fn)
            merger.write(str(pdf_out_fn).replace(" ", "_"))
            merger.close()




def get_volume(vrun):
    # a bit hacky ...
    vrun_data = vrun.to_dict()
    structures = vrun_data["data"]["modeling"]["structure"]
    for struct in structures:
        if struct["@name"] == "finalpos":
            structure = struct
            break
    else:
        raise RuntimeError("found no 'finalpos'")
    vol = structure["crystal"]["i"]
    assert vol["@name"] == "volume"
    vol = float(vol["#text"])

    return vol

def is_sorted_descending(lst):
    return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))

def get_data_to_plot(jarvis_at, vasprun_fn, outcar_fn, broadening, format_for_print=False, omega_step=1, individual=False):


    jarvis_id = jarvis_at.info["jarvis_jid"]

    try:
        vrun = Vasprun(vasprun_fn)
        dfpt = Vasprun(vasprun_fn).dfpt_data
        out = Outcar(outcar_fn)
    except:
        return None, None, None, None, None

    vol = get_volume(vrun)  # ang^3
    vol *= 1e-30  # m^3

    # doesn't necesserily match ase/jarvis number of atoms
    masses = np.array(dfpt["masses"])  # amu
    masses *= physical_constants["atomic mass constant"][0]  # kg
    num_masses = len(masses)

    # len_masses x 3 x 3
    bec = dfpt["born_charges"]  # multiples of elementary charge
    bec *= elementary_charge  # coulomb

    # list, len(3*masses), each num_masses x 3 shape
    # orthonormal
    evecs = np.array(dfpt["phonon_eigenvectors"])

    # eigenvalues rom vasprun are weird, read the outcar
    # array, (3*num_masses,)
    # eigenvalues = dfpt["phonon_eigenvalues"]

    # outcar eigenvalues reported twice
    # once for evecs with 1/sqrt(m),
    # once without
    evals = out.phonon_eigenvalues
    assert len(evals) % 2 == 0
    num_evals = int(len(evals) / 2)
    evals = evals[:num_evals]  # thz

    # ---------
    # skip smallest three evecs
    # ------
    # check that the last three are smallest    
    assert is_sorted_descending(evals) 
    evals = evals[:-3]
    evecs = evecs[:-3]

    # pick omega range
    upper_lim = 1000 / util.THz_to_inv_cm
    max_omega = evals[0]
    assert np.max(evals) == max_omega
    if max_omega < upper_lim:
        max_omega = upper_lim
    omega_range = np.arange(
        50 / util.THz_to_inv_cm, max_omega * 1.1,  omega_step / util.THz_to_inv_cm
    )  # thz

    # reporte "dfpt" epsilon is "epsilon" + "epsilon_ion". so
    # "epsilon" electornic bit.
    epsilon_inf = dfpt["epsilon"]["epsilon"]
    # for reference
    eps_0_ref = dfpt["epsilon"]["epsilon_ion"]

    S = util.permittivity.get_born_along_displacements(
        gamma_evecs=evecs,  # expect shape of  num_masses*3 x num_masses x 3
        masses=masses,  # kg
        born_charges=bec,  # expect shape of num_masses x 3 x 3, coulomb
    )  # c/sqrt(kg)

    numerator = util.permittivity.get_numerator(S)  # c^2/kg

    eps_for_omega = np.array(
        [
            epsilon_for_omega(
                omega=omega,
                gamma_frequencies=evals,
                numerator=numerator,
                volume=vol,
                gamma=broadening[0],
                broadening_type=broadening[1],
            )
            for omega in omega_range
        ]
    )

    eps_for_omega += epsilon_inf

    if individual:
        ref_omega_range = deepcopy(omega_range)

        # create individual plots
        individual_eps_for_omega = []
        for idx, (num, freq) in enumerate(zip(numerator, evals)):
            single_eps_for_omega = []
            for omega in omega_range:
                if idx==13 and omega > 217/util.THz_to_inv_cm and omega < 224/util.THz_to_inv_cm:
                    pass
                eps = epsilon_for_omega(
                        omega=omega,
                        gamma_frequencies=np.array([freq]),
                        numerator=np.array([num]),
                        volume=vol,
                        gamma=broadening[0],
                        broadening_type=broadening[1],
                    )
                single_eps_for_omega.append(eps)
            #import pdb; pdb.set_trace()
            single_eps_for_omega = np.array(single_eps_for_omega)

            # commented out for debugging:
#             single_eps_for_omega = np.array(
#                 [
#                     epsilon_for_omega(
#                         omega=omega,
#                         gamma_frequencies=np.array([freq]),
#                         numerator=np.array([num]),
#                         volume=vol,
#                         gamma=broadening[0],
#                         broadening_type=broadening[1],
#                     )
#                     for omega in omega_range
#                 ]
#             )
            single_eps_for_omega += epsilon_inf
            individual_eps_for_omega.append(single_eps_for_omega)

#            if idx == 6:
#                id1=1
#                id2=1
#                fig = plt.figure()
#                plt.plot(omega_range*util.THz_to_inv_cm, single_eps_for_omega[:,id1,id2].real)
#                plt.savefig(f"debug_eps_{idx}.{id1}{id2}.png")

        individual_eps_for_omega = np.array(individual_eps_for_omega)
    else:
        individual_eps_for_omega=None


    # check which s's don't contribute much and only return those phonon frequencies to be plotted 
    max_numerator = numerator.max(axis=1).max(axis=1)
    selected_freqs = evals[max_numerator > 1e-13]

    coupling_strength_df = util.permittivity.compute_dft_normalised_coupling_strengths(
        eps_infty=epsilon_inf, 
        phonon_freq = evals, 
        S=S, 
        volume=vol, 
        scattering_gamma=broadening[0], 
        broadening_type=broadening[1], 
        format_for_print=format_for_print
    )
    assert np.all(ref_omega_range == omega_range)

    #import pdb; pdb.set_trace()
    #return omega_range, eps_for_omega, selected_freqs, coupling_strength_df, individual_eps_for_omega
    return omega_range, eps_for_omega, evals, coupling_strength_df, individual_eps_for_omega




def extract_vasp_data(jarvis_at, vasprun_fn, outcar_fn):
    """Follows 'get_data_to_plot', but only extracts the needed data and assigns it to the atoms objects"""

    jarvis_id = jarvis_at.info["jarvis_jid"]

    try:
        vrun = Vasprun(vasprun_fn)
        dfpt = Vasprun(vasprun_fn).dfpt_data
        out = Outcar(outcar_fn)
    except:
        return None


    vol = get_volume(vrun)  # ang^3 needs to be converted to m^3 for plotting
    #vol *= 1e-30  # m^3
    jarvis_at.info["vasp_calc_volume"] = vol

    # doesn't necesserily match ase/jarvis number of atoms
    masses = np.array(dfpt["masses"])  # amu
    # for permittivity calc:
    #masses *= physical_constants["atomic mass constant"][0]  # kg
    jarvis_at.arrays["vasp_calc_masses"] = masses
    num_masses = len(masses)

    # len_masses x 3 x 3
    bec = dfpt["born_charges"]  # multiples of elementary charge
    # for plotting:
    #bec *= elementary_charge  # coulomb
    jarvis_at.arrays["vasp_calc_becs"] = bec #.reshape((-1, 9))

    # list, len(3*masses), each num_masses x 3 shape
    # orthonormal
    evecs = np.array(dfpt["phonon_eigenvectors"])
    # above is of the shape (3N, N, 3), which we can't assign to atoms
    # transpose and then we'll need to transpose it back
    # by executing the same (1, 0, 2) permutation
    assert evecs.shape[1] == num_masses
    #evecs_for_ats = np.transpose(evecs, (1, 0, 2)).reshape(num_masses, num_masses * 9)
    jarvis_at.arrays["vasp_calc_evecs"] = evecs #evecs_for_ats

    # eigenvalues rom vasprun are weird, read the outcar
    # array, (3*num_masses,)
    # eigenvalues = dfpt["phonon_eigenvalues"]

    # outcar eigenvalues reported twice
    # once for evecs with 1/sqrt(m),
    # once without
    evals = out.phonon_eigenvalues
    assert len(evals) % 2 == 0
    num_evals = int(len(evals) / 2)
    evals = evals[:num_evals]  # thz
    jarvis_at.arrays["vasp_calc_evals"] = evals

    #import pdb; pdb.set_trace()

    # reporte "dfpt" epsilon is "epsilon" + "epsilon_ion". so
    # "epsilon" electornic bit.
    epsilon_inf = dfpt["epsilon"]["epsilon"]
    # for reference
    eps_0_ref = dfpt["epsilon"]["epsilon_ion"]
    jarvis_at.info["vasp_calc_epsilon_inf"] = epsilon_inf
    jarvis_at.info["vasp_calc_eps_0"] = eps_0_ref

    return jarvis_at


# from plotted position to epsilon position


def plot_plot(at, vasprun_fn, outcar_fn, broadening, out_fn, xyz_file):
    "read data from jarvis xml files and plot permittivity"

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        run_warn = False
        try:
            omega_range, eps_for_omega = get_data_to_plot(
                at=at, vasprun_fn=vasprun_fn, outcar_fn=outcar_fn, broadening=broadening
            )
            if omega_range is None and eps_for_omega is None:
                return
        except RuntimeWarning:
            run_warn = True

    if run_warn:
        omega_range, eps_for_omega = get_data_to_plot(
            at=at, vasprun_fn=vasprun_fn, outcar_fn=outcar_fn, broadening=broadening
        )
        if omega_range is None and eps_for_omega is None:
            return

    omega_range *= util.THz_to_inv_cm
    ax_eps, ax_atoms = prepare_axes(omega_range)

    plot_eps(ax_eps, omega_range, eps_for_omega)

    util.plot.plot_atoms(at, ax_atoms)

    num_ase_at = len(at)
    num_vasp_at = at.info["jarvis_nat"]
    formula = at.info["jarvis_formula"]
    reference = at.info["jarvis_reference"]
    crystal = at.info["jarvis_crys"]
    dimensionality = at.info["jarvis_dimensionality"]
    jid = at.info["jarvis_jid"]

    # save atoms somehow

    title = f"{formula}, {num_vasp_at} atoms, {crystal}, {dimensionality}, {jid} {reference}, {broadening[1]} broadening\n"
    title += "cell: " + " ".join([f"{num:.2f}" for num in at.cell.lengths()])
    title += "\nangles: " + " ".join([f"{num:.0f}" for num in at.cell.angles()])
    if run_warn:
        title = "! " + title

    plt.suptitle(title, fontsize="18")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.savefig(out_fn, bbox_inches="tight")
    plt.close()

    write(xyz_file, at, append=True)


def plot_eps(ax_eps, omega_range, eps_for_omega):

    conv = "xyz"
    for ie in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]:

        # all this labeling is a bit messy...
        id1 = ie[0]
        id2 = ie[1]
        title = conv[id1] + conv[id2]

        ax_r = ax_eps[title]["real"]
        ax_i = ax_eps[title]["imag"]

        ys_r = eps_for_omega[..., id1, id2].real
        ys_i = eps_for_omega[..., id1, id2].imag

        ax_r.plot(omega_range, ys_r)
        ax_i.plot(omega_range, ys_i)

        if title in ["xx", "yy", "zz"]:
            ax_r.fill_between(
                omega_range,
                0,
                1,
                where=ys_r < 0,
                alpha=0.2,
                transform=ax_r.get_xaxis_transform(),
            )
            ax_i.fill_between(
                omega_range,
                0,
                1,
                where=ys_r < 0,
                alpha=0.2,
                transform=ax_i.get_xaxis_transform(),
            )

        for ax in [ax_r, ax_i]:
            y_min, y_max = ax.get_ylim()
            if np.abs(y_min) < 1 and np.abs(y_max) < 1:
                ax.set_ylim(-1, 1)
