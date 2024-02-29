from ase.dft import kpoints
from ase.io import read

from vibes.phonopy import postprocess
from vibes.phonopy import wrapper as phonopy_wrapper
from vibes.structure.convert import to_Atoms
from vibes.helpers import brillouinzone as bz

from util import phonopy as u_phonopy

def get_bands(atoms, paths=None, npoints=50, eps=0.0002):
    """Get the recommended BZ path(s) for atoms

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the recommended high-symmetry point path
    paths: list of np.ndarray
        Paths connecting high-symmetry points
    npoints: int
        Number of points for each band

    Returns
    -------
    bands: list of np.ndarrays
        The recommended BZ path(s) for atoms
    """
    if paths is None:
        paths = get_paths(atoms)
    bands = []
    for path in paths:
        points = kpoints.parse_path_string(path)[0]  # [:-1]
        ps = [points.pop(0)]
        #print(len(points))
        for _, p in enumerate(points):
            ps.append(p)
            #print(eps)
            bands.append(atoms.cell.bandpath("".join(ps), npoints=npoints, eps=eps).kpts)
            ps.pop(0)
    return bands


def plot_total_dos(
    ax,
    frequency_points,
    total_dos,
    freq_Debye=None,
    Debye_fit_coef=None,
    xlabel=None,
    ylabel=None,
    draw_grid=True,
    flip_xy=False,
    fmt="r-"
):
    """Plot total DOS."""
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(which="both", direction="in")
    ax.yaxis.set_tick_params(which="both", direction="in")

    if flip_xy:
        ax.plot(total_dos, frequency_points, fmt, linewidth=1)
        if freq_Debye:
            ax.plot(
                np.append(Debye_fit_coef * freqs**2, 0),
                np.append(freqs, freq_Debye),
                "b-",
                linewidth=1,
            )
    else:
        ax.plot(frequency_points, total_dos, fmt, linewidth=1)
        if freq_Debye:
            ax.plot(
                np.append(freqs, freq_Debye),
                np.append(Debye_fit_coef * freqs**2, 0),
                "b-",
                linewidth=1,
            )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(draw_grid)



def phonon_and_labels(trajectory, born_file=None, eps=0.001, q_mesh=None):

    if born_file is not None:
        phonon = postprocess(
            trajectory_file=trajectory,
            born_charges_file=born_file
        )
    else:
        phonon = postprocess(
            trajectory_file=trajectory,
        )

    at = to_Atoms(phonon.primitive)
    lat = at.cell.get_bravais_lattice(eps=eps)
    paths = lat.special_path.split(",")

    print(trajectory, ":")
    print(lat)
    print(paths)

    bands = u_phonopy.get_bands(at, paths=paths, eps=eps)
    labels = bz.get_labels(paths)
    phonon.run_band_structure(bands, labels=labels)

    if q_mesh is None:
        q_mesh = [45, 45, 45]

    phonon.run_mesh(q_mesh)
    phonon.run_total_dos(use_tetrahedron_method=True)

    return phonon, labels


def get_wfl_phonons(phonon, evaled_fn, prop_prefix, skip_first_image=True, eps=0.001, q_mesh=None):

    if skip_first_image:
        stride = "1:"
    else:
        stride = ":"

    calc_out_ats = read(evaled_fn, stride)
    force_sets = [at.arrays[f"{prop_prefix}forces"] for at in calc_out_ats]
    phonon.produce_force_constants(
        force_sets,
        calculate_full_force_constants=False        # vibes default
    )

    # ---------- Gamma point frequencies

    gamma_freq, max_freq = phonopy_wrapper.summarize_bandstructure(phonon)

    print(max_freq, "THz")
    print(gamma_freq, "THz")

    # ---------- band_structure 

    at = to_Atoms(phonon.primitive)
    lat = at.cell.get_bravais_lattice(eps=eps)
    paths = lat.special_path.split(",")

    print(f"{prop_prefix}:")
    print(lat)
    print(paths)

    bands = u_phonopy.get_bands(at, paths=paths, eps=eps)
    labels = bz.get_labels(paths)
    phonon.run_band_structure(bands, labels=labels)

    if q_mesh is None:
        q_mesh = [45, 45, 45]
    phonon.run_mesh(q_mesh)
    phonon.run_total_dos(use_tetrahedron_method=True)

    return phonon, labels



