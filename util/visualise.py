import numpy as np

from ase.io import write


def evecs(ref_at, evecs, fname, irrep_labels=None):

    ats_out = []
    for mode_idx, nmode in enumerate(evecs):
        at = ref_at.copy()
        at.info[f"mode"] = mode_idx
        if irrep_labels is not None:
            at.info[f"symmetry"] = irrep_labels[mode_idx]
        at.arrays["Displacement"] = np.real(nmode)
        ats_out.append(at)

    write(fname, ats_out)

