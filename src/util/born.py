from pathlib import Path
import numpy as np

from scipy.constants import elementary_charge

from ase.io import read
import util.aims



def assign_becs(at_base, ats_displaced, born_displacement_magnitude):

    volume = at_base.get_volume()

    bec_for_at_entry = []
    all_becs = []

    for disp_at in ats_displaced:

        displacement_direction = disp_at.info["bec_displacement_direction"]

        polarization = disp_at.info["aims_polarization"] # units  e/Ang^2
        #              Ang^3    e/Ang^2        Ang
        #born_charges = volume * polarization / born_displacement_magnitude # multiples of unit charge

        #          Ang^3    Ang^3 -> m^3  C/m^2          Ang                     Ang -> m
        born_charges = volume * 1e-30 *       polarization / (born_displacement_magnitude * 1e-10) #  C
        born_charges /= elementary_charge

        bec_for_at_entry.append([float(bec) for bec in born_charges])

        if displacement_direction == "z":

            all_becs.append(np.array(bec_for_at_entry).flatten())
            bec_for_at_entry = []

    all_becs = np.array(all_becs)
    at_base.arrays["aims_born_effective_charges"] = all_becs
    at_base.info["bec_displacement_magnitude"] = born_displacement_magnitude
    return at_base



def get_born_charges(homedir, displacement_magnitude, wdir="workdir", aims_output_filename="aims.out", units="Coulomb"):

    assert units in ["Coulomb", "multiple_of_elementary_charge"]

    out_born_charges = []

    homedir = Path(homedir)
    workdir = homedir / wdir

    atoms = read(homedir / 'geometry.in')
    non_equivalent_atoms = np.arange(len(atoms))

    displacement_magnitude = displacement_magnitude # Angstrom

    volume = atoms.get_volume() # A^3

    for idx, atom in enumerate(non_equivalent_atoms):

        entry = []
        for cart_idx, cart in enumerate(["x", "y", "z"]):

            atom_symbol = atoms.get_chemical_symbols()[atom]
            aims_output_file_path = workdir / f"{idx}.{atom_symbol}.{cart}" / aims_output_filename

            polarization = util.aims.get_polarization(aims_output_file_path) # Units C/m^2
                #          Ang^3    Ang^3 -> m^3  C/m^2          Ang                     Ang -> m
            born_charges = volume * 1e-30 *       polarization / (displacement_magnitude * 1e-10) #  C

            if units == "multiple_of_elementary_charge":
                born_charges /= elementary_charge

            entry.append([float(bec) for bec in born_charges])
        out_born_charges.append(entry)

    out_born_charges = np.array(out_born_charges)
    # to match the shape of array of reference methods
    out_born_charges = np.moveaxis(out_born_charges, 1, 2)
    #print(f"born charges shape is {out_born_charges.shape} n_atoms x 3_polarization_directions x 3_cart_at_displacements")
    #print(out_born_charges[0]/elementary_charge) # matches https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033102
    return out_born_charges, units

