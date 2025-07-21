from ase.atoms import Atoms
import spglib as spg
import numpy as np



def get_primitive_conventional(atoms, sym_thresh=1e-5):

    n_atoms = len(atoms)

    assert np.all(atoms.pbc)

    extra_info = {}
    extra_arrays = {}

    lattice = atoms.get_cell()[:]
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    magmoms = atoms.get_initial_magnetic_moments()
    spg_cell = (lattice, positions, numbers, magmoms)

    unit_cell_parameters = np.around(atoms.cell.cellpar(), 12)
    extra_info["unit_cell_parameters"] = unit_cell_parameters

    bravais = atoms.cell.get_bravais_lattice(eps=sym_thresh)
    extra_info["bravais"] = f"{bravais.longname} {bravais}"

    ats_prim = None
    ats_conv = None

    dataset = spg.get_symmetry_dataset(spg_cell, symprec=sym_thresh)
    if dataset:

        prim_lattice, prim_scaled_positions, prim_numbers = spg.find_primitive(
            spg_cell, symprec=sym_thresh
        )
        ats_prim = Atoms(
            cell=prim_lattice,
            scaled_positions=prim_scaled_positions,
            numbers=prim_numbers,
            pbc=True,
        )

        conv_lattice, conv_scaled_positions, conv_numbers = spg.standardize_cell(spg_cell, symprec=sym_thresh)
        ats_conv = Atoms(
            cell=conv_lattice,
            scaled_positions=conv_scaled_positions,
            numbers=conv_numbers,
            pbc=True,
        )
        
        extra_info = {"sym_thresh":sym_thresh}
        extra_info = {"spacegroup": dataset["number"]}
        extra_info = {"hall_symbol": dataset["hall"]}

        extra_info = {"occupied_wyckoffs": np.unique(dataset["wyckoffs"])}

        extra_info["equivalent_atoms"] = dataset["equivalent_atoms"]
        #extra_info["unique_equivalent_atoms"] = np.unique(dataset["equivalent_atoms"]) + 1
        extra_info["is_primitive"] = len(atoms) == len(prim_numbers)


        # get mapping to primitive 
        lattice = ats_conv.get_cell()[:]
        positions = ats_conv.get_scaled_positions()
        numbers = ats_conv.get_atomic_numbers()
        magmoms = ats_conv.get_initial_magnetic_moments()
        spg_cell = (lattice, positions, numbers, magmoms)
        dataset = spg.get_symmetry_dataset(spg_cell, symprec=sym_thresh)

        ats_conv.arrays["map_to_prim"] = dataset["mapping_to_primitive"]

        for at in [ats_prim, ats_conv]:
            at.info.update(extra_info)



    return ats_prim, ats_conv 
