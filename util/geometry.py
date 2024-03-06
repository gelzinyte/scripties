import numpy as np
from ase.geometry import wrap_positions
from ase.build import cut
import util
from pathlib import Path

minkowski_rotation = np.array(
    [[-1,  1,  0],
     [ 0,  0,  1],
     [ 1,  0,  0]])

cut_rotation =  np.array(
    [[ 0.5, -0.5,  0. ],
     [ 0.5,  0.5,  0. ],
     [ 0. ,  0. ,  1. ]])

ff = Path(util.__file__).parent / "data/prim_to_conventional_permutation_mx.npy" 

permutation_mx = np.load(ff)


def primitive_to_conventional_cell_op(at_prim_mink_red):

    at_prim = at_prim_mink_red.copy()

    # undo minkowski reduction
    primitive_cell_from_minkowski = np.linalg.inv(minkowski_rotation) @ at_prim_mink_red.cell
    at_prim.cell = primitive_cell_from_minkowski
    at_prim.positions = wrap_positions(at_prim.positions, primitive_cell_from_minkowski)


    # double the unit cell
    undo_cut_cell = np.linalg.inv(cut_rotation)
    back_to_normal_at = cut(at_prim, a=undo_cut[0], b=undo_cut[1], c=undo_cut[2])

    # swap atoms back
    ## remove all entries but necessary
    info_to_remove = [key for key in back_to_normal_at.info.keys()]
    arrays_to_remove = [key for key in back_to_normal_at.arrays.keys() if key not in ["numbers", "positions"]]
    print(f"removing info entries: {info_to_remove}")
    print(f"removing arrays entries: {arrays_to_remove}")
    for key in info_to_remove:
        del at.info[key]
    for key in arrays_to_remove:
        del at.arrays[key]

    final_at = back_to_normal_at.copy()
    final_at.positions = permutation_mx @ final_at.positions
    final_at.numbers = permutation_mx @ final_at.numbers

    return final_at 

