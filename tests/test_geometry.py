from ase.io import read, write
import numpy as np
from ase.geometry import wrap_positions

import util.geometry

minkowski_rotation = np.array(
    [[-1,  1,  0],
     [ 0,  0,  1],
     [ 1,  0,  0]])


def test_primitive_to_conventional():

    at_ref = read("assets/sd_1238510.xyz")
    at_test = read("assets/sd_1238510.primitive_cell.minkowski_reduced.xyz")

    back_converted_at = util.geometry.primitive_to_conventional_cell_op(at_test)

    assert np.all(at_ref.cell == back_converted_at.cell)
    assert np.all(np.isclose(at_ref.positions, back_converted_at.positions))
    assert np.all(at_ref.numbers == back_converted_at.numbers)


def test_primitive_to_conventional_arrays():


    at_ref = read("assets/sd_1238510.xyz")
    at_test = read("assets/sd_1238510.primitive_cell.minkowski_reduced.xyz")

    f_ref = at_ref.arrays["aims_pw-lda_light_forces"]
    f_test = at_test.arrays["aims_pw-lda_light_forces"]

    array_perm_mx = util.geometry.array_perm_duplicate_mx()


    ref_numbers = at_ref.numbers
    test_numbers = at_test.numbers

    back_unpermuted_numbers = array_perm_mx @ test_numbers

    assert np.all(ref_numbers == back_unpermuted_numbers)

    f_test_permuted_doubled = array_perm_mx @ f_test

    print(np.max(np.abs(f_test_permuted_doubled - f_ref)))

    assert np.all(np.isclose(f_test_permuted_doubled, f_ref, atol=0.0003))

