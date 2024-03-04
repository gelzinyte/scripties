from ase.io import read, write
import numpy as np

import util.geometry


def test_primitive_to_conventional():

    op = util.geometry.primitive_to_conventional_cell_op()
    at_ref = read("assets/sd_1238510.xyz")
    at_test = read("assets/sd_1238510.primitive_cell.minkowski_reduced.xyz")

    rotated_cell = op @ at_test.cell
    assert np.all(at_ref.cell == rotated_cell)
