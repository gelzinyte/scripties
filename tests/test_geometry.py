from ase.io import read, write
import numpy as np

import util.geometry


def test_primitive_to_conventional():

    at_ref = read("assets/sd_1238510.xyz")
    at_test = read("assets/sd_1238510.primitive_cell.minkowski_reduced.xyz")

    unconverted_at = util.geometry.primitive_to_conventional_cell_op(at_test)

    assert np.all(at_ref.cell == rotated_cell)
    assert np.all(np.is_close(at_ref.positions, unconverted_at.positions))
    assert np.all(at_ref.numbers == unconverted_at.numbers)
