import numpy as np

minkowski_rotation = np.array(
    [[-1  1  0]
     [ 0  0  1]
     [ 1  0  0]])

cut_rotation =  np.array(
    [[ 0.5, -0.5,  0. ],
     [ 0.5,  0.5,  0. ],
     [ 0. ,  0. ,  1. ]])


def primitive_to_conventional_cell(primitive_minkowski_cell):

    primitive_cell = np.linalg.inv(minkowski_rotation) @ primitive_minkowski_cell 
    conventional_cell = np.dot(np.linalg.inv(cut_rotation), primitive_cell)

    return conventional_cell

