from copy import deepcopy
from itertools import permutations

import numpy as np


x = 0
y = 1
z = 2



def get_dot_product(evec, ref_evec):
    dot_products = np.array([[np.dot(evec[:,i], ref_evec[:,j]) for i in [x, y, z]] for j in [x, y, z]])
    return dot_products



def match_permutation(evec, ref_evec):

    dot_products = get_dot_product(evec, ref_evec)
        
    best_trace = np.trace(np.abs(dot_products))
    best_evec = evec

    if  best_trace > 2.9:
        return evec

    for perm in permutations(range(3)):
        permuted_evec = deepcopy(evec)
        permuted_evec = permuted_evec[:, perm]

        dot_products = get_dot_product(permuted_evec, ref_evec)
        trace = np.trace(np.abs(dot_products))

        if trace > best_trace:
            best_trace = trace
            best_evec = permuted_evec


    return best_evec


def match_directions(evec, ref_evec):

    dot_products = get_dot_product(evec, ref_evec)
    
    for i in [x, y, z]:
        if dot_products[i][i] < 0:
            evec[:, i] *= -1

    return evec

def match_permutation_direction(evec, ref_evec):
    evec = match_permutation(evec, ref_evec)
    evec = match_directions(evec, ref_evec)
    return evec


