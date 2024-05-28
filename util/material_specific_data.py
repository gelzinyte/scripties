import numpy as np
import warnings

import util.geometry


# rotate the axes so they match the paper                               
ga2o3_R = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]])

#  rank 0:  b' = b                                                      
#                                                                       
#  rank 1: T'_i = T_j R_ji                                              
#  (rank 1: T'_i = R_ij  T_j)                                           
#                                                                       
# rank 2: s'_ij =  R_ij a_kl s_kl                                       
#         


def get_ga2o3_irreps_and_relevant_modes(phonon):
    phonon.set_irreps(q=[0,0,0])
    irrep_labels = phonon.irreps._get_ir_labels()
    if len(irrep_labels)!=30:
        warnings.warn(f"Got {len(irrep_labels)} labels, not 30 as expected for Ga2O3. Overwriting manually.")
        irrep_labels = np.array(['Bu', "Bu", "Au", 'Ag', 'Bg', 'Bg', 'Au', 'Ag', 'Ag', 'Bu', 'Bu', 'Bu', 'Au', 'Ag', 'Ag', 'Bu', 'Bg', 'Ag', 'Bu', 'Au', 'Ag', 'Bg', 'Bu', 'Ag', 'Bg', 'Ag', 'Au', 'Bu', 'Bu', 'Ag'])

    relevant_mode_idcs = np.array([idx for idx, label in enumerate(irrep_labels) if label in ["Bu", "Au"]])
    # skip the first three Acoustic modes
    relevant_mode_idcs = relevant_mode_idcs[3:]

    return irrep_labels, relevant_mode_idcs


def get_conventional_ga2o3_gamma_eigenvectors(qpoint_dict):
    # eigenvectors of dyn matrix
    # different eigenvectors as columns, so
    # shape - [n_at x 3] x n_eigenvectors,
    # 30 x 30, eigenvectors as columns
    # dot products are np.eye
    gamma_evecs = qpoint_dict["eigenvectors"][0]

    # transpose so different eigenvectors are enumerated by the first axis (rows)
    gamma_evecs = gamma_evecs.transpose()
    # reshape so each eigenvector has the shape of n_at x 3D
    gamma_evecs = gamma_evecs.reshape((30, 10, 3))   # [30,10,3]

    # For each eigenvector, convert from primitive to conventional
    prim_to_conv_perm_mx = util.geometry.array_perm_duplicate_mx()  # [20,10]
    # for each eigenvector the transformation is
    # prim_to_conv_perm_mx @ eigenvector
    # So to rewrite for everyone
    conv_evecs = np.einsum("ij,hjk->hik", prim_to_conv_perm_mx, gamma_evecs)
    return conv_evecs






