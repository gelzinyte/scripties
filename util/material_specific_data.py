import numpy as np


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


