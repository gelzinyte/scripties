from scipy.constants import speed_of_light
from scipy.constants import physical_constants

def by_label(ats, info_label):
    if info_label == None:
        return {"no_label":ats}
    data = {}
    for at in ats:
        if info_label not in at.info.keys():
            label = "no_label"
        else:
            label = at.info[info_label]

        if label not in data.keys():
            data[label] = []
        data[label].append(at)

    return data


THz_to_inv_cm = 10 ** 12 / speed_of_light / 100 # Thz -> s-1, c, m -> cm


Planck_in_eVHz = physical_constants["Planck constant in eV/Hz"][0]

#               cm*m-1  m*s-1           eV * s           eV->meV
inv_cm_to_meV = 100 * speed_of_light * Planck_in_eVHz * 1e3

#            THz-> Hz   eV Hz-1          eV->meV
THz_to_meV = 1e12    * Planck_in_eVHz * 1e3


def print_cross_mx(evecs):

    np.set_printoptions(precision=2, suppress=True)

    num = evecs.shape[0]
    mm = np.full((num,num), np.nan)
    for idx1 in range(num):
        for idx2 in range(num):
            mm[idx1][idx2] = np.dot(evecs[idx1].flatten(), evecs[idx2].flatten())

    print(mm)


