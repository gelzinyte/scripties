import re
import numpy as np


def read_dielectric(aims_out):
    dielectric_regex = re.compile(
        "(?:DFPT for dielectric_constant:--->  # PARSE DFPT_dielectric_tensor)\n"\
        "\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\n"\
        "\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\n"\
        "\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\n"
        )

    with open(aims_out) as f:
        lines = f.read()
    match = dielectric_regex.search(lines)
    return np.array([float(mm) for mm in match.groups()]).reshape((3,3))




def get_polarization(aims_output_file):

    regex_patt = re.compile("(?:Cartesian Polarization)\s+(-?[\d.E-]+)\s+(-?[\d.E-]+)\s+(-?[\d.E-]+)")

    with open(aims_output_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Cartesian Polarization" in line:
                re_match = regex_patt.search(line)
                polarization = np.array([float(mm) for mm in re_match.groups()])
                return polarization
    raise RuntimeError("No cartesian polarization found")


