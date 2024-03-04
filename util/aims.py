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

