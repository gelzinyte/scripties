from util.permittivity import get_single_orhogonal_lorentz_root


def test_solve_single_lorentzian():
    eps_infty_nn = 1.2
    phonon_freq = 0.5
    S_nn = 2.3
    V = 1.1
    scattering_gamma = 1.2


    root1, root2 = get_single_orhogonal_lorentz_root(
        eps_infty_nn = eps_infty_nn,
        phonon_freq = phonon_freq,
        S_nn = S_nn,
        V = V,
        const_gamma = scattering_gamma,
        prop_gamma = None,
    )


    expected1 = 0.6315361907216146
    expected2 = 1.6336271904521513

    assert root1 == expected1
    assert root2 == expected2
