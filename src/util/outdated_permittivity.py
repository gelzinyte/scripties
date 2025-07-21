



def get_zero_crossing(one_ph_eps, omega_range, id1, id2, threshold):

    assert len(one_ph_eps) == len(omega_range)

    # mask small values to be zero
    one_ph_eps[np.abs(one_ph_eps) < threshold] = 0

    # where crossess zero?
    zero_crossings = np.where(np.diff(np.sign(one_ph_eps[:, id1, id2].real)))[0]
    if len(zero_crossings) == 0:
        freq_eps_0 = "na"
    elif len(zero_crossings) != 2:
        raise RuntimeError(
            f"got {len(zero_crossings)} zero crossings for phonon {phonon_idx} - extend omega range?"
        )
    else:
        freq_eps_0 = omega_range[zero_crossings[1]]

    return freq_eps_0



def get_normalised_coupling_strentghts(
    omega_range,
    gamma_frequencies,
    numerator,
    volume,
    gamma,
    broadening_type,
    epsilon_inf,
    threshold=1e-14,
):

    #     out_dir =  Path(
    #     "/u/egg/mounted/high-throughput-permittivity/ionic_permittivities_from_jarvis_diagonalized_epsilon/tmp"
    #         )
    #     out_dir.mkdir(exist_ok=True)

    etas_dict = {
        "etas": [],
        "omega_phonon": [],
        "omega_eps0": [],
        "response_direction": [],
        "phonon_idx": [],
    }

    for phonon_idx, phonon_freq in enumerate(gamma_frequencies):

        ph_numerator = numerator[phonon_idx]

        assert np.all(ph_numerator.imag == 0)
        ph_numerator = ph_numerator.real

        if np.all(np.abs(ph_numerator) < threshold):
            continue

        orig_response_in, _, _ = get_direction_of_response(
            ph_numerator, threshold=threshold
        )

        ph_numerator = np.array([diagonalise_mx(ph_numerator)])
        _, id1, id2 = get_direction_of_response(ph_numerator[0])

        one_ph_eps = np.array(
            [
                epsilon_for_omega(
                    omega=omega,
                    gamma_frequencies=np.array(
                        [phonon_freq]
                    ),  # compute for this phonon only
                    numerator=ph_numerator,
                    volume=volume,
                    gamma=gamma,
                    broadening_type=broadening_type,
                )
                for omega in omega_range
            ]
        )
        one_ph_eps += epsilon_inf
        #
        #         out_fn = out_dir / f"phonon_{phonon_idx}.{phonon_freq*util.THz_to_inv_cm:.0f}.png"
        #         axs = prepare_axes(omega_range)
        #         omega_range_plt = omega_range * util.THz_to_inv_cm
        #         axs["xx"]["real"].plot(omega_range_plt, one_ph_eps[:,0,0].real)
        #         axs["xx"]["imag"].plot(omega_range_plt, one_ph_eps[:,0,0].imag)
        #         axs["yy"]["real"].plot(omega_range_plt, one_ph_eps[:,1,1].real)
        #         axs["yy"]["imag"].plot(omega_range_plt, one_ph_eps[:,1,1].imag)
        #         axs["zz"]["real"].plot(omega_range_plt, one_ph_eps[:,2,2].real)
        #         axs["zz"]["imag"].plot(omega_range_plt, one_ph_eps[:,2,2].imag)
        #         axs["xy"]["real"].plot(omega_range_plt, one_ph_eps[:,0,1].real)
        #         axs["xy"]["imag"].plot(omega_range_plt, one_ph_eps[:,0,1].imag)
        #
        #         plt.savefig(out_fn)
        #

        freq_eps_0 = get_zero_crossing(one_ph_eps, omega_range, id1, id2, threshold)

        if freq_eps_0 == "na":
            eta = "na"
        else:
            eta = get_normalized_coupling_strength(
                omega_ph=phonon_freq, omega_zero=freq_eps_0
            )
            freq_eps_0 *= util.THz_to_inv_cm

        etas_dict["etas"].append(eta)
        etas_dict["omega_phonon"].append(phonon_freq * util.THz_to_inv_cm)
        etas_dict["omega_eps0"].append(freq_eps_0)
        etas_dict["response_direction"].append(orig_response_in)
        etas_dict["phonon_idx"].append(phonon_idx)

    return etas_dict_to_df(etas_dict)






def diagonalise_mx(mx):
    evals, evecs = np.linalg.eig(mx)
    D = np.eye(len(evals)) * evals
    P = evecs
    A = mx
    assert np.allclose(D, np.linalg.inv(P) @ A @ P)
    return np.linalg.inv(P) @ mx @ P





def get_single_orhogonal_lorentz_root(eps_infty_nn, phonon_freq, S_nn, V, const_gamma):

    assert S_nn.imag == 0
    S_nn = S_nn.real
    
    const = -1 * eps_infty_nn * epsilon_0 * (2*np.pi)**2 * V / (S_nn**2 * 1e-24)

    A = const 
    B = const * (const_gamma ** 2 - 2 * phonon_freq ** 2) + 1
    C = const * phonon_freq ** 4 - phonon_freq ** 2

    #print(f"const: {const:.3g}, V: {V:.3g}, const_gamma: {const_gamma:.3g}, phonon_freq: {phonon_freq:.3g}")
    #print(f"A: {A:.3g}, B: {B:.3g}, C: {C:.3g}")
    
    return solve_quadratic(A=A, B=B, C=C)


