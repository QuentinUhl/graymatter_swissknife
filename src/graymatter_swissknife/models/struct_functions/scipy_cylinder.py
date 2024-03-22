import numpy as np


def cylinder_perpendicular_signal(r, D0, b, big_delta, small_delta):
    """
    Calculates diffusion attenuation, mlnS = - ln(S/S0), inside a perfectly reflecting cylinder of radius r, free
    diffusion coefficient D0, bvalue b (in IS units of s/m), with pulse width delta and distance big_delta between the fronts
    of the pulses, according to Murday and Cotts, JCP 1968
    (c) Quentin Uhl, July 2023

    :param r: The radius of the cylinder (in microns)
    :param D0: The diffusion coefficient inside the cylinder (in µm²/ms)
    :param big_delta: The time of the second pulse (in ms)
    :param small_delta: The pulse width (in ms)
    :param b: The b-value (in ms/µm²)
    :return mlnS: The diffusion attenuation, mlnS = - ln(S/S0)
    """
    # Make sure all inputs are numpy arrays
    b, big_delta, small_delta = np.array(b), np.array(big_delta), np.array(small_delta)
    # Define reference values
    g = np.sqrt(b / (big_delta - small_delta / 3)) / small_delta  # in 1/µm*ms
    t_ref = r ** 2 / D0  # Compute t_ref (in ms)
    bar_delta = np.expand_dims(small_delta / t_ref, axis=-1)  # Compute bar_delta (unitless)
    bar_bigdelta = np.expand_dims(big_delta / t_ref, axis=-1)  # Compute bar_bigdelta (unitless)
    # Cylindrical roots precomputed
    alpha = np.array([1.841183781340659, 5.331442773525032, 8.536316366346286, 11.706004902592063, 14.863588633909032,
                      18.015527862681804, 21.164369859188788, 24.311326857210776, 27.457050571059245,
                      30.601922972669094, 33.746182898667385, 36.889987409236809, 40.033444053350678,
                      43.17662896544882, 46.319597561173914, 49.462391139702753, 52.605041111556687,
                      55.747571792251009, 58.890002299185703, 62.032347870661987, 65.174620802544453,
                      68.316831125951808, 71.458987105850994, 74.601095613456408, 77.743162408196767,
                      80.885192353878438, 84.027189586293531, 87.169157644540277, 90.311099574903423,
                      93.45301801376003, 96.594915254291138, 99.736793300573908, 102.878653911754455,
                      106.020498638360806, 109.162328852340863, 112.304145772055051, 115.44595048318557,
                      118.587743956319926, 121.729527061810202, 124.871300582387889])
    # Put the alphas in the last dimension
    alpha_shape = alpha.shape
    for dim in range(np.max(np.ndim(big_delta), np.ndim(small_delta))):
        alpha_shape = (1,) + alpha_shape
    alpha = np.reshape(alpha, alpha_shape)
    # Compute the signal
    mlnS = np.sum((2 / (alpha ** 6 * (alpha ** 2 - 1))) * (-2 + 2 * alpha ** 2 * bar_delta +
                                                           2 * (np.exp(-alpha ** 2 * bar_delta) +
                                                                np.exp(-alpha ** 2 * bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bar_delta + bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bar_bigdelta - bar_delta))), axis=-1)
    signal = np.exp(-mlnS * D0 * g ** 2 * t_ref ** 3)
    return signal


def cylinder_perpendicular_jacobian(r, D0, b, big_delta, small_delta):
    """

    :param r: The radius of the cylinder (in microns)
    :param D0: The diffusion coefficient inside the cylinder (in microns^2/ms)
    :param big_delta: The time of the second pulse (in ms)
    :param small_delta: The pulse width (in ms)
    :param b: The b-value (in ms/µm²)
    :return mlnS: The diffusion attenuation, mlnS = - ln(S/S0)
    :return mlnS_dr: The jacobian of the diffusion attenuation with respect to r
    """
    # Make sure all inputs are numpy arrays
    b, big_delta, small_delta = np.array(b), np.array(big_delta), np.array(small_delta)
    # Define reference values
    g = np.sqrt(b / (big_delta - small_delta / 3)) / small_delta  # in 1/µm*ms
    t_ref = r ** 2 / D0  # Compute t_ref (in ms)
    bar_delta = np.expand_dims(small_delta / t_ref, axis=-1)  # Compute bar_delta (unitless)
    bar_bigdelta = np.expand_dims(big_delta / t_ref, axis=-1)  # Compute bar_bigdelta (unitless)
    # Compute the derivatives of the reference values
    dt_ref_dr = 2 * r / D0
    d_inv_t_ref_dr = -2 * D0 / (r ** 3)
    bar_delta_dr = np.expand_dims(small_delta * d_inv_t_ref_dr, axis=-1)
    bar_bigdelta_dr = np.expand_dims(big_delta * d_inv_t_ref_dr, axis=-1)
    # Cylindrical roots precomputed
    alpha = np.array([1.841183781340659, 5.331442773525032, 8.536316366346286, 11.706004902592063, 14.863588633909032,
                      18.015527862681804, 21.164369859188788, 24.311326857210776, 27.457050571059245,
                      30.601922972669094, 33.746182898667385, 36.889987409236809, 40.033444053350678,
                      43.17662896544882, 46.319597561173914, 49.462391139702753, 52.605041111556687,
                      55.747571792251009, 58.890002299185703, 62.032347870661987, 65.174620802544453,
                      68.316831125951808, 71.458987105850994, 74.601095613456408, 77.743162408196767,
                      80.885192353878438, 84.027189586293531, 87.169157644540277, 90.311099574903423,
                      93.45301801376003, 96.594915254291138, 99.736793300573908, 102.878653911754455,
                      106.020498638360806, 109.162328852340863, 112.304145772055051, 115.44595048318557,
                      118.587743956319926, 121.729527061810202, 124.871300582387889])
    # Put the alphas in the last dimension
    alpha_shape = alpha.shape
    for dim in range(np.max(np.ndim(big_delta), np.ndim(small_delta))):
        alpha_shape = (1,) + alpha_shape
    alpha = np.reshape(alpha, alpha_shape)
    # Compute the signal
    mlnS = np.sum((2 / (alpha ** 6 * (alpha ** 2 - 1))) * (-2 + 2 * alpha ** 2 * bar_delta +
                                                           2 * (np.exp(-alpha ** 2 * bar_delta) +
                                                                np.exp(-alpha ** 2 * bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bar_delta + bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bar_bigdelta - bar_delta))), axis=-1)
    mlnS_dr = np.sum((2 / (alpha ** 6 * (alpha ** 2 - 1))) *
                     (2 * alpha ** 2 * bar_delta_dr + 2 * ((-alpha ** 2 * bar_delta_dr) * np.exp(-alpha ** 2 * bar_delta) +
                                                          (-alpha ** 2 * bar_bigdelta_dr) * np.exp(-alpha ** 2 * bar_bigdelta)) -
                      (-alpha ** 2 * (bar_delta_dr + bar_bigdelta_dr)) * np.exp(-alpha ** 2 * (bar_delta + bar_bigdelta)) -
                      (-alpha ** 2 * (bar_bigdelta_dr - bar_delta_dr)) * np.exp(-alpha ** 2 * (bar_bigdelta - bar_delta))), axis=-1)
    mlnS_dr = mlnS_dr * D0 * g ** 2 * t_ref ** 3 + 3 * mlnS * D0 * g ** 2 * t_ref ** 2 * dt_ref_dr
    mlnS = mlnS * D0 * g ** 2 * t_ref ** 3
    signal = np.exp(-mlnS)
    jacobian = -mlnS_dr * signal
    return signal, jacobian


# PGSE First Pulse on a cylinder Compartment: Derivative of S_s,imper over S_s,imper
# dlog_cylinder_signal_first_pulse
def dlog_cylinder_signal_first_pulse(t, radius, Ds0, q, small_delta):
    # t, big_delta and small_delta in ms, r in microns, D0 in µm²/ms, b in ms/µm²
    # g = np.sqrt(b / (big_delta - small_delta/ 3)) / small_delta  # in 1/µm*ms
    g = q / small_delta  # in 1/µm*ms
    t_ref = radius ** 2 / Ds0
    bar_t = t / t_ref
    bar_t = np.expand_dims(bar_t, axis=-1)
    # Cylindrical roots precomputed
    alpha = np.array([1.841183781340659, 5.331442773525032, 8.536316366346286, 11.706004902592063, 14.863588633909032,
                      18.015527862681804, 21.164369859188788, 24.311326857210776, 27.457050571059245,
                      30.601922972669094, 33.746182898667385, 36.889987409236809, 40.033444053350678,
                      43.17662896544882, 46.319597561173914, 49.462391139702753, 52.605041111556687,
                      55.747571792251009, 58.890002299185703, 62.032347870661987, 65.174620802544453,
                      68.316831125951808, 71.458987105850994, 74.601095613456408, 77.743162408196767,
                      80.885192353878438, 84.027189586293531, 87.169157644540277, 90.311099574903423,
                      93.45301801376003, 96.594915254291138, 99.736793300573908, 102.878653911754455,
                      106.020498638360806, 109.162328852340863, 112.304145772055051, 115.44595048318557,
                      118.587743956319926, 121.729527061810202, 124.871300582387889])
    q2Ds_total = np.divide(2, (alpha ** 4 * (alpha ** 2 - 1))) * (1 - np.exp(-alpha ** 2 * bar_t))
    q2Ds_total = np.sum(q2Ds_total, axis=1)
    q2Ds_total = q2Ds_total * g ** 2 * Ds0 * t_ref ** 2
    return q2Ds_total


def jacobian_dlog_cylinder_signal_first_pulse(t, radius, Ds0, q, small_delta):
    # t, big_delta and small_delta in ms, r in microns, D0 in µm²/ms, b in ms/µm²
    # g = np.sqrt(b / (big_delta - small_delta/ 3)) / small_delta  # in 1/µm*ms
    g = q / small_delta  # in 1/µm*ms
    t_ref = radius ** 2 / Ds0
    bar_t = t / t_ref
    bar_t = np.expand_dims(bar_t, axis=-1)
    # Compute the derivatives of the reference values
    dt_ref_dr = 2 * radius / Ds0
    d_inv_t_ref_dr = -2 * Ds0 / (radius ** 3)
    bar_t_dr = np.expand_dims(t * d_inv_t_ref_dr, axis=-1)
    # Cylindrical roots precomputed
    alpha = np.array([1.841183781340659, 5.331442773525032, 8.536316366346286, 11.706004902592063, 14.863588633909032,
                      18.015527862681804, 21.164369859188788, 24.311326857210776, 27.457050571059245,
                      30.601922972669094, 33.746182898667385, 36.889987409236809, 40.033444053350678,
                      43.17662896544882, 46.319597561173914, 49.462391139702753, 52.605041111556687,
                      55.747571792251009, 58.890002299185703, 62.032347870661987, 65.174620802544453,
                      68.316831125951808, 71.458987105850994, 74.601095613456408, 77.743162408196767,
                      80.885192353878438, 84.027189586293531, 87.169157644540277, 90.311099574903423,
                      93.45301801376003, 96.594915254291138, 99.736793300573908, 102.878653911754455,
                      106.020498638360806, 109.162328852340863, 112.304145772055051, 115.44595048318557,
                      118.587743956319926, 121.729527061810202, 124.871300582387889])
    q2Ds_sum = np.divide(2, (alpha ** 4 * (alpha ** 2 - 1))) * (1 - np.exp(-alpha ** 2 * bar_t))
    q2Ds_sum = np.sum(q2Ds_sum, axis=1)
    der_q2Ds_radius_sum = np.divide(2, (alpha ** 2 * (alpha ** 2 - 1))) * bar_t_dr * np.exp(-alpha ** 2 * bar_t)
    der_q2Ds_radius_sum = np.sum(der_q2Ds_radius_sum, axis=1)
    q2Ds_total = q2Ds_sum * g ** 2 * Ds0 * t_ref ** 2
    der_q2Ds_radius_total = (der_q2Ds_radius_sum * g ** 2 * Ds0 * t_ref ** 2 +
                             2 * q2Ds_sum * g ** 2 * Ds0 * dt_ref_dr * t_ref)
    return q2Ds_total, der_q2Ds_radius_total


# PGSE Second Pulse on a cylinder Compartment : Derivative of S_s,imper over S_s,imper
def dlog_cylinder_signal_second_pulse(t, radius, Ds0, q, big_delta, small_delta):
    # t, big_delta and small_delta in ms, r in microns, D0 in µm²/ms, b in ms/µm²
    # g = np.sqrt(b / (big_delta - small_delta/ 3)) / small_delta  # in 1/µm*ms
    g = q / small_delta  # in 1/µm*ms
    t_ref = radius ** 2 / Ds0  # Compute t_ref
    bar_delta = small_delta / t_ref  # Compute bar_delta
    bar_bigdelta = big_delta / t_ref  # Compute bar_bigdelta
    bar_t = t / t_ref
    bar_t = np.expand_dims(bar_t, axis=-1)
    # Cylindrical roots precomputed
    alpha = np.array([1.841183781340659, 5.331442773525032, 8.536316366346286, 11.706004902592063, 14.863588633909032,
                      18.015527862681804, 21.164369859188788, 24.311326857210776, 27.457050571059245,
                      30.601922972669094, 33.746182898667385, 36.889987409236809, 40.033444053350678,
                      43.17662896544882, 46.319597561173914, 49.462391139702753, 52.605041111556687,
                      55.747571792251009, 58.890002299185703, 62.032347870661987, 65.174620802544453,
                      68.316831125951808, 71.458987105850994, 74.601095613456408, 77.743162408196767,
                      80.885192353878438, 84.027189586293531, 87.169157644540277, 90.311099574903423,
                      93.45301801376003, 96.594915254291138, 99.736793300573908, 102.878653911754455,
                      106.020498638360806, 109.162328852340863, 112.304145772055051, 115.44595048318557,
                      118.587743956319926, 121.729527061810202, 124.871300582387889])
    q2Ds_total = np.divide(2, (alpha ** 4 * (alpha ** 2 - 1))) * (1 - np.exp(-alpha ** 2 * (bar_t - bar_delta)) +
                                                                  np.exp(-alpha ** 2 * bar_t) -
                                                                  np.exp(-alpha ** 2 * (bar_t - bar_bigdelta)))
    q2Ds_total = np.sum(q2Ds_total, axis=1)
    q2Ds_total = q2Ds_total * g ** 2 * Ds0 * t_ref ** 2
    return q2Ds_total


def jacobian_dlog_cylinder_signal_second_pulse(t, radius, Ds0, q, big_delta, small_delta):
    # t, big_delta and small_delta in ms, r in microns, D0 in µm²/ms, b in ms/µm²
    # g = np.sqrt(b / (big_delta - small_delta/ 3)) / small_delta  # in 1/µm*ms
    g = q / small_delta  # in 1/µm*ms
    t_ref = radius ** 2 / Ds0  # Compute t_ref
    bar_delta = np.expand_dims(small_delta / t_ref, axis=-1)  # Compute bar_delta (unitless)
    bar_bigdelta = np.expand_dims(big_delta / t_ref, axis=-1)  # Compute bar_bigdelta (unitless)
    bar_t = np.expand_dims(t / t_ref, axis=-1)
    # Compute the derivatives of the reference values
    dt_ref_dr = 2 * radius / Ds0
    d_inv_t_ref_dr = -2 * Ds0 / (radius ** 3)
    bar_delta_dr = np.expand_dims(small_delta * d_inv_t_ref_dr, axis=-1)
    bar_bigdelta_dr = np.expand_dims(big_delta * d_inv_t_ref_dr, axis=-1)
    bar_t_dr = np.expand_dims(t * d_inv_t_ref_dr, axis=-1)
    # Cylindrical roots precomputed
    alpha = np.array([1.841183781340659, 5.331442773525032, 8.536316366346286, 11.706004902592063, 14.863588633909032,
                      18.015527862681804, 21.164369859188788, 24.311326857210776, 27.457050571059245,
                      30.601922972669094, 33.746182898667385, 36.889987409236809, 40.033444053350678,
                      43.17662896544882, 46.319597561173914, 49.462391139702753, 52.605041111556687,
                      55.747571792251009, 58.890002299185703, 62.032347870661987, 65.174620802544453,
                      68.316831125951808, 71.458987105850994, 74.601095613456408, 77.743162408196767,
                      80.885192353878438, 84.027189586293531, 87.169157644540277, 90.311099574903423,
                      93.45301801376003, 96.594915254291138, 99.736793300573908, 102.878653911754455,
                      106.020498638360806, 109.162328852340863, 112.304145772055051, 115.44595048318557,
                      118.587743956319926, 121.729527061810202, 124.871300582387889])
    q2Ds_sum = np.divide(2, (alpha ** 4 * (alpha ** 2 - 1))) * (1 - np.exp(-alpha ** 2 * (bar_t - bar_delta)) +
                                                                np.exp(-alpha ** 2 * bar_t) -
                                                                np.exp(-alpha ** 2 * (bar_t - bar_bigdelta)))
    q2Ds_sum = np.sum(q2Ds_sum, axis=1)
    der_q2Ds_radius_sum = (np.divide(2, (alpha ** 2 * (alpha ** 2 - 1))) *
                           ((bar_t_dr - bar_delta_dr) * np.exp(-alpha ** 2 * (bar_t - bar_delta)) -
                            bar_t_dr * np.exp(-alpha ** 2 * bar_t) +
                            (bar_t_dr - bar_bigdelta_dr) * np.exp(-alpha ** 2 * (bar_t - bar_bigdelta))))
    der_q2Ds_radius_sum = np.sum(der_q2Ds_radius_sum, axis=1)
    q2Ds_total = q2Ds_sum * g ** 2 * Ds0 * t_ref ** 2
    der_q2Ds_radius_total = (der_q2Ds_radius_sum * g ** 2 * Ds0 * t_ref ** 2 +
                             2 * q2Ds_sum * g ** 2 * Ds0 * dt_ref_dr * t_ref)
    return q2Ds_total, der_q2Ds_radius_total
