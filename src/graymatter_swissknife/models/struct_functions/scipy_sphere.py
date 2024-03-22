import numpy as np


def sphere_murdaycotts(r, D0, b, big_delta, small_delta):
    """
    Calculates diffusion attenuation, mlnS = - ln(S/S0), inside a perfectly reflecting sphere of radius r, free
    diffusion coefficient D0, bvalue b (in IS units of s/m), with pulse width delta and distance big_delta between the fronts
    of the pulses, according to Murday and Cotts, JCP 1968
    Reference value: g = 0.01070 for 40 mT/m

    Derived and optimized from (c) Dmitry Novikov, June 2021
    Quentin Uhl, July 2023

    :param r: The radius of the sphere (in microns)
    :param D0: The diffusion coefficient inside the sphere (in µm²/ms)
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
    bardelta = np.expand_dims(small_delta / t_ref, axis=-1)  # Compute bardelta (unitless)
    bar_bigdelta = np.expand_dims(big_delta / t_ref, axis=-1)  # Compute bar_bigdelta (unitless)
    # Spherical roots precomputed
    alpha = np.array([2.0815759778181, 5.940369990572712, 9.205840142936665, 12.404445021901974, 15.579236410387185,
                      18.742645584774756, 21.899696479492778, 25.052825280992952, 28.203361003952356,
                      31.352091726564478, 34.499514921366952, 37.645960323086392, 40.791655231271882,
                      43.936761471419779, 47.081397412154182, 50.225651649183071, 53.369591820490818,
                      56.513270462198577, 59.656729003527936, 62.800000556519777, 65.94311190465524, 69.086084946645187,
                      72.228937762015434, 75.371685409287323, 78.514340531930912, 81.656913824036792,
                      84.799414392202507, 87.941850039659869, 91.084227491468639, 94.226552574568288,
                      97.368830362900979, 100.511065295271166, 103.653261271734152, 106.795421732944149,
                      109.937549725876437, 113.079647958579201, 116.221718846032573, 119.363764548756691,
                      122.505787005472015, 125.647787960854458, 128.789768989223461, 131.931731514842539,
                      135.07367682938397, 138.215606107009307, 141.357520417436831, 144.49942073730486,
                      147.641307960078819, 150.783182904723986, 153.925046323311989, 157.066898907714517,
                      160.208741295510009, 163.350574075206424, 166.492397790873781, 169.634212946261414,
                      172.776020008465338, 175.917819411202572, 179.059611557740794, 182.201396823524391,
                      185.343175558533943, 188.48494808940859, 191.626714721360713, 194.768475739904318,
                      197.910231412418966])
    # Put the alphas in the last dimension
    alpha_shape = alpha.shape
    for dim in range(np.max(np.ndim(big_delta), np.ndim(small_delta))):
        alpha_shape = (1,) + alpha_shape
    alpha = np.reshape(alpha, alpha_shape)
    # Compute the signal
    mlnS = np.sum((2 / (alpha ** 6 * (alpha ** 2 - 2))) * (-2 + 2 * alpha ** 2 * bardelta +
                                                           2 * (np.exp(-alpha ** 2 * bardelta) +
                                                                np.exp(-alpha ** 2 * bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bardelta + bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bar_bigdelta - bardelta))), axis=-1)
    signal = np.exp(-mlnS * D0 * g ** 2 * t_ref ** 3)
    return signal


def sphere_jacobian(r, D0, b, big_delta, small_delta):
    """
    (c) Quentin Uhl, July 2023

    :param r: The radius of the sphere (in microns)
    :param D0: The diffusion coefficient inside the sphere (in microns^2/ms)
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
    bardelta = np.expand_dims(small_delta / t_ref, axis=-1)  # Compute bardelta (unitless)
    bar_bigdelta = np.expand_dims(big_delta / t_ref, axis=-1)  # Compute bar_bigdelta (unitless)
    # Compute the derivatives of the reference values
    dt_ref_dr = 2 * r / D0
    d_inv_t_ref_dr = -2 * D0 / (r ** 3)
    bardelta_dr = np.expand_dims(small_delta * d_inv_t_ref_dr, axis=-1)
    bar_bigdelta_dr = np.expand_dims(big_delta * d_inv_t_ref_dr, axis=-1)
    # Spherical roots precomputed
    alpha = np.array([2.0815759778181, 5.940369990572712, 9.205840142936665, 12.404445021901974, 15.579236410387185,
                      18.742645584774756, 21.899696479492778, 25.052825280992952, 28.203361003952356,
                      31.352091726564478, 34.499514921366952, 37.645960323086392, 40.791655231271882,
                      43.936761471419779, 47.081397412154182, 50.225651649183071, 53.369591820490818,
                      56.513270462198577, 59.656729003527936, 62.800000556519777, 65.94311190465524, 69.086084946645187,
                      72.228937762015434, 75.371685409287323, 78.514340531930912, 81.656913824036792,
                      84.799414392202507, 87.941850039659869, 91.084227491468639, 94.226552574568288,
                      97.368830362900979, 100.511065295271166, 103.653261271734152, 106.795421732944149,
                      109.937549725876437, 113.079647958579201, 116.221718846032573, 119.363764548756691,
                      122.505787005472015, 125.647787960854458, 128.789768989223461, 131.931731514842539,
                      135.07367682938397, 138.215606107009307, 141.357520417436831, 144.49942073730486,
                      147.641307960078819, 150.783182904723986, 153.925046323311989, 157.066898907714517,
                      160.208741295510009, 163.350574075206424, 166.492397790873781, 169.634212946261414,
                      172.776020008465338, 175.917819411202572, 179.059611557740794, 182.201396823524391,
                      185.343175558533943, 188.48494808940859, 191.626714721360713, 194.768475739904318,
                      197.910231412418966])
    # Put the alphas in the last dimension
    alpha_shape = alpha.shape
    for dim in range(np.max(np.ndim(big_delta), np.ndim(small_delta))):
        alpha_shape = (1,) + alpha_shape
    alpha = np.reshape(alpha, alpha_shape)
    # Compute the signal
    mlnS = np.sum((2 / (alpha ** 6 * (alpha ** 2 - 2))) * (-2 + 2 * alpha ** 2 * bardelta +
                                                           2 * (np.exp(-alpha ** 2 * bardelta) +
                                                                np.exp(-alpha ** 2 * bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bardelta + bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bar_bigdelta - bardelta))), axis=-1)
    mlnS_dr = np.sum((2 / (alpha ** 6 * (alpha ** 2 - 2))) *
                     (2 * alpha ** 2 * bardelta_dr + 2 * ((-alpha ** 2 * bardelta_dr) * np.exp(-alpha ** 2 * bardelta) +
                                                          (-alpha ** 2 * bar_bigdelta_dr) * np.exp(-alpha ** 2 * bar_bigdelta)) -
                      (-alpha ** 2 * (bardelta_dr + bar_bigdelta_dr)) * np.exp(-alpha ** 2 * (bardelta + bar_bigdelta)) -
                      (-alpha ** 2 * (bar_bigdelta_dr - bardelta_dr)) * np.exp(-alpha ** 2 * (bar_bigdelta - bardelta))), axis=-1)
    mlnS_dr = mlnS_dr * D0 * g ** 2 * t_ref ** 3 + 3 * mlnS * D0 * g ** 2 * t_ref ** 2 * dt_ref_dr
    mlnS = mlnS * D0 * g ** 2 * t_ref ** 3
    signal = np.exp(-mlnS)
    jacobian = -mlnS_dr * signal
    return signal, jacobian


# PGSE First Pulse on a Sphere Compartment: Derivative of S_s,imper over S_s,imper
# dlog_sphere_signal_first_pulse
def dlog_sphere_signal_first_pulse(t, radius, Ds0, q, small_delta):
    # t, big_delta and small_delta in ms, r in microns, D0 in µm²/ms, b in ms/µm²
    # g = np.sqrt(b / (big_delta - small_delta/ 3)) / small_delta  # in 1/µm*ms
    g = q / small_delta  # in 1/µm*ms
    t_ref = radius ** 2 / Ds0
    bar_t = t / t_ref
    bar_t = np.expand_dims(bar_t, axis=-1)
    alpha = np.array([[2.0815759778181, 5.940369990572712, 9.205840142936665, 12.404445021901974, 15.579236410387185,
                       18.742645584774756, 21.899696479492778, 25.052825280992952, 28.203361003952356,
                       31.352091726564478, 34.499514921366952, 37.645960323086392, 40.791655231271882,
                       43.936761471419779, 47.081397412154182, 50.225651649183071, 53.369591820490818,
                       56.513270462198577, 59.656729003527936, 62.800000556519777]])
    q2Ds_total = np.divide(2, (alpha ** 4 * (alpha ** 2 - 2))) * (1 - np.exp(-alpha ** 2 * bar_t))
    q2Ds_total = np.sum(q2Ds_total, axis=1)
    q2Ds_total = q2Ds_total * g ** 2 * Ds0 * t_ref ** 2
    return q2Ds_total


def jacobian_dlog_sphere_signal_first_pulse(t, radius, Ds0, q, small_delta):
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
    # Spherical roots precomputed
    alpha = np.array([[2.0815759778181, 5.940369990572712, 9.205840142936665, 12.404445021901974, 15.579236410387185,
                       18.742645584774756, 21.899696479492778, 25.052825280992952, 28.203361003952356,
                       31.352091726564478, 34.499514921366952, 37.645960323086392, 40.791655231271882,
                       43.936761471419779, 47.081397412154182, 50.225651649183071, 53.369591820490818,
                       56.513270462198577, 59.656729003527936, 62.800000556519777]])
    q2Ds_sum = np.divide(2, (alpha ** 4 * (alpha ** 2 - 2))) * (1 - np.exp(-alpha ** 2 * bar_t))
    q2Ds_sum = np.sum(q2Ds_sum, axis=1)
    der_q2Ds_radius_sum = np.divide(2, (alpha ** 2 * (alpha ** 2 - 2))) * bar_t_dr * np.exp(-alpha ** 2 * bar_t)
    der_q2Ds_radius_sum = np.sum(der_q2Ds_radius_sum, axis=1)
    q2Ds_total = q2Ds_sum * g ** 2 * Ds0 * t_ref ** 2
    der_q2Ds_radius_total = (der_q2Ds_radius_sum * g ** 2 * Ds0 * t_ref ** 2 +
                             2 * q2Ds_sum * g ** 2 * Ds0 * dt_ref_dr * t_ref)
    return q2Ds_total, der_q2Ds_radius_total


# PGSE Second Pulse on a Sphere Compartment : Derivative of S_s,imper over S_s,imper
def dlog_sphere_signal_second_pulse(t, radius, Ds0, q, big_delta, small_delta):
    # t, big_delta and small_delta in ms, r in microns, D0 in µm²/ms, b in ms/µm²
    # g = np.sqrt(b / (big_delta - small_delta/ 3)) / small_delta  # in 1/µm*ms
    g = q / small_delta  # in 1/µm*ms
    t_ref = radius ** 2 / Ds0  # Compute t_ref
    bar_delta = small_delta / t_ref  # Compute bardelta
    bar_bigdelta = big_delta / t_ref  # Compute bar_bigdelta
    bar_t = t / t_ref
    if np.ndim(bar_t) != np.ndim(bar_bigdelta):
        bar_t = np.expand_dims(bar_t, axis=-1)
    alpha = np.array([[2.0815759778181, 5.940369990572712, 9.205840142936665, 12.404445021901974, 15.579236410387185,
                       18.742645584774756, 21.899696479492778, 25.052825280992952, 28.203361003952356,
                       31.352091726564478, 34.499514921366952, 37.645960323086392, 40.791655231271882,
                       43.936761471419779, 47.081397412154182, 50.225651649183071, 53.369591820490818,
                       56.513270462198577, 59.656729003527936, 62.800000556519777]])
    q2Ds_total = np.divide(2, (alpha ** 4 * (alpha ** 2 - 2))) * (1 - np.exp(-alpha ** 2 * (bar_t - bar_delta)) +
                                                                  np.exp(-alpha ** 2 * bar_t) -
                                                                  np.exp(-alpha ** 2 * (bar_t - bar_bigdelta)))
    q2Ds_total = np.sum(q2Ds_total, axis=1)
    q2Ds_total = q2Ds_total * g ** 2 * Ds0 * t_ref ** 2
    return q2Ds_total


def jacobian_dlog_sphere_signal_second_pulse(t, radius, Ds0, q, big_delta, small_delta):
    # t, big_delta and small_delta in ms, r in microns, D0 in µm²/ms, b in ms/µm²
    # g = np.sqrt(b / (big_delta - small_delta/ 3)) / small_delta  # in 1/µm*ms
    g = q / small_delta  # in 1/µm*ms
    t_ref = radius ** 2 / Ds0  # Compute t_ref
    bar_delta = np.expand_dims(small_delta / t_ref, axis=-1)  # Compute bardelta (unitless)
    bar_bigdelta = np.expand_dims(big_delta / t_ref, axis=-1)  # Compute bar_bigdelta (unitless)
    bar_t = np.expand_dims(t / t_ref, axis=-1)
    # Compute the derivatives of the reference values
    dt_ref_dr = 2 * radius / Ds0
    d_inv_t_ref_dr = -2 * Ds0 / (radius ** 3)
    bardelta_dr = np.expand_dims(small_delta * d_inv_t_ref_dr, axis=-1)
    bar_bigdelta_dr = np.expand_dims(big_delta * d_inv_t_ref_dr, axis=-1)
    bar_t_dr = np.expand_dims(t * d_inv_t_ref_dr, axis=-1)
    # Spherical roots precomputed
    alpha = np.array([[2.0815759778181, 5.940369990572712, 9.205840142936665, 12.404445021901974, 15.579236410387185,
                       18.742645584774756, 21.899696479492778, 25.052825280992952, 28.203361003952356,
                       31.352091726564478, 34.499514921366952, 37.645960323086392, 40.791655231271882,
                       43.936761471419779, 47.081397412154182, 50.225651649183071, 53.369591820490818,
                       56.513270462198577, 59.656729003527936, 62.800000556519777]])
    q2Ds_sum = np.divide(2, (alpha ** 4 * (alpha ** 2 - 2))) * (1 - np.exp(-alpha ** 2 * (bar_t - bar_delta)) +
                                                                np.exp(-alpha ** 2 * bar_t) -
                                                                np.exp(-alpha ** 2 * (bar_t - bar_bigdelta)))
    q2Ds_sum = np.sum(q2Ds_sum, axis=1)
    der_q2Ds_radius_sum = (np.divide(2, (alpha ** 2 * (alpha ** 2 - 2))) *
                           ((bar_t_dr - bardelta_dr) * np.exp(-alpha ** 2 * (bar_t - bar_delta)) -
                            bar_t_dr * np.exp(-alpha ** 2 * bar_t) +
                            (bar_t_dr - bar_bigdelta_dr) * np.exp(-alpha ** 2 * (bar_t - bar_bigdelta))))
    der_q2Ds_radius_sum = np.sum(der_q2Ds_radius_sum, axis=1)
    q2Ds_total = q2Ds_sum * g ** 2 * Ds0 * t_ref ** 2
    der_q2Ds_radius_total = (der_q2Ds_radius_sum * g ** 2 * Ds0 * t_ref ** 2 +
                             2 * q2Ds_sum * g ** 2 * Ds0 * dt_ref_dr * t_ref)
    return q2Ds_total, der_q2Ds_radius_total


########################################################################################################################
# If you allow Ds0 to vary with the radius, you need to add the following functions
########################################################################################################################
def moving_ds0_jacobian_dlog_sphere_signal_first_pulse(t, radius, Ds0, q, small_delta):
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

    dt_ref_dds0 = -radius ** 2 / Ds0 ** 2
    d_inv_t_ref_dds0 = 1 / radius ** 2
    bar_t_dds0 = np.expand_dims(t * d_inv_t_ref_dds0, axis=-1)
    # Spherical roots precomputed
    alpha = np.array([[2.0815759778181, 5.940369990572712, 9.205840142936665, 12.404445021901974, 15.579236410387185,
                       18.742645584774756, 21.899696479492778, 25.052825280992952, 28.203361003952356,
                       31.352091726564478, 34.499514921366952, 37.645960323086392, 40.791655231271882,
                       43.936761471419779, 47.081397412154182, 50.225651649183071, 53.369591820490818,
                       56.513270462198577, 59.656729003527936, 62.800000556519777]])
    q2Ds_sum = np.divide(2, (alpha ** 4 * (alpha ** 2 - 2))) * (1 - np.exp(-alpha ** 2 * bar_t))
    q2Ds_sum = np.sum(q2Ds_sum, axis=1)

    der_q2Ds_radius_sum = np.divide(2, (alpha ** 2 * (alpha ** 2 - 2))) * bar_t_dr * np.exp(-alpha ** 2 * bar_t)
    der_q2Ds_radius_sum = np.sum(der_q2Ds_radius_sum, axis=1)

    der_q2Ds_ds0_sum = np.divide(2, (alpha ** 2 * (alpha ** 2 - 2))) * bar_t_dds0 * np.exp(-alpha ** 2 * bar_t)
    der_q2Ds_ds0_sum = np.sum(der_q2Ds_ds0_sum, axis=1)

    q2Ds_total = q2Ds_sum * g ** 2 * Ds0 * t_ref ** 2
    der_q2Ds_radius_total = (der_q2Ds_radius_sum * g ** 2 * Ds0 * t_ref ** 2 +
                             2 * q2Ds_sum * g ** 2 * Ds0 * dt_ref_dr * t_ref)
    der_q2Ds_ds0_total = (der_q2Ds_ds0_sum * g ** 2 * Ds0 * t_ref ** 2 +
                          q2Ds_sum * g ** 2 * t_ref ** 2 +
                          2 * q2Ds_sum * g ** 2 * Ds0 * t_ref * dt_ref_dds0)
    return q2Ds_total, der_q2Ds_radius_total, der_q2Ds_ds0_total


def moving_ds0_jacobian_dlog_sphere_signal_second_pulse(t, radius, Ds0, q, big_delta, small_delta):
    # t, big_delta and small_delta in ms, r in microns, D0 in µm²/ms, b in ms/µm²
    # g = np.sqrt(b / (big_delta - small_delta/ 3)) / small_delta  # in 1/µm*ms
    g = q / small_delta  # in 1/µm*ms
    t_ref = radius ** 2 / Ds0  # Compute t_ref
    bar_delta = np.expand_dims(small_delta / t_ref, axis=-1)  # Compute bardelta (unitless)
    bar_bigdelta = np.expand_dims(big_delta / t_ref, axis=-1)  # Compute bar_bigdelta (unitless)
    bar_t = np.expand_dims(t / t_ref, axis=-1)
    # Compute the derivatives of the reference values
    dt_ref_dr = 2 * radius / Ds0
    d_inv_t_ref_dr = -2 * Ds0 / (radius ** 3)
    bardelta_dr = np.expand_dims(small_delta * d_inv_t_ref_dr, axis=-1)
    bar_bigdelta_dr = np.expand_dims(big_delta * d_inv_t_ref_dr, axis=-1)
    bar_t_dr = np.expand_dims(t * d_inv_t_ref_dr, axis=-1)

    dt_ref_dds0 = -radius ** 2 / Ds0 ** 2
    d_inv_t_ref_dds0 = 1 / radius ** 2
    bardelta_dds0 = np.expand_dims(small_delta * d_inv_t_ref_dds0, axis=-1)
    bar_bigdelta_dds0 = np.expand_dims(big_delta * d_inv_t_ref_dds0, axis=-1)
    bar_t_dds0 = np.expand_dims(t * d_inv_t_ref_dds0, axis=-1)
    # Spherical roots precomputed
    alpha = np.array([[2.0815759778181, 5.940369990572712, 9.205840142936665, 12.404445021901974, 15.579236410387185,
                       18.742645584774756, 21.899696479492778, 25.052825280992952, 28.203361003952356,
                       31.352091726564478, 34.499514921366952, 37.645960323086392, 40.791655231271882,
                       43.936761471419779, 47.081397412154182, 50.225651649183071, 53.369591820490818,
                       56.513270462198577, 59.656729003527936, 62.800000556519777]])
    q2Ds_sum = np.divide(2, (alpha ** 4 * (alpha ** 2 - 2))) * (1 - np.exp(-alpha ** 2 * (bar_t - bar_delta)) +
                                                                np.exp(-alpha ** 2 * bar_t) -
                                                                np.exp(-alpha ** 2 * (bar_t - bar_bigdelta)))
    q2Ds_sum = np.sum(q2Ds_sum, axis=1)

    der_q2Ds_radius_sum = (np.divide(2, (alpha ** 2 * (alpha ** 2 - 2))) *
                           ((bar_t_dr - bardelta_dr) * np.exp(-alpha ** 2 * (bar_t - bar_delta)) -
                            bar_t_dr * np.exp(-alpha ** 2 * bar_t) +
                            (bar_t_dr - bar_bigdelta_dr) * np.exp(-alpha ** 2 * (bar_t - bar_bigdelta))))
    der_q2Ds_radius_sum = np.sum(der_q2Ds_radius_sum, axis=1)

    der_q2Ds_ds0_sum = (np.divide(2, (alpha ** 2 * (alpha ** 2 - 2))) *
                        ((bar_t_dds0 - bardelta_dds0) * np.exp(-alpha ** 2 * (bar_t - bar_delta)) -
                         bar_t_dds0 * np.exp(-alpha ** 2 * bar_t) +
                         (bar_t_dds0 - bar_bigdelta_dds0) * np.exp(-alpha ** 2 * (bar_t - bar_bigdelta))))
    der_q2Ds_ds0_sum = np.sum(der_q2Ds_ds0_sum, axis=1)

    q2Ds_total = q2Ds_sum * g ** 2 * Ds0 * t_ref ** 2
    der_q2Ds_radius_total = (der_q2Ds_radius_sum * g ** 2 * Ds0 * t_ref ** 2 +
                             2 * q2Ds_sum * g ** 2 * Ds0 * dt_ref_dr * t_ref)
    der_q2Ds_ds0_total = (der_q2Ds_ds0_sum * g ** 2 * Ds0 * t_ref ** 2 +
                          q2Ds_sum * g ** 2 * t_ref ** 2 +
                          2 * q2Ds_sum * g ** 2 * Ds0 * t_ref * dt_ref_dds0)
    return q2Ds_total, der_q2Ds_radius_total, der_q2Ds_ds0_total
