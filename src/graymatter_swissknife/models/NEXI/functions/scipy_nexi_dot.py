"""
Apache-2.0 license

Copyright (c) 2023 Quentin Uhl, Ileana O. Jelescu

Please cite 
"Quantifying human gray matter microstructure using neurite exchange imaging (NEXI) and 300 mT/m gradients"
Quentin Uhl, Tommaso Pavan, Malwina Molendowska, Derek K. Jones, Marco Palombo, Ileana Ozana Jelescu
Imaging Neuroscience (2024) 2: 1-19.
https://doi.org/10.1162/imag_a_00104
"""

# Implementation of the Nexi Dot model in scipy
import numpy as np
import scipy.integrate

# utilitary functions for jacobian and MSE jacobian
broad4 = lambda matrix: np.tile(matrix[..., np.newaxis], 4)


#######################################################################################################################
# Nexi Dot Model
#######################################################################################################################


def M_dot(x, b, t, tex, Di, De, Dp, f, f_dot):
    D1 = Di * b * (x ** 2)
    D2 = Dp * b + (De - Dp) * b * (x ** 2)
    Discr = np.square((t / tex) * (1 - 2 * f) + D1 - D2) + 4 * np.square(t / tex) * f * (1 - f)
    Discr_12 = np.sqrt(Discr)  # Discriminant ** (1/2)
    lp = (t / tex + D1 + D2) / 2 + Discr_12 / 2
    lm = (t / tex + D1 + D2) / 2 - Discr_12 / 2
    Pp = np.divide(f * D1 + (1 - f) * D2 - lm, Discr_12)
    Pm = 1 - Pp
    return (1-f_dot) * (Pp * np.exp(- lp) + Pm * np.exp(- lm)) + f_dot


def nexi_dot_signal(tex, Di, De, Dp, f, f_dot, b, t):
    if tex == 0:
        return np.ones_like(b)
    else:
        return scipy.integrate.quad_vec(lambda x: M_dot(x, b, t, tex, Di, De, Dp, f, f_dot), 0, 1, epsabs=1e-14)[0]


nexi_dot_signal_from_vector = lambda param, b, t: nexi_dot_signal(param[0], param[1], param[2], param[2], param[3], param[4], b, t)  # [tex, Di, De, De, f, f_dot]


#######################################################################################################################
# Nexi Dot jacobian
#######################################################################################################################


def M_dot_jac(x, b, t, tex, Di, De, Dp, f, f_dot):
    D1 = Di * b * (x ** 2)
    D2 = Dp * b + (De - Dp) * b * (x ** 2)
    Discr = np.square((t / tex) * (1 - 2 * f) + D1 - D2) + 4 * np.square(t / tex) * f * (1 - f)
    Discr_12 = np.sqrt(Discr)  # Discr ** (1/2)
    Discr_32 = Discr * Discr_12  # Discr ** (3/2)
    lp = (t / tex + D1 + D2) / 2 + Discr_12 / 2
    lm = (t / tex + D1 + D2) / 2 - Discr_12 / 2
    Pp = np.divide(f * D1 + (1 - f) * D2 - lm, Discr_12)
    Pm = 1 - Pp
    M = Pp * np.exp(- lp) + Pm * np.exp(- lm)

    # Derivatives of t_ex
    Discr_tex = ((t / tex) * (1 - 2 * f) + D1 - D2) * (-2 * (1 - 2 * f) * t / (tex ** 2)) - 8 * np.square(t) * f * (
            1 - f) / (tex ** 3)
    lp_tex = -(t / (tex ** 2)) / 2 + np.divide(Discr_tex, Discr_12) / 4
    lm_tex = -(t / (tex ** 2)) / 2 - np.divide(Discr_tex, Discr_12) / 4
    Pp_tex = np.divide(- lm_tex, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_tex, 2 * Discr_32)
    # Derivatives of Di
    D1_Di = b * (x ** 2)
    Discr_Di = 2 * D1_Di * ((t / tex) * (1 - 2 * f) + D1 - D2)
    lp_Di = D1_Di / 2 + np.divide(Discr_Di, Discr_12) / 4
    lm_Di = D1_Di / 2 - np.divide(Discr_Di, Discr_12) / 4
    Pp_Di = np.divide(f * D1_Di - lm_Di, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_Di, 2 * Discr_32)
    # Derivatives of De
    D2_De = b
    Discr_De = -2 * D2_De * ((t / tex) * (1 - 2 * f) + D1 - D2)
    lp_De = D2_De / 2 + np.divide(Discr_De, Discr_12) / 4
    lm_De = D2_De / 2 - np.divide(Discr_De, Discr_12) / 4
    Pp_De = np.divide((1 - f) * D2_De - lm_De, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_De,
                                                                     2 * Discr_32)
    # Derivatives of f
    Discr_f = np.multiply(D1 - D2, -4 * t / tex)
    lp_f = np.divide(Discr_f, Discr_12) / 4
    lm_f = -np.divide(Discr_f, Discr_12) / 4
    Pp_f = np.divide(D1 - D2 - lm_f, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_f, 2 * Discr_32)

    # Regroup jacobians
    lp_jac = np.zeros(b.shape + (4,))
    lp_jac[..., 0], lp_jac[..., 1], lp_jac[..., 2], lp_jac[..., 3] = lp_tex, lp_Di, lp_De, lp_f
    lm_jac = np.zeros(b.shape + (4,))
    lm_jac[..., 0], lm_jac[..., 1], lm_jac[..., 2], lm_jac[..., 3] = lm_tex, lm_Di, lm_De, lm_f
    Pp_jac = np.zeros(b.shape + (4,))
    Pp_jac[..., 0], Pp_jac[..., 1], Pp_jac[..., 2], Pp_jac[..., 3] = Pp_tex, Pp_Di, Pp_De, Pp_f

    Pm_jac = - Pp_jac

    Pp4, Pm4, lm4, lp4 = broad4(Pp), broad4(Pm), broad4(lm), broad4(lp)

    M_dot_jac_vector = Pp_jac * np.exp(-lp4) - Pp4 * lp_jac * np.exp(-lp4) + Pm_jac * np.exp(
        -lm4) - Pm4 * lm_jac * np.exp(-lm4)

    desired_shape = list(M_dot_jac_vector.shape)
    desired_shape[-1] += 1
    M_dot_jac_vector_with_dot = np.empty(desired_shape)
    M_dot_jac_vector_with_dot[..., 0:4] = (1-f_dot) * M_dot_jac_vector
    M_dot_jac_vector_with_dot[..., 4] = np.ones_like(b) - M
    return M_dot_jac_vector_with_dot


# nexi_dot_jac=lambda b,t,tex,Di,De,Dp,f: scipy.integrate.quad_vec(lambda x: M_dot_jac(x,b,t,tex,Di,De,Dp,f),0,1, epsabs = 1e-14)[0]


def nexi_dot_jacobian(tex, Di, De, Dp, f, f_dot, b, t):
    if tex == 0:
        nexi_dot_jacobian_array = broad4(np.ones_like(b))
        nexi_dot_jacobian_array[..., 0] = -np.divide(f * Di * b / 3 + (1 - f) * Dp * b, t)
        nexi_dot_jacobian_array[..., 1] = -b * f / 3
        nexi_dot_jacobian_array[..., 2] = -b * (1 - f)
        nexi_dot_jacobian_array[..., 3] = b * (De - Di / 3)
        nexi_dot_jacobian_array[..., 0:4] = (1 - f_dot) * nexi_dot_jacobian_array[..., 0:4]
        nexi_dot_jacobian_array[..., 4] = 0
        return nexi_dot_jacobian_array
    else:
        return scipy.integrate.quad_vec(lambda x: M_dot_jac(x, b, t, tex, Di, De, Dp, f, f_dot), 0, 1, epsabs=1e-14)[0]


nexi_dot_jacobian_from_vector = lambda param, b, t: nexi_dot_jacobian(param[0], param[1], param[2], param[2], param[3], param[4], b, t)  # [tex, Di, De, De, f, f_dot]


#######################################################################################################################
# Optimized Nexi Dot for computation of MSE jacobian
#######################################################################################################################


broad5 = lambda matrix: np.tile(matrix[..., np.newaxis], 5)


def nexi_dot_optimized_mse_jacobian(param, b, td, Y, b_td_dimensions=2):
    nexi_dot_vec_jac_concatenation = nexi_dot_jacobian_concatenated_from_vector(param, b, td)
    nexi_dot_vec = nexi_dot_vec_jac_concatenation[..., 0]
    nexi_dot_vec_jac = nexi_dot_vec_jac_concatenation[..., 1:6]
    if b_td_dimensions == 1:
        mse_jacobian = np.sum(2 * nexi_dot_vec_jac * broad5(nexi_dot_vec - Y), axis=0)
    elif b_td_dimensions == 2:
        mse_jacobian = np.sum(2 * nexi_dot_vec_jac * broad5(nexi_dot_vec - Y), axis=(0, 1))
    else:
        raise NotImplementedError
    return mse_jacobian


def M_dot_jac_concat(x, b, t, tex, Di, De, Dp, f, f_dot):
    D1 = Di * b * (x ** 2)
    D2 = Dp * b + (De - Dp) * b * (x ** 2)
    Discr = np.square((t / tex) * (1 - 2 * f) + D1 - D2) + 4 * np.square(t / tex) * f * (1 - f)
    Discr_12 = np.sqrt(Discr)  # Discr ** (1/2)
    Discr_32 = Discr * Discr_12  # Discr ** (3/2)
    lp = (t / tex + D1 + D2) / 2 + Discr_12 / 2
    lm = (t / tex + D1 + D2) / 2 - Discr_12 / 2
    Pp = np.divide(f * D1 + (1 - f) * D2 - lm, Discr_12)
    Pm = 1 - Pp
    M = Pp * np.exp(- lp) + Pm * np.exp(- lm)

    # Derivatives of t_ex
    Discr_tex = ((t / tex) * (1 - 2 * f) + D1 - D2) * (-2 * (1 - 2 * f) * t / (tex ** 2)) - 8 * np.square(t) * f * (
            1 - f) / (tex ** 3)
    lp_tex = -(t / (tex ** 2)) / 2 + np.divide(Discr_tex, Discr_12) / 4
    lm_tex = -(t / (tex ** 2)) / 2 - np.divide(Discr_tex, Discr_12) / 4
    Pp_tex = np.divide(- lm_tex, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_tex, 2 * Discr_32)
    # Derivatives of Di
    D1_Di = b * (x ** 2)
    Discr_Di = 2 * D1_Di * ((t / tex) * (1 - 2 * f) + D1 - D2)
    lp_Di = D1_Di / 2 + np.divide(Discr_Di, Discr_12) / 4
    lm_Di = D1_Di / 2 - np.divide(Discr_Di, Discr_12) / 4
    Pp_Di = np.divide(f * D1_Di - lm_Di, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_Di, 2 * Discr_32)
    # Derivatives of De
    D2_De = b
    Discr_De = -2 * D2_De * ((t / tex) * (1 - 2 * f) + D1 - D2)
    lp_De = D2_De / 2 + np.divide(Discr_De, Discr_12) / 4
    lm_De = D2_De / 2 - np.divide(Discr_De, Discr_12) / 4
    Pp_De = np.divide((1 - f) * D2_De - lm_De, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_De,
                                                                     2 * Discr_32)
    # Derivatives of f
    Discr_f = np.multiply(D1 - D2, -4 * t / tex)
    lp_f = np.divide(Discr_f, Discr_12) / 4
    lm_f = -np.divide(Discr_f, Discr_12) / 4
    Pp_f = np.divide(D1 - D2 - lm_f, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_f, 2 * Discr_32)

    # Regroup jacobians
    lp_jac = np.zeros(b.shape + (4,))
    lp_jac[..., 0], lp_jac[..., 1], lp_jac[..., 2], lp_jac[..., 3] = lp_tex, lp_Di, lp_De, lp_f
    lm_jac = np.zeros(b.shape + (4,))
    lm_jac[..., 0], lm_jac[..., 1], lm_jac[..., 2], lm_jac[..., 3] = lm_tex, lm_Di, lm_De, lm_f
    Pp_jac = np.zeros(b.shape + (4,))
    Pp_jac[..., 0], Pp_jac[..., 1], Pp_jac[..., 2], Pp_jac[..., 3] = Pp_tex, Pp_Di, Pp_De, Pp_f

    Pm_jac = - Pp_jac

    Pp4, Pm4, lm4, lp4 = broad4(Pp), broad4(Pm), broad4(lm), broad4(lp)

    M_dot_jac_vector = Pp_jac * np.exp(-lp4) - Pp4 * lp_jac * np.exp(-lp4) + Pm_jac * np.exp(
        -lm4) - Pm4 * lm_jac * np.exp(-lm4)

    desired_shape = list(M_dot_jac_vector.shape)
    desired_shape[-1] += 2
    M_dot_concat = np.empty(desired_shape)
    M_dot_concat[..., 0] = (1-f_dot) * M + f_dot
    M_dot_concat[..., 1:5] = (1-f_dot) * M_dot_jac_vector
    M_dot_concat[..., 5] = np.ones_like(M) - M

    return M_dot_concat


def nexi_dot_jacobian_concatenated(tex, Di, De, Dp, f, f_dot, b, t):
    if tex == 0:
        nexi_dot_jacobian_concat = np.tile(np.ones_like(b)[..., np.newaxis], 6)
        # nexi_dot_jacobian_concat[..., 0] = 1.0 -> already in np.ones
        nexi_dot_jacobian_concat[..., 1] = -np.divide(f * Di * b / 3 + (1 - f) * Dp * b, t)
        nexi_dot_jacobian_concat[..., 2] = -b * f / 3
        nexi_dot_jacobian_concat[..., 3] = -b * (1 - f)
        nexi_dot_jacobian_concat[..., 4] = b * (De - Di / 3)
        nexi_dot_jacobian_concat[..., 1:5] = (1 - f_dot) * nexi_dot_jacobian_concat[..., 1:5]
        nexi_dot_jacobian_concat[..., 5] = 0
        return nexi_dot_jacobian_concat
    else:
        return \
            scipy.integrate.quad_vec(lambda x: M_dot_jac_concat(x, b, t, tex, Di, De, Dp, f, f_dot), 0, 1, epsabs=1e-14)[0]


nexi_dot_jacobian_concatenated_from_vector = lambda param, b, t: nexi_dot_jacobian_concatenated(param[0], param[1], param[2], param[2],
                                                                                            param[3], param[4], b, t)  # [tex, Di, De, De, f, bias]
