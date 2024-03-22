"""
Apache-2.0 license

Copyright (c) 2023 Quentin Uhl, Ileana O. Jelescu

Please cite 
"Quantifying human gray matter microstructure using neurite exchange imaging (NEXI) and 300 mT/m gradients"
Quentin Uhl, Tommaso Pavan, Malwina Molendowska, Derek K. Jones, Marco Palombo, Ileana Ozana Jelescu
Imaging Neuroscience (2024) 2: 1-19.
https://doi.org/10.1162/imag_a_00104
"""

# Implementation of the Neurite Exchange Imaging model in scipy (Narrow Pulse approximation)

import numpy as np
import scipy.integrate

# utilitary functions for jacobian, hessian and MSE jacobian
broad4 = lambda matrix: np.tile(matrix[..., np.newaxis], 4)
broad4T = lambda matrix: np.repeat(matrix[..., np.newaxis, :], 4, -2)  # Use only on jacobians !
broad44 = lambda matrix: np.tile(broad4(matrix)[..., np.newaxis], 4)


#######################################################################################################################
# Nexi Model
#######################################################################################################################


def M(x, b, t, tex, Di, De, Dp, f):
    """
    The integrand inside the NEXI signal integral
    :param x: the direction of the gradient
    :param b: the b value
    :param t: the diffusion time
    :param tex: the exchange time
    :param Di: the intra-neurite diffusivity
    :param De: the extra-neurite diffusivity
    :param Dp: the extra-neurite diffusivity perpendicular to the neurites (usually equal to De)
    :param f: the volume fraction of neurites
    :return: the integrand
    """
    D1 = Di * b * (x ** 2)
    D2 = Dp * b + (De - Dp) * b * (x ** 2)
    Discr = np.square((t / tex) * (1 - 2 * f) + D1 - D2) + 4 * np.square(t / tex) * f * (1 - f)
    Discr_12 = np.sqrt(Discr)  # Discriminant ** (1/2)
    lp = (t / tex + D1 + D2) / 2 + Discr_12 / 2
    lm = (t / tex + D1 + D2) / 2 - Discr_12 / 2
    Pp = np.divide(f * D1 + (1 - f) * D2 - lm, Discr_12)
    Pm = 1 - Pp
    return Pp * np.exp(-lp) + Pm * np.exp(-lm)


def nexi_signal(tex, Di, De, Dp, f, b, t):
    """
    The NEXI signal
    :param tex: the exchange time
    :param Di: the intra-neurite diffusivity
    :param De: the extra-neurite diffusivity
    :param Dp: the extra-neurite diffusivity perpendicular to the neurites (usually equal to De)
    :param f: the volume fraction of neurites
    :param b: the b value
    :param t: the diffusion time
    :return: the Neurite Exchange Imaging signal
    """
    if tex == 0:
        return np.ones_like(b)
    else:
        return scipy.integrate.quad_vec(lambda x: M(x, b, t, tex, Di, De, Dp, f), 0, 1, epsabs=1e-14)[0]


nexi_signal_from_vector = lambda param, b, t: nexi_signal(
    param[0], param[1], param[2], param[2], param[3], b, t
)  # [tex, Di, De, De, f] since De,parallel = De,perpendicular


#######################################################################################################################
# Nexi jacobian
#######################################################################################################################


def M_jac(x, b, t, tex, Di, De, Dp, f):
    D1 = Di * b * (x ** 2)
    D2 = Dp * b + (De - Dp) * b * (x ** 2)
    Discr = np.square((t / tex) * (1 - 2 * f) + D1 - D2) + 4 * np.square(t / tex) * f * (1 - f)
    Discr_12 = np.sqrt(Discr)  # Discr ** (1/2)
    Discr_32 = Discr * Discr_12  # Discr ** (3/2)
    lp = (t / tex + D1 + D2) / 2 + Discr_12 / 2
    lm = (t / tex + D1 + D2) / 2 - Discr_12 / 2
    Pp = np.divide(f * D1 + (1 - f) * D2 - lm, Discr_12)
    Pm = 1 - Pp

    # Derivatives of t_ex
    Discr_tex = ((t / tex) * (1 - 2 * f) + D1 - D2) * (-2 * (1 - 2 * f) * t / (tex ** 2)) - 8 * np.square(t) * f * (
        1 - f
    ) / (tex ** 3)
    lp_tex = -(t / (tex ** 2)) / 2 + np.divide(Discr_tex, Discr_12) / 4
    lm_tex = -(t / (tex ** 2)) / 2 - np.divide(Discr_tex, Discr_12) / 4
    Pp_tex = np.divide(-lm_tex, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_tex, 2 * Discr_32)
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
    Pp_De = np.divide((1 - f) * D2_De - lm_De, Discr_12) - np.divide(
        (f * D1 + (1 - f) * D2 - lm) * Discr_De, 2 * Discr_32
    )
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

    Pm_jac = -Pp_jac

    Pp4, Pm4, lm4, lp4 = broad4(Pp), broad4(Pm), broad4(lm), broad4(lp)

    return Pp_jac * np.exp(-lp4) - Pp4 * lp_jac * np.exp(-lp4) + Pm_jac * np.exp(-lm4) - Pm4 * lm_jac * np.exp(-lm4)


def nexi_jacobian(tex, Di, De, Dp, f, b, t):
    """
    The NEXI signal jacobian
    :param tex: the exchange time
    :param Di: the intra-neurite diffusivity
    :param De: the extra-neurite diffusivity
    :param Dp: the extra-neurite diffusivity perpendicular to the neurites (usually equal to De)
    :param f: the volume fraction of neurites
    :param b: the b value
    :param t: the diffusion time
    :return: the Neurite Exchange Imaging jacobian
    """
    if tex == 0:
        nexi_jacobian_array = broad4(np.ones_like(b))
        nexi_jacobian_array[..., 0] = -np.divide(f * Di * b / 3 + (1 - f) * Dp * b, t)
        nexi_jacobian_array[..., 1] = -b * f / 3
        nexi_jacobian_array[..., 2] = -b * (1 - f)
        nexi_jacobian_array[..., 3] = b * (De - Di / 3)
        return nexi_jacobian_array
    else:
        return scipy.integrate.quad_vec(lambda x: M_jac(x, b, t, tex, Di, De, Dp, f), 0, 1, epsabs=1e-14)[0]


nexi_jacobian_from_vector = lambda param, b, t: nexi_jacobian(
    param[0], param[1], param[2], param[2], param[3], b, t
)  # [tex, Di, De, De, f]


#######################################################################################################################
# Optimized Nexi for computation of MSE jacobian
#######################################################################################################################

# Optimisation of the optimisation function derivatives in scipy
# Only relevant function are nexi_optimized_mse_jacobian and nexi_optimized_mse_hessian. Others are subfunction to simplify computation


def nexi_optimized_mse_jacobian(param, b, td, Y, b_td_dimensions=2):
    '''
    Compute the jacobian of the MSE of the NEXI model with respect to the parameters
    :param param: [tex, Di, De, f]
    :param b: b-values
    :param td: diffusion times
    :param Y: ground truth signal
    :param b_td_dimensions: 1 if b and td are 1D arrays, 2 if b and td are 2D arrays
    :return: jacobian of the MSE of the NEXI model with respect to the parameters
    '''
    nexi_vec_jac_concatenation = nexi_jacobian_concatenated_from_vector(param, b, td)
    nexi_signal_vec = nexi_vec_jac_concatenation[..., 0]
    nexi_vec_jac = nexi_vec_jac_concatenation[..., 1:5]
    if b_td_dimensions == 1:
        mse_jacobian = np.sum(2 * nexi_vec_jac * broad4(nexi_signal_vec - Y), axis=0)
    elif b_td_dimensions == 2:
        mse_jacobian = np.sum(2 * nexi_vec_jac * broad4(nexi_signal_vec - Y), axis=(0, 1))
    else:
        raise NotImplementedError
    return mse_jacobian


def nexi_optimized_mse_hessian(param, b, td, Y, b_td_dimensions=2):
    nexi_vec_hess_concatenation = nexi_hessian_concatenated_from_vector(param, b, td)
    nexi_signal_vec = nexi_vec_hess_concatenation[..., 0, 0]
    nexi_vec_jac = nexi_vec_hess_concatenation[..., 1]
    nexi_vec_hess = nexi_vec_hess_concatenation[..., 2:6]
    if b_td_dimensions == 1:
        mse_hessian = np.sum(
            2 * nexi_vec_hess * broad44(nexi_signal_vec - Y) + 2 * broad4(nexi_vec_jac) * broad4T(nexi_vec_jac),
            axis=(0, 1),
        )
    elif b_td_dimensions == 2:
        mse_hessian = np.sum(
            2 * nexi_vec_hess * broad44(nexi_signal_vec - Y) + 2 * broad4(nexi_vec_jac) * broad4T(nexi_vec_jac), axis=0
        )
    else:
        raise NotImplementedError
    return mse_hessian


#######################################################################################################################
# Optimised concatenated NEXI jacobian
#######################################################################################################################


def M_jac_concat(x, b, t, tex, Di, De, Dp, f):
    D1 = Di * b * (x ** 2)
    D2 = Dp * b + (De - Dp) * b * (x ** 2)
    Discr = np.square((t / tex) * (1 - 2 * f) + D1 - D2) + 4 * np.square(t / tex) * f * (1 - f)
    Discr_12 = np.sqrt(Discr)  # Discr ** (1/2)
    Discr_32 = Discr * Discr_12  # Discr ** (3/2)
    lp = (t / tex + D1 + D2) / 2 + Discr_12 / 2
    lm = (t / tex + D1 + D2) / 2 - Discr_12 / 2
    Pp = np.divide(f * D1 + (1 - f) * D2 - lm, Discr_12)
    Pm = 1 - Pp

    # Derivatives of t_ex
    Discr_tex = ((t / tex) * (1 - 2 * f) + D1 - D2) * (-2 * (1 - 2 * f) * t / (tex ** 2)) - 8 * np.square(t) * f * (
        1 - f
    ) / (tex ** 3)
    lp_tex = -(t / (tex ** 2)) / 2 + np.divide(Discr_tex, Discr_12) / 4
    lm_tex = -(t / (tex ** 2)) / 2 - np.divide(Discr_tex, Discr_12) / 4
    Pp_tex = np.divide(-lm_tex, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_tex, 2 * Discr_32)
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
    Pp_De = np.divide((1 - f) * D2_De - lm_De, Discr_12) - np.divide(
        (f * D1 + (1 - f) * D2 - lm) * Discr_De, 2 * Discr_32
    )
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

    Pm_jac = -Pp_jac

    Pp4, Pm4, lm4, lp4 = broad4(Pp), broad4(Pm), broad4(lm), broad4(lp)

    M_jac = Pp_jac * np.exp(-lp4) - Pp4 * lp_jac * np.exp(-lp4) + Pm_jac * np.exp(-lm4) - Pm4 * lm_jac * np.exp(-lm4)

    # Adding lines not required in the sole hessian calculation starting from here

    # Compute Original integrand
    M = Pp * np.exp(-lp) + Pm * np.exp(-lm)

    # Concatenating M4, M_jac and nexi_hess to be integered together
    # M4 = M_concat[0] , M_jac = M_concat[1] and M_hess = M_concat[2:6]
    desired_shape = list(M_jac.shape)
    desired_shape[-1] += 1
    M_concat = np.empty(desired_shape)
    M_concat[..., 0] = M
    M_concat[..., 1:5] = M_jac
    return M_concat


def nexi_jacobian_concatenated(tex, Di, De, Dp, f, b, t):
    """
    The NEXI signal together with its jacobian with respect to the microstructure parameters
    :param tex: the exchange time
    :param Di: the intra-neurite diffusivity
    :param De: the extra-neurite diffusivity
    :param Dp: the extra-neurite diffusivity perpendicular to the neurites (usually equal to De)
    :param f: the volume fraction of neurites
    :param b: the b value
    :param t: the diffusion time
    :return: the Neurite Exchange Imaging signal concatenated with its jacobian with respect to the
    microstructure parameters
    """
    if tex == 0:
        nexi_jacobian_concat = np.tile(np.ones_like(b)[..., np.newaxis], 5)
        # nexi_jacobian_concat[..., 0] = 1.0 -> already in np.ones
        nexi_jacobian_concat[..., 1] = -np.divide(f * Di * b / 3 + (1 - f) * Dp * b, t)
        nexi_jacobian_concat[..., 2] = -b * f / 3
        nexi_jacobian_concat[..., 3] = -b * (1 - f)
        nexi_jacobian_concat[..., 4] = b * (De - Di / 3)
        return nexi_jacobian_concat
    else:
        return scipy.integrate.quad_vec(lambda x: M_jac_concat(x, b, t, tex, Di, De, Dp, f), 0, 1, epsabs=1e-14)[0]


nexi_jacobian_concatenated_from_vector = lambda param, b, t: nexi_jacobian_concatenated(
    param[0], param[1], param[2], param[2], param[3], b, t
)  # [tex, Di, De, De, f] since De,parallel = De,perpendicular


#######################################################################################################################
# Nexi Hessian
#######################################################################################################################


def M_hess(x, b, t, tex, Di, De, Dp, f):
    D1 = Di * b * (x ** 2)
    D2 = Dp * b + (De - Dp) * b * (x ** 2)
    Discr = np.square((t / tex) * (1 - 2 * f) + D1 - D2) + 4 * np.square(t / tex) * f * (1 - f)
    Discr_12 = np.sqrt(Discr)  # Discr ** (1/2)
    Discr_32 = Discr * Discr_12  # Discr ** (3/2)
    Discr_52 = Discr * Discr * np.sqrt(Discr)  # Discr ** (5/2)

    lp = (t / tex + D1 + D2) / 2 + Discr_12 / 2
    lm = (t / tex + D1 + D2) / 2 - Discr_12 / 2
    Pp = np.divide(f * D1 + (1 - f) * D2 - lm, Discr_12)
    Pm = 1 - Pp

    # Derivatives of t_ex
    Discr_tex = ((t / tex) * (1 - 2 * f) + D1 - D2) * (-2 * (1 - 2 * f) * t / (tex ** 2)) - 8 * np.square(t) * f * (
        1 - f
    ) / (tex ** 3)
    lp_tex = -(t / (tex ** 2)) / 2 + np.divide(Discr_tex, Discr_12) / 4
    lm_tex = -(t / (tex ** 2)) / 2 - np.divide(Discr_tex, Discr_12) / 4
    Pp_tex = np.divide(-lm_tex, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_tex, 2 * Discr_32)
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
    Pp_De = np.divide((1 - f) * D2_De - lm_De, Discr_12) - np.divide(
        (f * D1 + (1 - f) * D2 - lm) * Discr_De, 2 * Discr_32
    )
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

    Pm_jac = -Pp_jac

    # Hessian of Discr
    Discr_tex_tex = (
        2 * np.square(-t / (tex ** 2) * (1 - 2 * f))
        + 2 * ((t / tex) * (1 - 2 * f) + D1 - D2) * (2 * t / (tex ** 3) * (1 - 2 * f))
        + 24 * np.square(t / (tex ** 2)) * f * (1 - f)
    )
    Discr_tex_Di = 2 * D1_Di * (-(t / (tex ** 2)) * (1 - 2 * f))
    Discr_tex_De = -2 * D2_De * (-t / (tex ** 2) * (1 - 2 * f))
    Discr_tex_f = (
        2 * (-2 * t / tex) * (-t / (tex ** 2) * (1 - 2 * f))
        + 2 * ((t / tex) * (1 - 2 * f) + D1 - D2) * (2 * t / (tex ** 2))
        - 8 * np.square(t) / (tex ** 3) * (1 - 2 * f)
    )
    Discr_Di_Di = 2 * (D1_Di ** 2)
    Discr_Di_De = -2 * D1_Di * D2_De
    Discr_Di_f = -4 * D1_Di * t / tex
    Discr_De_De = 2 * (D2_De ** 2)
    Discr_De_f = 4 * D2_De * t / tex
    # Discr_f_f = 0

    # Hessian of lp
    lp_tex_tex = t / (tex ** 3) + Discr_tex_tex / (4 * Discr_12) - Discr_tex * Discr_tex / (8 * Discr_32)
    lp_tex_Di = Discr_tex_Di / (4 * Discr_12) - Discr_tex * Discr_Di / (8 * Discr_32)
    lp_tex_De = Discr_tex_De / (4 * Discr_12) - Discr_tex * Discr_De / (8 * Discr_32)
    lp_tex_f = Discr_tex_f / (4 * Discr_12) - Discr_tex * Discr_f / (8 * Discr_32)
    lp_Di_Di = Discr_Di_Di / (4 * Discr_12) - Discr_Di * Discr_Di / (8 * Discr_32)
    lp_Di_De = Discr_Di_De / (4 * Discr_12) - Discr_Di * Discr_De / (8 * Discr_32)
    lp_Di_f = Discr_Di_f / (4 * Discr_12) - Discr_Di * Discr_f / (8 * Discr_32)
    lp_De_De = Discr_De_De / (4 * Discr_12) - Discr_De * Discr_De / (8 * Discr_32)
    lp_De_f = Discr_De_f / (4 * Discr_12) - Discr_De * Discr_f / (8 * Discr_32)
    lp_f_f = -Discr_f * Discr_f / (8 * Discr_32)

    # Hessian of lm
    lm_tex_tex = t / (tex ** 3) - Discr_tex_tex / (4 * Discr_12) + Discr_tex * Discr_tex / (8 * Discr_32)
    lm_tex_Di = -Discr_tex_Di / (4 * Discr_12) + Discr_tex * Discr_Di / (8 * Discr_32)
    lm_tex_De = -Discr_tex_De / (4 * Discr_12) + Discr_tex * Discr_De / (8 * Discr_32)
    lm_tex_f = -Discr_tex_f / (4 * Discr_12) + Discr_tex * Discr_f / (8 * Discr_32)
    lm_Di_Di = -Discr_Di_Di / (4 * Discr_12) + Discr_Di * Discr_Di / (8 * Discr_32)
    lm_Di_De = -Discr_Di_De / (4 * Discr_12) + Discr_Di * Discr_De / (8 * Discr_32)
    lm_Di_f = -Discr_Di_f / (4 * Discr_12) + Discr_Di * Discr_f / (8 * Discr_32)
    lm_De_De = -Discr_De_De / (4 * Discr_12) + Discr_De * Discr_De / (8 * Discr_32)
    lm_De_f = -Discr_De_f / (4 * Discr_12) + Discr_De * Discr_f / (8 * Discr_32)
    lm_f_f = Discr_f * Discr_f / (8 * Discr_32)

    # Hessian of Pp
    Pp_tex_tex = (
        -lm_tex_tex / Discr_12
        + lm_tex * Discr_tex / (2 * Discr_32)
        + lm_tex * Discr_tex / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_tex_tex / (2 * Discr_32) - 3 * Discr_tex * Discr_tex / (4 * Discr_52))
    )
    Pp_tex_Di = (
        -lm_tex_Di / Discr_12
        + lm_tex * Discr_Di / (2 * Discr_32)
        - (f * D1_Di - lm_Di) * Discr_tex / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_tex_Di / (2 * Discr_32) - 3 * Discr_tex * Discr_Di / (4 * Discr_52))
    )
    Pp_tex_De = (
        -lm_tex_De / Discr_12
        + lm_tex * Discr_De / (2 * Discr_32)
        - ((1 - f) * D2_De - lm_De) * Discr_tex / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_tex_De / (2 * Discr_32) - 3 * Discr_tex * Discr_De / (4 * Discr_52))
    )
    Pp_tex_f = (
        -lm_tex_f / Discr_12
        + lm_tex * Discr_f / (2 * Discr_32)
        - (D1 - D2 - lm_f) * Discr_tex / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_tex_f / (2 * Discr_32) - 3 * Discr_tex * Discr_f / (4 * Discr_52))
    )

    Pp_Di_Di = (
        -lm_Di_Di / Discr_12
        - (f * D1_Di - lm_Di) * Discr_Di / (2 * Discr_32)
        - (f * D1_Di - lm_Di) * Discr_Di / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_Di_Di / (2 * Discr_32) - 3 * Discr_Di * Discr_Di / (4 * Discr_52))
    )
    Pp_Di_De = (
        -lm_Di_De / Discr_12
        - (f * D1_Di - lm_Di) * Discr_De / (2 * Discr_32)
        - ((1 - f) * D2_De - lm_De) * Discr_Di / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_Di_De / (2 * Discr_32) - 3 * Discr_Di * Discr_De / (4 * Discr_52))
    )
    Pp_Di_f = (
        (-lm_Di_f + D1_Di) / Discr_12
        - (f * D1_Di - lm_Di) * Discr_f / (2 * Discr_32)
        - (D1 - D2 - lm_f) * Discr_Di / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_Di_f / (2 * Discr_32) - 3 * Discr_Di * Discr_f / (4 * Discr_52))
    )

    Pp_De_De = (
        -lm_De_De / Discr_12
        - ((1 - f) * D2_De - lm_De) * Discr_De / (2 * Discr_32)
        - ((1 - f) * D2_De - lm_De) * Discr_De / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_De_De / (2 * Discr_32) - 3 * Discr_De * Discr_De / (4 * Discr_52))
    )
    Pp_De_f = (
        (-lm_De_f - D2_De) / Discr_12
        - ((1 - f) * D2_De - lm_De) * Discr_f / (2 * Discr_32)
        - (D1 - D2 - lm_f) * Discr_De / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_De_f / (2 * Discr_32) - 3 * Discr_De * Discr_f / (4 * Discr_52))
    )

    Pp_f_f = (
        -lm_f_f / Discr_12
        - (D1 - D2 - lm_f) * Discr_f / (2 * Discr_32)
        - (D1 - D2 - lm_f) * Discr_f / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (-3 * Discr_f * Discr_f / (4 * Discr_52))
    )

    # Regroup hessains
    lp_hess = np.zeros(b.shape + (4, 4))
    lp_hess[..., 0, 0], lp_hess[..., 0, 1], lp_hess[..., 0, 2], lp_hess[..., 0, 3] = (
        lp_tex_tex,
        lp_tex_Di,
        lp_tex_De,
        lp_tex_f,
    )
    lp_hess[..., 1, 0], lp_hess[..., 1, 1], lp_hess[..., 1, 2], lp_hess[..., 1, 3] = (
        lp_tex_Di,
        lp_Di_Di,
        lp_Di_De,
        lp_Di_f,
    )
    lp_hess[..., 2, 0], lp_hess[..., 2, 1], lp_hess[..., 2, 2], lp_hess[..., 2, 3] = (
        lp_tex_De,
        lp_Di_De,
        lp_De_De,
        lp_De_f,
    )
    lp_hess[..., 3, 0], lp_hess[..., 3, 1], lp_hess[..., 3, 2], lp_hess[..., 3, 3] = lp_tex_f, lp_Di_f, lp_De_f, lp_f_f

    lm_hess = np.zeros(b.shape + (4, 4))
    lm_hess[..., 0, 0], lm_hess[..., 0, 1], lm_hess[..., 0, 2], lm_hess[..., 0, 3] = (
        lm_tex_tex,
        lm_tex_Di,
        lm_tex_De,
        lm_tex_f,
    )
    lm_hess[..., 1, 0], lm_hess[..., 1, 1], lm_hess[..., 1, 2], lm_hess[..., 1, 3] = (
        lm_tex_Di,
        lm_Di_Di,
        lm_Di_De,
        lm_Di_f,
    )
    lm_hess[..., 2, 0], lm_hess[..., 2, 1], lm_hess[..., 2, 2], lm_hess[..., 2, 3] = (
        lm_tex_De,
        lm_Di_De,
        lm_De_De,
        lm_De_f,
    )
    lm_hess[..., 3, 0], lm_hess[..., 3, 1], lm_hess[..., 3, 2], lm_hess[..., 3, 3] = lm_tex_f, lm_Di_f, lm_De_f, lm_f_f

    Pp_hess = np.zeros(b.shape + (4, 4))
    Pp_hess[..., 0, 0], Pp_hess[..., 0, 1], Pp_hess[..., 0, 2], Pp_hess[..., 0, 3] = (
        Pp_tex_tex,
        Pp_tex_Di,
        Pp_tex_De,
        Pp_tex_f,
    )
    Pp_hess[..., 1, 0], Pp_hess[..., 1, 1], Pp_hess[..., 1, 2], Pp_hess[..., 1, 3] = (
        Pp_tex_Di,
        Pp_Di_Di,
        Pp_Di_De,
        Pp_Di_f,
    )
    Pp_hess[..., 2, 0], Pp_hess[..., 2, 1], Pp_hess[..., 2, 2], Pp_hess[..., 2, 3] = (
        Pp_tex_De,
        Pp_Di_De,
        Pp_De_De,
        Pp_De_f,
    )
    Pp_hess[..., 3, 0], Pp_hess[..., 3, 1], Pp_hess[..., 3, 2], Pp_hess[..., 3, 3] = Pp_tex_f, Pp_Di_f, Pp_De_f, Pp_f_f

    Pm_hess = -Pp_hess

    Pp44, Pm44, lm44, lp44 = broad44(Pp), broad44(Pm), broad44(lm), broad44(lp)
    Pp_jac4, Pm_jac4, lp_jac4, lm_jac4 = broad4(Pp_jac), broad4(Pm_jac), broad4(lp_jac), broad4(lm_jac)
    Pp_jac4T, Pm_jac4T, lp_jac4T, lm_jac4T = broad4T(Pp_jac), broad4T(Pm_jac), broad4T(lp_jac), broad4T(lm_jac)

    return (Pp_hess - Pp_jac4 * lp_jac4T - Pp_jac4T * lp_jac4 - Pp44 * lp_hess + Pp44 * lp_jac4 * lp_jac4T) * np.exp(
        -lp44
    ) + (Pm_hess - Pm_jac4 * lm_jac4T - Pm_jac4T * lm_jac4 - Pm44 * lm_hess + Pm44 * lm_jac4 * lm_jac4T) * np.exp(-lm44)


nexi_hessian = lambda tex, Di, De, Dp, f, b, t: scipy.integrate.quad_vec(
    lambda x: M_hess(x, b, t, tex, Di, De, Dp, f), 0, 1, epsabs=1e-14
)[0]

nexi_hessian_from_vector = lambda param, b, t: nexi_hessian(
    param[0], param[1], param[2], param[2], param[3], b, t
)  # [tex, Di, De, De, f]


#######################################################################################################################
# Optimised concatenated NEXI Hessian
#######################################################################################################################


def M_hess_concat(x, b, t, tex, Di, De, Dp, f):
    D1 = Di * b * (x ** 2)
    D2 = Dp * b + (De - Dp) * b * (x ** 2)
    Discr = np.square((t / tex) * (1 - 2 * f) + D1 - D2) + 4 * np.square(t / tex) * f * (1 - f)
    Discr_12 = np.sqrt(Discr)  # Discr ** (1/2)
    Discr_32 = Discr * Discr_12  # Discr ** (3/2)
    Discr_52 = Discr * Discr * np.sqrt(Discr)  # Discr ** (5/2)

    lp = (t / tex + D1 + D2) / 2 + Discr_12 / 2
    lm = (t / tex + D1 + D2) / 2 - Discr_12 / 2
    Pp = np.divide(f * D1 + (1 - f) * D2 - lm, Discr_12)
    Pm = 1 - Pp

    # Derivatives of t_ex
    Discr_tex = ((t / tex) * (1 - 2 * f) + D1 - D2) * (-2 * (1 - 2 * f) * t / (tex ** 2)) - 8 * np.square(t) * f * (
        1 - f
    ) / (tex ** 3)
    lp_tex = -(t / (tex ** 2)) / 2 + np.divide(Discr_tex, Discr_12) / 4
    lm_tex = -(t / (tex ** 2)) / 2 - np.divide(Discr_tex, Discr_12) / 4
    Pp_tex = np.divide(-lm_tex, Discr_12) - np.divide((f * D1 + (1 - f) * D2 - lm) * Discr_tex, 2 * Discr_32)
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
    Pp_De = np.divide((1 - f) * D2_De - lm_De, Discr_12) - np.divide(
        (f * D1 + (1 - f) * D2 - lm) * Discr_De, 2 * Discr_32
    )
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

    Pm_jac = -Pp_jac

    # Hessian of Discr
    Discr_tex_tex = (
        2 * np.square(-t / (tex ** 2) * (1 - 2 * f))
        + 2 * ((t / tex) * (1 - 2 * f) + D1 - D2) * (2 * t / (tex ** 3) * (1 - 2 * f))
        + 24 * np.square(t / (tex ** 2)) * f * (1 - f)
    )
    Discr_tex_Di = 2 * D1_Di * (-(t / (tex ** 2)) * (1 - 2 * f))
    Discr_tex_De = -2 * D2_De * (-t / (tex ** 2) * (1 - 2 * f))
    Discr_tex_f = (
        2 * (-2 * t / tex) * (-t / (tex ** 2) * (1 - 2 * f))
        + 2 * ((t / tex) * (1 - 2 * f) + D1 - D2) * (2 * t / (tex ** 2))
        - 8 * np.square(t) / (tex ** 3) * (1 - 2 * f)
    )
    Discr_Di_Di = 2 * (D1_Di ** 2)
    Discr_Di_De = -2 * D1_Di * D2_De
    Discr_Di_f = -4 * D1_Di * t / tex
    Discr_De_De = 2 * (D2_De ** 2)
    Discr_De_f = 4 * D2_De * t / tex
    # Discr_f_f = 0

    # Hessian of lp
    lp_tex_tex = t / (tex ** 3) + Discr_tex_tex / (4 * Discr_12) - Discr_tex * Discr_tex / (8 * Discr_32)
    lp_tex_Di = Discr_tex_Di / (4 * Discr_12) - Discr_tex * Discr_Di / (8 * Discr_32)
    lp_tex_De = Discr_tex_De / (4 * Discr_12) - Discr_tex * Discr_De / (8 * Discr_32)
    lp_tex_f = Discr_tex_f / (4 * Discr_12) - Discr_tex * Discr_f / (8 * Discr_32)
    lp_Di_Di = Discr_Di_Di / (4 * Discr_12) - Discr_Di * Discr_Di / (8 * Discr_32)
    lp_Di_De = Discr_Di_De / (4 * Discr_12) - Discr_Di * Discr_De / (8 * Discr_32)
    lp_Di_f = Discr_Di_f / (4 * Discr_12) - Discr_Di * Discr_f / (8 * Discr_32)
    lp_De_De = Discr_De_De / (4 * Discr_12) - Discr_De * Discr_De / (8 * Discr_32)
    lp_De_f = Discr_De_f / (4 * Discr_12) - Discr_De * Discr_f / (8 * Discr_32)
    lp_f_f = -Discr_f * Discr_f / (8 * Discr_32)

    # Hessian of lm
    lm_tex_tex = t / (tex ** 3) - Discr_tex_tex / (4 * Discr_12) + Discr_tex * Discr_tex / (8 * Discr_32)
    lm_tex_Di = -Discr_tex_Di / (4 * Discr_12) + Discr_tex * Discr_Di / (8 * Discr_32)
    lm_tex_De = -Discr_tex_De / (4 * Discr_12) + Discr_tex * Discr_De / (8 * Discr_32)
    lm_tex_f = -Discr_tex_f / (4 * Discr_12) + Discr_tex * Discr_f / (8 * Discr_32)
    lm_Di_Di = -Discr_Di_Di / (4 * Discr_12) + Discr_Di * Discr_Di / (8 * Discr_32)
    lm_Di_De = -Discr_Di_De / (4 * Discr_12) + Discr_Di * Discr_De / (8 * Discr_32)
    lm_Di_f = -Discr_Di_f / (4 * Discr_12) + Discr_Di * Discr_f / (8 * Discr_32)
    lm_De_De = -Discr_De_De / (4 * Discr_12) + Discr_De * Discr_De / (8 * Discr_32)
    lm_De_f = -Discr_De_f / (4 * Discr_12) + Discr_De * Discr_f / (8 * Discr_32)
    lm_f_f = Discr_f * Discr_f / (8 * Discr_32)

    # Hessian of Pp
    Pp_tex_tex = (
        -lm_tex_tex / Discr_12
        + lm_tex * Discr_tex / (2 * Discr_32)
        + lm_tex * Discr_tex / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_tex_tex / (2 * Discr_32) - 3 * Discr_tex * Discr_tex / (4 * Discr_52))
    )
    Pp_tex_Di = (
        -lm_tex_Di / Discr_12
        + lm_tex * Discr_Di / (2 * Discr_32)
        - (f * D1_Di - lm_Di) * Discr_tex / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_tex_Di / (2 * Discr_32) - 3 * Discr_tex * Discr_Di / (4 * Discr_52))
    )
    Pp_tex_De = (
        -lm_tex_De / Discr_12
        + lm_tex * Discr_De / (2 * Discr_32)
        - ((1 - f) * D2_De - lm_De) * Discr_tex / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_tex_De / (2 * Discr_32) - 3 * Discr_tex * Discr_De / (4 * Discr_52))
    )
    Pp_tex_f = (
        -lm_tex_f / Discr_12
        + lm_tex * Discr_f / (2 * Discr_32)
        - (D1 - D2 - lm_f) * Discr_tex / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_tex_f / (2 * Discr_32) - 3 * Discr_tex * Discr_f / (4 * Discr_52))
    )

    Pp_Di_Di = (
        -lm_Di_Di / Discr_12
        - (f * D1_Di - lm_Di) * Discr_Di / (2 * Discr_32)
        - (f * D1_Di - lm_Di) * Discr_Di / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_Di_Di / (2 * Discr_32) - 3 * Discr_Di * Discr_Di / (4 * Discr_52))
    )
    Pp_Di_De = (
        -lm_Di_De / Discr_12
        - (f * D1_Di - lm_Di) * Discr_De / (2 * Discr_32)
        - ((1 - f) * D2_De - lm_De) * Discr_Di / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_Di_De / (2 * Discr_32) - 3 * Discr_Di * Discr_De / (4 * Discr_52))
    )
    Pp_Di_f = (
        (-lm_Di_f + D1_Di) / Discr_12
        - (f * D1_Di - lm_Di) * Discr_f / (2 * Discr_32)
        - (D1 - D2 - lm_f) * Discr_Di / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_Di_f / (2 * Discr_32) - 3 * Discr_Di * Discr_f / (4 * Discr_52))
    )

    Pp_De_De = (
        -lm_De_De / Discr_12
        - ((1 - f) * D2_De - lm_De) * Discr_De / (2 * Discr_32)
        - ((1 - f) * D2_De - lm_De) * Discr_De / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_De_De / (2 * Discr_32) - 3 * Discr_De * Discr_De / (4 * Discr_52))
    )
    Pp_De_f = (
        (-lm_De_f - D2_De) / Discr_12
        - ((1 - f) * D2_De - lm_De) * Discr_f / (2 * Discr_32)
        - (D1 - D2 - lm_f) * Discr_De / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (Discr_De_f / (2 * Discr_32) - 3 * Discr_De * Discr_f / (4 * Discr_52))
    )

    Pp_f_f = (
        -lm_f_f / Discr_12
        - (D1 - D2 - lm_f) * Discr_f / (2 * Discr_32)
        - (D1 - D2 - lm_f) * Discr_f / (2 * Discr_32)
        - (f * D1 + (1 - f) * D2 - lm) * (-3 * Discr_f * Discr_f / (4 * Discr_52))
    )

    # Regroup hessians
    lp_hess = np.zeros(b.shape + (4, 4))
    lp_hess[..., 0, 0], lp_hess[..., 0, 1], lp_hess[..., 0, 2], lp_hess[..., 0, 3] = (
        lp_tex_tex,
        lp_tex_Di,
        lp_tex_De,
        lp_tex_f,
    )
    lp_hess[..., 1, 0], lp_hess[..., 1, 1], lp_hess[..., 1, 2], lp_hess[..., 1, 3] = (
        lp_tex_Di,
        lp_Di_Di,
        lp_Di_De,
        lp_Di_f,
    )
    lp_hess[..., 2, 0], lp_hess[..., 2, 1], lp_hess[..., 2, 2], lp_hess[..., 2, 3] = (
        lp_tex_De,
        lp_Di_De,
        lp_De_De,
        lp_De_f,
    )
    lp_hess[..., 3, 0], lp_hess[..., 3, 1], lp_hess[..., 3, 2], lp_hess[..., 3, 3] = lp_tex_f, lp_Di_f, lp_De_f, lp_f_f

    lm_hess = np.zeros(b.shape + (4, 4))
    lm_hess[..., 0, 0], lm_hess[..., 0, 1], lm_hess[..., 0, 2], lm_hess[..., 0, 3] = (
        lm_tex_tex,
        lm_tex_Di,
        lm_tex_De,
        lm_tex_f,
    )
    lm_hess[..., 1, 0], lm_hess[..., 1, 1], lm_hess[..., 1, 2], lm_hess[..., 1, 3] = (
        lm_tex_Di,
        lm_Di_Di,
        lm_Di_De,
        lm_Di_f,
    )
    lm_hess[..., 2, 0], lm_hess[..., 2, 1], lm_hess[..., 2, 2], lm_hess[..., 2, 3] = (
        lm_tex_De,
        lm_Di_De,
        lm_De_De,
        lm_De_f,
    )
    lm_hess[..., 3, 0], lm_hess[..., 3, 1], lm_hess[..., 3, 2], lm_hess[..., 3, 3] = lm_tex_f, lm_Di_f, lm_De_f, lm_f_f

    Pp_hess = np.zeros(b.shape + (4, 4))
    Pp_hess[..., 0, 0], Pp_hess[..., 0, 1], Pp_hess[..., 0, 2], Pp_hess[..., 0, 3] = (
        Pp_tex_tex,
        Pp_tex_Di,
        Pp_tex_De,
        Pp_tex_f,
    )
    Pp_hess[..., 1, 0], Pp_hess[..., 1, 1], Pp_hess[..., 1, 2], Pp_hess[..., 1, 3] = (
        Pp_tex_Di,
        Pp_Di_Di,
        Pp_Di_De,
        Pp_Di_f,
    )
    Pp_hess[..., 2, 0], Pp_hess[..., 2, 1], Pp_hess[..., 2, 2], Pp_hess[..., 2, 3] = (
        Pp_tex_De,
        Pp_Di_De,
        Pp_De_De,
        Pp_De_f,
    )
    Pp_hess[..., 3, 0], Pp_hess[..., 3, 1], Pp_hess[..., 3, 2], Pp_hess[..., 3, 3] = Pp_tex_f, Pp_Di_f, Pp_De_f, Pp_f_f

    Pm_hess = -Pp_hess

    Pp44, Pm44, lm44, lp44 = broad44(Pp), broad44(Pm), broad44(lm), broad44(lp)
    Pp_jac4, Pm_jac4, lp_jac4, lm_jac4 = broad4(Pp_jac), broad4(Pm_jac), broad4(lp_jac), broad4(lm_jac)
    Pp_jac4T, Pm_jac4T, lp_jac4T, lm_jac4T = broad4T(Pp_jac), broad4T(Pm_jac), broad4T(lp_jac), broad4T(lm_jac)

    M_hess = (Pp_hess - Pp_jac4 * lp_jac4T - Pp_jac4T * lp_jac4 - Pp44 * lp_hess + Pp44 * lp_jac4 * lp_jac4T) * np.exp(
        -lp44
    ) + (Pm_hess - Pm_jac4 * lm_jac4T - Pm_jac4T * lm_jac4 - Pm44 * lm_hess + Pm44 * lm_jac4 * lm_jac4T) * np.exp(-lm44)

    # Adding lines not required in the sole hessian calculation starting from here
    # Compute Jacobian
    Pp4, Pm4, lm4, lp4 = broad4(Pp), broad4(Pm), broad4(lm), broad4(lp)
    M_jac = Pp_jac * np.exp(-lp4) - Pp4 * lp_jac * np.exp(-lp4) + Pm_jac * np.exp(-lm4) - Pm4 * lm_jac * np.exp(-lm4)
    # Compute Original integrand
    M4 = broad4(Pp * np.exp(-lp) + Pm * np.exp(-lm))

    # Concatenating M4, M_jac and M_hess to be integered together
    # M4 = M_concat[0] , M_jac = M_concat[1] and M_hess = M_concat[2:6]
    desired_shape = list(M_hess.shape)
    desired_shape[-1] += 2
    M_concat = np.empty(desired_shape)
    M_concat[..., 0] = M4
    M_concat[..., 1] = M_jac
    M_concat[..., 2:6] = M_hess
    return M_concat


nexi_hessian_concatenated = lambda tex, Di, De, Dp, f, b, t: scipy.integrate.quad_vec(
    lambda x: M_hess_concat(x, b, t, tex, Di, De, Dp, f), 0, 1, epsabs=1e-14
)[0]

nexi_hessian_concatenated_from_vector = lambda param, b, t: nexi_hessian_concatenated(
    param[0], param[1], param[2], param[2], param[3], b, t
)  # [tex, Di, De, De, f] since De,parallel = De,perpendicular
