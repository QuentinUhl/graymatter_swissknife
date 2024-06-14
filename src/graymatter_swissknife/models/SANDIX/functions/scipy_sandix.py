"""
Apache-2.0 license

Copyright (c) 2023 Quentin Uhl, Ileana O. Jelescu
smex_signal function is adapted and optimized from the code of Jonas Olesen, jonas@phys.au.dk

Please cite 
"Diffusion time dependence, power-law scaling, and exchange in gray matter"
NeuroImage, Volume 251, 118976 (2022).
https://doi.org/10.1016/j.neuroimage.2022.118976 

as well as my 2024 ISMRM abstract for this implementation:
"GEM: a unifying model for Gray Matter microstructure", 
Uhl, Q., Pavan, T., de Ridematten I., Nguyen-Duc, J., Jelescu, I.O., 2024. 
in: Proc. Intl. Soc. Mag. Reson. Med. 2024. 
Presented at the Annual Meeting of the ISMRM, Singapore, Singapore, p. 7970.
"""

# Implementation of the SANDIX model in scipy
import numpy as np
from ...NEXI.functions.scipy_smex import smex_signal, smex_jacobian_concatenated
from ...struct_functions.scipy_sphere import sphere_murdaycotts, sphere_jacobian


#######################################################################################################################
# SANDIX Model
#######################################################################################################################


def sandix_signal(tex, Di, De, f, rs, fs, b, delta, small_delta):
    """
    This function computes the SANDIX model. fs_tilde is the volume fraction of the spheres over
    the total volume fraction of both the extra-neurite and intra-axonal compartments. f_tilde is the volume fraction
    of the intra-axonal compartment over the total volume fraction of both the extra-neurite and intra-axonal.

    :param tex: The exchange time (in ms)
    :param Di: The intra-axonal/dendrite diffusivity (in µm²/ms)
    :param De: The extra-axonal/dendrite diffusivity (in µm²/ms)
    :param f: The volume fraction of the intra-neurite compartment (spheres and intra-axonal/dendrite)
    :param rs: The radius of the spheres (in µm)
    :param fs: The volume fraction of the spheres over the total volume fraction of both the intra-axonal/dendrite and
    the spheres.
    :param b: The b-value (in ms/µm²)
    :param delta: The diffusion time (in ms)
    :param small_delta: The pulse width (in ms)
    :return: The signal from the Nexi Sphere model
    """
    fs_tilde = f * fs
    f_tilde = f * (1 - fs) / (1 - fs_tilde)
    return fs_tilde*sphere_murdaycotts(rs, 3, b, delta, small_delta) + (1-fs_tilde)*smex_signal(tex, Di, De, f_tilde, b, delta, small_delta)


def sandix_signal_from_vector(param, b, delta, small_delta):
    # param: [tex, Di, De, De, f, rs, fs]
    return sandix_signal(param[0], param[1], param[2], param[3], param[4], param[5], b, delta, small_delta)

#######################################################################################################################
# SANDIX jacobian
#######################################################################################################################


def sandix_jacobian(tex, Di, De, f, rs, fs, b, delta, small_delta):
    fs_tilde = f * fs
    f_tilde = f * (1 - fs) / (1 - fs_tilde)
    nexi_finite_pulses_signal_vec, nexi_vec_jac = smex_jacobian_concatenated(tex, Di, De, f_tilde, b, delta, small_delta)
    # print(nexi_vec_jac_concatenation)
    # nexi_finite_pulses_signal_vec = nexi_vec_jac_concatenation[..., 0]
    # nexi_vec_jac = nexi_vec_jac_concatenation[..., 1:5]

    s_sphere, ds_sphere_dr = sphere_jacobian(rs, 3, b, delta, small_delta)

    s_sandix_jacobian_proxy = np.ones(nexi_vec_jac.shape[:-1] + (6,))
    s_sandix_jacobian_proxy[..., 0:4] = (1 - fs_tilde) * nexi_vec_jac
    s_sandix_jacobian_proxy[..., 4] = fs_tilde * ds_sphere_dr
    s_sandix_jacobian_proxy[..., 5] = s_sphere - nexi_finite_pulses_signal_vec

    s_sandix_jacobian = np.copy(s_sandix_jacobian_proxy)
    s_sandix_jacobian[..., 3] = ((1-fs)/((1-fs_tilde)**2) * s_sandix_jacobian_proxy[..., 3] +
                                      fs * s_sandix_jacobian_proxy[..., 5])
    s_sandix_jacobian[..., 5] = (-f*(1-f)/((1-fs_tilde)**2) * s_sandix_jacobian_proxy[..., 3] +
                                      f * s_sandix_jacobian_proxy[..., 5])

    return s_sandix_jacobian


def sandix_jacobian_from_vector(param, b, delta, small_delta):
    # param: [tex, Di, De, De, f, rs, fs]
    return sandix_jacobian(param[0], param[1], param[2], param[3], param[4], param[5], b, delta, small_delta)


#######################################################################################################################
# SANDIX signal & jacobian concatenated
#######################################################################################################################


def sandix_concat(tex, Di, De, f, rs, fs, b, delta, small_delta):

    fs_tilde = f * fs
    f_tilde = f * (1 - fs) / (1 - fs_tilde)

    nexi_signal_vec, nexi_vec_jac = smex_jacobian_concatenated(tex, Di, De, f_tilde, b, delta, small_delta)
    # nexi_signal_vec = nexi_vec_jac_concatenation[..., 0]
    # nexi_vec_jac = nexi_vec_jac_concatenation[..., 1:5]

    s_sphere, ds_sphere_dr = sphere_jacobian(rs, 3, b, delta, small_delta)

    s_sandix_concatenated_proxy = np.ones(nexi_vec_jac.shape[:-1] + (7,))
    s_sandix_concatenated_proxy[..., 0] = fs_tilde * s_sphere + (1 - fs_tilde) * nexi_signal_vec
    s_sandix_concatenated_proxy[..., 1:5] = (1 - fs_tilde) * nexi_vec_jac
    s_sandix_concatenated_proxy[..., 5] = fs_tilde * ds_sphere_dr
    s_sandix_concatenated_proxy[..., 6] = s_sphere - nexi_signal_vec

    s_sandix_concatenated = np.copy(s_sandix_concatenated_proxy)
    s_sandix_concatenated[..., 4] = ((1-fs)/((1-fs_tilde)**2) * s_sandix_concatenated_proxy[..., 4] +
                                          fs * s_sandix_concatenated_proxy[..., 6])
    s_sandix_concatenated[..., 6] = (-f*(1-f)/((1-fs_tilde)**2) * s_sandix_concatenated_proxy[..., 4] +
                                          f * s_sandix_concatenated_proxy[..., 6])

    return s_sandix_concatenated


def sandix_concatenated_from_vector(param, b, delta, small_delta):
    # param: [tex, Di, De, De, f, rs, fs]
    return sandix_concat(param[0], param[1], param[2], param[3], param[4], param[5], b, delta, small_delta)


#######################################################################################################################
# Optimized SANDIX for computation of MSE jacobian
#######################################################################################################################


broad6 = lambda matrix: np.tile(matrix[..., np.newaxis], 6)
broad7 = lambda matrix: np.tile(matrix[..., np.newaxis], 7)


def sandix_optimized_mse_jacobian(param, b, delta, small_delta, Y, b_delta_dimensions=2):
    sandix_vec_jac_concatenation = sandix_concatenated_from_vector(param, b, delta, small_delta)
    sandix_vec = sandix_vec_jac_concatenation[..., 0]
    sandix_vec_jac = sandix_vec_jac_concatenation[..., 1:7]
    if b_delta_dimensions == 1:
        mse_jacobian = np.sum(2 * sandix_vec_jac * broad6(sandix_vec - Y), axis=0)
    elif b_delta_dimensions == 2:
        mse_jacobian = np.sum(2 * sandix_vec_jac * broad6(sandix_vec - Y), axis=(0, 1))
    else:
        raise NotImplementedError
    return mse_jacobian


