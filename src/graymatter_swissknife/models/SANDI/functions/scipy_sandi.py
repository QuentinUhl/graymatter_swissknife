"""
Apache-2.0 license

Copyright (c) 2023 Quentin Uhl, Ileana O. Jelescu
sandi_signal function is adapted and optimized from the code of Marco Palombo, PalomboM@cardiff.ac.uk

Please cite 
"SANDI: A compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI"
NeuroImage, Volume 215, 116835 (2020).
https://doi.org/10.1016/j.neuroimage.2020.116835
"""

import numpy as np
import scipy.special
from ....models.struct_functions.scipy_sphere import sphere_murdaycotts, sphere_jacobian


def neurite_signal(Di, b):
    #return np.sqrt(np.pi / (4 * b * Di))*scipy.special.erf(np.sqrt(b * Di))
    return(np.divide(np.sqrt(np.pi)*scipy.special.erf(np.sqrt(b*Di)), np.sqrt(4*b*Di), out=np.ones_like(b).astype(float), where=Di*b!=0))


def neurite_jacobian(Di, b):
    # return(np.exp(-b*Di)/(2*Di) - 1/4*np.sqrt(np.pi/(b*Di))/Di*scipy.special.erf(np.sqrt(b*Di)))
    return np.divide(np.exp(-b * Di), 2 * Di, out=np.zeros_like(b).astype(float), where=Di * b != 0) - np.divide(
        1 / 4 * np.sqrt(np.pi) * scipy.special.erf(np.sqrt(b * Di)), np.sqrt(b * Di) * Di,
        out=np.zeros_like(b).astype(float), where=Di * b != 0)


def extracellular_signal(De, b):
    return np.exp(-b * De)


def extracellular_jacobian(De, b):
    return -b*np.exp(-b * De)


#######################################################################################################################
# Sandi Model
#######################################################################################################################


def sandi_signal(Di, De, f, rs, fs, b, td, small_delta):
    s_sphere = sphere_murdaycotts(rs, 3, b, td, small_delta)
    s_neurite = neurite_signal(Di, b)
    s_extracellular = extracellular_signal(De, b)
    return f * (fs * s_sphere + (1 - fs) * s_neurite) + (1 - f) * s_extracellular


def sandi_signal_from_vector(param, b, td, small_delta):
    # param: [Di, De, f, rs, fs]
    return sandi_signal(param[0], param[1], param[2], param[3], param[4], b, td, small_delta)


#######################################################################################################################
# Sandi signal & jacobian concatenated
#######################################################################################################################


def sandi_jacobian(Di, De, f, rs, fs, b, td, small_delta):
    # s_sphere = sphere_murdaycotts(rs, 3, b, td, small_delta)
    s_neurite = neurite_signal(Di, b)
    s_extracellular = extracellular_signal(De, b)
    s_sphere, s_sphere_jac = sphere_jacobian(rs, 3, b, td, small_delta)
    s_neurite_jac = neurite_jacobian(Di, b)
    s_extracellular_jac = extracellular_jacobian(De, b)
    signal = f * (fs * s_sphere + (1 - fs) * s_neurite) + (1 - f) * s_extracellular
    jacobian = np.array([f * (1 - fs) * s_neurite_jac,
                         (1 - f) * s_extracellular_jac,
                         fs * s_sphere + (1 - fs) * s_neurite - s_extracellular,
                         f * fs * s_sphere_jac,
                         f * (s_sphere - s_neurite)]).T
    return signal, jacobian


def sandi_jacobian_from_vector(param, b, td, small_delta):
    # param: [tex, Di, De, De, f, rs, fs]
    return sandi_jacobian(param[0], param[1], param[2], param[3], param[4], b, td, small_delta)


#######################################################################################################################
# Optimized Sandi for computation of MSE jacobian
#######################################################################################################################


broad5 = lambda matrix: np.tile(matrix[..., np.newaxis], 5)


def sandi_optimized_mse_jacobian(param, b, td, small_delta, Y, b_td_dimensions=2):
    sandi_vec, sandi_vec_jac = sandi_jacobian_from_vector(param, b, td, small_delta)
    if b_td_dimensions == 1:
        mse_jacobian = np.sum(2 * sandi_vec_jac * broad5(sandi_vec - Y), axis=0)
    elif b_td_dimensions == 2:
        mse_jacobian = np.sum(2 * sandi_vec_jac * broad5(sandi_vec - Y), axis=(0, 1))
    else:
        raise NotImplementedError
    return mse_jacobian
