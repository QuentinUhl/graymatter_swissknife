"""
Apache-2.0 license

Copyright (c) 2023 Quentin Uhl

Please cite 
"GEM: a unifying model for Gray Matter microstructure", 
Uhl, Q., Pavan, T., de Ridematten I., Nguyen-Duc, J., Jelescu, I.O., 2024. 
in: Proc. Intl. Soc. Mag. Reson. Med. 2024. 
Presented at the Annual Meeting of the ISMRM, Singapore, Singapore, p. 7970.
"""

# Implementation of the Generalized Exchange Model in scipy
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre
from scipy.linalg import expm
from ...struct_functions.scipy_sphere import (dlog_sphere_signal_first_pulse,
                                                                 dlog_sphere_signal_second_pulse,
                                                                 jacobian_dlog_sphere_signal_first_pulse,
                                                                 jacobian_dlog_sphere_signal_second_pulse)

# utilitary functions for jacobian, hessian (and Fsumsquares in NLS)
broad7 = lambda matrix: np.tile(matrix[..., np.newaxis], 7)
broad8 = lambda matrix: np.tile(matrix[..., np.newaxis], 8)


def legpts(n, interval=np.array([-1, 1])):
    a, b = interval[0], interval[-1]
    roots, weights = roots_legendre(n)
    return np.multiply((b - a) / 2, roots) + (a + b) / 2, weights * (b - a) / 2


#######################################################################################################################
# Generalized Exchange Model (GEM)
#######################################################################################################################


def gem_signal(texs, texd, Ds0, Dd, De, fn, radius, fsn, b, td, small_delta):
    """
    Computes the signal of the Generalized Exchange Model from the microstructure parameters texs, texd, Ds0, Dd, De,
    fn, radius and fsn and the acquisition parameters b, td and small_delta. b and td must have the same shape.
    """
    # If Δ and b are single, expand them
    if np.isscalar(b):
        b = np.array([b])
    if np.isscalar(td):
        td = np.array([td])
    assert len(b) == len(td), "b and td must have the same length"
    assert np.isscalar(small_delta), "𝛿 must be a scalar"
    len_b = len(b)
    return_scalar = True if len_b == 1 else False
    # Define the parameters from the other compartment
    fe = 1 - fn
    fs = fn * fsn
    fd = fn * (1 - fsn)
    # Import initial parameters from [tex, Di, De, f]
    rs = (fe / (fe + fs)) / texs
    rd = (fe / (fe + fd)) / texd
    re = (fs / (fe + fs)) / texs + (fd / (fe + fd)) / texd
    # can solve differential equation for all eps simultaneously
    len_w = 20
    [eps, w] = legpts(len_w, np.array([0, 1]))
    D = np.zeros((len_w, 3, 3))
    D[:, 0, 0] = 0
    D[:, 1, 1] = Dd * eps ** 2
    D[:, 2, 2] = De
    R = np.array([[-rs, 0, fs / fe * rs], [0, -rd, fd / fe * rd], [rs, rd, -re]])
    # Define the signal at time 0
    f = np.zeros((len_w*len_b, 3))
    f[:, 0] = fs
    f[:, 1] = fd
    f[:, 2] = fe
    # Define the q-value
    q2 = b / (td - small_delta / 3)
    q = np.sqrt(q2)
    # Compute the signal from the differential equation
    flat_f = f.flatten()
    # Expand dimensions to allow broadcasting
    q = np.expand_dims(q, axis=(1, 2, 3))
    q2 = np.expand_dims(q2, axis=(1, 2, 3))
    td = np.expand_dims(td, axis=(1, 2, 3))
    R = np.expand_dims(R, axis=(0, 1))
    D = np.expand_dims(D, axis=0)
    # Define the shape of the solution
    S_shape = (len_b, len_w, 3)
    # Compute the signal from the differential equation
    # S = np.zeros((len_b, len_w))
    # S_s = np.zeros((len_b, len_w))
    # S_d = np.zeros((len_b, len_w))
    # S_e = np.zeros((len_b, len_w))
    St = first_pulse(flat_f, D, R, radius, Ds0, small_delta, q, S_shape)
    St = St.reshape(S_shape)
    St = np.einsum('ijkl, ijl->ijk', expm((td - small_delta) * (R - q2 * D)), St)
    St = St.flatten()
    St = second_pulse(St, D, R, radius, Ds0, small_delta, td, q, S_shape)
    St = St.reshape(S_shape)
    S = St[..., 0] + St[..., 1] + St[..., 2]
    # S_s = St[..., 0]
    # S_d = St[..., 1]
    # S_e = St[..., 2]
    S = np.dot(S, np.transpose(w))
    # S_s = np.dot(S_s, np.transpose(w))
    # S_d = np.dot(S_d, np.transpose(w))
    # S_e = np.dot(S_e, np.transpose(w))
    S = np.squeeze(S)
    if return_scalar:
        S = S.item()
    return S  # , S_s, S_d, S_e


def first_pulse(f, D, R, radius, Ds0, small_delta, q, S_shape):
    """ Modelize the first gradient pulse function and solve the differential equation for GEM. """
    flat_q = q.flatten()
    # Define the gradient function
    def dSdt(t, S):
        S = S.reshape(S_shape)
        q2D_t = (q * t / small_delta) ** 2 * D
        q2D_t[..., 0, 0] = np.expand_dims(dlog_sphere_signal_first_pulse(t, radius, Ds0, flat_q, small_delta), axis=-1)
        return np.einsum('ijkl, ijl->ijk', (R - q2D_t), S).flatten()
    # Solve the differential equation starting from the initial solution f at time Δ+𝛿
    S = solve_ivp(dSdt, t_span=np.array([0, small_delta]), y0=f, method='RK45',
                  t_eval=np.array([0, small_delta]), rtol=1e-9, atol=1e-9).y[:, -1]
    return S


def second_pulse(S_td, D, R, radius, Ds0, small_delta, td, q, S_shape):
    """ Modelize the second gradient pulse function and solve the differential equation for GEM. """
    flat_q = q.flatten()
    td = td.flatten()[:, np.newaxis]
    def dSdt(t, S):
        S = S.reshape(S_shape)
        q2D_t = (q * (small_delta - t) / small_delta) ** 2 * D
        q2D_t[..., 0, 0] = np.expand_dims(dlog_sphere_signal_second_pulse(t+td, radius, Ds0, flat_q, td, small_delta), axis=-1)
        return np.einsum('ijkl, ijl->ijk', (R - q2D_t), S).flatten()
    # Solve the differential equation starting from the initial solution f at time Δ+𝛿
    S = solve_ivp(dSdt, t_span=np.array([0, small_delta]), y0=S_td, method='RK45',
                  t_eval=np.array([0, small_delta]), rtol=1e-9, atol=1e-9).y[:, -1]
    return S


def gem_signal_from_vector(parameters, b, td, small_delta):
    """Get signal from single Ground Truth."""
    texs = parameters[0]
    texd = parameters[1]
    Dd = parameters[2]
    De = parameters[3]
    fn = parameters[4]
    radius = parameters[5]
    fsn = parameters[6]
    return gem_signal(texs, texd, 3, Dd, De, fn, radius, fsn, b, td, small_delta)


#######################################################################################################################
# GEM jacobian
#######################################################################################################################

def gem_jacobian_from_vector(parameters, b, td, small_delta):
    """Get signal from single Ground Truth."""
    texs = parameters[0]
    texd = parameters[1]
    Dd = parameters[2]
    De = parameters[3]
    fn = parameters[4]
    radius = parameters[5]
    fsn = parameters[6]
    _, gem_jac = gem_jacobian(texs, texd, 3, Dd, De, fn, radius, fsn, b, td, small_delta)
    return gem_jac


def gem_jacobian_concatenated_from_vector(parameters, b, td, small_delta):
    """Get signal from single Ground Truth."""
    texs = parameters[0]
    texd = parameters[1]
    Dd = parameters[2]
    De = parameters[3]
    fn = parameters[4]
    radius = parameters[5]
    fsn = parameters[6]
    gem_signal, gem_jac = gem_jacobian(texs, texd, 3, Dd, De, fn, radius, fsn, b, td, small_delta)
    return gem_signal, gem_jac


def gem_jacobian(texs, texd, Ds0, Dd, De, fn, radius, fsn, b, td, small_delta):
    """
    Computes the jacobian of the Generalized Exchange Model from the microstructure parameters texs, texd, Ds0, Dd, De,
    fn, radius and fsn and the acquisition parameters b, td and small_delta. b and td must have the same shape.
    """
    # If Δ and b are single, expand them
    if np.isscalar(b):
        b = np.array([b])
    if np.isscalar(td):
        td = np.array([td])
    assert len(b) == len(td), "b and td must have the same length"
    assert np.isscalar(small_delta), "𝛿 must be a scalar"
    len_b = len(b)
    return_scalar = True if len_b == 1 else False
    n_param = 7
    # can solve differential equation for all eps simultaneously
    len_w = 20
    [eps, w] = legpts(len_w, np.array([0, 1]))
    # Define the fraction matrix, the rates matrix and the diffusivity matrix, with their derivatives
    S0_double, D, R, R_texs, R_texd, R_fn, R_fsn, D_Dd, D_De = compute_gem_matrices_derivatives(texs, texd,
                                                                                                     Dd, De,
                                                                                                     fn, fsn,
                                                                                                     eps, len_b, len_w)
    # Define the q-value
    q2 = b / (td - small_delta / 3)
    q = np.sqrt(q2)
    # Flatten the initial solution
    S0_double = S0_double.flatten()
    # Expand dimensions to allow broadcasting
    q = np.expand_dims(q, axis=(1, 2, 3))
    q2 = np.expand_dims(q2, axis=(1, 2, 3))
    td = np.expand_dims(td, axis=(1, 2, 3))
    # Compute the signal from the differential equation
    S_double_shape = (1+n_param, len_b, len_w, 3)
    St = first_pulse_jacobian(S0_double, D, R, radius, Ds0,
                              R_texs, R_texd, R_fn, R_fsn,
                              D_Dd, D_De,
                              small_delta, q, S_double_shape)
    St = St.reshape(S_double_shape)
    St = plateau_jacobian(St, D, R,
                          R_texs, R_texd, R_fn, R_fsn,
                          D_Dd, D_De,
                          small_delta, td, q2)
    St = second_pulse_jacobian(St, D, R, radius, Ds0,
                               R_texs, R_texd, R_fn, R_fsn,
                               D_Dd, D_De,
                               small_delta, td, q, S_double_shape)
    St = St.reshape(S_double_shape)
    S_all = (St[..., 0] + St[..., 1] + St[..., 2]).astype(float)
    S_all = np.einsum('ijk, k->ij', S_all, w)
    S = S_all[0, :].astype(float)
    S_jac = S_all[1:, :].T.astype(float)
    # 1D case (1 b-value, diffusion time and small_delta)
    S = np.squeeze(S)
    S_jac = np.squeeze(S_jac)
    if return_scalar:
        S = S.item()
    return S, S_jac


def compute_gem_matrices_derivatives(texs, texd, Dd, De, fn, fsn, eps, len_b, len_w):
    # Define the parameters from the other compartment
    fe = 1 - fn
    fs = fn * fsn
    fd = fn * (1 - fsn)
    # Import initial parameters from [tex, Di, De, f]
    rs = (fe / (fe + fs)) / texs
    rd = (fe / (fe + fd)) / texd
    re = (fs / (fe + fs)) / texs + (fd / (fe + fd)) / texd
    # Derivative of R w.r.t. texs
    rs_texs = -(fe / (fe + fs)) / texs ** 2
    re_texs = fs / fe * rs_texs
    as_re_texs = - (fs / (fe + fs)) / texs ** 2
    # Derivative of R w.r.t. texd
    rd_texd = -(fe / (fe + fd)) / texd ** 2
    re_texd = fd / fe * rd_texd
    as_rd_texd = - (fd / (fe + fd)) / texd ** 2
    # Derivative of R w.r.t. fn
    rs_fn = - fsn / ((fe + fs) ** 2) / texs
    rd_fn = - (1 - fsn) / ((fe + fd) ** 2) / texd
    as_re_fn = -rs_fn  # fsn / ((fe + fs)**2) / texs
    ad_re_fn = -rd_fn  # (1-fsn) / ((fe + fd)**2) / texd
    re_fn = as_re_fn + ad_re_fn
    # Derivative of R w.r.t. fsn
    rs_fsn = - fn * fe / ((fe + fs) ** 2) / texs
    rd_fsn = fn * fe / ((fe + fd) ** 2) / texd
    as_re_fsn = -rs_fsn
    ad_re_fsn = -rd_fsn
    re_fsn = as_re_fsn + ad_re_fsn
    # Define the diffusivity matrix (missing equation from the sphere)
    D = np.zeros((len_w, 3, 3))
    D[:, 0, 0] = 0
    D[:, 1, 1] = Dd * eps ** 2
    D[:, 2, 2] = De
    # Derivative of D w.r.t. Dd
    D_Dd = np.zeros((len_w, 3, 3))
    D_Dd[:, 1, 1] = eps ** 2
    # Derivative of D w.r.t. De
    D_De = np.zeros((len_w, 3, 3))
    D_De[:, 2, 2] = 1
    # Define the rates matrix
    R = np.array([[-rs, 0, fs / fe * rs], [0, -rd, fd / fe * rd], [rs, rd, -re]])
    R_texs = np.array([[-rs_texs, 0, as_re_texs], [0, 0, 0], [rs_texs, 0, -re_texs]])
    R_texd = np.array([[0, 0, 0], [0, -rd_texd, as_rd_texd], [0, rd_texd, -re_texd]])
    R_fn = np.array([[-rs_fn, 0, as_re_fn], [0, -rd_fn, ad_re_fn], [rs_fn, rd_fn, -re_fn]])
    R_fsn = np.array([[-rs_fsn, 0, as_re_fsn], [0, -rd_fsn, ad_re_fsn], [rs_fsn, rd_fsn, -re_fsn]])
    # Define the signal at time 0
    f = np.zeros((len_b, len_w, 3))
    f[..., 0] = fs
    f[..., 1] = fd
    f[..., 2] = fe
    # Derivative of f w.r.t. fn
    f_fn = np.zeros((len_b, len_w, 3))
    f_fn[..., 0] = fsn
    f_fn[..., 1] = 1 - fsn
    f_fn[..., 2] = -1
    # Derivative of f w.r.t. fsn
    f_fsn = np.zeros((len_b, len_w, 3))
    f_fsn[..., 0] = fn
    f_fsn[..., 1] = -fn
    # Define the signal at time 0
    S0_double = np.zeros((8, len_b, len_w, 3))
    S0_double[0, ...] = f
    S0_double[5, ...] = f_fn
    S0_double[7, ...] = f_fsn
    # Expand dimensions to allow broadcasting
    R = np.expand_dims(R, axis=(0, 1))
    D = np.expand_dims(D, axis=0)
    R_texs = np.expand_dims(R_texs, axis=(0, 1))
    R_texd = np.expand_dims(R_texd, axis=(0, 1))
    R_fn = np.expand_dims(R_fn, axis=(0, 1))
    R_fsn = np.expand_dims(R_fsn, axis=(0, 1))
    D_Dd = np.expand_dims(D_Dd, axis=0)
    D_De = np.expand_dims(D_De, axis=0)
    return S0_double, D, R, R_texs, R_texd, R_fn, R_fsn, D_Dd, D_De

#######################################################################################################################
# First pulse jacobian
#######################################################################################################################


# Define the function of the differential equation
def dSdouble_dt_first_pulse(t, S_double, factor, radius, Ds0, flat_q, small_delta,
                            D, D_Dd, D_De, R, R_texs, R_texd, R_fn, R_fsn, S_double_shape):
    S_double = S_double.reshape(S_double_shape).astype(float)
    new_S_double = np.zeros(S_double_shape)
    S, S_texs, S_texd, S_Dd, S_De, S_fn, S_radius, S_fsn = (S_double[0, ...], S_double[1, ...], S_double[2, ...],
                                                             S_double[3, ...], S_double[4, ...], S_double[5, ...],
                                                             S_double[6, ...], S_double[7, ...])
    # Define the gradient function
    q2D_fun = factor * t ** 2 * D
    q2Ds, der_q2Ds_radius = jacobian_dlog_sphere_signal_first_pulse(t, radius, Ds0, flat_q, small_delta)
    q2D_fun[..., 0, 0] = np.expand_dims(q2Ds, axis=-1)
    q2D_radius = np.zeros((S_double_shape[1], S_double_shape[2], 3, 3))
    q2D_radius[..., 0, 0] = np.expand_dims(der_q2Ds_radius, axis=-1)
    q2D_Dd = factor * t ** 2 * D_Dd
    q2D_De = factor * t ** 2 * D_De
    new_S_double[0, ...] = np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S)
    new_S_double[1, ...] = np.einsum('ijkl,ijl->ijk', R_texs, S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_texs)
    new_S_double[2, ...] = np.einsum('ijkl,ijl->ijk', R_texd, S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_texd)
    new_S_double[3, ...] = np.einsum('ijkl,ijl->ijk', (- q2D_Dd), S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_Dd)
    new_S_double[4, ...] = np.einsum('ijkl,ijl->ijk', (- q2D_De), S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_De)
    new_S_double[5, ...] = np.einsum('ijkl,ijl->ijk', R_fn, S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_fn)
    new_S_double[6, ...] = np.einsum('ijkl,ijl->ijk', (- q2D_radius), S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_radius)
    new_S_double[7, ...] = np.einsum('ijkl,ijl->ijk', R_fsn, S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_fsn)
    return new_S_double.flatten()


def first_pulse_jacobian(S0_double, D, R, radius, Ds0, R_texs, R_texd, R_fn, R_fsn, D_Dd, D_De,
                         small_delta, q, S_double_shape):
    factor = (q / small_delta) ** 2
    flat_q = q.flatten()
    # Solution at time 𝛿
    S_double = solve_ivp(dSdouble_dt_first_pulse, t_span=np.array([0, small_delta], dtype=object),
                         y0=S0_double, method='RK45', t_eval=np.array([0, small_delta], dtype=object),
                         rtol=1e-9, atol=1e-9,
                         args=(factor, radius, Ds0, flat_q, small_delta,
                               D, D_Dd, D_De, R, R_texs, R_texd, R_fn, R_fsn, S_double_shape)).y[:, -1]
    return S_double

#######################################################################################################################
# Plateau jacobian
#######################################################################################################################


def lie(A_matrix, B_matrix):
    return np.einsum('ijkl,ijlm->ijkm', A_matrix, B_matrix) - np.einsum('ijkl,ijlm->ijkm', B_matrix, A_matrix)


def expansion(X, dX_dp):
    lie_1 = 1/2*lie(X, dX_dp)
    lie_2 = 1/3*lie(X, lie_1)
    lie_3 = 1/4*lie(X, lie_2)
    lie_4 = 1/5*lie(X, lie_3)
    lie_5 = 1/6*lie(X, lie_4)
    lie_6 = 1/7*lie(X, lie_5)
    return dX_dp-lie_1+lie_2-lie_3+lie_4-lie_5+lie_6


def plateau_jacobian(S_double_small_delta, D, R, R_texs, R_texd, R_fn, R_fsn, D_Dd, D_De, small_delta, td, q2):
    S_double_small_delta = S_double_small_delta.astype(float)
    A = R - q2 * D
    exponential = expm((td - small_delta) * A)
    # Initialisation of the exponential solution
    S_small_delta = S_double_small_delta[0, ...]
    # Define solution at time Δ
    S_double_delta = np.einsum('ijkl,pijl->pijk', exponential, S_double_small_delta)
    # Add the contribution of the (R-q2D) derivatives to the derivative of the exponential
    S_double_delta[1, ...] = S_double_delta[1, ...] + np.einsum('ijkl,ijl->ijk', np.einsum('ijkl,ijlm->ijkm', exponential, expansion((td - small_delta) * A, (td - small_delta) * R_texs)), S_small_delta)  # S_texs
    S_double_delta[2, ...] = S_double_delta[2, ...] + np.einsum('ijkl,ijl->ijk', np.einsum('ijkl,ijlm->ijkm', exponential, expansion((td - small_delta) * A, (td - small_delta) * R_texd)), S_small_delta)  # S_texd
    S_double_delta[3, ...] = S_double_delta[3, ...] + np.einsum('ijkl,ijl->ijk', np.einsum('ijkl,ijlm->ijkm', exponential, expansion((td - small_delta) * A, -(td - small_delta) * q2 * D_Dd)), S_small_delta)  # S_Dd
    S_double_delta[4, ...] = S_double_delta[4, ...] + np.einsum('ijkl,ijl->ijk', np.einsum('ijkl,ijlm->ijkm', exponential, expansion((td - small_delta) * A, -(td - small_delta) * q2 * D_De)), S_small_delta)  # S_De
    S_double_delta[5, ...] = S_double_delta[5, ...] + np.einsum('ijkl,ijl->ijk', np.einsum('ijkl,ijlm->ijkm', exponential, expansion((td - small_delta) * A, (td - small_delta) * R_fn)), S_small_delta)  # S_fn
    S_double_delta[7, ...] = S_double_delta[7, ...] + np.einsum('ijkl,ijl->ijk', np.einsum('ijkl,ijlm->ijkm', exponential, expansion((td - small_delta) * A, (td - small_delta) * R_fsn)), S_small_delta)  # S_fsn
    # Flatten the solution
    S_double_delta = S_double_delta.flatten()
    return S_double_delta

#######################################################################################################################
# Second pulse jacobian
#######################################################################################################################


# Define the function of the differential equation
def dSdouble_dt_second_pulse(t, S_double, factor, radius, Ds0, flat_q, td, small_delta, D, D_Dd, D_De, R, R_texs, R_texd, R_fn, R_fsn, S_double_shape):
    S_double = S_double.reshape(S_double_shape).astype(float)
    new_S_double = np.zeros(S_double_shape)
    S, S_texs, S_texd, S_Dd, S_De, S_fn, S_radius, S_fsn = (S_double[0, ...], S_double[1, ...], S_double[2, ...],
                                                            S_double[3, ...], S_double[4, ...], S_double[5, ...],
                                                            S_double[6, ...], S_double[7, ...])
    # Define the gradient function
    q2D_fun = factor * (small_delta - t) ** 2 * D
    q2Ds, der_q2Ds_radius = jacobian_dlog_sphere_signal_second_pulse(t+td, radius, Ds0, flat_q, td, small_delta)
    q2D_fun[..., 0, 0] = np.expand_dims(q2Ds, axis=-1)
    q2D_radius = np.zeros((S_double_shape[1], S_double_shape[2], 3, 3))
    q2D_radius[..., 0, 0] = np.expand_dims(der_q2Ds_radius, axis=-1)
    q2D_Dd = factor * (small_delta - t) ** 2 * D_Dd
    q2D_De = factor * (small_delta - t) ** 2 * D_De
    new_S_double[0, ...] = np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S)
    new_S_double[1, ...] = np.einsum('ijkl,ijl->ijk', R_texs, S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_texs)
    new_S_double[2, ...] = np.einsum('ijkl,ijl->ijk', R_texd, S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_texd)
    new_S_double[3, ...] = np.einsum('ijkl,ijl->ijk', (- q2D_Dd), S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_Dd)
    new_S_double[4, ...] = np.einsum('ijkl,ijl->ijk', (- q2D_De), S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_De)
    new_S_double[5, ...] = np.einsum('ijkl,ijl->ijk', R_fn, S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_fn)
    new_S_double[6, ...] = np.einsum('ijkl,ijl->ijk', (- q2D_radius), S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_radius)
    new_S_double[7, ...] = np.einsum('ijkl,ijl->ijk', R_fsn, S) + np.einsum('ijkl,ijl->ijk', (R - q2D_fun), S_fsn)
    return new_S_double.flatten()


def second_pulse_jacobian(S_delta, D, R, radius, Ds0, R_texs, R_texd, R_fn, R_fsn, D_Dd, D_De, small_delta, td, q, S_double_shape):
    factor = (q / small_delta) ** 2
    flat_q = q.flatten()
    td = td.flatten()
    # Solve the differential equation starting from the initial solution S_delta
    # Solution at time Δ+𝛿
    S_double = solve_ivp(dSdouble_dt_second_pulse, t_span=np.array([0, small_delta], dtype=object),
                         y0=S_delta, method='RK45', t_eval=np.array([0, small_delta], dtype=object),
                         rtol=1e-9, atol=1e-9,
                         args=(factor, radius, Ds0, flat_q, td, small_delta,
                               D, D_Dd, D_De, R, R_texs, R_texd, R_fn, R_fsn, S_double_shape)).y[:, -1]
    return S_double


#######################################################################################################################
# Optimized GEM for computation of MSE jacobian
#######################################################################################################################


def gem_optimized_mse_jacobian(parameters, acq_parameters, signal_gt):
    gem_vec, gem_jac_vec = gem_jacobian_concatenated_from_vector(parameters, acq_parameters.b,
                                                                 acq_parameters.td,
                                                                 acq_parameters.small_delta)
    if acq_parameters.ndim == 1:
        mse_jacobian = np.sum(2 * gem_jac_vec * broad7(gem_vec - signal_gt), axis=0)
    elif acq_parameters.ndim == 2:
        mse_jacobian = np.sum(2 * gem_jac_vec * broad7(gem_vec - signal_gt), axis=(0, 1))
    else:
        raise NotImplementedError
    return mse_jacobian
