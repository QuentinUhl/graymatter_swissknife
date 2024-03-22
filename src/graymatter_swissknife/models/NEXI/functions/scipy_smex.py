"""
Apache-2.0 license

Copyright (c) 2023 Quentin Uhl, Ileana O. Jelescu
smex_signal function is adapted and optimized from the code of Jonas Olesen, jonas@phys.au.dk

Please cite 
"Diffusion time dependence, power-law scaling, and exchange in gray matter"
NeuroImage, Volume 251, 118976 (2022).
https://doi.org/10.1016/j.neuroimage.2022.118976 

as well as my 2024 ISMRM abstract 
"NEXI for the quantification of human gray matter microstructure on a clinical MRI scanner", Uhl et al.
"""

# Implementation of SMEX / Nexi Wide Pulses model in scipy
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre
from scipy.linalg import expm
from numpy.lib.scimath import sqrt as csqrt

# utilitary functions for jacobian, hessian (and Fsumsquares in NLS)
broad4 = lambda matrix: np.tile(matrix[..., np.newaxis], 4)


def legpts(n, interval=np.array([-1, 1])):
    a, b = interval[0], interval[-1]
    roots, weights = roots_legendre(n)
    return np.multiply((b - a)/2, roots) + (a + b)/2, weights*(b-a)/2

# Overwrite scipy.linalg.expm to make sure to handle 2x2 matrices as a 2x2 case.
def expm_2_2(A):
    """Compute the matrix exponential of an array.
    Parameters
    ----------
    A : ndarray
        Input with last two dimensions are square ``(..., n, n)``.
    Returns
    -------
    eA : ndarray
        The resulting matrix exponential with the same shape of ``A``
    Notes
    -----
    Implements the algorithm given in [1], which is essentially a Pade
    approximation with a variable order that is decided based on the array
    data.
    For input with size ``n``, the memory usage is in the worst case in the
    order of ``8*(n**2)``. If the input data is not of single and double
    precision of real and complex dtypes, it is copied to a new array.
    For cases ``n >= 400``, the exact 1-norm computation cost, breaks even with
    1-norm estimation and from that point on the estimation scheme given in
    [2] is used to decide on the approximation order.
    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham, (2009), "A New Scaling
           and Squaring Algorithm for the Matrix Exponential", SIAM J. Matrix
           Anal. Appl. 31(3):970-989, :doi:`10.1137/09074721X`
    .. [2] Nicholas J. Higham and Francoise Tisseur (2000), "A Block Algorithm
           for Matrix 1-Norm Estimation, with an Application to 1-Norm
           Pseudospectra." SIAM J. Matrix Anal. Appl. 21(4):1185-1201,
           :doi:`10.1137/S0895479899356080`
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import expm, sinm, cosm
    Matrix version of the formula exp(0) = 1:
    >>> expm(np.zeros((3, 2, 2)))
    array([[[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]]])
    Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
    applied to a matrix:
    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
    >>> expm(1j*a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
    >>> cosm(a) + 1j*sinm(a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
    """
    a = np.asarray(A)
    if a.size == 1 and a.ndim < 2:
        return np.array([[np.exp(a.item())]])

    if a.ndim < 2:
        raise NotImplementedError('The input array must be at least two-dimensional')
    if a.shape[-1] != a.shape[-2]:
        raise NotImplementedError('Last 2 dimensions of the array must be square')
    n = a.shape[-1]
    # Empty array
    if min(*a.shape) == 0:
        return np.empty_like(a)

    # Scalar case
    if a.shape[-2:] == (1, 1):
        return np.exp(a)

    if not np.issubdtype(a.dtype, np.inexact):
        a = a.astype(float)
    elif a.dtype == np.float16:
        a = a.astype(np.float32)

    # Explicit formula for 2x2 case, formula (2.2) in [1]
    # without Kahan's method numerical instabilities can occur.
    if a.shape[-2:] == (2, 2):
        a1, a2, a3, a4 = (a[..., [0], [0]],
                          a[..., [0], [1]],
                          a[..., [1], [0]],
                          a[..., [1], [1]])
        mu = csqrt((a1-a4)**2 + 4*a2*a3)/2.  # csqrt slow but handles neg.vals

        eApD2 = np.exp((a1+a4)/2.)
        AmD2 = (a1 - a4)/2.
        coshMu = np.cosh(mu)
        sinchMu = np.ones_like(coshMu)
        mask = mu != 0
        sinchMu[mask] = np.sinh(mu[mask]) / mu[mask]
        eA = np.empty((a.shape), dtype=mu.dtype)
        eA[..., [0], [0]] = eApD2 * (coshMu + AmD2*sinchMu)
        eA[..., [0], [1]] = eApD2 * a2 * sinchMu
        eA[..., [1], [0]] = eApD2 * a3 * sinchMu
        eA[..., [1], [1]] = eApD2 * (coshMu - AmD2*sinchMu)
        if np.isrealobj(a):
            return eA.real
        return eA
    else:
        raise NotImplemented

#######################################################################################################################
# SMEX / Nexi Wide Pulses Model
#######################################################################################################################


def smex_signal(tex, Di, De, fi, b, big_delta, small_delta):
    """
    Computes the signal of the SMEX / Nexi Wide Pulses model from the microstructure parameters tex, Di, De and fi
    and the acquisition parameters b, big_delta and small_delta. b and big_delta must have the same shape.
    """
    if np.isscalar(b):
        b = np.array([b])
    if np.isscalar(big_delta):
        big_delta = np.array([big_delta])
    assert len(b) == len(big_delta), "b and big_delta must have the same length"
    assert np.isscalar(small_delta), "𝛿 must be a scalar"
    len_b = len(b)
    return_scalar = True if len_b == 1 else False
    # Import initial parameters from [tex, Di, De, f]
    rs = (1 - fi) * 1 / tex
    # Define the parameters from the other compartment
    fb = 1 - fi
    rb = fi / tex
    # can solve differential equation for all eps simultaneously
    len_w = 20
    [eps, w] = legpts(len_w, np.array([0, 1]))
    D = np.zeros((len_w, 2, 2))
    D[:, 0, 0] = Di * eps ** 2
    D[:, 1, 1] = De
    R = np.array([[-rs, rb], [rs, -rb]])
    # Define the signal at time 0
    f = np.zeros((len_w*len_b, 2))
    f[:, 0] = fi
    f[:, 1] = fb
    # Define the q-value
    q2 = b / (big_delta - small_delta / 3)
    q = np.sqrt(q2)
    # Flatten the initial solution
    flat_f = f.flatten()
    # Expand dimensions to allow broadcasting
    q = np.expand_dims(q, axis=(1, 2, 3))
    q2 = np.expand_dims(q2, axis=(1, 2, 3))
    big_delta = np.expand_dims(big_delta, axis=(1, 2, 3))
    R = np.expand_dims(R, axis=(0, 1))
    D = np.expand_dims(D, axis=0)
    # Define the shape of the solution
    S_shape = (len_b, len_w, 2)
    # Solve the differential equation during the first pulse between time 0 and 𝛿
    St = gradient_pulse(flat_f, D, R, small_delta, q, S_shape)
    # Solve the differential equation between the two pulses between time 𝛿 and Δ
    exp_R_qD = expm_2_2((big_delta - small_delta) * (R - q2 * D))
    St = St.reshape(S_shape)
    St = np.einsum('ijkl, ijl->ijk', exp_R_qD, St.astype(float))
    St = St.flatten()
    # Solve the differential equation during the second pulse between time Δ and Δ+𝛿
    St = reversed_gradient_pulse(St, D, R, small_delta, q, S_shape)
    # Compute the signal
    S = St[::2] + St[1::2]
    S = S.reshape(len_b, len_w)
    S = np.dot(S, np.transpose(w)).astype(float)
    S = np.squeeze(S)
    if return_scalar:
        S = S.item()
    return S


def gradient_pulse(f, D, R, small_delta, q, S_shape):
    """ Modelize the gradient pulse function and solve the differential equation for the wide pulses model. """
    factor = (q / small_delta) ** 2
    # Define the gradient function
    def dSdt(t, S):
        S = S.reshape(S_shape)
        new_S = np.einsum('ijkl, ijl->ijk', R - factor * t ** 2 * D, S)
        new_S = new_S.flatten()
        return new_S
    # Solve the differential equation starting from the initial solution f at time 𝛿
    S = solve_ivp(dSdt, t_span=np.array([0, small_delta], dtype=object), y0=f, method='RK45',
                  t_eval=np.array([0, small_delta], dtype=object), rtol=1e-9, atol=1e-9).y[:, -1]
    return S


def reversed_gradient_pulse(S_delta, D, R, small_delta, q, S_shape):
    """ Modelize the gradient pulse function and solve the differential equation for the wide pulses model. """
    factor = (q / small_delta) ** 2
    # Define the gradient function
    def dSdt(t, S):
        S = S.reshape(S_shape)
        new_S = np.einsum('ijkl, ijl->ijk', R - factor * (small_delta - t) ** 2 * D, S)
        new_S = new_S.flatten()
        return new_S
    # Solve the differential equation starting from the initial solution f at time Δ+𝛿
    S = solve_ivp(dSdt, t_span=np.array([0, small_delta], dtype=object), y0=S_delta, method='RK45',
                  t_eval=np.array([0, small_delta], dtype=object), rtol=1e-9, atol=1e-9).y[:, -1]
    return S


def smex_signal_from_vector(param, b, big_delta, small_delta):
    """ Computes the signal of the Nexi Wide Pulses model from the microstructure parameters vector x and 
    the acquisition parameters b, big_delta and small_delta. """
    return smex_signal(param[0], param[1], param[2], param[3], b, big_delta, small_delta)  # [tex, Di, De, f]


#######################################################################################################################
# SMEX / Nexi Wide Pulses jacobian
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


def smex_jacobian(tex, Di, De, fi, b, big_delta, small_delta):
    _, S_jac = smex_jacobian_concatenated(tex, Di, De, fi, b, big_delta, small_delta)
    return S_jac


def gradient_pulse_jacobian(f, D, R, R_tex, D_Di, D_De, R_fi, f_fi, small_delta, q, Sd_shape):
    factor = (q / small_delta) ** 2

    # Define the function of the differential equation
    def dSdouble_dt(t, S_double):
        S_double = S_double.reshape(Sd_shape).astype(float)
        S, S_tex, S_Di, S_De, S_fi = (S_double[0, ...], S_double[1, ...], S_double[2, ...], S_double[3, ...],
                                      S_double[4, ...])
        # Define the gradient function
        q2_fun = factor * t ** 2
        new_S = np.concatenate((np.einsum('ijkl, ijl->ijk', (R - q2_fun * D), S),
                                np.einsum('ijkl, ijl->ijk', R_tex, S) + np.einsum('ijkl, ijl->ijk', (R - q2_fun * D),
                                                                                  S_tex),
                                np.einsum('ijkl, ijl->ijk', (- q2_fun * D_Di), S) + np.einsum('ijkl, ijl->ijk',
                                                                                              (R - q2_fun * D), S_Di),
                                np.einsum('ijkl, ijl->ijk', (- q2_fun * D_De), S) + np.einsum('ijkl, ijl->ijk',
                                                                                              (R - q2_fun * D), S_De),
                                np.einsum('ijkl, ijl->ijk', R_fi, S) + np.einsum('ijkl, ijl->ijk', (R - q2_fun * D),
                                                                                 S_fi)))
        new_S = new_S.flatten()
        return new_S

    # Solve the differential equation starting from the initial solution f
    len_s = len(f)
    S_double_0 = np.zeros((5, len_s))
    S_double_0[0, :] = f
    S_double_0[4, :] = f_fi
    S_double_0 = S_double_0.flatten()
    # Solution at time 𝛿
    S_double = solve_ivp(dSdouble_dt, t_span=np.array([0, small_delta], dtype=object), y0=S_double_0, method='RK45',
                         t_eval=np.array([0, small_delta], dtype=object), rtol=1e-9, atol=1e-9).y[:, -1]
    return S_double


def gradient_pulse_jacobian_middle(S_double_small_delta, D, R, R_tex, D_Di, D_De, R_fi, small_delta, big_delta, q2, Sd_shape):
    # Define solution at time Δ
    S_double_small_delta = S_double_small_delta.reshape(Sd_shape).astype(float)
    A = R - q2 * D
    exponential = expm((big_delta - small_delta) * A)

    S_small_delta = S_double_small_delta[0, ...]
    S_double_delta = np.einsum('ijkl,pijl->pijk', exponential, S_double_small_delta)

    S_double_delta[1, ...] = (S_double_delta[1, ...] +
                              np.einsum('ijkl,ijl->ijk',
                                        np.einsum('ijkl,ijlm->ijkm', exponential, expansion((big_delta - small_delta) * A, (big_delta - small_delta) * R_tex)),
                                        S_small_delta))
    S_double_delta[2, ...] = S_double_delta[2, ...] + np.einsum('ijkl,ijl->ijk',
        np.einsum('ijkl,ijlm->ijkm', exponential, expansion((big_delta - small_delta) * A, -(big_delta - small_delta) * q2 * D_Di)), S_small_delta)
    S_double_delta[3, ...] = S_double_delta[3, ...] + np.einsum('ijkl,ijl->ijk',
        np.einsum('ijkl,ijlm->ijkm', exponential, expansion((big_delta - small_delta) * A, -(big_delta - small_delta) * q2 * D_De)), S_small_delta)
    S_double_delta[4, ...] = S_double_delta[4, ...] + np.einsum('ijkl,ijl->ijk',
        np.einsum('ijkl,ijlm->ijkm', exponential, expansion((big_delta - small_delta) * A, (big_delta - small_delta) * R_fi)), S_small_delta)

    S_double_delta_flat = S_double_delta.flatten()

    return S_double_delta_flat


def gradient_pulse_jacobian_reversed(S_double_delta, D, R, R_tex, D_Di, D_De, R_fi, small_delta, q, Sd_shape):
    factor = (q / small_delta) ** 2

    # Define the function of the differential equation
    def dSdouble_dt(t, S_double):
        S_double = S_double.reshape(Sd_shape).astype(float)
        S, S_tex, S_Di, S_De, S_fi = (S_double[0, ...], S_double[1, ...], S_double[2, ...], S_double[3, ...],
                                      S_double[4, ...])
        # Define the gradient function
        q2_fun = factor * (small_delta - t) ** 2
        new_S = np.concatenate((np.einsum('ijkl, ijl->ijk', (R - q2_fun * D), S),
                                np.einsum('ijkl, ijl->ijk', R_tex, S) + np.einsum('ijkl, ijl->ijk', (R - q2_fun * D), S_tex),
                                np.einsum('ijkl, ijl->ijk', (- q2_fun * D_Di), S) + np.einsum('ijkl, ijl->ijk', (R - q2_fun * D), S_Di),
                                np.einsum('ijkl, ijl->ijk', (- q2_fun * D_De), S) + np.einsum('ijkl, ijl->ijk', (R - q2_fun * D), S_De),
                                np.einsum('ijkl, ijl->ijk', R_fi, S) + np.einsum('ijkl, ijl->ijk', (R - q2_fun * D), S_fi)))
        new_S = new_S.flatten()
        return new_S

    # Solve the differential equation starting from the initial solution S_double_delta
    # Solution at time Δ+𝛿
    S_double = solve_ivp(dSdouble_dt, t_span=np.array([0, small_delta], dtype=object), y0=S_double_delta, method='RK45',
                         t_eval=np.array([0, small_delta], dtype=object), rtol=1e-9, atol=1e-9).y[:, -1]
    return S_double


def smex_jacobian_from_vector(param, b, big_delta, small_delta):
    return smex_jacobian(param[0], param[1], param[2], param[3], b, big_delta, small_delta)  # [tex, Di, De, f]


#######################################################################################################################
# Optimized SMEX / Nexi Wide Pulses for computation of MSE jacobian
#######################################################################################################################


def smex_optimized_mse_jacobian(parameters, acq_parameters, signal_gt):
    nexi_vec_jac_concatenation = smex_jacobian_concatenated_from_vector(parameters, acq_parameters)
    nexi_vec = nexi_vec_jac_concatenation[0]
    nexi_vec_jac = nexi_vec_jac_concatenation[1]
    if acq_parameters.ndim == 1:
        mse_jacobian = np.sum(2 * nexi_vec_jac * broad4(nexi_vec - signal_gt), axis=0)
    elif acq_parameters.ndim == 2:
        mse_jacobian = np.sum(2 * nexi_vec_jac * broad4(nexi_vec - signal_gt), axis=(0, 1))
    else:
        raise NotImplementedError
    return mse_jacobian


def smex_jacobian_concatenated(tex, Di, De, fi, b, big_delta, small_delta):
    assert len(b) == len(big_delta), "b and big_delta must have the same length"
    assert np.isscalar(small_delta), "𝛿 must be a scalar"
    len_b = len(b)
    return_scalar = True if len_b == 1 else False
    # Import initial parameters from [tex, Di, De, f]
    rs = (1 - fi) / tex
    rs_tex = -(1 - fi) / (tex ** 2)
    rs_fi = -1 / tex
    # Define the parameters from the other compartment
    fb = 1 - fi
    rb = fi / tex  # rs * fi / fb
    rb_tex = -fi / (tex ** 2)  # rs_tex * fi / fb
    rb_fi = -rs_fi
    # can solve differential equation for all eps simultaneously
    len_w = 20
    [eps, w] = legpts(len_w, np.array([0, 1]))
    # Define the diffusion and rate tensors
    D = np.zeros((len_w, 2, 2))
    D[:, 0, 0] = Di * eps ** 2
    D[:, 1, 1] = De
    R = np.array([[-rs, rb], [rs, -rb]])
    # Define the diffusion and rate tensors derivatives
    D_Di = np.zeros((len_w, 2, 2))
    D_Di[:, 0, 0] = eps ** 2
    D_De = np.zeros((len_w, 2, 2))
    D_De[:, 1, 1] = 1
    R_tex = np.array([[-rs_tex, rb_tex], [rs_tex, -rb_tex]])
    R_fi = np.array([[-rs_fi, rb_fi], [rs_fi, -rb_fi]])
    # Define the signal at time 0
    f = np.zeros((len_w * len_b, 2))
    f[:, 0] = fi
    f[:, 1] = fb
    f_fi = np.zeros((len_w * len_b, 2))
    f_fi[:, 0] = 1
    f_fi[:, 1] = -1
    # Define the q-value
    q2 = b / (big_delta - small_delta / 3)
    q = np.sqrt(q2)
    # Compute the signal from the differential equation
    flat_f = f.flatten()
    flat_fi = f_fi.flatten()
    # Expand dimensions to allow broadcasting
    q = np.expand_dims(q, axis=(1, 2, 3))
    q2 = np.expand_dims(q2, axis=(1, 2, 3))
    big_delta = np.expand_dims(big_delta, axis=(1, 2, 3))
    R = np.expand_dims(R, axis=(0, 1))
    D = np.expand_dims(D, axis=0)
    R_tex = np.expand_dims(R_tex, axis=(0, 1))
    R_fi = np.expand_dims(R_fi, axis=(0, 1))
    D_Di = np.expand_dims(D_Di, axis=0)
    D_De = np.expand_dims(D_De, axis=0)
    # Define the shape of the solution
    Sd_shape = (5, len_b, len_w, 2)
    # Compute the signal from the differential equation
    St = gradient_pulse_jacobian(flat_f, D, R, R_tex, D_Di, D_De, R_fi, flat_fi, small_delta, q, Sd_shape)
    St = gradient_pulse_jacobian_middle(St, D, R, R_tex, D_Di, D_De, R_fi, small_delta, big_delta, q2, Sd_shape)
    St = gradient_pulse_jacobian_reversed(St, D, R, R_tex, D_Di, D_De, R_fi, small_delta, q, Sd_shape)
    # Reshape the signal and
    St = St.reshape(Sd_shape)
    # Compute the sum of the compartment signals
    St = (St[..., 0] + St[..., 1])
    # Compute the sum of the quadrature weights
    S_double = St.reshape(5, len_b, len_w).astype(float)
    S_double = np.einsum('ijk,k->ij', S_double, w)
    # Extract the jacobian
    S = S_double[0, ...].astype(float)
    S_jac = S_double[1:, ...].T.astype(float)
    S = np.squeeze(S)
    S_jac = np.squeeze(S_jac)
    if return_scalar:
        S = S.item()
    return S, S_jac


def smex_jacobian_concatenated_from_vector(param, acq_parameters):
    return smex_jacobian_concatenated(param[0], param[1], param[2], param[3],
                                                    acq_parameters.b, acq_parameters.big_delta,
                                                    acq_parameters.small_delta)  # [tex, Di, De, f]
