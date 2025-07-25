# Implementation of Non-linear Least Squares
import logging
import numpy as np
import scipy.optimize as op
from tqdm import tqdm
from joblib import Parallel, delayed
from ..models.noise.rice_mean import rice_mean

####################################################################
# Parallelized NLS
####################################################################


def xgboost_powered_nls_loop(xgboost_model, target_signal, microstruct_model, nls_param_lim, initial_gt=None):
    """
    Non-linear least squares algorithm for a single ground truth.
    :param target_signal: signal for all b-values (shells) and diffusion times (Deltas)
    :param microstruct_model: microstructure model
    :param nls_param_lim: bounds for the estimation in the non-linear least squares algorithm
    :param initial_gt: the initial ground truth to start the optimization from. If None, random initializations are used.
    :return: x_sol: estimated ground truth
    :return: x_0: initial ground truth
    """
    # Initial solution
    assert initial_gt is not None, "Initial ground truth must be provided for XGBoost-powered NLS."

    # Function to optimize using the XGBoost model
    if not microstruct_model.has_noise_correction:
        x_0 = initial_gt
        optim_fun = lambda x: np.sum(np.square(xgboost_model.predict(np.array([x]))[0] - target_signal))
        # Optimisation function
        result = op.minimize(fun=optim_fun, x0=x_0, method='L-BFGS-B', bounds=nls_param_lim, tol=1e-14)
    else:
        x_0 = initial_gt[:-1]
        sigma = initial_gt[-1]
        optim_fun = lambda x: np.sum(np.square(rice_mean(xgboost_model.predict(np.array([x]))[0], sigma) - target_signal))
        # Optimisation function
        result = op.minimize(fun=optim_fun, x0=x_0, method='L-BFGS-B', bounds=nls_param_lim, tol=1e-14)
    x_sol = result.x
    return [x_sol, x_0]


def xgboost_powered_nls_parallel(xgboost_model, signal, N, microstruct_model, nls_param_lim, max_nls_verif=5, initial_gt=None):
    """
    Parallelized version of the non-linear least squares algorithm.

    :param signal: signal for all b-values (shells) and diffusion times (Deltas)
    :param N: number of ground truth to estimate
    :param microstruct_model: microstructure model to use
    :param nls_param_lim: bounds for the estimation in the non-linear least squares algorithm
    :param max_nls_verif: maximum number of times the non-linear least squares algorithm is run for each ground truth
    :param initial_gt: the initial ground truth to start the optimization from. If None, random initializations are used.
    :return: x_sol: estimated ground truth
    :return: x_0: initial ground truth
    """
    x_0 = np.empty([N, microstruct_model.n_params])
    x_sol = np.empty([N, microstruct_model.n_params])
    logging.info('Running XGBoost powered Non-linear Least Squares')

    # Change the limits to [0, 1] for the XGBoost model
    if not microstruct_model.has_noise_correction:
        xgb_nls_param_lim = np.array([[0, 1] for _ in range(microstruct_model.n_params)])
    else:
        sigma = initial_gt[:, -1]
        xgb_nls_param_lim = np.array([[0, 1] for _ in range(microstruct_model.n_params - 1)])

    x_parallel = []
    for irunning in tqdm(range(N)):
        x_parallel.append(xgboost_powered_nls_loop_verified(
            xgboost_model, signal[irunning], microstruct_model, xgb_nls_param_lim, initial_gt[irunning], max_nls_verif
        ))

    # NLS_loop_verified is NLS_loop run several times until NLS doesn't output any boundary. Defined below.
    verif_failed_count = 0
    if not microstruct_model.has_noise_correction:
        for index in range(N):
            x_sol[index] = x_parallel[index][0]
            x_0[index] = x_parallel[index][1]
            verif_failed_count += x_parallel[index][2]
    else:
        for index in range(N):
            x_sol[index, :-1] = x_parallel[index][0]
            x_0[index, :-1] = x_parallel[index][1]
            verif_failed_count += x_parallel[index][2]
        x_sol[:, -1] = sigma
        x_0[:, -1] = sigma

    # Multiply the results by the limits to get the real values (XGBoost version)
    # x_sol and x_0 are in [0, 1] so we need to multiply by the limits to get the real values
    if not microstruct_model.has_noise_correction:
        for param_index in range(microstruct_model.n_params):
            x_sol[:, param_index] = (
                x_sol[:, param_index] * (nls_param_lim[param_index][1] - nls_param_lim[param_index][0])
                + nls_param_lim[param_index][0]
            )
            x_0[:, param_index] = (
                x_0[:, param_index] * (nls_param_lim[param_index][1] - nls_param_lim[param_index][0])
                + nls_param_lim[param_index][0]
            )
    else:
        for param_index in range(microstruct_model.n_params - 1):
            x_sol[:, param_index] = (
                x_sol[:, param_index] * (nls_param_lim[param_index][1] - nls_param_lim[param_index][0])
                + nls_param_lim[param_index][0]
            )
            x_0[:, param_index] = (
                x_0[:, param_index] * (nls_param_lim[param_index][1] - nls_param_lim[param_index][0])
                + nls_param_lim[param_index][0]
            )
        # sigma is the last parameter
        x_sol[:, -1] = sigma
        x_0[:, -1] = sigma

    logging.info(f"Failed {max_nls_verif}-times border verification procedure : {verif_failed_count} out of {N}")
    logging.info('Non-linear Least Squares completed')
    return x_sol, x_0


####################################################################
# Verification of each NLS loop max_nls_verif times
####################################################################


def touch_border(x_sol, nls_param_lim, n_param):
    """
    Check if the NLS algorithm has touched the border of the parameter space.
    :param x_sol: estimated ground truth
    :param nls_param_lim: bounds for the estimation in the non-linear least squares algorithm
    :param n_param: number of parameters
    :return: True if the NLS algorithm has touched the border of the parameter space, False otherwise
    """
    for ind_p in range(n_param):
        if nls_param_lim[ind_p][0] != nls_param_lim[ind_p][1]:
            if x_sol[ind_p] == nls_param_lim[ind_p][0] or x_sol[ind_p] == nls_param_lim[ind_p][1]:
                return True
    return False


def xgboost_powered_nls_loop_verified(xgboost_model, target_signal, microstruct_model, nls_param_lim, initial_gt=None, max_nls_verif=5):
    """
    Verification of each NLS loop max_nls_verif times.
    :param target_signal: signal for all b-values (shells) and diffusion times (Deltas)
    :param microstruct_model: microstructure model
    :param nls_param_lim: bounds for the estimation in the non-linear least squares algorithm
    :param initial_gt: the initial ground truth to start the optimization from. If None, random initializations are used.
    :param max_nls_verif: maximum number of times the non-linear least squares algorithm is run for each ground truth
    :return: x_sol: estimated ground truth
    :return: x_0: initial ground truth
    :return: verif_failed: array of booleans indicating if the NLS algorithm has touched the border of the parameter space
    """
    nls_loop_return = xgboost_powered_nls_loop(xgboost_model, target_signal, microstruct_model, nls_param_lim, initial_gt)
    x_sol, x_0 = nls_loop_return[0], nls_loop_return[1]
    iter_nb = 1
    if not microstruct_model.has_noise_correction:
        n_moving_param = microstruct_model.n_params
    else:
        n_moving_param = microstruct_model.n_params - 1

    while touch_border(x_sol, nls_param_lim, n_moving_param) and iter_nb < max_nls_verif:
        nls_loop_return = xgboost_powered_nls_loop(xgboost_model, target_signal, microstruct_model, nls_param_lim, initial_gt=None)
        x_sol, x_0 = nls_loop_return[0], nls_loop_return[1]
        iter_nb += 1
    if touch_border(x_sol, nls_param_lim, n_moving_param) and iter_nb == max_nls_verif:
        verif_failed = 1
        # logging.info("Border touched " + str(int(max_nls_verif)) + " times !")
    else:
        verif_failed = 0
    return [x_sol, x_0, verif_failed]
