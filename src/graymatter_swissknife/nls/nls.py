# Implementation of Non-linear Least Squares
import logging
import numpy as np
import scipy.optimize as op
from tqdm import tqdm
from joblib import Parallel, delayed

####################################################################
# Parallelized NLS
####################################################################


def nls_loop(target_signal, microstruct_model, acq_param, nls_param_lim, initial_gt=None):
    """
    Non-linear least squares algorithm for a single ground truth.
    :param target_signal: signal for all b-values (shells) and diffusion times (Deltas)
    :param microstruct_model: microstructure model
    :param acq_param: acquisition parameters
    :param nls_param_lim: bounds for the estimation in the non-linear least squares algorithm
    :param initial_gt: the initial ground truth to start the optimization from. If None, random initializations are used.
    :return: x_sol: estimated ground truth
    :return: x_0: initial ground truth
    """
    # Initial solution
    if not (initial_gt is None):
        x_0 = initial_gt
    else:
        x_0 = np.empty((microstruct_model.n_params,))
        for ind_p in range(microstruct_model.n_params):
            x_0[ind_p] = np.random.uniform(nls_param_lim[ind_p][0], nls_param_lim[ind_p][1])
        for constraint in microstruct_model.constraints:
            x_0 = constraint(x_0)

    # Function to optimize
    optim_fun = lambda x: np.sum(np.square(microstruct_model.get_signal(x, acq_param) - target_signal))

    # Hessian of this function
    # fun_hess = lambda x: optim_fun_hess(x, b, td, Y, acq_dimension)

    # Optimisation function
    if hasattr(microstruct_model, 'get_mse_jacobian'):
        # Jacobian of the Mean Square Error
        fun_jac = lambda x: microstruct_model.get_mse_jacobian(x, acq_param, target_signal)
        result = op.minimize(fun=optim_fun, x0=x_0, method='L-BFGS-B', jac=fun_jac, bounds=nls_param_lim, tol=1e-14)
    else:
        result = op.minimize(fun=optim_fun, x0=x_0, method='L-BFGS-B', bounds=nls_param_lim, tol=1e-14)
    x_sol = result.x
    return [x_sol, x_0]


def nls_parallel(signal, N, microstruct_model, acq_param, nls_param_lim, max_nls_verif=5, initial_gt=None, n_cores=-1):
    """
    Parallelized version of the non-linear least squares algorithm.

    :param signal: signal for all b-values (shells) and diffusion times (Deltas)
    :param N: number of ground truth to estimate
    :param microstruct_model: microstructure model to use
    :param acq_param: acquisition parameters
    :param nls_param_lim: bounds for the estimation in the non-linear least squares algorithm
    :param max_nls_verif: maximum number of times the non-linear least squares algorithm is run for each ground truth
    :param initial_gt: the initial ground truth to start the optimization from. If None, random initializations are used.
    :param n_cores: number of cores to use for the parallelization. If -1, all available cores are used.
    :return: x_sol: estimated ground truth
    :return: x_0: initial ground truth
    """
    x_0 = np.empty([N, microstruct_model.n_params])
    x_sol = np.empty([N, microstruct_model.n_params])
    if initial_gt is None:
        initial_gt = [None for _ in range(N)]
    logging.info('Running Parallelized Non-linear Least Squares')
    if hasattr(microstruct_model, 'get_mse_jacobian'):
        logging.info("Optimization using implemented jacobian.")
    else:
        logging.info("Warning : Jacobian not implemented for this model. Optimization will be slower.")
    x_parallel = Parallel(n_jobs=n_cores)(
        delayed(nls_loop_verified)(
            signal[irunning], microstruct_model, acq_param, nls_param_lim, initial_gt[irunning], max_nls_verif
        )
        for irunning in tqdm(range(N))
    )  # n_jobs=-1, verbose=50 for verbose
    # NLS_loop_verified is NLS_loop run several times until NLS doesn't output any boundary. Defined below.
    verif_failed_count = 0
    for index in range(N):
        x_sol[index] = x_parallel[index][0]
        x_0[index] = x_parallel[index][1]
        verif_failed_count += x_parallel[index][2]
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
        if x_sol[ind_p] == nls_param_lim[ind_p][0] or x_sol[ind_p] == nls_param_lim[ind_p][1]:
            return True
    return False


def nls_loop_verified(target_signal, microstruct_model, acq_param, nls_param_lim, initial_gt=None, max_nls_verif=5):
    """
    Verification of each NLS loop max_nls_verif times.
    :param target_signal: signal for all b-values (shells) and diffusion times (Deltas)
    :param microstruct_model: microstructure model
    :param acq_param: acquisition parameters
    :param nls_param_lim: bounds for the estimation in the non-linear least squares algorithm
    :param initial_gt: the initial ground truth to start the optimization from. If None, random initializations are used.
    :param max_nls_verif: maximum number of times the non-linear least squares algorithm is run for each ground truth
    :return: x_sol: estimated ground truth
    :return: x_0: initial ground truth
    :return: verif_failed: array of booleans indicating if the NLS algorithm has touched the border of the parameter space
    """
    nls_loop_return = nls_loop(target_signal, microstruct_model, acq_param, nls_param_lim, initial_gt)
    x_sol, x_0 = nls_loop_return[0], nls_loop_return[1]
    iter_nb = 1
    while touch_border(x_sol, nls_param_lim, microstruct_model.n_params) and iter_nb < max_nls_verif:
        nls_loop_return = nls_loop(target_signal, microstruct_model, acq_param, nls_param_lim, initial_gt=None)
        x_sol, x_0 = nls_loop_return[0], nls_loop_return[1]
        iter_nb += 1
    if touch_border(x_sol, nls_param_lim, microstruct_model.n_params) and iter_nb == max_nls_verif:
        verif_failed = 1
        # logging.info("Border touched " + str(int(max_nls_verif)) + " times !")
    else:
        verif_failed = 0
    return [x_sol, x_0, verif_failed]
