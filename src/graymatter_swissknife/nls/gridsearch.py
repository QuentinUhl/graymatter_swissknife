import logging
import numpy as np
from tqdm import tqdm
import itertools
from joblib import Parallel, delayed
from ..models.noise.rice_mean import rice_mean
from ..models.noise.folded_normal_mean import folded_normal_mean


def find_nls_initialization(
    signal, sigma, nb_estimates, acq_param, microstruct_model, grid_search_param_lim, grid_search_nb_points, debug=False
):
    """
    Find the initialization of the Non-linear least squares algorithm.
    :param signal: signal for all b-values (shells) and diffusion times (Deltas)
    :param sigma: noise standard deviation
    :param nb_estimates: number of ground truth to estimate
    :param acq_param: acquisition parameters
    :param microstruct_model: microstructure model
    :param grid_search_param_lim: bounds for the grid search
    :param debug: if True, print debug information
    :return: initial_gt: array of optimized initial ground truths for the NLS algorithm
    """
    # Access to grid search hyperparameters
    n_cores = -1  # multiprocessing.cpu_count()
    overlap = 1  # from 0 to 1 : 0 mean no overlap of the grid ranges, 1 means complete overlap

    # Check that the number of parameters in the grid search and in the model are the same
    assert microstruct_model.n_params == len(
        grid_search_param_lim
    ), "Number of parameters in the grid search and in the model are different"

    # Define number of moving parameters
    if microstruct_model.has_noise_correction:
        n_moving_param = microstruct_model.n_params - 1  # -1 because sigma is fixed
    else:
        n_moving_param = microstruct_model.n_params

    logging.info(
        f"Initializing first grid search inside :\n{grid_search_param_lim}\nusing model {microstruct_model.name}\n..."
    )

    # Generate the parameter grid combinations (avoiding the limits of the bounds)
    param_grid = []
    for p in range(n_moving_param):
        if grid_search_param_lim[p][0] == grid_search_param_lim[p][1]:
            param_grid.append([grid_search_param_lim[p][0]])
        else:
            param_grid.append(
                np.linspace(grid_search_param_lim[p][0], grid_search_param_lim[p][1], grid_search_nb_points[p] + 2)[1:-1].tolist()
            )
    grid_combinations = list(itertools.product(*param_grid))
    parameters = np.array(grid_combinations)

    # Generate the non-corrected signal dictionary (or grid)
    if microstruct_model.has_noise_correction:
        non_corrected_model = microstruct_model.non_corrected_model
        logging.info(f"Generating {non_corrected_model.name} signal dictionary")
        signal_dict = np.array(
            Parallel(n_jobs=n_cores)(
                delayed(non_corrected_model.get_signal)(params, acq_param) for params in tqdm(grid_combinations)
            )
        )
    else:
        signal_dict = np.array(
            Parallel(n_jobs=n_cores)(
                delayed(microstruct_model.get_signal)(params, acq_param) for params in tqdm(grid_combinations)
            )
        )
    if debug:
        logging.info(f"Signal dictionary shape : {signal_dict.shape}")

    # Find the initial ground truth whether the model has a Rician mean correction or not
    logging.info("Extracting initial Ground Truth from dictionary")
    if microstruct_model.has_noise_correction:
        # If the model has noise correction, we need to find the initial GT with the noise correction
        if microstruct_model.name == "NEXI_Folded_Normal":
            # Noise correction case : Folded Normal
            initial_gt = np.array(
                Parallel(n_jobs=n_cores)(
                    delayed(least_square_argmin_folded_normal_corrected)(
                        signal[i], sigma[i], signal_dict, parameters, acq_param.ndim
                    )
                    for i in tqdm(range(nb_estimates))
                )
            )
        else:
            # Noise correction case : Rice noise
            initial_gt = np.array(
                Parallel(n_jobs=n_cores)(
                    delayed(least_square_argmin_rician_corrected)(
                        signal[i], sigma[i], signal_dict, parameters, acq_param.ndim
                    )
                    for i in tqdm(range(nb_estimates))
                )
            )
        initial_gt = np.hstack((initial_gt, np.array([sigma]).T))
    else:
        initial_gt = np.array(
            Parallel(n_jobs=n_cores)(
                delayed(least_square_argmin)(signal[i], signal_dict, parameters, acq_param.ndim)
                for i in tqdm(range(nb_estimates))
            )
        )
    if debug:
        logging.info(f'initial_gt shape : {initial_gt.shape}')

    # Add randomness to the initial GT
    if microstruct_model.has_noise_correction:
        # If the model has noise correction, we need to add randomness to the initial GT except to sigma (-1)
        for parameter in range(len(grid_search_param_lim) - 1):
            initial_gt[:, parameter] += (
                (1 + overlap)
                * (grid_search_param_lim[parameter][1] - grid_search_param_lim[parameter][0])
                / (grid_search_nb_points[parameter] + 1)
                * (np.random.rand(nb_estimates) - 0.5)
            )
    else:
        for parameter in range(len(grid_search_param_lim)):
            initial_gt[:, parameter] += (
                (1 + overlap)
                * (grid_search_param_lim[parameter][1] - grid_search_param_lim[parameter][0])
                / (grid_search_nb_points[parameter] + 1)
                * (np.random.rand(nb_estimates) - 0.5)
            )

    # Print the first initial ground truth to check
    if debug:
        for p_ind, param_name in enumerate(microstruct_model.param_names):
            logging.info(f'First initial_gt {param_name} : {initial_gt[0, p_ind]}')
    logging.info("First Grid Search completed\n")
    return initial_gt


def least_square_argmin(signal_i, signal_dict, parameters, acq_param_ndim):
    """
    Compute the argmin of the least squares function for a given signal.
    :param signal_i: signal for all b-values (shells) and diffusion times (Deltas)
    :param signal_dict: signal dictionary
    :param parameters: parameters of the signal dictionary
    :param acq_param_ndim: number of dimensions of the acquisition parameters
    :return: argmin of the least squares function
    """
    if acq_param_ndim == 2:
        lstsqr = np.sum(np.square(np.expand_dims(signal_i, axis=0) - signal_dict), axis=(-2, -1))
    elif acq_param_ndim == 1:
        lstsqr = np.sum(np.square(np.expand_dims(signal_i, axis=0) - signal_dict), axis=-1)
    else:
        raise NotImplementedError
    return parameters[np.unravel_index(lstsqr.argmin(), lstsqr.shape)]


def least_square_argmin_rician_corrected(signal_i, sigma_i, initial_signal_dict, parameters, acq_param_ndim):
    """
    Compute the argmin of the least squares function for a given not yet corrected signal and noise level.
    :param signal_i: signal for all b-values (shells) and diffusion times (Deltas)
    :param signal_dict: signal dictionary
    :param parameters: parameters of the signal dictionary
    :param acq_param_ndim: number of dimensions of the acquisition parameters
    :return: argmin of the least squares function
    """
    # Update the signal dictionary with the noise level
    corrected_signal_dict = rice_mean(initial_signal_dict, sigma_i)
    # Compute the argmin of the least squares function
    if acq_param_ndim == 2:
        lstsqr = np.sum(np.square(np.expand_dims(signal_i, axis=0) - corrected_signal_dict), axis=(-2, -1))
    elif acq_param_ndim == 1:
        lstsqr = np.sum(np.square(np.expand_dims(signal_i, axis=0) - corrected_signal_dict), axis=-1)
    else:
        raise NotImplementedError
    return parameters[np.unravel_index(lstsqr.argmin(), lstsqr.shape)]


def least_square_argmin_folded_normal_corrected(signal_i, sigma_i, initial_signal_dict, parameters, acq_param_ndim):
    """
    Compute the argmin of the least squares function for a given not yet corrected signal and noise level.
    :param signal_i: signal for all b-values (shells) and diffusion times (Deltas)
    :param signal_dict: signal dictionary
    :param parameters: parameters of the signal dictionary
    :param acq_param_ndim: number of dimensions of the acquisition parameters
    :return: argmin of the least squares function
    """
    # Update the signal dictionary with the noise level
    corrected_signal_dict = folded_normal_mean(initial_signal_dict, sigma_i)
    # Compute the argmin of the least squares function
    if acq_param_ndim == 2:
        lstsqr = np.sum(np.square(np.expand_dims(signal_i, axis=0) - corrected_signal_dict), axis=(-2, -1))
    elif acq_param_ndim == 1:
        lstsqr = np.sum(np.square(np.expand_dims(signal_i, axis=0) - corrected_signal_dict), axis=-1)
    else:
        raise NotImplementedError
    return parameters[np.unravel_index(lstsqr.argmin(), lstsqr.shape)]
