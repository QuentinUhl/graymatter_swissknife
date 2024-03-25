import os
import logging
import argparse
import numpy as np
from .powderaverage.powderaverage import powder_average, normalize_sigma, save_data_as_npz
from .models.find_model import find_model
from .models.parameters.acq_parameters import AcquisitionParameters
from .models.parameters.save_parameters import save_estimations_as_nifti
from .nls.nls import nls_parallel
from .nls.gridsearch import find_nls_initialization


def estimate_model(model_name, dwi_path, bvals_path, td_path, small_delta, lowb_noisemap_path, out_path, 
                   mask_path=None, fixed_parameters=None, n_cores=-1 ,debug=False):
    """
    Estimate the model parameters for a given set of preprocessed signals,
    providing the b-values, diffusion times and low b-values noise map. A mask is optional but highly recommended.

    Parameters
    ----------
    dwi_path : str
        Path to the preprocessed DWI signal.
    bvals_path : str
        Path to the b-values file. b-values must be provided in ms/µm².
    td_path : str
        Path to the diffusion time file. Diffusion time must be provided in ms.
    small_delta : float
        Small delta value in ms. This value is optional for NEXI (NEXI with Narrow Pulse Approximation (NPA)) but is mandatory for other models.
        If using NEXI with NPA, set small_delta to None.
    lowb_noisemap_path : str
        Path to the low b-values (b < 2ms/µm²) noise map.
    out_path : str
        Path to the output directory. If the directory does not exist, it will be created.
    mask_path : str, optional
        Path to the mask file. The default is None.
    fixed_parameters : tuple
        Allows to fix some parameters of the model if not set to None. Tuple of fixed parameters for the model. 
        The tuple must have the same length as the number of parameters of the model (without noise correction).
        Example of use: Fix Di to 2.0µm²/ms and De to 1.0µm²/ms in the NEXI model by specifying fixed_parameters=(None, 2.0 , 1.0, None)
    n_cores : int, optional
        Number of cores to use for the parallelization. If -1, all available cores are used. The default is -1.
    debug : bool, optional
        Debug mode. The default is False.

    Returns
    -------
    None.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')  # Set the logging level to INFO or desired level

    # Create the output directory if it does not exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Convert into powder average
    powder_average_path, updated_bvals_path, updated_td_path = powder_average(
        dwi_path, bvals_path, td_path, mask_path, out_path, debug=debug
    )
    # NEXI with Rician Mean correction
    normalized_sigma_filename = normalize_sigma(dwi_path, lowb_noisemap_path, bvals_path, out_path)
    powder_average_signal_npz_filename = save_data_as_npz(
        powder_average_path,
        updated_bvals_path,
        updated_td_path,
        out_path,
        normalized_sigma_filename=normalized_sigma_filename,
        debug=debug,
    )

    # Load the powder average signal, normalized sigma, b-values and diffusion time (acquisition parameters)
    powder_average_signal_npz = np.load(powder_average_signal_npz_filename)
    signal = powder_average_signal_npz['signal']
    voxel_nb = len(signal)
    sigma = powder_average_signal_npz['sigma']
    bvals = powder_average_signal_npz['b']
    td = powder_average_signal_npz['td']
    acq_param = AcquisitionParameters(bvals, td, small_delta=small_delta)

    # Estimate the model parameters

    # Define the parameter limits for the Non-Linear Least Squares
    microstruct_model = find_model(model_name + 'RiceMean')
    nls_param_lim = microstruct_model.param_lim
    grid_search_nb_points = microstruct_model.grid_search_nb_points
    max_nls_verif = 1

    # Replace the NLS parameter limits for the fixed parameters if provided
    if fixed_parameters is not None:
        # Assert that the number of parameters in the model and the number of fixed parameters are the same
        assert (len(fixed_parameters) == microstruct_model.n_params - 1) or (len(fixed_parameters) == microstruct_model.n_params), "The number of parameters in the model and the length of fixed_parameters are different. Set the moving parameters in the fixed_parameters tuple to None."

        # Replace the NLS parameter limits for the fixed parameters
        for i, fixed_param in enumerate(fixed_parameters):
            if fixed_param is not None:
                nls_param_lim[i] = [fixed_param, fixed_param]

    # Compute the initial Ground Truth to start the NLS with if requested
    initial_grid_search = True
    initial_gt = None
    if initial_grid_search:
        initial_gt = find_nls_initialization(
            signal, sigma, voxel_nb, acq_param, microstruct_model, nls_param_lim, grid_search_nb_points, debug=debug
        )

    # Compute the NLS estimations
    estimations, estimation_init = nls_parallel(
        signal,
        voxel_nb,
        microstruct_model,
        acq_param,
        nls_param_lim=nls_param_lim,
        max_nls_verif=max_nls_verif,
        initial_gt=initial_gt,
        n_cores=n_cores
    )

    # Save the model parameters
    if debug:
        np.savez_compressed(
            f'{out_path}/{microstruct_model.name.lower()}_estimations.npz',
            estimations=estimations,
            estimation_init=estimation_init,
        )

    # Save the model parameters as nifti
    save_estimations_as_nifti(estimations, microstruct_model, powder_average_path, mask_path, out_path)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Estimate the NEXI model parameters for a given set of preprocessed '
        'signals, providing the b-values and diffusion time.'
    )
    parser.add_argument('dwi_path', help='path to the preprocessed signals')
    # For conversion from b-values in s/µm² to b-values in ms/µm², divide by 1000
    parser.add_argument('bvals_path', help='path to the b-values (in ms/µm²) txt file')
    parser.add_argument('td_path', help='path to the diffusion times (in ms) txt file')
    parser.add_argument('lowb_noisemap_path', help='path to the lowb noisemap')
    parser.add_argument('out_path', help='path to the output folder')
    # potential arguments
    # Set to None if not provided
    parser.add_argument('--small_delta', help='small delta (in ms)', required=False, type=float, default=None)
    parser.add_argument('--mask_path', help='path to the mask', required=False, default=None)
    parser.add_argument('--debug', help='debug mode', required=False, action='store_true')
    args = parser.parse_args()

    # estimate_nexi(**vars(parser.parse_args()))
    estimate_model(
        args.dwi_path,
        args.bvals_path,
        args.td_path,
        args.lowb_noisemap_path,
        args.out_path,
        args.small_delta,
        args.mask_path,
        args.debug,
    )
