import os
import logging
import argparse
import numpy as np
from .powderaverage.powderaverage import powder_average, save_data_as_npz
from .models.find_model import find_model
from .models.parameters.acq_parameters import AcquisitionParameters
from .models.parameters.save_parameters import save_estimations_as_nifti
from .nls.nls import nls_parallel
from .nls.gridsearch import find_nls_initialization


def estimate_model_noiseless(model_name, dwi_path, bvals_path, td_path, small_delta, out_path, mask_path=None, debug=False):
    """
    Estimate the noiseless NEXI model parameters for a given set of preprocessed signals,
    providing the b-values and diffusion times. A mask is optional but highly recommended.

    Parameters
    ----------
    dwi_path : str
        Path to the preprocessed DWI signal.
    bvals_path : str
        Path to the b-values file. b-values must be provided in ms/µm².
    td_path : str
        Path to the diffusion time file. Diffusion time must be provided in ms.
    out_path : str
        Path to the output directory. If the directory does not exist, it will be created.
    mask_path : str, optional
        Path to the mask file. The default is None.
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
    # Model without Rician Mean correction
    powder_average_signal_npz_filename = save_data_as_npz(
        powder_average_path, updated_bvals_path, updated_td_path, out_path, debug=debug
    )

    # Load the powder average signal, b-values and diffusion time (acquisition parameters)
    powder_average_signal_npz = np.load(powder_average_signal_npz_filename)
    signal = powder_average_signal_npz['signal']
    voxel_nb = len(signal)
    bvals = powder_average_signal_npz['b']
    td = powder_average_signal_npz['td']
    acq_param = AcquisitionParameters(bvals, td, small_delta=small_delta)

    # Estimate the microstructure model parameters

    # Define the parameter limits for the Non-Linear Least Squares
    microstruct_model = find_model(model_name)
    nls_param_lim = microstruct_model.param_lim
    grid_search_nb_points = microstruct_model.grid_search_nb_points
    max_nls_verif = 1

    # Compute the initial Ground Truth to start the NLS with if requested
    initial_grid_search = True
    initial_gt = None
    if initial_grid_search:
        initial_gt = find_nls_initialization(
            signal, None, voxel_nb, acq_param, microstruct_model, nls_param_lim, grid_search_nb_points, debug=debug
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
    )

    # Save the NEXI model parameters
    if debug:
        np.savez_compressed(
            f'{out_path}/{microstruct_model.name.lower()}_estimations.npz',
            estimations=estimations,
            estimation_init=estimation_init,
        )

    # Save the NEXI model parameters as nifti
    save_estimations_as_nifti(estimations, microstruct_model, powder_average_path, mask_path, out_path)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Estimate the NEXI model parameters for a given set of preprocessed '
        'signals, providing the b-values and diffusion time.'
    )
    parser.add_argument('model_name', help='microstructure model name between Nexi, Smex, Sandi, Sandix and Gem. NexiDot is also possible.')
    parser.add_argument('dwi_path', help='path to the preprocessed signals')
    # For conversion from b-values in s/µm² to b-values in ms/µm², divide by 1000
    parser.add_argument('bvals_path', help='path to the b-values (in ms/µm²) txt file')
    parser.add_argument('td_path', help='path to the diffusion times (in ms) txt file')
    parser.add_argument('out_path', help='path to the output folder')
    # potential arguments
    # Set to None if not provided
    parser.add_argument('--mask_path', help='path to the mask', required=False, default=None)
    parser.add_argument('--debug', help='debug mode', required=False, action='store_true')
    args = parser.parse_args()

    # estimate_nexi(**vars(parser.parse_args()))
    estimate_model_noiseless(args.model_name ,args.dwi_path, args.bvals_path, args.td_path, args.out_path, args.mask_path, args.debug)
