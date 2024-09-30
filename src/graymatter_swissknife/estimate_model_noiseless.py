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


def estimate_model_noiseless(model_name, dwi_path, bvals_path, delta_path, small_delta, out_path, 
                             mask_path=None, fixed_parameters=None, adjust_parameter_limits=None, 
                             optimization_method='NLS', xgboost_model_path=None, retrain_xgboost=False,
                             n_cores=-1, force_cpu=False, debug=False):
    """
    Estimate the noiseless NEXI model parameters for a given set of preprocessed signals,
    providing the b-values and diffusion times. A mask is optional but highly recommended.

    Parameters
    ----------
    dwi_path : str
        Path to the preprocessed DWI signal.
    bvals_path : str
        Path to the b-values file. b-values must be provided in ms/µm².
    delta_path : str
        Path to the big delta file. Δ must be provided in ms.
    small_delta : float
        Small delta value in ms. This value is optional for NEXI (NEXI with Narrow Pulse Approximation (NPA)) but is mandatory for other models.
        If using NEXI with NPA, set small_delta to None.
    out_path : str
        Path to the output directory. If the directory does not exist, it will be created.
    mask_path : str, optional
        Path to the mask file. The default is None.
    fixed_parameters : tuple, optional
        Allows to fix some parameters of the model if not set to None. Tuple of fixed parameters for the model. 
        The tuple must have the same length as the number of parameters of the model.
        Example of use: Fix Di to 2.0µm²/ms and De to 1.0µm²/ms in the NEXI model by specifying fixed_parameters=(None, 2.0 , 1.0, None)
    adjust_parameter_limits : tuple, optional
        Allows to adjust the parameter limits for the Non-Linear Least Squares if not set to None. Tuple of adjusted parameter limits for the model.
        The tuple must have the same length as the number of parameters of the model.
        Example of use: Adjust the parameter limits for Di to [1.5, 2.5]µm²/ms and De to [0.5, 1.5]µm²/ms in the NEXI model by specifying adjust_parameter_limits=(None, [1.5, 2.5], [0.5, 1.5], None)
    optimization_method : string, optional
        Optimization method to use. 'NLS' for Non-Linear Least Squares. 'XGBoost' for XGBoost (Machine Learning). 
        The default is 'NLS'.
    xgboost_model_path : string, optional
        Path to your future or pretrained XGBoost model file. Allowed file extensions are json or ubj.
        Example: 'yourfolder/cohort_microstructuremodel.ubj'. 
        The default is None.
    retrain_xgboost : bool, optional
        If True, the XGBoost model will be retrained. If False, the XGBoost model will be loaded from the xgboost_model_path.
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
    powder_average_path, updated_bvals_path, updated_delta_path, updated_mask_path = powder_average(
        dwi_path, bvals_path, delta_path, mask_path, out_path, debug=debug
    )
    # Model without Rician Mean correction
    powder_average_signal_npz_filename = save_data_as_npz(
        powder_average_path, updated_bvals_path, updated_delta_path, updated_mask_path, out_path, debug=debug
    )

    # Load the powder average signal, b-values and diffusion time (acquisition parameters)
    powder_average_signal_npz = np.load(powder_average_signal_npz_filename)
    signal = powder_average_signal_npz['signal']
    voxel_nb = len(signal)
    bvals = powder_average_signal_npz['b']
    delta = powder_average_signal_npz['delta']
    acq_param = AcquisitionParameters(bvals, delta, small_delta=small_delta)

    # Estimate the microstructure model parameters

    # Define the parameter limits for the Non-Linear Least Squares
    microstruct_model = find_model(model_name)
    parameter_limits = microstruct_model.param_lim
    grid_search_nb_points = microstruct_model.grid_search_nb_points
    max_nls_verif = 1

    # Replace the NLS parameter limits if adjust_parameter_limits is provided
    if adjust_parameter_limits is not None:
        # Assert that the number of parameters in the model and the number of fixed parameters are the same
        assert (len(adjust_parameter_limits) == microstruct_model.n_params - 1) or (len(adjust_parameter_limits) == microstruct_model.n_params), "The number of parameters in the model and the length of adjust_parameter_limits are different. Set the unchanged parameter limits in the adjust_parameter_limits tuple to None."

        # Replace the NLS parameter limits for the fixed parameters
        for i, (adjust_parameter_limits) in enumerate(adjust_parameter_limits):
            if adjust_parameter_limits is not None:
                assert len(adjust_parameter_limits) == 2, "The adjusted parameter limits must be a tuple of two values."
                parameter_limits[i] = [adjust_parameter_limits[0], adjust_parameter_limits[1]]
    
    # Replace the NLS parameter limits for the fixed parameters if provided
    if fixed_parameters is not None:
        # Assert that the number of parameters in the model and the number of fixed parameters are the same
        assert microstruct_model.n_params == len(fixed_parameters), "The number of parameters in the model and the length of fixed_parameters are different. Set the moving parameters in the fixed_parameters tuple to None."

        # Replace the NLS parameter limits for the fixed parameters
        for i, fixed_param in enumerate(fixed_parameters):
            if fixed_param is not None:
                parameter_limits[i] = [fixed_param, fixed_param]
    
    # Update the parameter limits in the microstructure model
    microstruct_model.param_lim = parameter_limits
    
    # Check the optimization method
    optimization_method = optimization_method.lower()
    assert optimization_method in ['nls', 'xgboost'], "The optimization method must be 'NLS' or 'XGBoost'."

    if optimization_method == 'nls':

        # Compute the initial Ground Truth to start the NLS with if requested
        initial_grid_search = True
        initial_gt = None
        if initial_grid_search:
            initial_gt = find_nls_initialization(
                signal, None, voxel_nb, acq_param, microstruct_model, parameter_limits, grid_search_nb_points, debug=debug
            )
            # Print how many problems were found in the initialization
            if debug:
                problematic_init_mask = np.any(np.isinf(initial_gt), axis=1) | np.any(np.isnan(initial_gt), axis=1)
                number_of_problems = np.sum(problematic_init_mask)
                if number_of_problems > 0:
                    logging.info(f"Problems found in the initialization: {np.sum(np.isnan(initial_gt))} out of {voxel_nb} voxels.")
                    logging.info(f"Some of the problems are: {initial_gt[problematic_init_mask]}")

        # Compute the NLS estimations
        estimations, estimation_init = nls_parallel(
            signal,
            voxel_nb,
            microstruct_model,
            acq_param,
            nls_param_lim=parameter_limits,
            max_nls_verif=max_nls_verif,
            initial_gt=initial_gt,
            n_cores=n_cores
        )
    
    elif optimization_method == 'xgboost':

        # Import the XGBoost functions
        from .xgboost.define_xgboost_model import define_xgboost_model
        from .xgboost.apply_xgboost_model import apply_xgboost_model

        # No initialization in XGBoost
        estimation_init = None

        # Check if the XGBoost model path is provided
        assert xgboost_model_path is not None, "The XGBoost model path must be provided, either to save or load the model."

        # Define the XGBoost model from a file or generate and train it
        n_training_samples = 1000000
        xgboost_model = define_xgboost_model(xgboost_model_path, retrain_xgboost, 
                                             microstruct_model, acq_param, n_training_samples, None, force_cpu, n_cores)
        
        # Apply the XGBoost model
        estimations = apply_xgboost_model(xgboost_model, signal, microstruct_model)
    
    else:
        raise ValueError("The optimization method must be 'NLS' or 'XGBoost'.")

    # Save the NEXI model parameters
    if debug:
        np.savez_compressed(
            f'{out_path}/{microstruct_model.name.lower()}_estimations.npz',
            estimations=estimations,
            estimation_init=estimation_init,
        )

    # Save the NEXI model parameters as nifti
    save_estimations_as_nifti(estimations, microstruct_model, powder_average_path, updated_mask_path, out_path, optimization_method)


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
    parser.add_argument('delta_path', help='path to the diffusion times (in ms) txt file')
    parser.add_argument('out_path', help='path to the output folder')
    # potential arguments
    # Set to None if not provided
    parser.add_argument('--small_delta', help='small delta (in ms)', required=False, type=float, default=None)
    parser.add_argument('--mask_path', help='path to the mask', required=False, default=None)
    parser.add_argument('--fixed_parameters', help='tuple of fixed parameters', required=False, default=None)
    parser.add_argument('--adjust_parameter_limits', help='tuple of adjusted parameter limits', required=False, default=None)
    parser.add_argument('--optimization_method', help='optimization method to use', required=False, default='NLS')
    parser.add_argument('--xgboost_model_path', help='path to the XGBoost model file', required=False, default=None)
    parser.add_argument('--retrain_xgboost', help='retrain the XGBoost model', required=False, action='store_true')
    parser.add_argument('--n_cores', help='number of cores to use for the parallelization', required=False, type=int, default=-1)
    parser.add_argument('--force_cpu', help='debug mode - GPU low memory', required=False, action='store_true')
    parser.add_argument('--debug', help='debug mode', required=False, action='store_true')
    args = parser.parse_args()

    # estimate_nexi(**vars(parser.parse_args()))
    estimate_model_noiseless(model_name=args.model_name, dwi_path=args.dwi_path, bvals_path=args.bvals_path, delta_path=args.delta_path, 
                             small_delta=args.small_delta, out_path=args.out_path, mask_path=args.mask_path, 
                             fixed_parameters=args.fixed_parameters, adjust_parameter_limits=args.adjust_parameter_limits, 
                             optimization_method='NLS', xgboost_model_path=None, retrain_xgboost=False,
                             n_cores=-1, force_cpu=False, debug=args.debug)
