import os
import logging
import argparse
import nibabel as nib
import numpy as np
from .powderaverage.powderaverage import powder_average, normalize_sigma, save_data_as_npz
from .models.find_model import find_model
from .models.parameters.acq_parameters import AcquisitionParameters
from .models.parameters.save_parameters import save_estimations_as_nifti, save_initialization_as_nifti
from .nls.nls import nls_parallel
# from .nls.gridsearch_preinitialized import find_nls_initialization_with_preinitialization


def estimate_model_preinitialized(model_name, dwi_path, bvals_path, delta_path, small_delta, lowb_noisemap_path, out_path, 
                                  preinit_params=None,
                                  mask_path=None, 
                                  fixed_parameters=None, adjust_parameter_limits=None, 
                                  save_nls_initialization=False,
                                  n_cores=-1, debug=False):
    """
    Estimate the model parameters for a given set of preprocessed signals,
    providing the b-values, diffusion times and low b-values noise map. A mask is optional but highly recommended.

    Parameters
    ----------
    dwi_path : str
        Path to the preprocessed DWI signal.
    bvals_path : str
        Path to the b-values file. b-values must be provided in ms/µm².
    delta_path : str
        Path to the big delta file. Δ must be provided in ms.
    small_delta : float
        Small delta value in ms.
    lowb_noisemap_path : str
        Path to the low b-values (b < 2ms/µm²) noise map.
    out_path : str
        Path to the output directory. If the directory does not exist, it will be created.
    preinit_params : dict
        Dictionary specifying parameter names as keys and file paths to parameter maps as values.
        The expected keys depend on the chosen model.
        Examples:
            - NEXI model: {'t_ex': 'folder/file_t_ex.nii.gz', 'di': 'folder/file_di.nii.gz'}
            - SANDI model: {'radius': 'folder/file_radius.nii.gz'}
    mask_path : str, optional
        Path to the mask file. The default is None.
    fixed_parameters : tuple, optional
        Allows to fix some parameters of the model if not set to None. Tuple of fixed parameters for the model. 
        The tuple must have the same length as the number of parameters of the model (with or without noise correction).
        Example of use: Fix Di to 2.0µm²/ms and De to 1.0µm²/ms in the NEXI model by specifying fixed_parameters=(None, 2.0 , 1.0, None)
    adjust_parameter_limits : tuple, optional
        Allows to adjust the parameter limits for the Non-Linear Least Squares if not set to None. Tuple of adjusted parameter limits for the model.
        The tuple must have the same length as the number of parameters of the model (with or without noise correction).
        Example of use: Adjust the parameter limits for Di to [1.5, 2.5]µm²/ms and De to [0.5, 1.5]µm²/ms in the NEXI model by specifying adjust_parameter_limits=(None, [1.5, 2.5], [0.5, 1.5], None)
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

    ########################################################
    # Pre-initialization
    # Assert that the preinit_params contains at least one parameter
    microstruct_model = find_model(model_name + 'RiceMean')
    parameter_names = microstruct_model.param_names
    parameter_names_lower = parameter_names.lower()
    pre_initialization_warning = "The preinit_params must contain at least one parameter. The possible parameters are: " 
    pre_initialization_warning += ', '.join(parameter_names_lower) 
    pre_initialization_warning += ". The parameter names are case insensitive. The parameter names must be provided in the preinit_params dictionary in this format: {'parameter_name': 'path_to_file'}."
    
    # Assert that the preinit_params is not None
    assert preinit_params is not None, pre_initialization_warning
    # Assert that the preinit_params is a dictionary
    assert isinstance(preinit_params, dict), pre_initialization_warning
    # Lower the keys of the preinit_params dictionary
    preinit_params = {key.lower(): value for key, value in preinit_params.items()}
    # Assert that all the preinit_params are parameters from the model
    for preinit_param_name in preinit_params.keys():
        if preinit_param_name == 'sigma':
            logging.warning("The parameter name 'sigma' is reserved for the noise standard deviation. Please provide the noise standard deviation only in the 'lowb_noisemap_path' argument.")
        assert preinit_param_name in parameter_names_lower, f"The parameter name '{preinit_param_name}' is not a parameter of the {model_name} model. The possible parameters are: {', '.join(parameter_names_lower)}."
    # Warn that the preinitialization might be very slow if only a few parameters are provided.
    logging.info("Pre-initialization parameters are provided. Warning: The pre-initialization will be very slow if only a few parameters are provided.")
    if len(preinit_params.keys()) == len(parameter_names):
        logging.info("All parameters are provided for the pre-initialization. The pre-initialization will be faster.")
    else:
        # List the parameters that are not provided
        missing_parameters = [parameter_name for parameter_name in parameter_names_lower if parameter_name not in preinit_params.keys()]
        logging.info(f"The following parameters are not provided for the pre-initialization: {', '.join(missing_parameters)}. There is a risk that the pre-initialization will be very slow.")
    ########################################################

    # Create the output directory if it does not exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Convert into powder average
    powder_average_path, updated_bvals_path, updated_delta_path, updated_mask_path = powder_average(
        dwi_path, bvals_path, delta_path, mask_path, out_path, debug=debug
    )
    # NEXI with Rician Mean correction
    normalized_sigma_filename = normalize_sigma(dwi_path, lowb_noisemap_path, bvals_path, out_path)
    powder_average_signal_npz_filename = save_data_as_npz(
        powder_average_path,
        updated_bvals_path,
        updated_delta_path,
        updated_mask_path,
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
    delta = powder_average_signal_npz['delta']
    acq_param = AcquisitionParameters(bvals, delta, small_delta=small_delta)

    # Estimate the model parameters

    # Define the parameter limits for the Non-Linear Least Squares
    parameter_limits = microstruct_model.param_lim
    grid_search_nb_points = microstruct_model.grid_search_nb_points
    max_nls_verif = 1

    # Replace the NLS parameter limits if adjust_parameter_limits if provided
    if adjust_parameter_limits is not None:
        # Assert that the number of parameters in the model and the number of fixed parameters are the same
        assert (len(adjust_parameter_limits) == microstruct_model.n_params - 1) or (len(adjust_parameter_limits) == microstruct_model.n_params), "The number of parameters in the model and the length of adjust_parameter_limits are different. Set the unchanged parameter limits in the adjust_parameter_limits tuple to None."
        if microstruct_model.has_noise_correction and len(adjust_parameter_limits) == microstruct_model.n_params - 1:
            adjust_parameter_limits = list(adjust_parameter_limits) + [microstruct_model.classic_limits[-1]]
        
        # Replace the NLS parameter limits for the fixed parameters
        for i, (adjust_parameter_limits) in enumerate(adjust_parameter_limits):
            if adjust_parameter_limits is not None:
                assert len(adjust_parameter_limits) == 2, "The adjusted parameter limits must be a tuple of two values."
                parameter_limits[i] = [adjust_parameter_limits[0], adjust_parameter_limits[1]]
    
    # Replace the NLS parameter limits for the fixed parameters if provided
    if fixed_parameters is not None:
        # Assert that the number of parameters in the model and the number of fixed parameters are the same
        assert (len(fixed_parameters) == microstruct_model.n_params - 1) or (len(fixed_parameters) == microstruct_model.n_params), "The number of parameters in the model and the length of fixed_parameters are different. Set the moving parameters in the fixed_parameters tuple to None."

        # Replace the NLS parameter limits for the fixed parameters
        for i, fixed_param in enumerate(fixed_parameters):
            if fixed_param is not None:
                parameter_limits[i] = [fixed_param, fixed_param]
    
    # Update the parameter limits in the microstructure model
    microstruct_model.param_lim = parameter_limits
    
    # Define the optimization method
    optimization_method = 'nls'
    print("The optimization method will be 'NLS', since the preinitialized method is not available for XGBoost.")

    # Compute the initial Ground Truth to start the NLS with if requested
    initial_grid_search = False

    ########################################################
    # 
    # Get the indices of the preinit_params
    preinit_param_indices = np.array([parameter_names_lower.index(preinit_param_name) for preinit_param_name in preinit_params.keys()])
    # Sort the preinit_param_indices
    preinit_param_indices = np.sort(preinit_param_indices)
    # The remaining parameters are the moving parameters for the grid search
    moving_param_indices = np.array([i for i in range(len(parameter_names_lower)) if i not in preinit_param_indices])

    ########################################################
    
    initial_gt = None

    if len(preinit_params.keys()) == len(parameter_names):
        logging.info("Starting full preinitialization.")
        # Import the mask
        mask = nib.load(updated_mask_path).get_fdata().astype(bool)
        # Initialize the ground truth with the preinitialization
        initial_gt = np.zeros((voxel_nb, len(parameter_names)))
        for key, preinit_param_filename in preinit_params.items():
            parameter_index = parameter_names_lower.index(key)
            if key.lower() == 'sigma':
                continue
            initial_gt[:, parameter_index] = np.clip(nib.load(preinit_param_filename).get_fdata()[mask], parameter_limits[parameter_index][0], parameter_limits[parameter_index][1])
        # Noise case
        if 'sigma' in parameter_names_lower:
            initial_gt[:, parameter_names_lower.index('sigma')] = sigma
    ########################################################
    else:
        logging.warning("Starting partial preinitialization.")
        if not initial_grid_search:
            # Import the mask
            mask = nib.load(updated_mask_path).get_fdata().astype(bool)
            # Initialize the ground truth with the preinitialization
            initial_gt = np.zeros((voxel_nb, len(parameter_names)))
            for key, preinit_param_filename in preinit_params.items():
                parameter_index = parameter_names_lower.index(key)
                if key.lower() == 'sigma':
                    continue  # Skip the noise parameter, it will be handled later
                initial_gt[:, parameter_index] = np.clip(nib.load(preinit_param_filename).get_fdata()[mask], parameter_limits[parameter_index][0], parameter_limits[parameter_index][1])
            for parameter_index in moving_param_indices:
                # If the parameter is not in the preinit_params, initialize it randomly
                if parameter_index in moving_param_indices:
                    initial_gt[:, parameter_index] = np.random.uniform(parameter_limits[parameter_index][0], parameter_limits[parameter_index][1], voxel_nb)
                    # Implement the constraint of the diffusivities if needed: Not implemented yet

            # Noise case
            if 'sigma' in parameter_names_lower:
                initial_gt[:, parameter_names_lower.index('sigma')] = sigma
            
            logging.warning("The constraints on the diffusivities are not implemented in the case of a partial preinitialization. " \
            "Please provide the preinitialization parameters in the preinit_params dictionary by randomly initializing the other parameters following the constraints by yourself beforehand.")

            
        else:
            raise NotImplementedError("The preinitialization is not implemented for the grid search. Please provide the preinitialization parameters in the preinit_params dictionary.")

            # # Look for any parameter in the dictionnary {'t_ex':'t_ex_filename', 'Di':'Di_filename', 'De':'De_filename', 'f':'f_filename'}
            # # If the parameter is not found, the initialization will be done with the grid search

            # ########################################################
            # initial_gt = find_nls_initialization_with_preinitialization(
            #     signal, sigma, voxel_nb, acq_param, microstruct_model, 
            #     grid_search_param_lim=parameter_limits, 
            #     grid_search_nb_points=grid_search_nb_points, 
            #     moving_param_indices=moving_param_indices, 
            #     initial_gt=initial_gt,
            #     debug=debug
            # )
            # ########################################################

        # Print how many problems were found in the initialization
        if debug:
            problematic_init_mask = np.any(np.isinf(initial_gt), axis=1) | np.any(np.isnan(initial_gt), axis=1)
            number_of_problems = np.sum(problematic_init_mask)
            if number_of_problems > 0:
                logging.info(f"Problems found in the initialization: {np.sum(np.isnan(initial_gt))} out of {voxel_nb} voxels.")
                logging.info(f"Some of the problems are: {initial_gt[problematic_init_mask]}")

        # Save the initialization as nifti
        if save_nls_initialization:
            save_initialization_as_nifti(initial_gt, microstruct_model, powder_average_path, updated_mask_path, out_path)

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

    # Save the model parameters
    if debug:
        np.savez_compressed(
            f'{out_path}/{microstruct_model.name.lower()}_estimations.npz',
            estimations=estimations,
            estimation_init=estimation_init,
        )

    # Save the model parameters as nifti
    save_estimations_as_nifti(estimations, microstruct_model, powder_average_path, updated_mask_path, out_path, optimization_method)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Estimate the NEXI model parameters for a given set of preprocessed '
        'signals, providing the b-values and diffusion time.'
    )
    parser.add_argument('dwi_path', help='path to the preprocessed signals')
    # For conversion from b-values in s/µm² to b-values in ms/µm², divide by 1000
    parser.add_argument('bvals_path', help='path to the b-values (in ms/µm²) txt file')
    parser.add_argument('delta_path', help='path to the diffusion times (in ms) txt file')
    parser.add_argument('lowb_noisemap_path', help='path to the lowb noisemap')
    parser.add_argument('out_path', help='path to the output folder')
    # potential arguments
    # Set to None if not provided
    parser.add_argument('--small_delta', help='small delta (in ms)', required=False, type=float, default=None)
    parser.add_argument('--mask_path', help='path to the mask', required=False, default=None)
    parser.add_argument('--fixed_parameters', help='tuple of fixed parameters', required=False, default=None)
    parser.add_argument('--adjust_parameter_limits', help='tuple of adjusted parameter limits', required=False, default=None)
    parser.add_argument('--save_nls_initialization', help='boolean to save the Non-Linear Least Square initialization', required=False, default=False)
    parser.add_argument('--n_cores', help='number of cores to use for the parallelization', required=False, type=int, default=-1)
    parser.add_argument('--debug', help='debug mode', required=False, action='store_true')
    args = parser.parse_args()

    # estimate_nexi(**vars(parser.parse_args()))
    estimate_model_preinitialized(model_name=args.model_name, dwi_path=args.dwi_path, bvals_path=args.bvals_path, delta_path=args.delta_path, 
                   small_delta=args.small_delta, lowb_noisemap_path=args.lowb_noisemap_path, out_path=args.out_path, 
                   mask_path=args.mask_path, fixed_parameters=args.fixed_parameters, adjust_parameter_limits=args.adjust_parameter_limits, 
                   save_nls_initialization=args.save_nls_initialization,
                   n_cores=args.n_cores, debug=args.debug)
