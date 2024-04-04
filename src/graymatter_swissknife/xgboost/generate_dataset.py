import numpy as np
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
from ..models.parameters.mist_parameters import MIcroSTructureParameters


def generate_dataset(microstruct_model, acq_param, n_samples, sigma=None, n_cores=-1):
    """Generate a dataset for the XGBoost model.

    Args:
        model_name (str): Name of the microstructure model.
        n_samples (int): Number of samples to generate.
        n_cores (int): Number of cores to use for the parallel computation.

    Returns:
        np.ndarray: The generated dataset.
    """
    # Initialize the parameters given the parameter limits    
    synthetic_param = MIcroSTructureParameters(microstruct_model, None)
    synthetic_param, normalized_synthetic_param = synthetic_param.initialize_random_parameters(n_samples, microstruct_model)

    # Generate the synthetic signals from the parameters and the microstructure model   
    if microstruct_model.has_rician_mean_correction:
        assert sigma is not None, "The standard deviation of the Rician noise must be provided"
        non_corrected_model = microstruct_model.non_corrected_model
        logging.info(f"Generating {non_corrected_model.name} signal dictionary")
        synthetic_signal = np.array(
            Parallel(n_jobs=n_cores)(
                delayed(non_corrected_model.get_signal)(params, acq_param) for params in tqdm(synthetic_param)
            )
        )
        # Select random sigma values n_samples times with replacement from the given sigma values
        random_sigma = np.random.choice(sigma, n_samples, replace=True)
        # Add Rician noise to the synthetic signals
        real_part = synthetic_signal + np.random.randn(synthetic_signal.shape[0], synthetic_signal.shape[1]) * random_sigma[:, None]
        imag_part = np.random.randn(synthetic_signal.shape[0], synthetic_signal.shape[1]) * random_sigma[:, None]
        synthetic_signal = np.sqrt(real_part ** 2 + imag_part ** 2)
    else:
        synthetic_signal = np.array(
            Parallel(n_jobs=n_cores)(
                delayed(microstruct_model.get_signal)(params, acq_param) for params in tqdm(synthetic_param)
            )
        )

    return synthetic_signal, synthetic_param, normalized_synthetic_param
