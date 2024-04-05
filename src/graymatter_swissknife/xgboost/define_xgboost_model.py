import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from .generate_dataset import generate_dataset
import numpy as np

def define_xgboost_model(xgboost_model_path, retrain_xgboost, 
                         microstruct_model, acq_param, n_training_samples, sigma=None, n_cores=-1):
    """Load a XGBoost model from a file or generate it.

    Args:
        xgboost_model_path (str): Path to the XGBoost model, either to load or to save it.
        retrain_xgboost (bool): Whether to retrain the XGBoost model.
        microstruct_model (object): The microstructure model.
        acq_param (object): The acquisition parameters.
        n_training_samples (int): Number of training samples.
        sigma (np.ndarray): The standard deviation of the Rician noise.
        n_cores (int): Number of cores to use for the parallel computation.

    Returns:
        object: The XGBoost model.        
    """
    
    xgboost_model = XGBRegressor(
                                    tree_method="hist", 
                                    n_estimators=128,
                                    n_jobs=16,
                                    max_depth=8,
                                    multi_strategy="one_output_per_tree",
                                    subsample=0.6,
                                    device='gpu'  # Use GPU acceleration
                                )

    # Check if the path leads to an existing file
    if os.path.exists(xgboost_model_path) and not retrain_xgboost:
        # Load the XGBoost model
        xgboost_model.load_model(xgboost_model_path)
    else:
        # Generate the dataset
        n_samples = 125*n_training_samples//100  # 25% more samples to account for the test and validation sets
        synthetic_signal, _, normalized_param = generate_dataset(microstruct_model, acq_param, n_samples, sigma, n_cores)

        # Split the dataset into train, validation and test sets
        X_train, X_test, y_train, y_test = train_test_split(synthetic_signal, normalized_param, test_size=0.2, random_state=11)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=12)

        # Train the XGBoost model
        xgboost_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        
        # Save the XGBoost model
        xgboost_model.save_model(xgboost_model_path)

        # Plot the application of the model on the test set
        y_hat = xgboost_model.predict(X_test)
        y_hat = np.clip(y_hat, 0, 1)
        for param_index in range(microstruct_model.n_params):
            y_hat[:, param_index] = (
                y_hat[:, param_index]
                * (microstruct_model.param_lim[param_index][1] - microstruct_model.param_lim[param_index][0])
                + microstruct_model.param_lim[param_index][0]
            )
        
    return xgboost_model