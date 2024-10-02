import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from .generate_dataset import generate_dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

def define_xgboost_model(xgboost_model_path, retrain_xgboost, 
                         microstruct_model, acq_param, n_training_samples, sigma=None, force_cpu=False, n_cores=-1):
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
    # Check if the user wants to force the CPU if the GPU has low memory
    if force_cpu:
        device = 'cpu'
    else:
        device = 'gpu'  # Use GPU acceleration
    
    xgboost_model = XGBRegressor(
                                    tree_method="hist", 
                                    n_estimators=512,
                                    early_stopping_rounds=64,
                                    n_jobs=n_cores,
                                    max_depth=16,
                                    multi_strategy="one_output_per_tree",
                                    subsample=0.6,
                                    device=device
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
        n_params = microstruct_model.n_params if not microstruct_model.has_noise_correction else microstruct_model.n_params - 1
        for param_index in range(n_params):
            y_hat[:, param_index] = (
                y_hat[:, param_index]
                * (microstruct_model.param_lim[param_index][1] - microstruct_model.param_lim[param_index][0])
                + microstruct_model.param_lim[param_index][0]
            )
            y_test[:, param_index] = (
                y_test[:, param_index]
                * (microstruct_model.param_lim[param_index][1] - microstruct_model.param_lim[param_index][0])
                + microstruct_model.param_lim[param_index][0]
            )
        
        # Save the plot in the same directory as the XGBoost model
        xgboost_directory = os.path.dirname(xgboost_model_path)
        # Put "test_" in front of the filename and replace the extension (could be any extension) with ".png"
        xgboost_model_basename = os.path.basename(xgboost_model_path)
        # Remove the last extension and remerge everything with the new extension
        xgboost_test_plot_filename = ".".join(xgboost_model_basename.split(".")[:-1]) + ".png"
        xgboost_test_plot_path = '/'.join([xgboost_directory, xgboost_test_plot_filename])
        
        # Create the scatter plot between the GT y_test and the predicted y_hat
        plt.figure(figsize=(10, 10))
        # Compute the number of variable parameters (non fixed in param_lim, i.e. that have two different values in their limits)
        n_varying_params = 0
        varying_params = []
        for param_index in range(n_params):
            if microstruct_model.param_lim[param_index][0] != microstruct_model.param_lim[param_index][1]:
                n_varying_params += 1
                varying_params.append(param_index)
        # Divide the plot depending on n_varying_params
        n_rows = 2
        if n_varying_params % 2 == 0:
            n_cols = n_varying_params // n_rows
        else:
            n_cols = n_varying_params // n_rows + 1
        for plot_index, param_index in enumerate(varying_params):
            plt.subplot(n_rows, n_cols, plot_index + 1)
            # Plot with colors from an histogram 
            bins = 20
            x = y_test[:, param_index]
            y = y_hat[:, param_index]
            data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
            z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                        data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            plt.scatter(x, y, c=z, s=0.2)
            # Plot identity line
            plt.plot([microstruct_model.param_lim[param_index][0], microstruct_model.param_lim[param_index][1]], 
                     [microstruct_model.param_lim[param_index][0], microstruct_model.param_lim[param_index][1]], 
                     color="black", linestyle="--")
            plt.xlabel(f"Target {microstruct_model.param_names[param_index]}")
            plt.ylabel(f"Predicted {microstruct_model.param_names[param_index]}")
            plt.title(f"{microstruct_model.param_names[param_index]}")
        plt.tight_layout()
        plt.savefig(xgboost_test_plot_path)

    return xgboost_model