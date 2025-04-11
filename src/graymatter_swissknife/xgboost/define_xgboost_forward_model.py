import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from .generate_dataset import generate_dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

def define_xgboost_forward_model(xgboost_model_path, retrain_xgboost, 
                                 microstruct_model, acq_param, n_training_samples, force_cpu=False, n_cores=-1):
    """Load a XGBoost model from a file or generate it.

    Args:
        xgboost_model_path (str): Path to the XGBoost model, either to load or to save it.
        retrain_xgboost (bool): Whether to retrain the XGBoost model.
        microstruct_model (object): The microstructure model.
        acq_param (object): The acquisition parameters.
        n_training_samples (int): Number of training samples.
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
        synthetic_signal, _, normalized_param = generate_dataset(microstruct_model, acq_param, n_samples, None, n_cores)

        # Split the dataset into train, validation and test sets
        X_train, X_test, y_train, y_test = train_test_split(normalized_param, synthetic_signal, test_size=0.2, random_state=11)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=12)

        # Train the XGBoost model
        xgboost_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        
        # Save the XGBoost model
        xgboost_model.save_model(xgboost_model_path)

        # Plot the application of the model on the test set
        y_hat = xgboost_model.predict(X_test)
        y_hat = np.clip(y_hat, 0, 1)
        
        # Save the plot in the same directory as the XGBoost model
        xgboost_directory = os.path.dirname(xgboost_model_path)
        # Put "test_" in front of the filename and replace the extension (could be any extension) with ".png"
        xgboost_model_basename = os.path.basename(xgboost_model_path)
        # Remove the last extension and remerge everything with the new extension
        xgboost_test_plot_filename = ".".join(xgboost_model_basename.split(".")[:-1]) + ".png"
        xgboost_test_plot_path = '/'.join([xgboost_directory, xgboost_test_plot_filename])
        
        # Create the scatter plot between the GT y_test and the predicted y_hat
        plt.figure(figsize=(10, 10))
        # Compute the number of acquisitions
        n_acquisition_parameters = len(acq_param.b)
        acquisition_parameters_indices = list(range(n_acquisition_parameters))
        # Divide the plot depending on n_acquisition_parameters
        n_rows = 2
        if n_acquisition_parameters % 2 == 0:
            n_cols = n_acquisition_parameters // n_rows
        else:
            n_cols = n_acquisition_parameters // n_rows + 1
        for plot_index, acquisition_index in enumerate(acquisition_parameters_indices):
            plt.subplot(n_rows, n_cols, plot_index + 1)
            # Plot with colors from an histogram 
            bins = 20
            x = y_test[:, acquisition_index]
            y = y_hat[:, acquisition_index]
            data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
            z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                        data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            plt.scatter(x, y, c=z, s=0.2)
            # Plot identity line
            max_val = max(np.max(x), np.max(y))
            min_val = min(np.min(x), np.min(y))
            plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
            plt.xlabel(f"Target signal")
            plt.ylabel(f"Predicted signal")
            plt.title(fr'$b={acq_param.b[acquisition_index]} ms/ \mu m^2$, $ \Delta={acq_param.delta[acquisition_index]} ms$, $ \delta={acq_param.small_delta} ms$')
        plt.tight_layout()
        plt.savefig(xgboost_test_plot_path)

    return xgboost_model
