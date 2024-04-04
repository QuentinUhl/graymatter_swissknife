import numpy as np

def apply_xgboost_model(xgboost_model, signal, microstruct_model):
    
    # Application of the model on the signals
    y_hat = xgboost_model.predict(signal)
    y_hat = np.clip(y_hat, 0, 1)
    for param_index in range(microstruct_model.n_params):
        y_hat[:, param_index] = (
            y_hat[:, param_index]
            * (microstruct_model.param_lim[param_index][1] - microstruct_model.param_lim[param_index][0])
            + microstruct_model.param_lim[param_index][0]
        )
    
    return y_hat