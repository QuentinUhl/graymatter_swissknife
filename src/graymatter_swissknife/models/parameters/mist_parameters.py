import numpy as np
from abc import ABC


class MIcroSTructureParameters(ABC):
    """Microstructure parameters."""

    def __init__(self, mist_model, param=None):
        """Initialize the microstructure parameters."""
        # number of parameters
        self.n_params = mist_model.n_params
        # parameter names
        self.names = mist_model.param_names
        # parameters
        self.param = param

    def initialize_fixed_parameters(self, n_set, ground_truth_params):
        # parameters
        param = np.empty([n_set, self.n_params])
        for param_index in range(self.n_params):
            param[:, param_index] = np.ones(n_set) * ground_truth_params[param_index]
        self.param = param
        return param

    def initialize_random_parameters(self, n_set, microstruct_model):
        # parameters
        param = np.empty([n_set, self.n_params])
        for param_index in range(microstruct_model.n_params):
            param[:, param_index] = (
                np.random.rand(n_set)
                * (microstruct_model.param_lim[param_index][1] - microstruct_model.param_lim[param_index][0])
                + microstruct_model.param_lim[param_index][0]
            )
        if microstruct_model.constraints is not None:
            for constr in microstruct_model.constraints:
                param = constr(param)
        self.param = param
        return param


class MIcroSTructureParametersException(Exception):
    """Handle exceptions related to microstructure parameters."""

    pass


class InvalidMIcroSTructureParameters(MIcroSTructureParametersException):
    """Handle exceptions related to wrong microstructure parameters."""

    pass
