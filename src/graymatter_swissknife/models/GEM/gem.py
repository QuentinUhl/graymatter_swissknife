import numpy as np
from ...models.microstructure_models import MicroStructModel
from .functions.scipy_gem import (gem_signal_from_vector, gem_jacobian_from_vector,
                                                           gem_optimized_mse_jacobian)


class Gem(MicroStructModel):
    """Generalized Exchange Model for diffusion MRI."""

    # Class attributes
    n_params = 7
    param_names = ["t_exs", "t_exd", "Dd", "De", "fn", "radius", "fsn"]
    classic_limits = np.array([[1, 150], [1, 150], [0.1, 3.5], [0.1, 3.5], [0.05, 0.95], [1, 30], [0.05, 0.5]])
    grid_search_nb_points = [5, 5, 5, 5, 5, 5, 5]  # Optimal would be [12, 8, 5, 5, 5, 5, 5], but it will depends on how much time your machine takes
    has_rician_mean_correction = False

    def __init__(self, param_lim=classic_limits, invert_tex=False):
        super().__init__(name='GEM')
        self.param_lim = param_lim
        self.invert_tex = invert_tex
        self.constraints = [self.constr_on_diffusivities]

    @staticmethod
    def constr_on_diffusivities(param):
        if np.array(param).ndim == 1:
            # Put Di>De
            if param[2] < param[3]:
                # exchange them
                param[2], param[3] = param[3], param[2]
        elif np.array(param).ndim == 2:
            # Put Di>De
            for iset in range(len(param)):
                # if Di < De of the set iset
                if param[iset, 2] < param[iset, 3]:
                    # exchange them
                    param[iset, 2], param[iset, 3] = param[iset, 3], param[iset, 2]
        else:
            raise ValueError('Wrong dimension of parameters')
        return param

    def set_param_lim(self, param_lim):
        self.param_lim = param_lim

    @classmethod
    def get_signal(cls, parameters, acq_parameters):
        """Get signal from single Ground Truth."""
        return gem_signal_from_vector(parameters, acq_parameters.b, acq_parameters.td, acq_parameters.small_delta)

    @classmethod
    def get_jacobian(cls, parameters, acq_parameters):
        """Get jacobian from single Ground Truth."""
        return gem_jacobian_from_vector(parameters, acq_parameters.b, acq_parameters.td, acq_parameters.small_delta)

    @classmethod
    def get_hessian(cls, parameters, acq_parameters):
        return None

    # Optimized methods for Non-linear Least Squares
    @classmethod
    def get_mse_jacobian(cls, parameters, acq_parameters, signal_gt):
        """Get jacobian of Mean Square Error from single Ground Truth."""
        return gem_optimized_mse_jacobian(parameters, acq_parameters, signal_gt)
