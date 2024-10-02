import numpy as np
from ...models.microstructure_models import MicroStructModel
from .gem import Gem
from .functions.scipy_gem import (gem_signal_from_vector, gem_jacobian_concatenated_from_vector, broad8)
from ..noise.rice_mean import rice_mean, rice_mean_and_jacobian


class GemRiceMean(MicroStructModel):
    """Generalized Exchange Model for diffusion MRI with Rician Mean correction."""

    # Class attributes
    n_params = 8
    param_names = ["t_exs", "t_exd", "Dd", "De", "fn", "radius", "fsn", "sigma"]
    classic_limits = np.array([[1, 150], [1, 150], [0.1, 3.5], [0.1, 3.5], [0.05, 0.95], [1, 30], [0.05, 0.5], [0, 100]])
    grid_search_nb_points = [5, 5, 5, 5, 5, 5, 5]  # Optimal would be [12, 8, 5, 5, 5, 5, 5], but it will depends on how much time your machine takes
    has_noise_correction = True
    non_corrected_model = Gem()

    def __init__(self, param_lim=classic_limits):
        super().__init__(name='GEM_Rice_Mean')
        self.param_lim = param_lim
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
        return rice_mean(gem_signal_from_vector(parameters[:-1], acq_parameters.b, acq_parameters.delta,
                                                acq_parameters.small_delta), parameters[-1])

    @classmethod
    def get_jacobian(cls, parameters, acq_parameters):
        """Get jacobian from single Ground Truth."""
        gem_signal, gem_jac = gem_jacobian_concatenated_from_vector(parameters[:-1],
                                                                    acq_parameters.b,
                                                                    acq_parameters.delta,
                                                                    acq_parameters.small_delta)
        # Turn last parameter jacobian to 0 to avoid updates
        _, gem_rm_jac = rice_mean_and_jacobian(gem_signal, parameters[-1], dnu=gem_jac)
        return gem_rm_jac

    @classmethod
    def get_hessian(cls, parameters, acq_parameters):
        """Get hessian from single Ground Truth."""
        return None  # KM_sphere_vec_hess(parameters, acq_parameters.b, acq_parameters.delta)

    # Optimized methods for Non-linear Least Squares
    @classmethod
    def get_mse_jacobian(cls, parameters, acq_parameters, signal_gt):
        """Get jacobian of Mean Square Error from single Ground Truth."""
        gem_signal, gem_jac = gem_jacobian_concatenated_from_vector(parameters[:-1],
                                                                    acq_parameters.b,
                                                                    acq_parameters.delta,
                                                                    acq_parameters.small_delta)
        gem_rm_signal, gem_rm_jac = rice_mean_and_jacobian(gem_signal, parameters[-1], dnu=gem_jac)
        if acq_parameters.ndim == 1:
            mse_jacobian = np.sum(2 * gem_rm_jac * broad8(gem_rm_signal - signal_gt), axis=0)
        elif acq_parameters.ndim == 2:
            mse_jacobian = np.sum(2 * gem_rm_jac * broad8(gem_rm_signal - signal_gt), axis=(0, 1))
        else:
            raise NotImplementedError
        # Turn last parameter jacobian to 0 to avoid updates
        mse_jacobian[..., -1] = np.zeros_like(mse_jacobian[..., -1])
        return mse_jacobian
