import numpy as np
from ...models.microstructure_models import MicroStructModel
from .sandix import Sandix
from .functions.scipy_sandix import (sandix_signal_from_vector, sandix_jacobian_from_vector,
                                                                sandix_concatenated_from_vector, broad7)
from ...models.rice_noise.rice_mean import rice_mean, rice_mean_and_jacobian


class SandixRiceMean(MicroStructModel):
    """Soma And Neurite Density Imaging model with Exchange for diffusion MRI. Corrected for Rician noise."""

    # Class attributes
    n_params = 7
    param_names = ["t_ex", "Di", "De", "f", "rs", "fs", "sigma"]
    classic_limits = np.array([[1, 150], [0.1, 3.5], [0.1, 3.5], [0.05, 0.95], [1, 30], [0.05, 0.5], [0, 100]])
    grid_search_nb_points = [7, 5, 5, 5, 5, 5]  # Optimal would be [12, 8, 5, 5, 5, 5], but it will depends on how much time your machine takes
    has_rician_mean_correction = True
    non_corrected_model = Sandix()

    def __init__(self, param_lim=classic_limits):
        super().__init__(name='SANDIX_Rice_Mean')
        self.param_lim = param_lim
        self.constraints = [self.constr_on_diffusivities]

    @staticmethod
    def constr_on_diffusivities(param):
        if np.array(param).ndim == 1:
            # Put Di>De
            if param[0] < param[1]:
                # exchange them
                param[0], param[1] = param[1], param[0]
        elif np.array(param).ndim == 2:
            # Put Di>De
            for iset in range(len(param)):
                # if Di < De of the set iset
                if param[iset, 0] < param[iset, 1]:
                    # exchange them
                    param[iset, 0], param[iset, 1] = param[iset, 1], param[iset, 0]
        else:
            raise ValueError('Wrong dimension of parameters')
        return param

    def set_param_lim(self, param_lim):
        self.param_lim = param_lim

    @classmethod
    def get_signal(cls, parameters, acq_parameters):
        """Get signal from single Ground Truth."""
        return rice_mean(sandix_signal_from_vector(parameters[:-1],
                                                   acq_parameters.b, acq_parameters.td, acq_parameters.small_delta),
                         parameters[-1])

    @classmethod
    def get_jacobian(cls, parameters, acq_parameters):
        """Get jacobian from single Ground Truth."""
        concatenated = sandix_concatenated_from_vector(parameters[:-1], acq_parameters.b, acq_parameters.td,
                                                       acq_parameters.small_delta)
        signal, jacobian = concatenated[..., 0], concatenated[..., 1:]
        # Turn last parameter jacobian to 0 to avoid updates
        _, sandix_rm_vec_jac = rice_mean_and_jacobian(signal, parameters[-1], dnu=jacobian)
        return sandix_rm_vec_jac

    @classmethod
    def get_hessian(cls, parameters, acq_parameters):
        """Get hessian from single Ground Truth."""
        return None

    # Optimized methods for Non-linear Least Squares
    @classmethod
    def get_mse_jacobian(cls, parameters, acq_parameters, signal_gt):
        """Get signal from single Ground Truth."""
        concatenated = sandix_concatenated_from_vector(parameters[:-1], acq_parameters.b, acq_parameters.td,
                                                       acq_parameters.small_delta)
        signal, jacobian = concatenated[..., 0], concatenated[..., 1:]
        sandix_rm_signal_vec, sandix_rm_vec_jac = rice_mean_and_jacobian(signal, parameters[-1], dnu=jacobian)
        if acq_parameters.ndim == 1:
            mse_jacobian = np.sum(2 * sandix_rm_vec_jac * broad7(sandix_rm_signal_vec - signal_gt), axis=0)
        elif acq_parameters.ndim == 2:
            mse_jacobian = np.sum(2 * sandix_rm_vec_jac * broad7(sandix_rm_signal_vec - signal_gt), axis=(0, 1))
        else:
            raise NotImplementedError
        return mse_jacobian
