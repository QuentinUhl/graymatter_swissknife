from ...models.microstructure_models import MicroStructModel
from ...models.NEXI.functions.scipy_nexi import nexi_signal_from_vector, nexi_jacobian_from_vector, \
    nexi_hessian_from_vector, nexi_optimized_mse_jacobian, nexi_optimized_mse_hessian
import numpy as np


class Nexi(MicroStructModel):
    """Neurite Exchange Imaging model for diffusion MRI. Narrow Pulse approximation."""

    # Class attributes
    n_params = 4
    param_names = ["t_ex", "Di", "De", "f"]
    classic_limits = np.array([[1, 150], [0.1, 3.5], [0.1, 3.5], [0.1, 0.9]])
    grid_search_nb_points = [15, 12, 8, 8]
    has_noise_correction = False

    def __init__(self, param_lim=classic_limits):
        super().__init__(name='NEXI')
        self.param_lim = param_lim
        self.constraints = [self.constr_on_diffusivities]

    @staticmethod
    def constr_on_diffusivities(param):
        if np.array(param).ndim == 1:
            # Put Di>De
            if param[1] < param[2]:
                # exchange them
                param[1], param[2] = param[2], param[1]
        elif np.array(param).ndim == 2:
            # Put Di>De
            for iset in range(len(param)):
                # if Di < De of the set iset
                if param[iset, 1] < param[iset, 2]:
                    # exchange them
                    param[iset, 1], param[iset, 2] = param[iset, 2], param[iset, 1]
        else:
            raise ValueError('Wrong dimension of parameters')
        return param

    def set_param_lim(self, param_lim):
        self.param_lim = param_lim

    @classmethod
    def get_signal(cls, parameters, acq_parameters):
        """Get signal from single Ground Truth."""
        return nexi_signal_from_vector(parameters, acq_parameters.b, acq_parameters.td)

    @classmethod
    def get_jacobian(cls, parameters, acq_parameters):
        """Get jacobian from single Ground Truth."""
        return nexi_jacobian_from_vector(parameters, acq_parameters.b, acq_parameters.td)

    @classmethod
    def get_hessian(cls, parameters, acq_parameters):
        """Get hessian from single Ground Truth."""
        return nexi_hessian_from_vector(parameters, acq_parameters.b, acq_parameters.td)

    # Optimized methods for Non-linear Least Squares
    @classmethod
    def get_mse_jacobian(cls, parameters, acq_parameters, signal_gt):
        """Get signal from single Ground Truth."""
        return nexi_optimized_mse_jacobian(parameters, acq_parameters.b, acq_parameters.td, signal_gt, acq_parameters.ndim)

    # Optimized methods for XGBoost
    @classmethod
    def get_mse_hessian(cls, parameters, acq_parameters, signal_gt):
        """Get signal from single Ground Truth."""
        return nexi_optimized_mse_hessian(parameters, acq_parameters.b, acq_parameters.td, signal_gt, acq_parameters.ndim)
