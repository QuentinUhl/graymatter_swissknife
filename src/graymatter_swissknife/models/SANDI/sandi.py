from ...models.microstructure_models import MicroStructModel
from .functions.scipy_sandi import sandi_signal_from_vector, sandi_jacobian_from_vector, sandi_optimized_mse_jacobian
# from models.SANDI.functions.torch_sandi import torch_sandi_signal_from_vector
import numpy as np


class Sandi(MicroStructModel):
    """Soma And Neurite Density Imaging model for diffusion MRI."""

    # Class attributes
    n_params = 5
    param_names = ["Di", "De", "f", "rs", "fs"]
    classic_limits = np.array([[0.1, 3.5], [0.1, 3.5], [0.05, 0.95], [1, 30], [0.05, 0.5]])
    grid_search_nb_points = [15, 8, 8, 12, 8]
    has_rician_mean_correction = False

    def __init__(self, param_lim=classic_limits):
        super().__init__(name='SANDI')
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
        return sandi_signal_from_vector(parameters, acq_parameters.b, acq_parameters.delta, acq_parameters.small_delta)

    @classmethod
    def get_jacobian(cls, parameters, acq_parameters):
        """Get jacobian from single Ground Truth."""
        signal, jacobian = sandi_jacobian_from_vector(parameters, acq_parameters.b, acq_parameters.delta,
                                                      acq_parameters.small_delta)
        return jacobian

    @classmethod
    def get_hessian(cls, parameters, acq_parameters):
        """Get hessian from single Ground Truth."""
        return None

    # Optimized methods for Non-linear Least Squares
    @classmethod
    def get_mse_jacobian(cls, parameters, acq_parameters, signal_gt):
        """Get signal from single Ground Truth."""
        return sandi_optimized_mse_jacobian(parameters, acq_parameters.b, acq_parameters.delta, acq_parameters.small_delta,
                                            signal_gt, acq_parameters.ndim)


    # @classmethod
    # def get_torch_signal(cls, parameters, acq_parameters):
    #     """Get signal from single Ground Truth."""
    #     return torch_sandi_signal_from_vector(parameters, acq_parameters.b, acq_parameters.delta, acq_parameters.small_delta)