import numpy as np
from ...models.microstructure_models import MicroStructModel
from .sandi import Sandi
from .functions.scipy_sandi import sandi_signal_from_vector, sandi_jacobian_from_vector, sandi_optimized_mse_jacobian
from ...models.rice_noise.rice_mean import rice_mean, rice_mean_and_jacobian, broad6


class SandiRiceMean(MicroStructModel):
    """Soma And Neurite Density Imaging model for diffusion MRI. Corrected for Rician noise."""

    # Class attributes
    n_params = 6
    param_names = ["Di", "De", "f", "rs", "fs", "sigma"]
    classic_limits = np.array([[0.1, 3.5], [0.1, 3.5], [0.05, 0.95], [1, 30], [0.05, 0.5], [0, 100]])
    grid_search_nb_points = [15, 8, 8, 12, 8]
    has_rician_mean_correction = True
    non_corrected_model = Sandi()

    def __init__(self, param_lim=classic_limits, invert_tex=False):
        super().__init__(name='SANDI_Rice_Mean')
        self.param_lim = param_lim
        self.invert_tex = False
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
        return rice_mean(sandi_signal_from_vector(parameters[:-1], acq_parameters.b, acq_parameters.td, acq_parameters.small_delta), parameters[-1])

    @classmethod
    def get_jacobian(cls, parameters, acq_parameters):
        """Get jacobian from single Ground Truth."""
        signal, jacobian = sandi_jacobian_from_vector(parameters, acq_parameters.b, acq_parameters.td,
                                                      acq_parameters.small_delta)
        # Turn last parameter jacobian to 0 to avoid updates
        _, sandi_rm_vec_jac = rice_mean_and_jacobian(signal, parameters[-1], dnu=jacobian)
        return sandi_rm_vec_jac

    @classmethod
    def get_hessian(cls, parameters, acq_parameters):
        """Get hessian from single Ground Truth."""
        return None

    # Optimized methods for Non-linear Least Squares
    @classmethod
    def get_mse_jacobian(cls, parameters, acq_parameters, signal_gt):
        """Get signal from single Ground Truth."""
        signal, jacobian = sandi_jacobian_from_vector(parameters, acq_parameters.b, acq_parameters.td,
                                                      acq_parameters.small_delta)
        sandi_rm_signal_vec, sandi_rm_vec_jac = rice_mean_and_jacobian(signal, parameters[-1], dnu=jacobian)
        if acq_parameters.ndim == 1:
            mse_jacobian = np.sum(2 * sandi_rm_vec_jac * broad6(sandi_rm_signal_vec - signal_gt), axis=0)
        elif acq_parameters.ndim == 2:
            mse_jacobian = np.sum(2 * sandi_rm_vec_jac * broad6(sandi_rm_signal_vec - signal_gt), axis=(0, 1))
        else:
            raise NotImplementedError
        return mse_jacobian
