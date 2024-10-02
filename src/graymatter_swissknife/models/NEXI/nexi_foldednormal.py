from ...models.microstructure_models import MicroStructModel
from .nexi import Nexi
from .functions.scipy_nexi import nexi_signal_from_vector, nexi_jacobian_concatenated_from_vector
from ...models.noise.folded_normal_mean import folded_normal_mean, folded_normal_mean_and_jacobian, broad5
import numpy as np


class NexiFoldedNormal(MicroStructModel):
    """Neurite Exchange Imaging model for diffusion MRI. Narrow Pulse approximation. Corrected for Rician noise."""

    # Class attributes
    n_params = 5
    param_names = ["t_ex", "Di", "De", "f", "sigma"]
    classic_limits = np.array([[1, 150], [0.1, 3.5], [0.1, 3.5], [0.1, 0.9], [0, 100]])
    grid_search_nb_points = [15, 12, 8, 8]
    has_noise_correction = True
    non_corrected_model = Nexi()

    def __init__(self, param_lim=classic_limits):
        super().__init__(name='NEXI_Folded_Normal')
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
        return folded_normal_mean(nexi_signal_from_vector(parameters[:4], acq_parameters.b, acq_parameters.td), parameters[4])

    @classmethod
    def get_jacobian(cls, parameters, acq_parameters):
        """Get jacobian from single Ground Truth."""
        nexi_vec_jac_concatenation = nexi_jacobian_concatenated_from_vector(parameters[:-1], acq_parameters.b, acq_parameters.td)
        nexi_signal_vec = nexi_vec_jac_concatenation[..., 0]
        nexi_vec_jac = nexi_vec_jac_concatenation[..., 1:]
        # Turn last parameter jacobian to 0 to avoid updates
        _, nexi_fn_vec_jac = folded_normal_mean_and_jacobian(nexi_signal_vec, parameters[-1], dnu=nexi_vec_jac)
        # nexi_fn_vec_jac[..., 4] = np.zeros_like(nexi_fn_vec_jac[..., 4])
        return nexi_fn_vec_jac

    @classmethod
    def get_hessian(cls, parameters, acq_parameters):
        """Get hessian from single Ground Truth."""
        return None

    # Optimized methods for Non-linear Least Squares
    @classmethod
    def get_mse_jacobian(cls, parameters, acq_parameters, signal_gt):
        """Get jacobian of Mean Square Error from single Ground Truth."""
        nexi_vec_jac_concatenation = nexi_jacobian_concatenated_from_vector(parameters[:-1], acq_parameters.b, acq_parameters.td)
        nexi_signal_vec = nexi_vec_jac_concatenation[..., 0]
        nexi_vec_jac = nexi_vec_jac_concatenation[..., 1:]
        nexi_fn_signal_vec, nexi_fn_vec_jac = folded_normal_mean_and_jacobian(nexi_signal_vec, parameters[-1], dnu=nexi_vec_jac)
        if acq_parameters.ndim == 1:
            mse_jacobian = np.sum(2 * nexi_fn_vec_jac * broad5(nexi_fn_signal_vec - signal_gt), axis=0)
        elif acq_parameters.ndim == 2:
            mse_jacobian = np.sum(2 * nexi_fn_vec_jac * broad5(nexi_fn_signal_vec - signal_gt), axis=(0, 1))
        else:
            raise NotImplementedError
        return mse_jacobian


        # Turn last parameter jacobian to 0 to avoid updates
        # if acq_parameters.ndim == 1:
        #     mse_jacobian[-1] = 0.0
        # else:
        #     mse_jacobian[..., -1] = np.zeros_like(mse_jacobian[..., -1])
