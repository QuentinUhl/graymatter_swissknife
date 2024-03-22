import numpy as np
from ...models.microstructure_models import MicroStructModel
from .functions.scipy_nexi_dot import nexi_dot_signal_from_vector, \
    nexi_dot_jacobian_from_vector, nexi_dot_optimized_mse_jacobian


class NexiDot(MicroStructModel):
    """Neurite Exchange Imaging model ( Narrow Pulse approximation) with an additional dot compartment for diffusion MRI."""

    # Class attributes
    n_params = 5
    param_names = ["t_ex", "Di", "De", "f", "f_dot"]
    classic_limits = np.array([[1, 150], [0.1, 3.5], [0.1, 3.5], [0.1, 0.9], [0.0, 0.3]])
    grid_search_nb_points = [15, 12, 8, 8, 6]
    has_rician_mean_correction = False

    def __init__(self, param_lim=classic_limits, invert_tex=False):
        super().__init__(name='NEXI_dot')
        self.param_lim = param_lim
        self.invert_tex = invert_tex
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
        return nexi_dot_signal_from_vector(parameters, acq_parameters.b, acq_parameters.td)

    @classmethod
    def get_jacobian(cls, parameters, acq_parameters):
        """Get jacobian from single Ground Truth."""
        return nexi_dot_jacobian_from_vector(parameters, acq_parameters.b, acq_parameters.td)

    @classmethod
    def get_hessian(cls, parameters, acq_parameters):
        """Get hessian from single Ground Truth."""
        return None  # KM_dot_vec_hess(parameters, acq_parameters.b, acq_parameters.td)

    # Optimized methods for Non-linear Least Squares
    @classmethod
    def get_mse_jacobian(cls, parameters, acq_parameters, signal_gt):
        """Get signal from single Ground Truth."""
        return nexi_dot_optimized_mse_jacobian(parameters, acq_parameters.b, acq_parameters.td, signal_gt, acq_parameters.ndim)
