from .NEXI import *
from .SANDI import *
from .SANDIX import *
from .GEM import *

def find_model(model_name):
    """Find the microstructure model."""
    if model_name == 'Nexi_Wide_Pulses':
        return Smex()
    elif model_name == 'Nexi_Wide_PulsesRiceMean':
        return SmexRiceMean()
    elif model_name == 'Nexi_Narrow_Pulses_Approximation':
        return Nexi()
    elif model_name == 'Nexi_Narrow_Pulses_ApproximationRiceMean':
        return NexiRiceMean()
    else:
        return eval(f'{model_name}()')
    