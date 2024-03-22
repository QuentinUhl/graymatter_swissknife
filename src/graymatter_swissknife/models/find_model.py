from .NEXI import *
from .SANDI import *
from .SANDIX import *
from .GEM import *

def find_model(model_name):
    """Find the microstructure model."""
    if model_name == 'NexiWidePulse':
        return Smex()
    elif model_name == 'NexiWidePulseRiceMean':
        return SmexRiceMean()
    elif model_name == 'NexiNarrowPulseApproximation':
        return Nexi()
    elif model_name == 'NexiNarrowPulseApproximationRiceMean':
        return NexiRiceMean()
    else:
        return eval(f'{model_name}()')
    