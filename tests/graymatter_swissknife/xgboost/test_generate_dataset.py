import numpy as np
from src.graymatter_swissknife.models.NEXI.nexi import Nexi
from src.graymatter_swissknife.models.parameters.acq_parameters import AcquisitionParameters
from src.graymatter_swissknife.xgboost.generate_dataset import generate_dataset

def test_generate_dataset():
    microstruct_model = Nexi()
    acq_param = AcquisitionParameters(b=[1, 2, 3], delta=[20, 20, 30])
    n_samples = 100
    sigma = None
    n_cores = -1

    signal, param, normalized_param = generate_dataset(microstruct_model, acq_param, n_samples, sigma, n_cores)
    print(generate_dataset(microstruct_model, acq_param, n_samples, sigma, n_cores))

    # Assert that there is no nan value in the generated dataset
    assert not np.isnan(signal).any()
    assert not np.isnan(param).any()
    assert not np.isnan(normalized_param).any()