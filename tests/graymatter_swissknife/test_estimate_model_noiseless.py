import os
import numpy as np
import nibabel as nib
from src.graymatter_swissknife.estimate_model_noiseless import estimate_model_noiseless


def test_estimate_model_noiseless():
    np.random.seed(123)
    if not os.path.isdir('tests/graymatter_swissknife/data/nexi'):
        os.mkdir('tests/graymatter_swissknife/data/nexi')
    estimate_model_noiseless(
        'Nexi',
        'tests/graymatter_swissknife/data/phantom.nii.gz',
        'tests/graymatter_swissknife/data/phantom.bval',
        'tests/graymatter_swissknife/data/phantom.delta',
        None,
        'tests/graymatter_swissknife/data/nexi',
        mask_path='tests/graymatter_swissknife/data/mask.nii.gz',
        debug=False,
    )

    powder_average_filename = 'tests/graymatter_swissknife/data/nexi/powderaverage_dwi.nii.gz'
    bval_filename = 'tests/graymatter_swissknife/data/nexi/powderaverage.bval'
    delta_filename = 'tests/graymatter_swissknife/data/nexi/powderaverage.delta'
    powder_average_npz_filename = 'tests/graymatter_swissknife/data/nexi/powderaverage_signal.npz'
    updated_mask_filename = 'tests/graymatter_swissknife/data/nexi/updated_mask.nii.gz'

    assert np.allclose(
        nib.load(powder_average_filename).get_fdata(),
        nib.load('tests/graymatter_swissknife/data/powderaverage_dwi_ref.nii.gz').get_fdata(),
        equal_nan=True,
    )
    assert np.allclose(np.loadtxt(bval_filename), np.loadtxt('tests/graymatter_swissknife/data/powderaverage_ref.bval'))
    assert np.allclose(np.loadtxt(delta_filename), np.loadtxt('tests/graymatter_swissknife/data/powderaverage_ref.delta'))

    os.remove(powder_average_filename)
    os.remove(bval_filename)
    os.remove(delta_filename)
    os.remove(powder_average_npz_filename)
    os.remove(updated_mask_filename)

    parameters = ["t_ex", "di", "de", "f"]
    for param in parameters:
        param_filename = f'tests/graymatter_swissknife/data/nexi/nexi_{param}.nii.gz'
        param_ref_filename = f'tests/graymatter_swissknife/data/models_ref/nexi_{param}_ref.nii.gz'
        assert np.allclose(
            nib.load(param_filename).get_fdata(), nib.load(param_ref_filename).get_fdata(), equal_nan=True
        )
        os.remove(param_filename)

    os.rmdir('tests/graymatter_swissknife/data/nexi')
