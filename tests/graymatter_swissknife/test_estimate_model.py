import os
import sys
import numpy as np
import nibabel as nib

# Get the absolute path to the root directory of your project
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Append the project root to sys.path
sys.path.append(project_root)

from src.graymatter_swissknife.estimate_model import estimate_model


def test_estimate_model():
    np.random.seed(123)
    if not os.path.isdir('tests/graymatter_swissknife/data/nexi_rice_mean'):
        os.mkdir('tests/graymatter_swissknife/data/nexi_rice_mean')
    estimate_model(
        'Nexi',
        'tests/graymatter_swissknife/data/phantom.nii.gz',
        'tests/graymatter_swissknife/data/phantom.bval',
        'tests/graymatter_swissknife/data/phantom.delta',
        None,
        'tests/graymatter_swissknife/data/lowb_noisemap.nii.gz',
        'tests/graymatter_swissknife/data/nexi_rice_mean',
        mask_path='tests/graymatter_swissknife/data/mask.nii.gz',
        debug=False,
    )

    powder_average_filename = 'tests/graymatter_swissknife/data/nexi_rice_mean/powderaverage_dwi.nii.gz'
    bval_filename = 'tests/graymatter_swissknife/data/nexi_rice_mean/powderaverage.bval'
    delta_filename = 'tests/graymatter_swissknife/data/nexi_rice_mean/powderaverage.delta'
    powder_average_npz_filename = 'tests/graymatter_swissknife/data/nexi_rice_mean/powderaverage_signal.npz'
    sigma_filename = 'tests/graymatter_swissknife/data/nexi_rice_mean/normalized_sigma.nii.gz'
    updated_mask_filename = 'tests/graymatter_swissknife/data/nexi_rice_mean/updated_mask.nii.gz'

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
    os.remove(sigma_filename)
    os.remove(updated_mask_filename)

    parameters = ["t_ex", "di", "de", "f"]  # , "sigma"]
    for param in parameters:
        param_filename = f'tests/graymatter_swissknife/data/nexi_rice_mean/nexi_rice_mean_{param}.nii.gz'
        param_ref_filename = f'tests/graymatter_swissknife/data/models_ref/nexi_rice_mean_{param}_ref.nii.gz'
        assert np.allclose(
            nib.load(param_filename).get_fdata(), nib.load(param_ref_filename).get_fdata(), equal_nan=True
        )
        os.remove(param_filename)

    os.rmdir('tests/graymatter_swissknife/data/nexi_rice_mean')
