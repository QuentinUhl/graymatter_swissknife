import os
import sys
import numpy as np
import nibabel as nib

# Get the absolute path to the root directory of your project
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Append the project root to sys.path
sys.path.append(project_root)

from src.graymatter_swissknife.powderaverage.powderaverage import powder_average, normalize_sigma


def test_powder_average():

    powder_average_filename, bval_filename, td_filename, mask_filename = powder_average(
        'tests/graymatter_swissknife/data/phantom.nii.gz',
        'tests/graymatter_swissknife/data/phantom.bval',
        'tests/graymatter_swissknife/data/phantom.td',
        'tests/graymatter_swissknife/data/mask.nii.gz',
        'tests/graymatter_swissknife/data',
        debug=False,
    )
    assert np.allclose(
        nib.load(powder_average_filename).get_fdata(),
        nib.load('tests/graymatter_swissknife/data/powderaverage_dwi_ref.nii.gz').get_fdata(),
        equal_nan=True,
    )
    assert np.allclose(
        nib.load(mask_filename).get_fdata(),
        nib.load('tests/graymatter_swissknife/data/mask_upd_ref.nii.gz').get_fdata(),
        equal_nan=True,
    )
    assert np.allclose(np.loadtxt(bval_filename), np.loadtxt('tests/graymatter_swissknife/data/powderaverage_ref.bval'))
    assert np.allclose(np.loadtxt(td_filename), np.loadtxt('tests/graymatter_swissknife/data/powderaverage_ref.td'))
    os.remove(powder_average_filename)
    os.remove(bval_filename)
    os.remove(td_filename)
    os.remove(mask_filename)


def test_normalize_sigma():
    # sys.path.append('../')
    normalized_sigma_filename = normalize_sigma(
        'tests/graymatter_swissknife/data/phantom.nii.gz',
        'tests/graymatter_swissknife/data/lowb_noisemap.nii.gz',
        'tests/graymatter_swissknife/data/phantom.bval',
        'tests/graymatter_swissknife/data',
    )
    assert np.allclose(
        nib.load(normalized_sigma_filename).get_fdata(),
        nib.load('tests/graymatter_swissknife/data/normalized_sigma_ref.nii.gz').get_fdata(),
        equal_nan=True,
    )
    os.remove(normalized_sigma_filename)
