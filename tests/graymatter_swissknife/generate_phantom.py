import os
import sys
import numpy as np
import nibabel as nib

# Get the absolute path to the root directory of your project
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Append the project root to sys.path
sys.path.append(project_root)

from src.graymatter_swissknife.models.NEXI.functions.scipy_nexi import nexi_signal


def generate_phantom():
    """
    Generate a phantom with 27 voxels
    """
    # Set the seed for the global random number generator
    np.random.seed(42)
    # Create a specific instance of RandomState for reproducibility
    rng = np.random.RandomState(42)
    # Generate random numbers using the rng instance : rng.rand(5)

    # Initialize
    signals = []
    parameters = []
    sigmas = []

    # Parameters
    b = np.array([0, 1, 0, 2, 4, 0, 1, 2, 5])
    t = np.array([20, 20, 30, 30, 30, 40, 40, 40, 40])

    for ind in range(27):
        tex = 2 + 98 * rng.rand()
        Di = 2 + rng.rand()
        De = 0.5 + rng.rand()
        Dp = De
        f = rng.rand()
        # Genreate associated signal
        single_signal = nexi_signal(tex, Di, De, Dp, f, b, t)
        magnitude = 10 + 200 * rng.rand()
        single_signal = magnitude * single_signal
        signals.append(single_signal)
        # Random sigma
        sigma = (0.01 + 0.3 * rng.rand()) * magnitude
        sigmas.append(sigma)
        parameters.append([tex, Di, De, f])
    signals = np.array(signals)
    sigmas = np.array(sigmas)
    parameters = np.array(parameters)

    # Reshape and add problematic values
    image = signals.reshape((3, 3, 3, len(b)))
    image[1, 2, 0, 2] = np.nan
    image[1, 2, 1, 0] = np.nan
    image[2, 1, 0, 1] = np.nan
    lowb_noisemap = sigmas.reshape(3, 3, 3)
    lowb_noisemap[0, 1, 1] = np.nan

    # Save the powder average
    image_nii = nib.Nifti1Image(image, affine=np.eye(4))
    nib.save(image_nii, 'tests/graymatter_swissknife/data/phantom.nii.gz')

    lowb_noisemap_nii = nib.Nifti1Image(lowb_noisemap, affine=np.eye(4))
    nib.save(lowb_noisemap_nii, 'tests/graymatter_swissknife/data/lowb_noisemap.nii.gz')

    # Create a mask
    mask = np.ones((3, 3, 3))
    mask[1, 2, 0] = 0
    mask[1, 1, 1] = 0
    mask_nii = nib.Nifti1Image(mask, affine=np.eye(4))
    nib.save(mask_nii, 'tests/graymatter_swissknife/data/mask.nii.gz')

    # Save diffusion times
    np.savetxt('tests/graymatter_swissknife/data/phantom.bval', b, fmt='%.4f')
    np.savetxt('tests/graymatter_swissknife/data/phantom.td', t, fmt='%.4f')
