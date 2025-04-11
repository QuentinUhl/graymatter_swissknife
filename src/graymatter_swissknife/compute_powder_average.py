import os
import logging
import argparse
from .powderaverage.powderaverage import powder_average, normalize_sigma, save_data_as_npz


def compute_powder_average(dwi_path, bvals_path, delta_path, lowb_noisemap_path, out_path, mask_path=None, debug=False):
    """
    Compute the powder average only. This is also done in the estimate_model functions. A mask is optional but highly recommended.

    Parameters
    ----------
    dwi_path : str
        Path to the preprocessed DWI signal.
    bvals_path : str
        Path to the b-values file. b-values must be provided in ms/µm².
    delta_path : str
        Path to the big delta file. Δ must be provided in ms.
    lowb_noisemap_path : str
        Path to the low b-values (b < 2ms/µm²) noise map.
    out_path : str
        Path to the output directory. If the directory does not exist, it will be created.
    mask_path : str, optional
        Path to the mask file. The default is None.
    debug : bool, optional
        Debug mode. The default is False.

    Returns
    -------
    None.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')  # Set the logging level to INFO or desired level

    # Create the output directory if it does not exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Convert into powder average
    powder_average_path, updated_bvals_path, updated_delta_path, updated_mask_path = powder_average(
        dwi_path, bvals_path, delta_path, mask_path, out_path, debug=debug
    )
    # NEXI with Rician Mean correction
    normalized_sigma_filename = normalize_sigma(dwi_path, lowb_noisemap_path, bvals_path, out_path)
    _ = save_data_as_npz(
        powder_average_path,
        updated_bvals_path,
        updated_delta_path,
        updated_mask_path,
        out_path,
        normalized_sigma_filename=normalized_sigma_filename,
        debug=debug,
    )


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Compute the powder average only. This is also done in the estimate_model functions. A mask is optional but highly recommended.'
    )
    parser.add_argument('dwi_path', help='path to the preprocessed signals')
    # For conversion from b-values in s/µm² to b-values in ms/µm², divide by 1000
    parser.add_argument('bvals_path', help='path to the b-values (in ms/µm²) txt file')
    parser.add_argument('delta_path', help='path to the diffusion times (in ms) txt file')
    parser.add_argument('lowb_noisemap_path', help='path to the lowb noisemap')
    parser.add_argument('out_path', help='path to the output folder')
    # potential arguments
    # Set to None if not provided
    parser.add_argument('--mask_path', help='path to the mask', required=False, default=None)
    parser.add_argument('--debug', help='debug mode', required=False, action='store_true')
    args = parser.parse_args()

    # estimate_nexi(**vars(parser.parse_args()))
    compute_powder_average(dwi_path=args.dwi_path, bvals_path=args.bvals_path, delta_path=args.delta_path, 
                           lowb_noisemap_path=args.lowb_noisemap_path, out_path=args.out_path,
                           mask_path=args.mask_path, debug=args.debug)
