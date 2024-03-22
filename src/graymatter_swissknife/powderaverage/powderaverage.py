import logging
import warnings
import numpy as np
import nibabel as nib


def powder_average(dwi_path, bvals_path, td_path, mask_path, out_path, debug=False):
    """
    Compute the powder average of the diffusion-weighted images.

    Parameters
    ----------
    dwi_path : str
        Path to the preprocessed signals.
    bvals_path : str
        Path to the b-values (in ms/µm²) txt file.
    td_path : str
        Path to the diffusion times (in ms) txt file.
    mask_path : str
        Path to the mask.
    out_path : str
        Path to the output folder.
    debug : bool, optional
        If True, print debug information. The default is False.

    Returns
    -------
    powder_average_filename : str
        Path to the powder average image.
    updated_bvals_path : str
        Path to the updated b-values (in ms/µm²) txt file.
    updated_td_path : str
        Path to the updated diffusion times (in ms) txt file.
    """

    powder_average_filename = f'{out_path}/powderaverage_dwi.nii.gz'
    updated_bvals_path = f'{out_path}/powderaverage.bval'
    updated_td_path = f'{out_path}/powderaverage.td'
    # Re-split the image into different diffusion times
    dwi_image_nii = nib.load(dwi_path)
    dwi_image = dwi_image_nii.get_fdata()
    aff, hdr = dwi_image_nii.affine, dwi_image_nii.header
    tdvalues = np.loadtxt(td_path).astype(float)
    bvalues = np.loadtxt(bvals_path).astype(float)
    # Convert the mask into booleans
    if mask_path is not None:
        mask = np.squeeze(nib.load(mask_path).get_fdata())
    else:
        mask = np.ones(dwi_image.shape[:-1])
    bool_mask = np.copy(mask) > 0.33
    # Initialize powder average image, b-values and diffusion times
    pa_image = []
    pa_b = []
    pa_td = []
    # Initialize nan mask
    nan_mask = np.copy(bool_mask).astype(float)
    for td in np.unique(tdvalues):
        # Find b0s
        b0_selection = (tdvalues == td) & (bvalues == 0)
        nb_of_b0 = np.sum(b0_selection)
        assert nb_of_b0 != 0
        # Compute the mean of all b0 images to have one unique b0 image
        if nb_of_b0 > 1:
            b0_image = np.mean(dwi_image[..., b0_selection], axis=-1)
        else:
            b0_image = dwi_image[..., b0_selection][..., 0]
        # Update mask with the positions in the b0 images where the signal is equal to zero (to avoid div. by 0)
        bool_mask = bool_mask & (b0_image != 0)
        # Convert the boolean mask into a 1.0/nan mask to be easily multiplied
        nan_mask = np.copy(bool_mask).astype(float)
        nan_mask[bool_mask == 0] = np.nan
        # Put the b0 image, b and td inside the lists
        pa_image.append((b0_image != 0).astype(float) * nan_mask)
        pa_b.append(0)
        pa_td.append(td)
        # Sort the unique (non-zero) b-values per diffusion time
        nonzero_b = np.sort(np.unique(bvalues[tdvalues == td]))[1:]
        for b in nonzero_b:
            selection = (tdvalues == td) & (bvalues == b)
            norm_image = np.mean(dwi_image[..., selection], axis=-1)
            # Compute the normalization inside the mask
            norm_image = np.divide(norm_image * nan_mask, b0_image * nan_mask)
            # Add all features to the lists
            pa_image.append(norm_image)
            pa_b.append(b)
            pa_td.append(td)
            # Update mask with the values where the diffusion-weighted signal was higher than the b0 image
            bool_mask = bool_mask & (norm_image < 1)
            nan_mask = np.copy(bool_mask).astype(float)
            nan_mask[bool_mask == 0] = np.nan
    # Convert the list of powder averaged images into numpy array
    pa_image = np.stack(pa_image, axis=-1)
    # Reapply the last updated mask to every volume
    for volume in range(pa_image.shape[-1]):
        pa_image[..., volume] = nan_mask * pa_image[..., volume]
    # Get rid of extreme values
    pa_image = np.clip(pa_image, 0, 1)

    # Check and convert b-values from s/mm² to ms/µm² if provided in s/mm²
    if np.max(pa_b) > 500:
        # Warn that the b-values were probably not in ms/µm²:
        warnings.warn(
            "b-values provided are suspiciously high. Please check that these values were in ms/µm². "
            "Continuing as if b-values provided were in s/mm²"
        )
        logging.info("Continuing as if b-values provided were in s/mm²")
        pa_b = np.divide(pa_b, 1000)

    # Save the powder averaged image with all b-values including b0s
    if debug:
        logging.info("Sanity check")
        logging.info(f"Image with b0s shape : {pa_image.shape}")
        powder_average_including_b0_nii = nib.Nifti1Image(pa_image, affine=aff, header=hdr)
        powder_average_including_b0_filename = f'{out_path}/powderaverage_incl_b0_dwi.nii.gz'
        nib.save(powder_average_including_b0_nii, powder_average_including_b0_filename)

    # Save without the b = 0 ms/µm² images
    without_b0 = np.array(pa_b) != 0
    pa_img_no_b0 = pa_image[..., without_b0]
    pa_b_no_b0 = np.array(pa_b)[without_b0]
    pa_td_no_b0 = np.array(pa_td)[without_b0]
    powder_average_nii = nib.Nifti1Image(pa_img_no_b0, affine=aff, header=hdr)
    nib.save(powder_average_nii, powder_average_filename)
    if debug:
        logging.info(
            f"Without b0s, image shape : {pa_img_no_b0.shape}  with maximum value of {np.nanmax(pa_img_no_b0)}"
        )
        logging.info(f"b-values :{pa_b_no_b0}")
        logging.info(f"td values :{pa_td_no_b0}")

    # Save b-values and diffusion times
    np.savetxt(updated_bvals_path, pa_b_no_b0, fmt='%1.3f')
    np.savetxt(updated_td_path, pa_td_no_b0, fmt='%1.3f')

    logging.info("Powder average finished !")

    return powder_average_filename, updated_bvals_path, updated_td_path


# Save to npz
def save_powder_average_as_npz(dwi_path, bvals_path, td_path, small_delta, out_path, debug=False):
    """
    Save the powder averaged image, b-values and diffusion times in a npz file

    Parameters
    ----------
    dwi_path : str
        Path to the dwi image
    bvals_path : str
        Path to the b-values file
    td_path : str
        Path to the diffusion times file
    small_delta : float
        Small delta value in ms
    out_path : str
        Path to the output folder
    debug : bool
        If True, print some information for debugging

    Returns
    -------
    powder_average_signal_npz_filename : str
        Path to the npz file containing the powder averaged image, b-values and diffusion times
    """
    powder_average_signal_npz_filename = f'{out_path}/powderaverage_signal.npz'
    # Extract the powder-averaged image, b-values and diffusion times
    pa_image_nii = nib.load(dwi_path)
    pa_image = np.clip(pa_image_nii.get_fdata(), 0, 1)
    bval = np.loadtxt(bvals_path)
    tdval = np.loadtxt(td_path)

    if debug:
        logging.info("Sanity Check")
        logging.info(f"b-values : {bval}")
        logging.info(f"diffusion times : {tdval}")
        logging.info(f"small delta : {small_delta}")

    # Initialize the extraction of the signal values where there is no NaN
    mask = pa_image.sum(axis=-1) > 0
    mask = mask.astype(bool) & np.invert(np.isnan(mask))
    locations = np.array(np.where(mask)).T  # (N, 3)
    signal = pa_image[mask, :]

    if debug:
        logging.info(f"Is there any nan ? {np.isnan(signal).sum()}")
        logging.info(f"Meaningful signal shape : {signal.shape}")

    np.savez_compressed(
        powder_average_signal_npz_filename,
        signal=signal,
        b=bval,
        td=tdval,
        small_delta=small_delta,
        locations=locations,
        img_shape=pa_image.shape[0:3],
    )
    return powder_average_signal_npz_filename


def normalize_sigma(dwi_path, lowb_noisemap_path, bvals_path, out_path):
    """
    Normalize the noisemap by the b0 image

    Parameters
    ----------
    dwi_path : str
        Path to the dwi image
    lowb_noisemap_path : str
        Path to the lowb noisemap
    bvals_path : str
        Path to the b-values file
    out_path : str
        Path to the output folder

    Returns
    -------
    normalized_sigma_filename : str
    """
    dwi_nii = nib.load(dwi_path)
    noisemap_nii = nib.load(lowb_noisemap_path)
    bval = np.loadtxt(bvals_path)
    aff, hdr = noisemap_nii.affine, noisemap_nii.header
    b0 = dwi_nii.get_fdata()[..., bval == 0]
    b0 = np.mean(b0, axis=-1)
    # Load the noisemap
    noisemap = noisemap_nii.get_fdata()
    if noisemap.ndim >= 4:
        noisemap = np.squeeze(noisemap)
    assert noisemap.ndim == 3, "Noisemap should be 3D"
    # Normalize the noisemap (sigma)
    norm_sigma = np.divide(noisemap, b0)
    # Sigma should be strictly higher than 0, we also clip it at 100
    norm_sigma = np.clip(norm_sigma, 1e-9, 100)
    norm_sigma[norm_sigma > 99] = np.nan
    # Save the normalized sigma map
    normalized_sigma_nii = nib.Nifti1Image(norm_sigma, affine=aff, header=hdr)
    normalized_sigma_filename = f'{out_path}/normalized_sigma.nii.gz'
    nib.save(normalized_sigma_nii, normalized_sigma_filename)

    return normalized_sigma_filename


def save_data_as_npz(
    powder_average_filename,
    bvals_path,
    td_path,
    out_path,
    normalized_sigma_filename=None,
    small_delta=None,
    debug=False,
):
    """
    Save the powder averaged image, b-values and diffusion times in a npz file

    Parameters
    ----------
    powder_average_filename : str
        Path to the powder-average image
    bvals_path : str
        Path to the b-values file (in ms/µm2)
    td_path : str
        Path to the diffusion times file (in ms)
    out_path : str
        Path to the output folder
    normalized_sigma_filename : str
        Path to the normalized sigma file
    small_delta : float
        Small delta value in ms
    debug : bool
        If True, print some information for debugging

    Returns
    -------
    powder_average_signal_npz_filename : str
    """
    powder_average_signal_npz_filename = f'{out_path}/powderaverage_signal.npz'
    # Extract the powder-averaged image, b-values and diffusion times
    pa_image_nii = nib.load(powder_average_filename)
    pa_image = np.clip(pa_image_nii.get_fdata(), 0, 1)
    bval = np.loadtxt(bvals_path)
    tdval = np.loadtxt(td_path)

    if debug:
        logging.info("Sanity Check")
        logging.info(f"b-values : {bval}")
        logging.info(f"diffusion times : {tdval}")

    # Initialize the extraction of the signal values where there is no NaN
    mask = pa_image.sum(axis=-1) > 0
    mask = mask.astype(bool) & np.invert(np.isnan(mask))
    locations = np.array(np.where(mask)).T  # (N, 3)
    signal = pa_image[mask, :]

    if debug:
        logging.info(f"Is there any NaN in the signal ? {np.isnan(np.sum(signal))}")
        logging.info(f"Meaningful signal shape : {signal.shape}")
        logging.info(f"Mask shape : {mask.shape}")

    if normalized_sigma_filename is not None:
        norm_sigma = nib.load(normalized_sigma_filename).get_fdata()
        norm_sigma = np.clip(norm_sigma, 1e-9, 100)
        if debug:
            logging.info(f"Normalized sigma array shape : {norm_sigma.shape}")
        sigma = norm_sigma[mask]
        np.savez_compressed(
            powder_average_signal_npz_filename,
            signal=signal,
            sigma=sigma,
            b=bval,
            td=tdval,
            small_delta=small_delta,
            locations=locations,
            img_shape=pa_image.shape[0:3],
        )
    else:
        np.savez_compressed(
            powder_average_signal_npz_filename,
            signal=signal,
            b=bval,
            td=tdval,
            small_delta=small_delta,
            locations=locations,
            img_shape=pa_image.shape[0:3],
        )
    return powder_average_signal_npz_filename
