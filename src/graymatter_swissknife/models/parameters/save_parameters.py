import logging
import numpy as np
import nibabel as nib


def save_estimations_as_nifti(estimations, model, powder_average_path, mask_path, out_path):
    aff, hdr = nib.load(powder_average_path).affine, nib.load(powder_average_path).header

    powder_average = nib.load(powder_average_path).get_fdata()
    if powder_average.ndim == 4:
        powder_average = np.sum(powder_average, axis=-1)

    if mask_path is not None:
        mask = nib.load(mask_path).get_fdata()
        mask = mask.astype(bool)
        mask = mask & (~np.isnan(powder_average))
    else:
        mask = ~np.isnan(powder_average)
    param_map_shape = mask.shape

    param_names = model.param_names
    for i, param_name in enumerate(param_names):
        param_map = np.zeros(param_map_shape)
        param_map[mask] = estimations[:, i]
        param_map_nifti = nib.Nifti1Image(param_map, aff, hdr)
        nib.save(param_map_nifti, f'{out_path}/{model.name.lower()}_{param_name.lower()}.nii.gz')
        logging.info(f'{model.name.lower()}_{param_name.lower()}.nii.gz saved in {out_path}')
