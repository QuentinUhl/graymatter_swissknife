import logging
import numpy as np
import nibabel as nib


def save_estimations_as_nifti(estimations, model, powder_average_path, mask_path, out_path, optimization_method):
    aff, hdr = nib.load(powder_average_path).affine, nib.load(powder_average_path).header

    powder_average = nib.load(powder_average_path).get_fdata()
    if powder_average.ndim == 4:
        powder_average = np.sum(powder_average, axis=-1)
    
    if mask_path is not None:
        # threshold for the mask (coulb also be 0.33)
        # If you touch it, please change it also in powderaverage.py
        mask_threshold = 0 
        mask = nib.load(mask_path).get_fdata()
        mask = (mask > mask_threshold).astype(bool)
        mask = mask & (~np.isnan(powder_average))
    else:
        mask = ~np.isnan(powder_average)
    param_map_shape = mask.shape

    param_names = model.param_names
    # Remove the last parameter (sigma) from the parameter names if the model has a Rician mean correction
    if model.has_rician_mean_correction:
        param_names = param_names[:-1]
    for i, param_name in enumerate(param_names):
        param_map = np.zeros(param_map_shape) * np.nan
        param_map[mask] = estimations[:, i]
        param_map_nifti = nib.Nifti1Image(param_map, aff, hdr)
        if optimization_method == 'nls':
            nib.save(param_map_nifti, f'{out_path}/{model.name.lower()}_{param_name.lower()}.nii.gz')
            logging.info(f'{model.name.lower()}_{param_name.lower()}.nii.gz saved in {out_path}')
        else:
            nib.save(param_map_nifti, f'{out_path}/{optimization_method}_{model.name.lower()}_{param_name.lower()}.nii.gz')
            logging.info(f'{optimization_method}_{model.name.lower()}_{param_name.lower()}.nii.gz saved in {out_path}')

