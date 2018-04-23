"""
Utilities to compute a brain mask from EPI images
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import numpy as np
from scipy import ndimage
import nibabel


###############################################################################
# Time series extraction
###############################################################################

def apply_mask(niimgs, mask_img, dtype=np.float32,
                     smoothing_fwhm=None, ensure_finite=True):
    """Extract signals from images using specified mask.

    Read the time series from the given nifti images or filepaths,
    using the mask.

    Parameters
    -----------
    niimgs: list of 4D nifti images
        Images to be masked. list of lists of 3D images are also accepted.

    mask_img: niimg
        3D mask array: True where a voxel should be used.

    smoothing_fwhm: float
        (optional) Gives the size of the spatial smoothing to apply to
        the signal, in voxels. Implies ensure_finite=True.

    ensure_finite: bool
        If ensure_finite is True (default), the non-finite values (NaNs and
        infs) found in the images will be replaced by zeros.

    Returns
    --------
    session_series: numpy.ndarray
        2D array of series with shape (image number, voxel number)

    Notes
    -----
    When using smoothing, ensure_finite is set to True, as non-finite
    values would spread accross the image.
    """

    if isinstance(mask_img, str):
        mask_img = nibabel.load(mask_img)
    mask_data = mask_img.get_data().astype(bool)
    mask_affine = mask_img.get_affine()

    if smoothing_fwhm is not None:
        ensure_finite = True

    if isinstance(niimgs, str):
        niimgs = nibabel.load(niimgs)
    affine = niimgs.get_affine()[:3, :3]

    if not np.allclose(mask_affine, niimgs.get_affine()):
        raise ValueError('Mask affine: \n%s\n is different from img affine:'
                         '\n%s' % (str(mask_affine),
                                   str(niimgs.get_affine())))

    if not mask_data.shape == niimgs.shape[:3]:
        raise ValueError('Mask shape: %s is different from img shape:%s'
                         % (str(mask_data.shape), str(niimgs.shape[:3])))

    # All the following has been optimized for C order.
    # Time that may be lost in conversion here is regained multiple times
    # afterward, especially if smoothing is applied.
    data = niimgs.get_data()
    series = np.asarray(data)
    del data, niimgs  # frees a lot of memory

    _smooth_array(series, affine, fwhm=smoothing_fwhm,
                  ensure_finite=ensure_finite, copy=False)
    return series[mask_data].T


def _smooth_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):
    """Smooth images by applying a Gaussian filter.

    Apply a Gaussian filter along the three first dimensions of arr.

    Parameters
    ==========
    arr: numpy.ndarray
        4D array, with image number as last dimension. 3D arrays are also
        accepted.

    affine: numpy.ndarray
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).

    fwhm: scalar or numpy.ndarray
        Smoothing strength, as a full-width at half maximum, in millimeters.
        If a scalar is given, width is identical on all three directions.
        A numpy.ndarray must have 3 elements, giving the FWHM along each axis.
        If fwhm is None, no filtering is performed (useful when just removal
        of non-finite values is needed)

    ensure_finite: bool
        if True, replace every non-finite values (like NaNs) by zero before
        filtering.

    copy: bool
        if True, input array is not modified. False by default: the filtering
        is performed in-place.

    Returns
    =======
    filtered_arr: numpy.ndarray
        arr, filtered.

    Notes
    =====
    This function is most efficient with arr in C order.
    """

    if copy:
        arr = arr.copy()

    # Keep only the scale part.
    affine = affine[:3, :3]

    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        arr[np.logical_not(np.isfinite(arr))] = 0

    if fwhm is not None:
        # Convert from a FWHM to a sigma:
        # Do not use /=, fwhm may be a numpy scalar
        fwhm = fwhm / np.sqrt(8 * np.log(2))
        vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
        sigma = fwhm / vox_size
        for n, s in enumerate(sigma):
            ndimage.gaussian_filter1d(arr, s, output=arr, axis=n)

    return arr


def unmask(X, mask_img, order="C"):
    """Take masked data and bring them back to 3D (space only).

    Parameters
    ==========
    X: numpy.ndarray
        Masked data. shape: (samples,)

    mask_img: niimg
        3D mask array: True where a voxel should be used.
    """

    if isinstance(mask_img, str):
        mask_img = nibabel.load(mask_img)
    mask_data = mask_img.get_data().astype(bool)

    if X.ndim == 1:
        data = np.zeros(
            (mask_data.shape[0], mask_data.shape[1], mask_data.shape[2]),
            dtype=X.dtype, order=order)
        data[mask_data] = X
        return data
    elif X.ndim == 2:
        data = np.zeros(mask_data.shape + (X.shape[0],), dtype=X.dtype, order=order)
        data[mask_data, :] = X.T
        return data
    raise ValueError("X must be 1-dimensional or 2-dimensional")
