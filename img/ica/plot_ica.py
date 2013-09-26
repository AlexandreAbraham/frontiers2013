"""
Independent component analysis of resting-state fMRI
=====================================================

An example applying ICA to resting-state data.
"""

### Load nyu_rest dataset #####################################################
from nilearn import datasets
dataset = datasets.fetch_adhd(n_subjects=10)
n_components = 10

# Index of the generated axial slice of the brain
z = 35

### Initialization ############################################################
import nibabel
import os
import pylab as pl
import numpy as np
import time
from nilearn.input_data import MultiNiftiMasker


def plot_ica_map(map_3d):
    # Mask the background
    map_3d = np.ma.masked_array(map_3d,
            np.logical_not(masker.mask_img_.get_data().astype(bool)))
    section = map_3d[:, :, z]
    vmax = np.max(np.abs(section))

    pl.figure(figsize=(3.8, 4.5))
    pl.axes([0, 0, 1, 1])
    pl.imshow(np.rot90(section), interpolation='nearest',
              vmax=vmax, vmin=-vmax, cmap='jet')
    pl.axis('off')


# Masker:
masker = MultiNiftiMasker(smoothing_fwhm=6.,
                          target_affine=np.diag((3, 3, 3.)),
                          memory="nilearn_cache",
                          memory_level=1, n_jobs=-1)
masker.fit(dataset.func)


### CanICA ####################################################################

if not os.path.exists('canica.nii.gz'):
    t0 = time.time()
    from nilearn.decomposition.canica import CanICA
    canica = CanICA(n_components=n_components, mask=masker,
                    target_affine=np.diag((3, 3, 3.)),
                    smoothing_fwhm=6.,
                    memory="nilearn_cache", memory_level=1,
                    threshold=None,
                    random_state=1, n_jobs=-1)
    canica.fit(dataset.func)
    print('Canica: %f' % (time.time() - t0))
    canica_components = masker.inverse_transform(canica.components_)
    nibabel.save(canica_components, 'canica.nii.gz')

if not os.path.exists('canica.pdf'):
    plot_ica_map(nibabel.load('canica.nii.gz').get_data()[..., 4])
    pl.savefig('canica.pdf')
    pl.savefig('canica.eps')


smooth_masked = masker.transform(dataset.func)

### Melodic ICA ############################################################
# To have MELODIC results, please use my melodic branch of nilearn

if not os.path.exists('melodic.nii.gz'):
    from nilearn.decomposition.melodic import MelodicICA
    smoothed_func = masker.inverse_transform(smooth_masked)
    melodic = MelodicICA(mask=masker.mask_img_, approach='concat')
    t0 = time.time()
    melodic.fit(smoothed_func)
    print('Melodic: %f' % (time.time() - t0))
    melodic_components = melodic.maps_img_
    nibabel.save(melodic_components, 'melodic.nii.gz')

if not os.path.exists('melodic.pdf'):
    plot_ica_map(nibabel.load('melodic.nii.gz').get_data()[..., 3])
    pl.savefig('melodic.pdf')
    pl.savefig('melodic.eps')

### FastICA ##################################################################

# Concatenate all the subjects
if not os.path.exists('ica.nii.gz'):
    from sklearn.decomposition import FastICA
    X = np.vstack(smooth_masked)
    ica = FastICA(n_components=n_components)
    t0 = time.time()
    ica.fit(X)
    print('FastICA: %f' % (time.time() - t0))
    ica_components = masker.inverse_transform(ica.components_)
    nibabel.save(ica_components, 'ica.nii.gz')

if not os.path.exists('ica.pdf'):
    plot_ica_map(nibabel.load('ica.nii.gz').get_data()[..., 2])
    pl.savefig('ica.pdf')
    pl.savefig('ica.eps')
