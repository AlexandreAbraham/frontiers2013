"""
Independent component analysis of resting-state fMRI
=====================================================

An example applying ICA to resting-state data.
"""

### Load nyu_rest dataset #####################################################
from nilearn import datasets
# Here we use only 3 subjects to get faster-running code. For better
# results, simply increase this number
dataset = datasets.fetch_adhd(n_subjects=10)
n_components = 20

### Preprocess ################################################################
import nibabel
import os
import pylab as pl
import numpy as np
import time

# CanICA
from nilearn.decomposition.canica import CanICA

t0 = time.time()
canica = CanICA(n_components=n_components,
                smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=1,
                threshold=None,
                random_state=1)
canica.fit(dataset.func)
print('Canica: %f' % (time.time() - t0))
masker = canica.masker_
affine = masker.mask_img_.get_affine()

if not os.path.exists('canica.nii.gz'):
    canica_components = masker.inverse_transform(canica.components_)
    nibabel.save(canica_components, 'canica.nii.gz')

if not os.path.exists('canica.pdf'):
    cc = nibabel.load('canica.nii.gz')

    c = cc.get_data()[:, :, :, 16]
    vmax = np.max(np.abs(c[:, :, 37]))
    c = np.ma.masked_equal(c, 0.)
    pl.imshow(np.rot90(c[:, :, 37]),
            interpolation='nearest', vmax=vmax, vmin=-vmax, cmap='jet')
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
    melodic_components = nibabel.load('melodic.nii.gz')
    c = melodic_components.get_data()[:, :, :, 8]
    vmax = np.max(np.abs(c[:, :, 37]))
    c = np.ma.masked_equal(c, 0.)
    pl.imshow(np.rot90(c[:, :, 37]), 
            interpolation='nearest', vmax=vmax, vmin=-vmax, cmap='jet')
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
    ica_components = nibabel.load('ica.nii.gz')
    c = ica_components.get_data()[:, :, :, 2]
    vmax = np.max(np.abs(c[:, :, 37]))
    c = np.ma.masked_equal(c, 0.)
    pl.imshow(np.rot90(c[:, :, 37]),
            interpolation='nearest', vmax=vmax, vmin=-vmax, cmap='jet')
    pl.savefig('ica.pdf')
    pl.savefig('ica.eps')
