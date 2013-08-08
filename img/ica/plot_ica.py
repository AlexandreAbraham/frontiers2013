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
from nipy.labs.viz import plot_map
import pylab as pl
import numpy as np

# CanICA
from nilearn.decomposition.canica import CanICA

canica = CanICA(n_components=n_components,
                smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=1,
                threshold=None,
                random_state=1)
canica.fit(dataset.func)
masker = canica.masker_

canica_components = masker.inverse_transform(canica.components_)
nibabel.save(canica_components, 'canica.nii.gz')

c = canica_components.get_data()[:, :, :, 16]
plot_map(c, canica_components.get_affine(), cut_coords=[37], threshold=0.002,
        slicer='z')

pl.savefig('canica.png')

smooth_masked = masker.transform(dataset.func)

### Melodic ICA ############################################################
# To have MELODIC results, please use my melodic branch of nilearn
from nilearn.decomposition.melodic import MelodicICA

if not os.path.exists('melodic.png'):
    smoothed_func = masker.inverse_transform(smooth_masked)
    melodic = MelodicICA(mask=masker.mask_img_)
    melodic.fit(smoothed_func)
    melodic_components = melodic.maps_img_
    nibabel.save(melodic_components, 'melodic.nii.gz')
    c = melodic_components.get_data()[:, :, :, 5]
    plot_map(c, melodic_components.get_affine(), cut_coords=[37], threshold=1.5,
            slicer='z')
    pl.savefig('melodic.png')

### FastICA ##################################################################

# Concatenate all the subjects
from sklearn.decomposition import FastICA
X = np.vstack(smooth_masked)
ica = FastICA(n_components=n_components)
ica.fit(X)
ica_components = masker.inverse_transform(ica.components_)
nibabel.save(ica_components, 'ica.nii.gz')
