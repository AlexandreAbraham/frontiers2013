"""
Independent component analysis of resting-state fMRI
=====================================================

An example applying ICA to resting-state data.
"""

### Load nyu_rest dataset #####################################################
from utils import datasets
dataset = datasets.fetch_adhd(n_subjects=10)
n_components = 10

# Index of the generated axial slice of the brain
z = 35
# Path
path = 'ica'

### Initialization ############################################################
import nibabel
from os.path import exists, join
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

if not exists(join(path, 'canica.nii.gz')):
    try:
        from nilearn.decomposition.canica import CanICA
        t0 = time.time()
        canica = CanICA(n_components=n_components, mask=masker,
                        target_affine=np.diag((3, 3, 3.)),
                        smoothing_fwhm=6.,
                        memory="nilearn_cache", memory_level=1,
                        threshold=None,
                        random_state=1, n_jobs=-1)
        canica.fit(dataset.func)
        print('Canica: %f' % (time.time() - t0))
        canica_components = masker.inverse_transform(canica.components_)
        nibabel.save(canica_components, join(path, 'canica.nii.gz'))
    except ImportError:
        import warnings
        warnings.warn('nilearn must be installed to run CanICA')


plot_ica_map(nibabel.load(join(path, 'canica.nii.gz')).get_data()[..., 4])
pl.savefig(join(path, 'canica.pdf'))
pl.savefig(join(path, 'canica.eps'))


smooth_masked = masker.transform(dataset.func)

### Melodic ICA ############################################################
# To have MELODIC results, please use my melodic branch of nilearn

plot_ica_map(nibabel.load(join(path, 'melodic.nii.gz')).get_data()[..., 3])
pl.savefig(join(path, 'melodic.pdf'))
pl.savefig(join(path, 'melodic.eps'))

### FastICA ##################################################################

# Concatenate all the subjects
if not exists(join(path, 'ica.nii.gz')):
    from sklearn.decomposition import FastICA
    X = np.vstack(smooth_masked)
    ica = FastICA(n_components=n_components)
    t0 = time.time()
    ica.fit(X)
    print('FastICA: %f' % (time.time() - t0))
    ica_components = masker.inverse_transform(ica.components_)
    nibabel.save(ica_components, join(path, 'ica.nii.gz'))

plot_ica_map(nibabel.load(join(path, 'ica.nii.gz')).get_data()[..., 2])
pl.savefig(join(path, 'ica.pdf'))
pl.savefig(join(path, 'ica.eps'))
