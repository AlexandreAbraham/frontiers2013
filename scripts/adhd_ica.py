"""
Independent component analysis of resting-state fMRI
=====================================================

An example applying ICA to resting-state data.
"""

### Load nyu_rest dataset #####################################################
from utils import datasets, masking, signal
import numpy as np
dataset = datasets.fetch_adhd(n_subjects=10)
n_components = 10

# Index of the generated axial slice of the brain
z = 35
# Path
path = 'ica'
cmap = 'jet'

### Initialization ############################################################
import nibabel
from os.path import exists, join
import pylab as pl
import time


mask_img = nibabel.load(join('utils', 'adhd_mask.nii.gz'))


def plot_ica_map(map_3d):
    # Mask the background
    map_3d = np.ma.masked_array(map_3d,
            np.logical_not(mask_img.get_data().astype(bool)))
    section = map_3d[:, :, z]
    vmax = np.max(np.abs(map_3d))
    pl.figure(figsize=(3.8, 4.5))
    pl.axes([0, 0, 1, 1])
    pl.imshow(np.rot90(section), interpolation='nearest',
              vmax=vmax, vmin=-vmax, cmap=cmap)
    pl.axis('off')

# Mask data
X_ = []
for x in dataset.func:
    X_.append(masking.apply_mask(x, mask_img, smoothing_fwhm=6.))
X = X_

# Clean signals
X_ = []
for x in X:
    X_.append(signal.clean(x, standardize=True, detrend=False))
X = X_

### CanICA ####################################################################

if not exists(join(path, 'canica.nii.gz')):
    try:
        from nilearn.decomposition.canica import CanICA
        t0 = time.time()
        canica = CanICA(n_components=n_components, mask=mask_img,
                        smoothing_fwhm=6.,
                        memory="nilearn_cache", memory_level=1,
                        threshold=None,
                        random_state=1, n_jobs=-1)
        canica.fit(dataset.func)
        print('Canica: %f' % (time.time() - t0))
        canica_components = masking.unmask(canica.components_, mask_img)
        nibabel.save(nibabel.Nifti1Image(canica_components,
            mask_img.get_affine()), join(path, 'canica.nii.gz'))
    except ImportError:
        import warnings
        warnings.warn('nilearn must be installed to run CanICA')


canica_dmn = nibabel.load(join(path, 'canica.nii.gz')).get_data()[..., 4]


### Melodic ICA ############################################################
# To have MELODIC results, please use my melodic branch of nilearn

melodic_dmn = nibabel.load(join(path, 'melodic.nii.gz')).get_data()[..., 3]

### FastICA ##################################################################

# Concatenate all the subjects
if not exists(join(path, 'ica.nii.gz')):
    from sklearn.decomposition import FastICA
    X = np.vstack(X)
    ica = FastICA(n_components=n_components, random_state=0)
    t0 = time.time()
    ica.fit(X)
    print('FastICA: %f' % (time.time() - t0))
    ica_components = masking.unmask(ica.components_, mask_img)
    nibabel.save(nibabel.Nifti1Image(ica_components,
            mask_img.get_affine()), join(path, 'ica.nii.gz'))

ica_dmn = nibabel.load(join(path, 'ica.nii.gz')).get_data()[..., 2]

### Plots ####################################################################

# Split the sign to harmonize maps
ica_dmn = -ica_dmn
canica_dmn = -canica_dmn

plot_ica_map(ica_dmn)
pl.savefig(join(path, 'ica.pdf'))
pl.savefig(join(path, 'ica.eps'))

plot_ica_map(canica_dmn)
pl.savefig(join(path, 'canica.pdf'))
pl.savefig(join(path, 'canica.eps'))

plot_ica_map(melodic_dmn)
pl.savefig(join(path, 'melodic.pdf'))
pl.savefig(join(path, 'melodic.eps'))

### Plot the colorbar #########################################################
import matplotlib as mpl


fig = pl.figure(figsize=(.4, 2.4))
norm = mpl.colors.Normalize(vmin=0., vmax=1.)
cb = mpl.colorbar.ColorbarBase(pl.gca(), cmap=cmap, norm=norm)
cb.set_ticks([])
#fig.subplots_adjust(bottom=0.05, top=1., left=0.08, right=.92)
pl.savefig(join(path, 'colorbar.pdf'))
pl.savefig(join(path, 'colorbar.eps'))
