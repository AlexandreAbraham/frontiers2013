### Load Haxby dataset ########################################################
import numpy as np
import nibabel
from nilearn import datasets

dataset_files = datasets.fetch_haxby_simple()
h = datasets.fetch_haxby()

fmri_img = nibabel.load(dataset_files.func)
y, session = np.loadtxt(dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']

### Restrict to faces and houses ##############################################
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')

fmri_img = nibabel.Nifti1Image(fmri_img.get_data()[..., condition_mask],
                               fmri_img.get_affine().copy())
y, session = y[condition_mask], session[condition_mask]
conditions = conditions[condition_mask]

### Prepare masks #############################################################
# - mask_img is the original mask
# - process_mask_img is a subset of mask_img, it contains the voxels that
#   should be processed (we only keep the slice z = 26 and the back of the
#   brain to speed up computation)

mask_img = nibabel.load(dataset_files.mask)

# .astype() makes a copy.
process_mask = mask_img.get_data().astype(np.int)
process_mask[..., 26:] = 0
process_mask[..., :24] = 0
#process_mask[:, 30:] = 0
process_mask_img = nibabel.Nifti1Image(process_mask, mask_img.get_affine())

### Searchlight computation ###################################################

# Make processing parallel
# /!\ As each thread will print its progress, n_jobs > 1 could mess up the
#     information output.
n_jobs = 1

### Define the score function used to evaluate classifiers
# Here we use precision which measures proportion of true positives among
# all positives results for one class.
from sklearn.metrics import precision_score
score_func = precision_score

### Define the cross-validation scheme used for validation.
# Here we use a KFold cross-validation on the session, which corresponds to
# splitting the samples in 4 folds and make 4 runs using each fold as a test
# set once and the others as learning sets
from sklearn.cross_validation import KFold
cv = KFold(y.size, k=4)

import nilearn.decoding
# The radius is the one of the Searchlight sphere that will scan the volume
searchlight = nilearn.decoding.SearchLight(mask_img,
                                      process_mask_img=process_mask_img,
                                      radius=5.6, n_jobs=n_jobs,
                                      score_func=score_func, verbose=1, cv=cv)
searchlight.fit(fmri_img, y)

### F-scores computation ######################################################
from nilearn.io import NiftiMasker

nifti_masker = NiftiMasker(mask=mask_img, sessions=session,
                           memory='nilearn_cache', memory_level=1)
fmri_masked = nifti_masker.fit_transform(fmri_img)

from sklearn.feature_selection import f_classif
f_values, p_values = f_classif(fmri_masked, y)
p_values = -np.log10(p_values)
p_values[np.isnan(p_values)] = 0
p_values[p_values > 10] = 10
p_unmasked = nifti_masker.inverse_transform(p_values).get_data()

### Visualization #############################################################
import pylab as pl
from matplotlib.patches import Rectangle

# Use the fmri mean image as a surrogate of anatomical data
mean_fmri = fmri_img.get_data().mean(axis=-1)

# Searchlight results
pl.figure(1, figsize=(5, 7))
# searchlight.scores_ contains per voxel cross validation scores
s_scores = np.ma.array(searchlight.scores_, mask=np.logical_not(process_mask))
pl.imshow(np.rot90(mean_fmri[..., 25]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.imshow(np.rot90(s_scores[..., 25]), interpolation='nearest',
          cmap=pl.cm.hot, vmax=1)

mask_house = nibabel.load(h.mask_house[0]).get_data()
mask_face = nibabel.load(h.mask_face[0]).get_data()

pl.contour(np.rot90(mask_house[..., 25].astype(np.bool)), contours=1,
        antialiased=False, linewidths=1., levels=[0], interpolation='nearest',
    colors=['indigo'])

pl.contour(np.rot90(mask_face[..., 25].astype(np.bool)), contours=1,
        antialiased=False, linewidths=1., levels=[0], interpolation='nearest',
    colors=['darkgreen'])

p_h = Rectangle((0, 0), 1, 1, fc="indigo")
p_f = Rectangle((0, 0), 1, 1, fc="darkgreen")
pl.legend([p_h, p_f], ["house", "face"])

pl.axis('off')
pl.title('Searchlight')
pl.tight_layout()
pl.savefig('haxby_searchlight.pdf')
pl.savefig('haxby_searchlight.eps')

### F_score results
pl.figure(2, figsize=(5, 7))
p_ma = np.ma.array(p_unmasked, mask=np.logical_not(process_mask))
pl.imshow(np.rot90(mean_fmri[..., 25]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.imshow(np.rot90(p_ma[..., 25]), interpolation='nearest',
          cmap=pl.cm.hot)
pl.contour(np.rot90(mask_house[..., 25].astype(np.bool)), contours=1,
        antialiased=False, linewidths=1., levels=[0], interpolation='nearest',
    colors=['indigo'])

pl.contour(np.rot90(mask_face[..., 25].astype(np.bool)), contours=1,
        antialiased=False, linewidths=1., levels=[0], interpolation='nearest',
    colors=['darkgreen'])
p_h = Rectangle((0, 0), 1, 1, fc="indigo")
p_f = Rectangle((0, 0), 1, 1, fc="darkgreen")
pl.legend([p_h, p_f], ["house", "face"])

pl.title('F-scores')
pl.axis('off')
pl.tight_layout()
pl.savefig('haxby_fscore.pdf')
pl.savefig('haxby_fscore.eps')
