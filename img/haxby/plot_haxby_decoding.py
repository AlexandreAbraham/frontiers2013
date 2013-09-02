### Visualization function ####################################################
import pylab as pl
from matplotlib.patches import Rectangle


def plot_haxby(activation, title):
    z = 25

    fig = pl.figure(figsize=(4, 5.4))
    fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
    pl.axis('off')
    # pl.title('SVM vectors')
    pl.imshow(np.rot90(mean_img[:, 4:58, z]), cmap=pl.cm.gray,
              interpolation='nearest')
    pl.imshow(np.rot90(activation[:, 4:58, z]), cmap=pl.cm.hot,
              interpolation='nearest')

    mask_house = nibabel.load(h.mask_house[0]).get_data()
    mask_face = nibabel.load(h.mask_face[0]).get_data()

    pl.contour(np.rot90(mask_house[:, 4:58, z].astype(np.bool)), contours=1,
            antialiased=False, linewidths=4., levels=[0],
            interpolation='nearest', colors=['blue'])

    pl.contour(np.rot90(mask_face[:, 4:58, z].astype(np.bool)), contours=1,
            antialiased=False, linewidths=4., levels=[0],
            interpolation='nearest', colors=['limegreen'])

    p_h = Rectangle((0, 0), 1, 1, fc="blue")
    p_f = Rectangle((0, 0), 1, 1, fc="limegreen")
    pl.legend([p_h, p_f], ["house", "face"])
    pl.title(title, x=.05, ha='left', y=.90, color='w', size=28)


### Load Haxby dataset ########################################################
from nilearn import datasets
import numpy as np
import nibabel

dataset_files = datasets.fetch_haxby_simple()
h = datasets.fetch_haxby()

# fmri_data and mask are copied to break any reference to the original object
bold_img = nibabel.load(dataset_files.func)
fmri_data = np.copy(bold_img.get_data())
affine = bold_img.get_affine()
y, session = np.loadtxt(dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']
mask = dataset_files.mask
# fmri_data.shape is (40, 64, 64, 1452)
# and mask.shape is (40, 64, 64)

### Preprocess data ###########################################################
# Build the mean image because we have no anatomic data
mean_img = fmri_data.mean(axis=-1)

### Restrict to faces and houses ##############################################
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
X = fmri_data[..., condition_mask]
y = y[condition_mask]
session = session[condition_mask]
conditions = conditions[condition_mask]

### Masking step ##############################################################
from nilearn.io import NiftiMasker
from nibabel import Nifti1Image
nifti_masker = NiftiMasker(mask=mask, sessions=session, smoothing_fwhm=4,
                           memory="nilearn_cache", memory_level=1)
X_img = Nifti1Image(X, affine)
X = nifti_masker.fit_transform(X_img)

###############################################################################
#                                                                             #
#   F-score                                                                   #
#                                                                             #
###############################################################################

from sklearn.feature_selection import f_classif
f_values, p_values = f_classif(X, y)
p_values = -np.log10(p_values)
p_values[np.isnan(p_values)] = 0
p_values[p_values > 10] = 10
p_unmasked = nifti_masker.inverse_transform(p_values).get_data()

plot_haxby(p_unmasked, 'F-score')
pl.savefig('haxby_fscore.pdf')
pl.savefig('haxby_fscore.eps')

###############################################################################
#                                                                             #
#   SVC                                                                       #
#                                                                             #
###############################################################################
### Define the estimator
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=0.01)

### Dimension reduction #######################################################

from sklearn.feature_selection import SelectKBest, f_classif

### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 500
feature_selection = SelectKBest(f_classif, k=500)

# We have our classifier (SVC), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])

### Fit and predict ###########################################################

anova_svc.fit(X, y)
y_pred = anova_svc.predict(X)

### Visualisation #############################################################

### Look at the discriminating weights
svc = clf.support_vectors_
# reverse feature selection
svc = feature_selection.inverse_transform(svc)
# reverse masking
svc = nifti_masker.inverse_transform(svc[0])

# We use a masked array so that the voxels at '-1' are displayed
# transparently
act = np.ma.masked_array(svc.get_data(), svc.get_data() == 0)

plot_haxby(act, 'SVC')
pl.savefig('haxby_svm.pdf')
pl.savefig('haxby_svm.eps')


###############################################################################
#                                                                             #
#   Searchlight                                                               #
#                                                                             #
###############################################################################

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
process_mask_img = nibabel.Nifti1Image(process_mask, mask_img.get_affine())

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
searchlight.fit(X_img, y)
s_scores = np.ma.array(searchlight.scores_, mask=np.logical_not(process_mask))
plot_haxby(s_scores, 'Searchlight')
pl.savefig('haxby_searchlight.pdf')
pl.savefig('haxby_searchlight.eps')
