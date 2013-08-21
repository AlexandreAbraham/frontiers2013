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

# Keep only data corresponding to faces or houses
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
X = fmri_data[..., condition_mask]
y = y[condition_mask]
session = session[condition_mask]
conditions = conditions[condition_mask]

# We have 2 conditions
n_conditions = np.size(np.unique(y))

### Loading step ##############################################################
from nilearn.io import NiftiMasker
from nibabel import Nifti1Image
nifti_masker = NiftiMasker(mask=mask, sessions=session, smoothing_fwhm=4,
                           memory="nilearn_cache", memory_level=1)
niimg = Nifti1Image(X, affine)
X = nifti_masker.fit_transform(niimg)

### Prediction function #######################################################

### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel and C=1
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
niimg = nifti_masker.inverse_transform(svc[0])

# We use a masked array so that the voxels at '-1' are displayed
# transparently
act = np.ma.masked_array(niimg.get_data(), niimg.get_data() == 0)

### Create the figure

z = 25

import pylab as pl
from matplotlib.patches import Rectangle
pl.figure(figsize=(5,7))
pl.axis('off')
# pl.title('SVM vectors')
pl.imshow(np.rot90(mean_img[..., z]), cmap=pl.cm.gray,
          interpolation='nearest')
pl.imshow(np.rot90(act[..., z]), cmap=pl.cm.hot,
          interpolation='nearest')

mask_house = nibabel.load(h.mask_house[0]).get_data()
mask_face = nibabel.load(h.mask_face[0]).get_data()

pl.contour(np.rot90(mask_house[..., z].astype(np.bool)), contours=1,
        antialiased=False, linewidths=3., levels=[0], interpolation='nearest',
    colors=['indigo'])

pl.contour(np.rot90(mask_face[..., z].astype(np.bool)), contours=1,
        antialiased=False, linewidths=3., levels=[0], interpolation='nearest',
    colors=['deeppink'])

p_h = Rectangle((0, 0), 1, 1, fc="indigo")
p_f = Rectangle((0, 0), 1, 1, fc="deeppink")
pl.legend([p_h, p_f], ["house", "face"])
pl.tight_layout()

pl.savefig('haxby_svm.pdf')
pl.savefig('haxby_svm.eps')
