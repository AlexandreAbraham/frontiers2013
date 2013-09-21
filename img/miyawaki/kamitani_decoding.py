### Init ######################################################################
import numpy as np

offset = 2

hand_made_affine = np.asarray(
        [[3, 0, 0, 98],
         [0, 3, 0, -112],
         [0, 0, 3, -26],
         [0, 0, 0, 1]])

### Load Kamitani dataset #####################################################
from nilearn import datasets
dataset = datasets.fetch_kamitani()

# Keep only random runs
X_random = dataset.func[12:]
y_random = dataset.label[12:]
y_shape = (10, 10)

### Preprocess data ###########################################################
from nilearn.io import MultiNiftiMasker

print "Preprocessing data"

# Load and mask fMRI data
masker = MultiNiftiMasker(mask=dataset.mask, detrend=True,
                          standardize=True)
masker.fit()
mask = masker.mask_img_.get_data().astype(bool)
X_train = masker.transform(X_random)

# Load target data
y_train = []
for y in y_random:
    y_train.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                              (-1,) + y_shape, order='F'))

X_train = [x[offset:] for x in X_train]
y_train = [y[:-offset] for y in y_train]


X_train = np.vstack(X_train)
y_train = np.vstack(y_train).astype(np.float)

n_pixels = y_train.shape[1]
n_features = X_train.shape[1]

# Flatten the stimuli
y_train = np.reshape(y_train, (-1, n_pixels * n_pixels))

# Remove rest period
X_train = X_train[y_train[:, 0] != -1]
y_train = y_train[y_train[:, 0] != -1]


### Prediction function #######################################################
import pylab as pl
import nibabel

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_selection import f_classif, SelectKBest

# Chosen pixel
p = (4, 2)
# Mask for chosen pixel
pixmask = np.zeros((10, 10), dtype=bool)
pixmask[p] = 1
# Get index of the chosen pixel
i_p = 42

lr = LR(penalty='l1', C=0.01)
svc = LinearSVC(penalty='l1', dual=False)
lr.fit(X_train, y_train[:, i_p])
svc.fit(X_train, y_train[:, i_p])

### Output ###################################################################
# Create a mask with chosen voxels to contour them
contour = np.zeros_like(mask)
for x, y in [(31, 9), (31, 10), (30, 10), (32, 10)]:
    contour[x, y, 10] = 1

from matplotlib.lines import Line2D


def plot_lines(mask, linewidth=3):
    for i, j in np.ndindex(mask.shape):
        if i + 1 < mask.shape[0] and mask[i, j] != mask[i + 1, j]:
            pl.gca().add_line(Line2D([j - .5, j + .5], [i + .5, i + .5],
                color='b', linewidth=linewidth))
        if j + 1 < mask.shape[1] and mask[i, j] != mask[i, j + 1]:
            pl.gca().add_line(Line2D([j + .5, j + .5], [i - .5, i + .5],
                color='b', linewidth=linewidth))


fig = pl.figure(figsize=(8, 8))
sbrain = masker.inverse_transform(lr.coef_)
sbrain = sbrain.get_data()[..., 0]
vmax = np.max(np.abs(sbrain))
bg = nibabel.load('bg_.nii.gz')
pl.imshow(bg.get_data()[:, :, 10], interpolation="nearest", cmap='gray')
pl.imshow(np.ma.masked_less(sbrain[:, :, 10], 1e-10),
        interpolation="nearest", cmap='autumn', vmin=-vmax, vmax=vmax)
plot_lines(contour[:, :, 10])
pl.axis('off')
fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
pl.savefig('pixel_logistic.pdf')
pl.savefig('pixel_logistic.eps')
pl.close()

fig = pl.figure(figsize=(8, 8))
sbrain = masker.inverse_transform(svc.coef_)
sbrain = sbrain.get_data()
vmax = np.max(np.abs(sbrain))
bg = nibabel.load('bg_.nii.gz')
pl.imshow(bg.get_data()[:, :, 10], interpolation="nearest", cmap='gray')
pl.imshow(np.ma.masked_less(sbrain[:, :, 10, 0], 1e-10),
        interpolation="nearest", cmap='autumn', vmin=-vmax, vmax=vmax)
plot_lines(contour[:, :, 10])
pl.axis('off')
fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
pl.savefig('pixel_omp.pdf')
pl.savefig('pixel_omp.eps')
pl.close()

# Scores
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Parallel, delayed

pipeline_LR = Pipeline([('selection', SelectKBest(f_classif, 500)),
                     ('clf', LR(penalty="l1", C=0.01))])
pipeline_SVC = Pipeline([('selection', SelectKBest(f_classif, 500)),
                     ('clf', LinearSVC(penalty='l1', dual=False))])


scores_log = Parallel(n_jobs=1)(delayed(cross_val_score)(pipeline_LR, X_train,
    y, cv=5, verbose=True) for y in y_train.T)
scores_omp = Parallel(n_jobs=1)(delayed(cross_val_score)(pipeline_SVC,
    X_train, y,
    cv=5, verbose=True) for y in y_train.T)


fig = pl.figure(figsize=(8, 8))
pl.imshow(np.array(scores_log).mean(1).reshape(10, 10),
        interpolation="nearest", vmin=.3, vmax=1.)
plot_lines(pixmask, linewidth=6)
pl.hot()
fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
pl.savefig('scores_log.pdf')
pl.savefig('scores_log.eps')
pl.close()


fig = pl.figure(figsize=(8, 8))
pl.imshow(np.array(scores_omp).mean(1).reshape(10, 10),
        interpolation="nearest", vmin=.3, vmax=1.)
plot_lines(pixmask, linewidth=6)
pl.hot()
fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
pl.savefig('scores_omp.pdf')
pl.savefig('scores_omp.eps')
pl.close()
