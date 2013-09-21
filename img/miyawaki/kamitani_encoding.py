import numpy as np
import pylab as pl
import nibabel

### Load Kamitani dataset #####################################################
from nilearn import datasets
dataset = datasets.fetch_kamitani()
X_random = dataset.func[12:]
X_figure = dataset.func[:12]
y_random = dataset.label[12:]
y_figure = dataset.label[:12]
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
X_test = masker.transform(X_figure)

# Load target data
y_train = []
for y in y_random:
    y_train.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                              (-1,) + y_shape, order='F'))

y_test = []
for y in y_figure:
    y_test.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                             (-1,) + y_shape, order='F'))
offset = 2
X_train = [x[offset:] for x in X_train]
y_train = [y[:-offset] for y in y_train]
X_test = [x[offset:] for x in X_test]
y_test = [y[:-offset] for y in y_test]


X_train = np.vstack(X_train)
y_train = np.vstack(y_train).astype(np.float)
X_test = np.vstack(X_test)
y_test = np.vstack(y_test).astype(np.float)

y_train = y_train.reshape(-1, 100)

n_pixels = y_train.shape[1]
n_features = X_train.shape[1]

### Very simple encoding using ridge regression

from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

print "Do ridge regression"
estimator = Ridge(alpha=100.)
cv = KFold(len(y_train), 10)
predictions = [
        Ridge(alpha=100.).fit(y_train.reshape(-1, 100)[train], X_train[train]
            ).predict(y_train.reshape(-1, 100)[test]) for train, test in cv]

print "Scoring"
scores = [1. - (((X_train[test] - pred) ** 2).sum(axis=0) /
           ((X_train[test] - X_train[test].mean(axis=0)) ** 2).sum(axis=0))
for pred, (train, test) in zip(predictions, cv)]

### Show scores

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


sbrain = masker.inverse_transform(np.array(scores).mean(0))

bg = nibabel.load('bg_.nii.gz')

pl.figure(figsize=(8, 8))
pl.imshow(bg.get_data()[:, :, 10], interpolation="nearest", cmap='gray')
pl.imshow(np.ma.masked_less(sbrain.get_data()[:, :, 10], 1e-6),
        interpolation="nearest", cmap='hot')
plot_lines(contour[:, :, 10])
pl.axis('off')
pl.subplots_adjust(left=0., right=1., bottom=0., top=1.)
pl.savefig('encoding_scores.pdf')
pl.savefig('encoding_scores.eps')
pl.clf()

### Compute receptive fields

from sklearn.linear_model import LassoLarsCV

lasso = LassoLarsCV(max_iter=10,)

p = (4, 2)
# Mask for chosen pixel
pixmask = np.zeros((10, 10), dtype=bool)
pixmask[p] = 1

for index in [1780, 1951, 2131, 1935]:
    rf = lasso.fit(y_train, X_train[:, index]).coef_.reshape(10, 10)
    pl.figure(figsize=(8, 8))
    pl.imshow(rf, vmin=0, interpolation="nearest", cmap='hot')
    plot_lines(pixmask, linewidth=6)
    pl.axis('off')
    pl.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    pl.savefig('encoding_%d.pdf' % index)
    pl.savefig('encoding_%d.eps' % index)
    pl.clf()
