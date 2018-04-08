import numpy as np
import pylab as pl
import nibabel
import os
import sys
import time


offset = 2


### Load Kamitani dataset #####################################################
from utils import datasets
dataset = datasets.fetch_miyawaki2008()

# Keep only random runs
X_random = dataset.func[12:]
y_random = dataset.label[12:]
y_shape = (10, 10)

### Preprocess data ###########################################################
from utils import masking, signal, cm


sys.stderr.write("Preprocessing data...")
t0 = time.time()

# Load and mask fMRI data
X_train = []
for x_random in X_random:
    # Mask data
    x_img = nibabel.load(x_random)
    x = masking.apply_mask(x_img, dataset.mask)
    x = signal.clean(x, standardize=True, detrend=True)
    X_train.append(x[offset:])

# Load target data and reshape it in 2D
y_train = []
for y in y_random:
    y_train.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
        (-1,) + y_shape, order='F')[:-offset].astype(float))

X_train = np.vstack(X_train)
y_train = np.vstack(y_train)

# Flatten the stimuli
y_train = np.reshape(y_train, (-1, y_shape[0] * y_shape[1]))

sys.stderr.write(" Done (%.2fs)\n" % (time.time() - t0))
n_pixels = y_train.shape[1]
n_features = X_train.shape[1]

### Very simple encoding using ridge regression

from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

print("Do ridge regression")
estimator = Ridge(alpha=100.)
cv = KFold(len(y_train), 10)
predictions = [
        Ridge(alpha=100.).fit(y_train.reshape(-1, 100)[train], X_train[train]
            ).predict(y_train.reshape(-1, 100)[test]) for train, test in cv]

print("Scoring")
scores = [1. - (((X_train[test] - pred) ** 2).sum(axis=0) /
           ((X_train[test] - X_train[test].mean(axis=0)) ** 2).sum(axis=0))
for pred, (train, test) in zip(predictions, cv)]

### Show scores

# Create a mask with chosen voxels to contour them
contour = np.zeros(nibabel.load(dataset.mask).shape, dtype=bool)
for x, y in [(31, 9), (31, 10), (30, 10), (32, 10)]:
    contour[x, y, 10] = 1


from matplotlib.lines import Line2D


def plot_lines(mask, linewidth=3, color='b'):
    for i, j in np.ndindex(mask.shape):
        if i + 1 < mask.shape[0] and mask[i, j] != mask[i + 1, j]:
            pl.gca().add_line(Line2D([j - .5, j + .5], [i + .5, i + .5],
                color=color, linewidth=linewidth))
        if j + 1 < mask.shape[1] and mask[i, j] != mask[i, j + 1]:
            pl.gca().add_line(Line2D([j + .5, j + .5], [i - .5, i + .5],
                color=color, linewidth=linewidth))


sbrain = masking.unmask(np.array(scores).mean(0), dataset.mask)

bg = nibabel.load(os.path.join('utils', 'bg.nii.gz'))

pl.figure(figsize=(8, 8))
ax1 = pl.axes([0., 0., 1., 1.])
pl.imshow(bg.get_data()[:, :, 10].T, interpolation="nearest", cmap='gray',
          origin='lower')
pl.imshow(np.ma.masked_less(sbrain[:, :, 10].T, 1e-6),
          interpolation="nearest", cmap='hot', origin="lower")
plot_lines(contour[:, :, 10].T)
pl.axis('off')
ax2 = pl.axes([.08, .5, .05, .47])
cb = pl.colorbar(cax=ax2, ax=ax1)
cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_tick_params(labelcolor='white')
cb.ax.yaxis.set_tick_params(labelsize=20)
cb.set_ticks(np.arange(0., .8, .2))
pl.savefig(os.path.join('miyawaki', 'encoding_scores.pdf'))
pl.savefig(os.path.join('miyawaki', 'encoding_scores.png'))
pl.savefig(os.path.join('miyawaki', 'encoding_scores.eps'))
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
    # Black background
    pl.imshow(np.zeros_like(rf), vmin=0., vmax=1., cmap='gray')
    pl.imshow(np.ma.masked_equal(rf, 0.), vmin=0., vmax=0.75,
            interpolation="nearest", cmap=cm.bluegreen)
    plot_lines(pixmask, linewidth=6, color='r')
    pl.axis('off')
    pl.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    pl.savefig(os.path.join('miyawaki', 'encoding_%d.pdf' % index))
    pl.savefig(os.path.join('miyawaki', 'encoding_%d.eps' % index))
    pl.clf()


### Plot the colorbar #########################################################
import matplotlib as mpl


fig = pl.figure(figsize=(2.4, .4))
norm = mpl.colors.Normalize(vmin=0., vmax=.75)
cb = mpl.colorbar.ColorbarBase(pl.gca(), cmap=cm.bluegreen, norm=norm,
                               orientation='horizontal')
#cb.ax.yaxis.set_ticks_position('left')
cb.set_ticks([0., 0.38, 0.75])
fig.subplots_adjust(bottom=0.5, top=1., left=0.08, right=.92)
pl.savefig(os.path.join('miyawaki', 'rf_colorbar.pdf'))
pl.savefig(os.path.join('miyawaki', 'rf_colorbar.png'))
pl.savefig(os.path.join('miyawaki', 'rf_colorbar.eps'))
