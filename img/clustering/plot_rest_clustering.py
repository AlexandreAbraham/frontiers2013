"""
Hierarchical clustering to learn a brain parcellation from rest fMRI
====================================================================

We use spatial-constrained Ward-clustering to create a set of
parcels. These parcels are particularly interesting for creating a
'compressed' representation of the data, replacing the data in the fMRI
images by mean on the parcellation.

This parcellation may be useful in a supervised learning, see for
instance: `A supervised clustering approach for fMRI-based inference of
brain states <http://hal.inria.fr/inria-00589201>`_, Michel et al,
Pattern Recognition 2011.

"""

### Load nyu_rest dataset #####################################################

import numpy as np
import pylab as pl
from nilearn import datasets, io
dataset = datasets.fetch_adhd(n_subjects=1)
nifti_masker = io.NiftiMasker(memory='nilearn_cache', memory_level=1,
                              standardize=True)
fmri_masked = nifti_masker.fit_transform(dataset.func[0])
mask = nifti_masker.mask_img_.get_data().astype(np.bool)
    
n_clusters = 42

def plot_labels(labels, seed):
    cut = labels[:, :, 45].astype(int)
    np.random.seed(seed)
    colors = np.random.random(size=(n_clusters + 1, 3))
    
    # Cluster '-1' should be black (it's outside the brain)
    colors[-1] = 0
    pl.figure(figsize=(4, 4.5))
    pl.axis('off')
    pl.tight_layout()
    pl.imshow(colors[np.rot90(cut)], interpolation='nearest')

### Ward ######################################################################

# Compute connectivity matrix: which voxel is connected to which
from sklearn.feature_extraction import image
shape = mask.shape
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)

# Computing the ward for the first time, this is long...
from sklearn.cluster import WardAgglomeration
import time
start = time.time()
ward = WardAgglomeration(n_clusters=n_clusters, connectivity=connectivity,
                         memory='nilearn_cache', compute_full_tree=True)
ward.fit(fmri_masked)
print "Ward agglomeration %d clusters: %.2fs" % (n_clusters, time.time() - start)

labels = ward.labels_ + 1
labels = nifti_masker.inverse_transform(labels).get_data()
# 0 is the background, putting it to -1
labels = labels - 1

# Display the labels
plot_labels(labels, 8)
pl.savefig('ward.eps')
pl.savefig('ward.pdf')

pl.figure(figsize=(4, 4.5))
first_epi = nifti_masker.inverse_transform(fmri_masked[0]).get_data()
# Outside the mask: a uniform value, smaller than inside the mask
vmax = np.max(np.abs(first_epi[..., 45]))
first_epi[np.logical_not(mask)] = -vmax - 1
pl.imshow(np.rot90(first_epi[..., 45]),
          interpolation='nearest', cmap=pl.cm.spectral, vmin=-vmax, vmax=vmax)
pl.axis('off')
pl.tight_layout()
pl.savefig('original.eps')
pl.savefig('original.pdf')

# dimension
fmri_reduced = ward.transform(fmri_masked)

# Display the corresponding data compressed using the parcellation
fmri_compressed = ward.inverse_transform(fmri_reduced)
compressed = nifti_masker.inverse_transform(
    fmri_compressed[0]).get_data()
compressed[np.logical_not(mask)] = -vmax - 1
pl.figure(figsize=(4, 4.5))
pl.imshow(np.rot90(compressed[:, :, 45]),
          interpolation='nearest', cmap=pl.cm.spectral, vmin=-vmax, vmax=vmax)
pl.axis('off')
pl.tight_layout()
pl.savefig('ward_compressed.eps')
pl.savefig('ward_compressed.pdf')

### Kmeans ####################################################################
from sklearn.cluster import MiniBatchKMeans

nifti_masker = io.NiftiMasker(memory='nilearn_cache', memory_level=1,
        smoothing_fwhm=6., standardize=True)
fmri_masked = nifti_masker.fit_transform(dataset.func[0])
mask = nifti_masker.mask_img_.get_data().astype(np.bool)

kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=1)
kmeans.fit(fmri_masked.T)

labels = kmeans.labels_ + 1
labels = nifti_masker.inverse_transform(labels).get_data()
# 0 is the background, putting it to -1
labels = labels - 1

plot_labels(labels, 5)
#pl.title('K-Means clustering')
#pl.tight_layout()
pl.savefig('kmeans.eps')
pl.savefig('kmeans.pdf')


### Spectral ##################################################################
"""
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction import image

graph = image.img_to_graph(
        nifti_masker.inverse_transform(fmri_masked).get_data().mean(axis=3),
        mask=mask)
graph.data = np.exp(-graph.data / graph.data.std())

spectral = SpectralClustering(n_clusters=n_clusters, random_state=1,
        assign_labels='discretize', affinity='precomputed')
spectral.fit(graph)

labels = spectral.labels_
labels = labels.reshape(X.shape)
pl.imshow(X[:, :, 20], cmap=plt.cm.gray)
for l in range(n_clusters):
    p.contour(labels[:, :, 20] == l, contours=1,
            colors=[pl.cm.spectral(l / float(n_clusters)), ])
pl.xticks(())
pl.yticks(())
pl.show()
"""



### Show result ###############################################################

# Unmask data
# Avoid 0 label
