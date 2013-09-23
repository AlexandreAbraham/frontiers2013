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
from nilearn import datasets, input_data
dataset = datasets.fetch_adhd(n_subjects=1)
nifti_masker = input_data.NiftiMasker(memory='nilearn_cache',
                        memory_level=1, standardize=True)
fmri_masked = nifti_masker.fit_transform(dataset.func[0])
mask = nifti_masker.mask_img_.get_data().astype(np.bool)

z = 42


def plot_labels(labels, seed):
    labels = labels.astype(int)
    n_colors = np.max(labels)
    cut = labels[:, :, z]
    np.random.seed(seed)
    colors = np.random.random(size=(n_colors + 1, 3))

    # Cluster '-1' should be white (it's outside the brain)
    colors[-1] = 1
    pl.figure(figsize=(3.8, 4.5))
    pl.axes([0, 0, 1, 1])
    pl.imshow(colors[np.rot90(cut)], interpolation='nearest')
    pl.axis('off')


# Compute connectivity matrix: which voxel is connected to which
from sklearn.feature_extraction import image
shape = mask.shape
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)

for n_clusters in 100, 1000:
    # Compute Ward clustering
    from sklearn.cluster import WardAgglomeration
    ward = WardAgglomeration(n_clusters=n_clusters, connectivity=connectivity,
                            memory='nilearn_cache', compute_full_tree=True)
    ward.fit(fmri_masked)

    labels = ward.labels_ + 1
    labels = nifti_masker.inverse_transform(labels).get_data()
    # 0 is the background, putting it to -1
    labels = labels - 1

    # Display the labels
    plot_labels(labels, 8)
    pl.savefig('ward_%i.eps' % n_clusters)
    pl.savefig('ward_%i.pdf' % n_clusters)

    # Compute Kmeans clustering
    from sklearn.cluster import MiniBatchKMeans

    nifti_masker = input_data.NiftiMasker(memory='nilearn_cache',
                    memory_level=1, smoothing_fwhm=6., standardize=True)
    fmri_masked = nifti_masker.fit_transform(dataset.func[0])
    mask = nifti_masker.mask_img_.get_data().astype(np.bool)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=1)
    kmeans.fit(fmri_masked.T)

    labels = kmeans.labels_ + 1
    labels = nifti_masker.inverse_transform(labels).get_data()
    # 0 is the background, putting it to -1
    labels = labels - 1

    plot_labels(labels, 2)
    pl.savefig('kmeans_%i.eps' % n_clusters)
    pl.savefig('kmeans_%i.pdf' % n_clusters)

