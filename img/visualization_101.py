import nibabel
import numpy as np
from matplotlib import pyplot as plt
from os.path import join


# Load image
bg_img = nibabel.load(join('utils', 'bg.nii.gz'))
bg = bg_img.get_data()
# Keep values over 4000 as activation map
act = bg.copy()
act[act < 6000] = 0.

# Display the background
plt.imshow(bg[..., 10].T, origin='lower', interpolation='nearest', cmap='gray')
# Mask background values of activation map
masked_act = np.ma.masked_equal(act, 0.)
plt.imshow(masked_act[..., 10].T, origin='lower', interpolation='nearest', cmap='hot')
# Cosmetics: disable axis
plt.axis('off')
plt.show()
# Save the activation map
nibabel.save(nibabel.Nifti1Image(act, bg_img.get_affine()), 'activation.nii.gz')
