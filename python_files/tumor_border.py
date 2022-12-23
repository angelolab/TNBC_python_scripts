import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
import skimage

import skimage.measure
from skimage import morphology
from scipy.ndimage import gaussian_filter

from ark.utils.io_utils import list_folders

channel_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/channel_data'

folders = list_folders(channel_dir)
test_img = 'TONIC_TMA1_R8C1'
test_img = folders[3]

min_size = 2000
hole_size = 1000
ecad = io.imread(os.path.join(channel_dir, test_img, 'ECAD.tiff'))


def create_cancer_boundary(img, min_size=2000, hole_size=1000, border_size=50):
    img_smoothed = gaussian_filter(img, sigma=10)
    img_mask = img_smoothed > 0.0015

    # clean up mask prior to analysis
    label_mask = skimage.measure.label(img_mask)
    label_mask = morphology.remove_small_objects(label_mask, min_size=min_size)
    label_mask = morphology.remove_small_holes(label_mask, area_threshold=hole_size)

    # define external borders
    external_boundary = morphology.binary_dilation(label_mask)

    for _ in range(border_size):
        external_boundary = morphology.binary_dilation(external_boundary)

    external_boundary = external_boundary.astype(int) - label_mask.astype(int)
    #plt.imshow(external_boundary)
    # create interior borders
    interior_boundary = morphology.binary_erosion(label_mask)

    for _ in range(border_size):
        interior_boundary = morphology.binary_erosion(interior_boundary)

    interior_boundary = interior_boundary.astype(int) - label_mask.astype(int)
    plt.imshow(interior_boundary)

    combined_mask = np.zeros_like(img_mask)
    combined_mask[label_mask] = 1
    combined_mask[external_boundary > 0] = 2
    combined_mask[interior_boundary > 0] = 3

    return combined_mask


combined_mask = create_cancer_boundary(ecad, min_size=min_size, hole_size=hole_size)

fig, ax = plt.subplots(1, 1)
ax.imshow(combined_mask)





