from ark.segmentation import marker_quantification
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
import skimage

import numpy as np
import pandas as pd

import skimage.measure
from skimage import morphology
from scipy.ndimage import gaussian_filter

from ark.utils.io_utils import list_folders


def smooth_seg_mask(seg_mask, cell_table, fov_name, cell_type, sigma=10):
    # get cell labels for fov and cell type
    cell_subset = cell_table[cell_table['fov'] == fov_name]
    cell_subset = cell_subset[cell_subset['cell_cluster_broad'] == cell_type]
    cell_labels = cell_subset['label'].values

    # create mask for cell type
    cell_mask = np.isin(seg_mask, cell_labels)

    # smooth mask
    cell_mask_smoothed = gaussian_filter(cell_mask.astype(float), sigma=sigma)

    cell_mask = cell_mask_smoothed > 0.3

    return cell_mask


def create_cancer_boundary(img, seg_mask, min_size=3500, hole_size=1000, border_size=50, channel_thresh=0.0015):
    img_smoothed = gaussian_filter(img, sigma=10)
    img_mask = img_smoothed > channel_thresh

    # clean up mask prior to analysis
    img_mask = np.logical_or(img_mask, seg_mask)
    label_mask = skimage.measure.label(img_mask)
    label_mask = morphology.remove_small_objects(label_mask, min_size=min_size)
    label_mask = morphology.remove_small_holes(label_mask, area_threshold=hole_size)

    # define external borders
    external_boundary = morphology.binary_dilation(label_mask)

    for _ in range(border_size):
        external_boundary = morphology.binary_dilation(external_boundary)

    external_boundary = external_boundary.astype(int) - label_mask.astype(int)
    # create interior borders
    interior_boundary = morphology.binary_erosion(label_mask)

    for _ in range(border_size):
        interior_boundary = morphology.binary_erosion(interior_boundary)

    interior_boundary = label_mask.astype(int) - interior_boundary.astype(int)

    combined_mask = np.zeros_like(img)
    combined_mask[label_mask] = 3
    combined_mask[external_boundary > 0] = 1
    combined_mask[interior_boundary > 0] = 2

    return combined_mask


channel_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/channel_data'
seg_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/segmentation_masks'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots'
data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'

cell_table_short = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_clusters_only.csv'))

folders = list_folders(channel_dir)
test_img = 'TONIC_TMA1_R8C1'
test_img = folders[3]

for folder in folders:
    ecad = io.imread(os.path.join(channel_dir, folder, 'ECAD.tiff'))
    seg_label = io.imread(os.path.join(seg_dir, folder + '_feature_0.tif'))[0]
    seg_mask = smooth_seg_mask(seg_label, cell_table_short, folder, 'Cancer')
    combined_mask = create_cancer_boundary(ecad, seg_mask, min_size=7000)
    combined_mask = combined_mask.astype(np.uint8)
    io.imsave(os.path.join(plot_dir, folder + '_cancer_boundary_combined.png'), combined_mask)


# create combined images for visualization
for folder in folders:
    boundary_img = io.imread(os.path.join(plot_dir, 'cancer_stroma_boundary', folder + '_cancer_boundary_combined.png'))
    overlay_img = io.imread(os.path.join(plot_dir, 'overlays', 'overlay_' + folder + '_test.png'))

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(boundary_img)
    ax[1].imshow(overlay_img)
    plt.savefig(os.path.join(plot_dir, folder + '_border_visualization.png'))
    plt.close()


# create folder with individual channels from combined images
output_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/stroma_masks'

for folder in folders:
    combined_img = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/cancer_stroma_boundary/' + folder + '_cancer_boundary_combined.png')
    output_folder = os.path.join(output_dir, folder)
    os.mkdir(output_folder)

    for idx, name in zip(range(1, 4), ['stroma_border', 'cancer_border', 'cancer_core']):
        channel_img = combined_img == idx
        io.imsave(os.path.join(output_folder, name + '.tiff'), channel_img.astype(np.uint8))


# create dataframe with cell assignments to mask
def assign_cells_to_mask(seg_dir, mask_dir, fovs):
    normalized_cell_table, _ = marker_quantification.generate_cell_table(segmentation_dir=seg_dir,
                                                               tiff_dir=mask_dir, fovs=fovs,
                                                               img_sub_folder='')
    # drop cell_size column
    normalized_cell_table = normalized_cell_table.drop(columns=['cell_size'])

    # move fov column to front
    fov_col = normalized_cell_table.pop('fov')
    normalized_cell_table.insert(0, 'fov', fov_col)

    # remove all columns after label
    normalized_cell_table = normalized_cell_table.loc[:, :'label']

    # move label column to front
    label_col = normalized_cell_table.pop('label')
    normalized_cell_table.insert(1, 'label', label_col)

    # get fraction of pixels per cell not assigned to any of the supplied masks
    cell_sum = normalized_cell_table.iloc[:, 2:].sum(axis=1)
    normalized_cell_table.insert(normalized_cell_table.shape[1], 'other', 1 - cell_sum)

    # create new column with name of column max for each row
    normalized_cell_table['mask_name'] = normalized_cell_table.iloc[:, 2:].idxmax(axis=1)

    return normalized_cell_table[['fov', 'label', 'mask_name']]


assignment_table = assign_cells_to_mask(seg_dir='/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/segmentation_masks',
                     mask_dir='/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/stroma_masks',
                     fovs=folders[:2])