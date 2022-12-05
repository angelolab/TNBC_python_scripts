# functions for creating image overlays
import os
import pandas as pd
import skimage.io as io
import numpy as np
from skimage.measure import block_reduce

from ark.utils import io_utils, data_utils

# get FOVs of size 2048 x 2048
image_dir = '/Volumes/Big_Boy/TONIC_Cohort/image_data/samples/'
folders = io_utils.list_folders(image_dir)

keep_folders = []

for folder in folders:
    test_img = io.imread(os.path.join(image_dir, folder, 'CD3.tiff'))
    if test_img.shape[0] == 2048:
        keep_folders.append(folder)

output_img = np.zeros((5120, 10240))

# final image size: 20 x 40 images
image_num = 0
channel_name = 'ECAD.tiff'

np.random.shuffle(keep_folders)
for col_num in range(40):
    for row_num in range(20):
        img = io.imread(os.path.join(image_dir, keep_folders[image_num], channel_name))
        img_small = block_reduce(img, block_size=(8, 8), func=np.mean)
        output_img[row_num * 256:(row_num + 1) * 256, col_num * 256:(col_num + 1) * 256] = img_small
        image_num += 1

output_img = output_img / np.max(output_img)
io.imsave('/Users/noahgreenwald/Downloads/ECAD.tiff', output_img)


# create image with segmentation mask colored by cell type
image_names = ['TONIC_TMA10_R11C6', 'TONIC_TMA10_R10C6']

base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'

cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_clusters_only.csv'))
cell_subset = cell_table[cell_table['fov'].isin(image_names)]
cell_subset['unique_ids'] = pd.factorize(cell_subset['cell_cluster'])[0] + 1


# import viridis colormap from mpl
num_categories = np.max(cell_subset.unique_ids)
cm_values = cm.get_cmap('Paired', num_categories - 1)

# get RGB values from cm_values
rgb_values = cm_values(np.arange(num_categories - 1))

# combine with all black for background
rgb_values = np.vstack((np.array([0, 0, 0, 1]), rgb_values))

new_cmap = colors.ListedColormap(rgb_values)

for image in image_names:
    seg_mask = io.imread(os.path.join(base_dir, 'segmentation_data/deepcell_output', image + '_feature_0.tif'))[0]

    # convert string entries in pandas df to unique integers
    cell_subset_plot = cell_subset[cell_subset['fov'] == image]
    labels_dict = dict(zip(cell_subset_plot['label'], cell_subset_plot['unique_ids']))

    # relabel the array
    relabeled_img_array = data_utils.relabel_segmentation(seg_mask, labels_dict)

    output = new_cmap(relabeled_img_array / np.max(relabeled_img_array))
    io.imsave(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/overlays', image + '_seg_overlay.png'), output)