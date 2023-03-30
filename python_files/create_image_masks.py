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

from alpineer.io_utils import list_folders
from python_files import utils

# real paths
channel_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/'
seg_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output'
mask_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/mask_dir/'
post_processing_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/post_processing/'
#cell_table_clusters = pd.read_csv(os.path.join())

folders = list_folders(channel_dir)
folders = missing_fovs

# create directories to hold masks
intermediate_dir = os.path.join(mask_dir, 'intermediate_masks')
if not os.path.exists(intermediate_dir):
    os.mkdir(intermediate_dir)

individual_dir = os.path.join(mask_dir, 'individual_masks')
if not os.path.exists(individual_dir):
    os.mkdir(individual_dir)

# loop over each FOV and generate the appropriate masks
for folder in folders:
    try:
        ecad = io.imread(os.path.join(channel_dir, folder, 'ECAD.tiff'))
    except:
        print('No ECAD channel for ' + folder)
        continue

    intermediate_folder = os.path.join(intermediate_dir, folder)
    if not os.path.exists(intermediate_folder):
        os.mkdir(intermediate_folder)

    # generate cancer/stroma mask by combining segmentation mask with ECAD channel
    seg_label = io.imread(os.path.join(seg_dir, folder + '_whole_cell.tiff'))[0]
    seg_mask = utils.create_cell_mask(seg_label, cell_table_clusters, folder, ['Cancer'])
    cancer_mask = utils.create_cancer_boundary(ecad, seg_mask, min_mask_size=7000)
    cancer_mask = cancer_mask.astype(np.uint8)
    io.imsave(os.path.join(intermediate_folder, 'cancer_mask.png'), cancer_mask,
              check_contrast=False)

    # create mask for TLS
    tls_mask = utils.create_cell_mask(seg_label, cell_table_clusters, folder, ['B', 'T'], sigma=4)
    tls_label_mask = skimage.measure.label(tls_mask)
    tls_label_mask = morphology.remove_small_objects(tls_label_mask, min_size=25000)
    tls_label_mask = morphology.remove_small_holes(tls_label_mask, area_threshold=7000)
    io.imsave(os.path.join(intermediate_folder, 'tls_mask.png'), tls_label_mask.astype(np.uint8),
              check_contrast=False)

    # create mask for slide background
    gold = io.imread(os.path.join(channel_dir, folder, 'Au.tiff'))

    gold_mask = utils.create_channel_mask(img=gold, sigma=2, intensity_thresh=350,
                                          min_mask_size=5000, max_hole_size=1000)

    for _ in range(5):
        gold_mask = morphology.binary_erosion(gold_mask)

    io.imsave(os.path.join(intermediate_folder, 'gold_mask.png'), gold_mask.astype(np.uint8),
                check_contrast=False)

folders = list_folders(intermediate_dir)

# remove any overlapping pixels from different masks, then save individually
for folder in folders:
    # read in generated masks
    intermediate_folder = os.path.join(intermediate_dir, folder)
    cancer_mask = io.imread(os.path.join(intermediate_folder, 'cancer_mask.png'))
    tls_mask = io.imread(os.path.join(intermediate_folder, 'tls_mask.png'))
    gold_mask = io.imread(os.path.join(intermediate_folder, 'gold_mask.png'))

    # create a single unified mask; TLS and background override tumor compartments
    cancer_mask[tls_mask == 1] = 5
    cancer_mask[gold_mask == 1] = 0

    # save individual masks
    processed_folder = os.path.join(individual_dir, folder)
    if not os.path.exists(processed_folder):
        os.mkdir(processed_folder)

    for idx, name in zip(range(0, 6), ['empty_slide', 'stroma_core', 'stroma_border',
                                       'cancer_border', 'cancer_core', 'tls']):
        channel_img = cancer_mask == idx
        io.imsave(os.path.join(processed_folder, name + '.tiff'), channel_img.astype(np.uint8),
                  check_contrast=False)

# compute the area of each mask
area_df = utils.calculate_mask_areas(mask_dir=individual_dir, fovs=folders)
area_df.to_csv(os.path.join(post_processing_dir, 'fov_annotation_mask_area.csv'), index=False)

# create combined images for visualization
for folder in folders[:50]:
    cluster_overlay = io.imread(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay', folder + '.png'))
    compartment_overlay = io.imread(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/compartment_overlay', folder + '.png'))
    gold_chan = io.imread(os.path.join(channel_dir, folder, 'Au.tiff'))
    border_mask = io.imread(os.path.join(intermediate_dir, folder, 'cancer_mask.png'))
    tls_mask = io.imread(os.path.join(intermediate_dir, folder, 'tls_mask.png'))
    gold_mask = io.imread(os.path.join(intermediate_dir, folder, 'gold_mask.png'))

    # create a single unified mask; TLS and background override tumor compartments
    border_mask[tls_mask == 1] = 5
    border_mask[gold_mask == 1] = 0

    # make top row shorter than bottom row
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 2]})
    ax[1, 0].imshow(cluster_overlay)
    ax[1, 0].axis('off')
    ax[0, 0].imshow(gold_chan)
    ax[0, 1].imshow(border_mask)
    ax[1, 1].imshow(compartment_overlay)
    ax[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/combined_mask_overlay', folder + '.png'))
    plt.close()


# assign cells to the correct compartment
assignment_table = utils.assign_cells_to_mask(seg_dir=seg_dir, mask_dir=individual_dir, fovs=folders)
assignment_table.to_csv(os.path.join(post_processing_dir, 'cell_annotation_mask.csv'), index=False)

#assignment_table = pd.read_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/assignment_table.csv'))

#
# cell_table_short = cell_table_short.loc[cell_table_short['fov'].isin(assignment_table.fov.unique()), :]
# cell_table_short = cell_table_short.merge(assignment_table, on=['fov', 'label'], how='left')
# cell_table_short = cell_table_short.rename(columns={'mask_name': 'tumor_region'})
# cell_table_short.to_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data', 'combined_cell_table_normalized_cell_labels_updated_clusters_only_kmeans_nh_mask.csv'), index=False)
#
# cell_table_func = pd.read_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/', 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'))
# cell_table_func = cell_table_func.loc[cell_table_func['fov'].isin(assignment_table.fov.unique()), :]
# cell_table_func = cell_table_func.merge(assignment_table, on=['fov', 'label'], how='left')
# cell_table_func = cell_table_func.rename(columns={'mask_name': 'tumor_region'})
# cell_table_func.to_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data', 'combined_cell_table_normalized_cell_labels_updated_functional_only_mask.csv'), index=False)
#


