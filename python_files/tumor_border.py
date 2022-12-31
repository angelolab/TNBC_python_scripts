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


def smooth_seg_mask(seg_mask, cell_table, fov_name, cell_types, sigma=10, smooth_thresh=0.3):
    # get cell labels for fov and cell type
    cell_subset = cell_table[cell_table['fov'] == fov_name]
    cell_subset = cell_subset[cell_subset['cell_cluster_broad'].isin(cell_types)]
    cell_labels = cell_subset['label'].values

    # create mask for cell type
    cell_mask = np.isin(seg_mask, cell_labels)

    # smooth mask
    cell_mask_smoothed = gaussian_filter(cell_mask.astype(float), sigma=sigma)

    cell_mask = cell_mask_smoothed > smooth_thresh

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

    combined_mask = np.ones_like(img)
    combined_mask[label_mask] = 4
    combined_mask[external_boundary > 0] = 2
    combined_mask[interior_boundary > 0] = 3

    return combined_mask


# for testing
channel_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/channel_data/'
seg_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/segmentation_masks'
out_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/mask_dir'
overlay_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/overlays'
cell_table_short = pd.read_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data', 'combined_cell_table_normalized_cell_labels_updated_clusters_only_kmeans_nh.csv'))

# real paths
channel_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/'
seg_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output'
out_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/mask_dir/'
cell_table_short = pd.read_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data', 'combined_cell_table_normalized_cell_labels_updated_clusters_only_kmeans_nh.csv'))

folders = list_folders(channel_dir)

intermediate_dir = os.path.join(out_dir, 'intermediate_masks')
if not os.path.exists(intermediate_dir):
    os.mkdir(intermediate_dir)


individual_dir = os.path.join(out_dir, 'individual_masks')
if not os.path.exists(individual_dir):
    os.mkdir(individual_dir)

# create masks for each FOV
for folder in folders:
    try:
        ecad = io.imread(os.path.join(channel_dir, folder, 'ECAD.tiff'))
    except:
        print('No ECAD channel for ' + folder)
        continue

    # generate mask by combining segmentation mask and channel mask
    seg_label = io.imread(os.path.join(seg_dir, folder + '_whole_cell.tiff'))[0]
    seg_mask = smooth_seg_mask(seg_label, cell_table_short, folder, ['Cancer'])
    cancer_mask = create_cancer_boundary(ecad, seg_mask, min_size=7000)
    cancer_mask = cancer_mask.astype(np.uint8)
    intermediate_folder = os.path.join(intermediate_dir, folder)
    if not os.path.exists(intermediate_folder):
        os.mkdir(intermediate_folder)

    io.imsave(os.path.join(intermediate_folder, 'cancer_mask.png'), cancer_mask,
              check_contrast=False)

    # create mask for TLS
    tls_mask = smooth_seg_mask(seg_label, cell_table_short, folder, ['B', 'T'], sigma=4)
    tls_label_mask = skimage.measure.label(tls_mask)
    tls_label_mask = morphology.remove_small_objects(tls_label_mask, min_size=25000)
    tls_label_mask = morphology.remove_small_holes(tls_label_mask, area_threshold=7000)

    io.imsave(os.path.join(intermediate_folder, 'tls_mask.png'), tls_label_mask.astype(np.uint8),
              check_contrast=False)

    # create mask for slide background
    gold = io.imread(os.path.join(channel_dir, folder, 'Au.tiff'))
    gold_smoothed = gaussian_filter(gold.astype(float), sigma=2)
    gold_mask = gold_smoothed > 350

    # clean up mask prior to analysis
    gold_label_mask = skimage.measure.label(gold_mask)
    gold_label_mask = morphology.remove_small_objects(gold_label_mask, min_size=5000)
    gold_label_mask = morphology.remove_small_holes(gold_label_mask, area_threshold=1000)

    for _ in range(5):
        gold_label_mask = morphology.binary_erosion(gold_label_mask)

    io.imsave(os.path.join(intermediate_folder, 'gold_mask.png'), gold_label_mask.astype(np.uint8),
                check_contrast=False)

folders = list_folders(intermediate_dir)

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
    individual_folder = os.path.join(individual_dir, folder)
    if not os.path.exists(individual_folder):
        os.mkdir(individual_folder)

    for idx, name in zip(range(0, 6), ['empty_slide', 'stroma_core', 'stroma_border',
                                       'cancer_border', 'cancer_core', 'tls']):
        channel_img = cancer_mask == idx
        io.imsave(os.path.join(individual_folder, name + '.tiff'), channel_img.astype(np.uint8),
                  check_contrast=False)


# create combined images for visualization
for folder in folders:
    overlay_img = io.imread(os.path.join(overlay_dir, 'overlay_' + folder + '_test.png'))
    gold_mask = io.imread(os.path.join(intermediate_dir, folder, 'gold_mask.png'))
    gold_chan = io.imread(os.path.join(channel_dir, folder, 'Au.tiff'))
    border_mask = io.imread(os.path.join(intermediate_dir, folder, 'cancer_mask.png'))
    tils_mask = io.imread(os.path.join(intermediate_dir, folder, 'tls_mask.png'))

    border_mask[gold_mask == 1] = 0
    border_mask[tils_mask == 1] = 5

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    ax[0, 0].imshow(overlay_img)
    ax[0, 1].imshow(gold_chan)
    ax[1, 0].imshow(border_mask)

    plt.tight_layout()
    plt.savefig(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/combined_overlays', folder + '_border_visualization.png'))
    plt.close()


# compute the area of each compartment per fov
def calculate_mask_areas(mask_dir):
    fovs = io_utils.list_folders(mask_dir)

    mask_files = io_utils.list_files(os.path.join(mask_dir, fovs[0]))
    mask_names = [os.path.splitext(os.path.basename(x))[0] for x in mask_files]

    area_dfs = []
    for fov in fovs:
        mask_areas = []
        for mask_file in mask_files:
            mask = io.imread(os.path.join(mask_dir, fov, mask_file))
            mask_areas.append(np.sum(mask))

        area_df = pd.DataFrame({'compartment': mask_names, 'area': mask_areas,
                                'fov': fov})

        # separately calculate size for non-background compartment
        bg_area = area_df[area_df['compartment'] == 'empty_slide']['area'].values[0]
        foreground_area = mask.shape[0] ** 2 - bg_area

        area_df = area_df.append({'compartment': 'all', 'area': foreground_area, 'fov': fov},
                       ignore_index=True)

        area_dfs.append(area_df)

    return pd.concat(area_dfs, axis=0)


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


assignment_table = assign_cells_to_mask(seg_dir=seg_dir,
                                        mask_dir=individual_dir,
                                        fovs=folders)
assignment_table.to_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/assignment_table.csv'), index=False)

assignment_table = pd.read_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/assignment_table.csv'))

cell_table_short = cell_table_short.loc[cell_table_short['fov'].isin(assignment_table.fov.unique()), :]
cell_table_short = cell_table_short.merge(assignment_table, on=['fov', 'label'], how='left')
cell_table_short = cell_table_short.rename(columns={'mask_name': 'tumor_region'})
cell_table_short.loc[cell_table_short['tumor_region'] == 'other', 'tumor_region'] = 'stroma_core'
cell_table_short.to_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data', 'combined_cell_table_normalized_cell_labels_updated_clusters_only_kmeans_nh_mask.csv'), index=False)

cell_table_func = pd.read_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/', 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'))
cell_table_func = cell_table_func.loc[cell_table_func['fov'].isin(assignment_table.fov.unique()), :]
cell_table_func = cell_table_func.merge(assignment_table, on=['fov', 'label'], how='left')
cell_table_func = cell_table_func.rename(columns={'mask_name': 'tumor_region'})
cell_table_func.loc[cell_table_func['tumor_region'] == 'other', 'tumor_region'] = 'stroma_core'
cell_table_func.to_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data', 'combined_cell_table_normalized_cell_labels_updated_functional_only_mask.csv'), index=False)


area_df = calculate_mask_areas(mask_dir=individual_dir)

test_df = core_df_cluster.loc[core_df_cluster.fov.isin(folders)]

area_df = area_df.rename(columns={'compartment': 'subset'})
test_df = test_df.merge(area_df, on=['fov', 'subset'], how='left')