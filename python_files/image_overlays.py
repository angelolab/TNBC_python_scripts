# functions for creating image overlays
import os

import matplotlib.pyplot as plt
import pandas as pd
import skimage.io as io
import numpy as np
from skimage.measure import block_reduce

from matplotlib import cm
from matplotlib import colors

from ark.utils import io_utils, data_utils
from skimage.segmentation import find_boundaries


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



def create_cell_overlay(cell_table, seg_folder, fovs, cluster_col, plot_dir, save_names):
    cell_subset = cell_table.copy()
    cell_subset['unique_ids'] = pd.factorize(cell_subset[cluster_col])[0] + 1

    categories = cell_subset[[cluster_col, 'unique_ids']].drop_duplicates()[cluster_col].values

    # import viridis colormap from mpl
    num_categories = np.max(cell_subset.unique_ids)
    cm_values = cm.get_cmap('Paired', num_categories)

    # get RGB values from cm_values
    rgb_values = cm_values(np.arange(num_categories))

    # combine with all black for background
    rgb_values = np.vstack((np.array([0, 0, 0, 1]), rgb_values))

    new_cmap = colors.ListedColormap(rgb_values)

    bounds = [i-0.5 for i in np.linspace(0, num_categories+1, num_categories+2)]
    norm = colors.BoundaryNorm(bounds, new_cmap.N + 1)

    for idx, image in enumerate(fovs):
        seg_mask = io.imread(os.path.join(seg_folder, image + '_whole_cell.tiff'))[0]

        edges = find_boundaries(seg_mask, mode='inner')
        seg_mask = np.where(edges == 0, seg_mask, 0)

        # convert string entries in pandas df to unique integers
        cell_subset_plot = cell_subset[cell_subset['fov'] == image]
        labels_dict = dict(zip(cell_subset_plot['label'], cell_subset_plot['unique_ids']))

        # relabel the array
        relabeled_img_array = data_utils.relabel_segmentation(seg_mask, labels_dict)

        #output = new_cmap(relabeled_img_array / np.max(relabeled_img_array))

        im = plt.imshow(relabeled_img_array, cmap=new_cmap, norm=norm)
        tick_names = ['Empty'] + categories.tolist()
        cbar = plt.colorbar(im, ticks=np.arange(len(tick_names)))
        cbar.set_ticks(cbar.ax.get_yticks())
        cbar.ax.set_yticklabels(tick_names)
        plt.savefig(os.path.join(plot_dir, save_names[idx]), dpi=300)
        plt.close()

        #io.imsave(os.path.join(plot_dir, save_names[idx]), output)


#folders = cell_table_clusters.fov.unique()
#np.random.shuffle(folders)

fovs = harmonized_metadata.loc[harmonized_metadata['primary_baseline'] == True, 'fov'].values
fovs = cell_table_clusters.fov.unique()
create_cell_overlay(cell_table=cell_table_clusters, seg_folder='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output',
                    fovs=fovs, cluster_col='cell_cluster_broad', plot_dir='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay',
                    save_names=['{}.png'.format(x) for x in fovs])


# create overlays based on microenvironment
fovs = cell_table_clusters.fov.unique()
create_cell_overlay(cell_table=cell_table_clusters, seg_folder='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output',
                    fovs=fov_subset, cluster_col='tumor_region', plot_dir='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/compartment_overlay',
                    save_names=['{}.png'.format(x) for x in fov_subset])



# create combined images for visualization
for fov in fov_subset:
    cluster_overlay = io.imread(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay', fov + '.png'))
    compartment_overlay = io.imread(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/compartment_overlay', fov + '.png'))
    ecm_overlay = io.imread(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/ecm_overlay', fov + '.png'))

    # plot a combined image with all three overlays in a row
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cluster_overlay)
    ax[0].set_title('Cell Type')
    ax[0].axis('off')
    ax[1].imshow(compartment_overlay)
    ax[1].set_title('Compartment')
    ax[1].axis('off')
    ax[2].imshow(ecm_overlay)
    ax[2].set_title('ECM')
    ax[2].axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/combined_ecm_overlay', fov + '.png'), dpi=300)
    plt.close()


# create combined images for visualization
for fov in fov_subset:
    ecm_overlay = io.imread(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/ecm_overlay', fov + '.png'))
    cluster_overlay = io.imread(
        os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay',
                     fov + '.png'))
    compartment_overlay = io.imread(
        os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/compartment_overlay',
                     fov + '.png'))
    collagen_img = io.imread(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples', fov,  'Collagen1.tiff'))
    collagen_img /= percentiles['Collagen1']
    collagen_img = np.where(collagen_img > 1, 1, collagen_img)

    FAP_img = io.imread(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples', fov,  'FAP.tiff'))
    FAP_img /= percentiles['FAP']
    FAP_img = np.where(FAP_img > 1, 1, FAP_img)

    Fibronectin_img = io.imread(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples', fov,  'Fibronectin.tiff'))
    Fibronectin_img /= percentiles['Fibronectin']
    Fibronectin_img = np.where(Fibronectin_img > 1, 1, Fibronectin_img)

    # plot a combined image with all three overlays in a row
    fig, ax = plt.subplots(2, 3, figsize=(12, 5))
    ax[0, 0].imshow(cluster_overlay)
    ax[0, 0].set_title('Cell Type')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(compartment_overlay)
    ax[0, 1].set_title('Compartment')
    ax[0, 1].axis('off')
    ax[0, 2].imshow(ecm_overlay)
    ax[0, 2].set_title('ECM')
    ax[0, 2].axis('off')


    ax[1, 0].imshow(collagen_img)
    ax[1, 0].set_title('Collagen')
    ax[1, 0].axis('off')
    ax[1, 1].imshow(FAP_img)
    ax[1, 1].set_title('FAP')
    ax[1, 1].axis('off')
    ax[1, 2].imshow(Fibronectin_img)
    ax[1, 2].set_title('Fibronectin')
    ax[1, 2].axis('off')
    plt.tight_layout()



    plt.savefig(os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/combined_ecm_overlay', fov + '.png'), dpi=300)
    plt.close()


