# ECM analysis
import skimage.io as io
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from ark.utils import load_utils, io_utils, data_utils

import itertools

from skimage.segmentation import find_boundaries
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from skimage import morphology

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import pickle

from python_files import utils


#
# This script is for generating the ECM assignments for image crops
#

out_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/ecm_masks'
channel_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples'
mask_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/mask_dir/individual_masks/'
fovs = io_utils.list_folders(channel_dir)
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots'

#
# Visualization to assess spatial patterns in signal
#


fov_subset = ['TONIC_TMA3_R2C5', 'TONIC_TMA20_R5C3', 'TONIC_TMA9_R7C6', 'TONIC_TMA9_R3C1',
              'TONIC_TMA6_R11C6', 'TONIC_TMA20_R5C4', 'TONIC_TMA13_R1C5', 'TONIC_TMA10_R5C4',
              'TONIC_TMA24_R5C1', 'TONIC_TMA9_R11C1', 'TONIC_TMA23_R12C2', 'TONIC_TMA22_R9C1',
              'TONIC_TMA13_R11C6', 'TONIC_TMA17_R8C5', 'TONIC_TMA12_R4C3', 'TONIC_TMA13_R10C6',
              'TONIC_TMA19_R3C6', 'TONIC_TMA24_R4C3', 'TONIC_TMA21_R9C2', 'TONIC_TMA11_R6C4',
              'TONIC_TMA13_R5C4', 'TONIC_TMA7_R4C5', 'TONIC_TMA21_R1C4', 'TONIC_TMA20_R8C2',
              'TONIC_TMA2_R10C6', 'TONIC_TMA8_R7C6', 'TONIC_TMA20_R10C5', 'TONIC_TMA16_R10C6',
              'TONIC_TMA14_R8C2', 'TONIC_TMA23_R9C4', 'TONIC_TMA12_R10C5', 'TONIC_TMA4_R2C3',
              'TONIC_TMA11_R8C6', 'TONIC_TMA11_R2C1', 'TONIC_TMA15_R1C5', 'TONIC_TMA9_R9C6',
              'TONIC_TMA15_R2C5', 'TONIC_TMA14_R4C1', 'TONIC_TMA7_R8C5', 'TONIC_TMA9_R6C3',
              'TONIC_TMA14_R8C1', 'TONIC_TMA2_R12C4']

# calculate image percentiles
percentiles = {}
for chan in image_data.channels.values:
    current_data = image_data.loc[:, :, :, chan]
    chan_percentiles = []
    for i in range(current_data.shape[0]):
        current_img = current_data.values[i, :, :]
        chan_percentiles.append(np.percentile(current_img[current_img > 0], 99.9))
    percentiles[chan] = (np.mean(chan_percentiles))

# save the percentiles
pd.DataFrame(percentiles, index=['percentile']).to_csv(os.path.join(out_dir, 'percentiles.csv'))

# load the percentiles
percentiles = pd.read_csv(os.path.join(out_dir, 'percentiles.csv'), index_col=0).to_dict(orient='records')[0]

# stitch images together to enable comparison
image_data = load_utils.load_imgs_from_tree(channel_dir,
                                            fovs=fov_subset,
                                            img_sub_folder='',
                                            max_image_size=2048,
                                            channels=['Collagen1', 'Fibronectin', 'FAP', 'SMA', 'Vim'])

stitched = data_utils.stitch_images(image_data, 6)

stitch_dir = os.path.join(out_dir, 'stitched_images_single_channel')
if not os.path.exists(stitch_dir):
    os.makedirs(stitch_dir)

for chan in stitched.channels.values:
        current_img = stitched.loc['stitched_image', :, :, chan].values
        io.imsave(os.path.join(stitch_dir, chan + '_16.tiff'), current_img.astype('float16'),
                  check_contrast=False)


# create mask for total foreground of all ECM channels


def create_combined_channel_mask(chans, channel_dir, percentiles, threshold, smooth_val,
                                 erode_val):
    """
    Creates a mask for the total foreground of all channels in chans
    """

    normalized_chans = []
    for chan in chans:
        current_img = io.imread(os.path.join(channel_dir, chan + '.tiff'))
        current_img /= percentiles[chan]
        current_img[current_img > 1] = 1
        normalized_chans.append(current_img)

    normalized_chans = np.stack(normalized_chans, axis=0)
    normalized_chans = np.sum(normalized_chans, axis=0)

    smoothed = gaussian_filter(normalized_chans, sigma=smooth_val)
    mask = smoothed > threshold
    for _ in range(erode_val):
        mask = morphology.binary_erosion(mask)

    return mask


# create ecm mask for each FOV
for fov in fov_subset:
    mask = create_combined_channel_mask(chans=['Collagen1', 'Fibronectin', 'FAP', 'SMA', 'Vim'],
                                        channel_dir=os.path.join(channel_dir, fov),
                                        percentiles=percentiles,
                                        threshold=0.1,
                                        smooth_val=5,
                                        erode_val=5)

    # create mask
    io.imsave(os.path.join(mask_dir, fov, 'total_ecm.tiff'), mask.astype('uint8'),
              check_contrast=False)

# generate crop sums
channels = ['Collagen1', 'Fibronectin', 'FAP', 'Vim']
crop_size = 256
tiled_crops = utils.generate_crop_sum_dfs(channel_dir=channel_dir,
                                          mask_dir=mask_dir,
                                          channels=channels,
                                          crop_size=crop_size, fovs=fov_subset, cell_table=None)
# normalize by ecm area
tiled_crops['ecm_fraction'] = (tiled_crops['ecm_mask'] + 1) / (crop_size ** 2)

tiled_crops.loc[:, channels] = tiled_crops.loc[:, channels].div(tiled_crops['ecm_fraction'], axis=0)

# create a pipeline for normalization and clustering the data
kmeans_pipe = make_pipeline(preprocessing.PowerTransformer(method='yeo-johnson', standardize=True),
                            KMeans(n_clusters=4, random_state=0))

# select subset of data to train on
no_ecm_mask = tiled_crops.ecm_fraction < 0.1
train_data = tiled_crops[~no_ecm_mask]
train_data = train_data.loc[:, channels]

# fit the pipeline on the data
kmeans_pipe.fit(train_data.values)

# save the trained pipeline
pickle.dump(kmeans_pipe, open(os.path.join(plot_dir, 'kmeans_pipe_new_4_all_data.pkl'), 'wb'))


# load the model
kmeans_pipe = pickle.load(open(os.path.join(plot_dir, 'kmeans_pipe_new.pkl'), 'rb'))

kmeans_preds = kmeans_pipe.predict(tiled_crops[channels].values)

# get the transformed intermediate data
transformed_data = kmeans_pipe.named_steps['powertransformer'].transform(tiled_crops[channels].values)
transformed_df = pd.DataFrame(transformed_data, columns=channels)
transformed_df['cluster'] = kmeans_preds
tiled_crops['cluster'] = kmeans_preds
tiled_crops.loc[no_ecm_mask, 'cluster'] = -1
transformed_df.loc[no_ecm_mask, 'cluster'] = -1

# generate average image for each cluster
cluster_means = transformed_df[~no_ecm_mask].groupby('cluster').mean()


# plot the average images
cluster_means_clustermap = sns.clustermap(cluster_means, cmap='Reds', figsize=(10, 10))
plt.savefig(os.path.join(plot_dir, 'cluster_means.png'), dpi=300)
plt.close()

# plot distribution of clusters in each fov
cluster_counts = tiled_crops.groupby('fov').value_counts(['cluster'])
cluster_counts = cluster_counts.reset_index()
cluster_counts.columns = ['fov', 'cluster', 'count']
cluster_counts = cluster_counts.pivot(index='fov', columns='cluster', values='count')
cluster_counts = cluster_counts.fillna(0)
cluster_counts = cluster_counts.apply(lambda x: x / x.sum(), axis=1)

# plot the cluster counts
cluster_counts_clustermap = sns.clustermap(cluster_counts, cmap='Reds', figsize=(10, 10))
plt.savefig(os.path.join(plot_dir, 'cluster_fov_counts.png'), dpi=300)
plt.close()

# save dfs
tiled_crops.to_csv(os.path.join(plot_dir, 'tiled_crops.csv'), index=False)


# create a stitched image with example images from each cluster
channels = cluster_means.columns[cluster_means_clustermap.dendrogram_col.reordered_ind]
n_examples = 20
for cluster in tiled_crops.cluster.unique():
    cluster_data = tiled_crops[(~no_ecm_mask) & (tiled_crops.cluster == cluster)]
    cluster_data = cluster_data.sample(n=n_examples, random_state=0)

    stitched_img = np.zeros((crop_size * n_examples, crop_size * (len(channels) + 1)))
    for i in range(n_examples):
        fov_name = cluster_data.iloc[i]['fov']
        row_start = cluster_data.iloc[i]['row_coord']
        col_start = cluster_data.iloc[i]['col_coord']

        for j, chan in enumerate(channels):
            img = io.imread(os.path.join(channel_dir, fov_name, chan + '.tiff'))
            img_subset = img[row_start:row_start + crop_size, col_start:col_start + crop_size]
            img_subset = img_subset / percentiles[chan]
            img_subset[img_subset > 1] = 1

            stitched_img[i * crop_size:(i + 1) * crop_size, j * crop_size:(j + 1) * crop_size] = img_subset

        # do the same thing for the ecm mask
        img = io.imread(os.path.join(mask_dir, fov_name, 'total_ecm.tiff'))
        img_subset = img[row_start:row_start + crop_size, col_start:col_start + crop_size]
        stitched_img[i * crop_size:(i + 1) * crop_size, -crop_size:] = img_subset

    io.imsave(os.path.join(plot_dir, 'cluster_' + str(cluster) + '.tiff'), stitched_img.astype('float32'),
                check_contrast=False)


# generate crops around cells to classify using the trained model
cell_table_clusters = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated.csv'))
cell_table_clusters = cell_table_clusters[cell_table_clusters.fov.isin(fov_subset)]
cell_table_clusters = cell_table_clusters[['fov', 'centroid-0', 'centroid-1', 'label']]

cell_crops = utils.generate_crop_sum_dfs(channel_dir=channel_dir,
                                         mask_dir=mask_dir,
                                         channels=channels,
                                         crop_size=crop_size, fovs=fov_subset,
                                         cell_table=cell_table_clusters)

# normalize based on ecm area
cell_crops['ecm_fraction'] = (cell_crops['ecm_mask'] + 1) / (crop_size ** 2)
cell_crops.loc[:, channels] = cell_crops.loc[:, channels].div(cell_crops['ecm_fraction'], axis=0)

cell_classifications = kmeans_pipe.predict(cell_crops[channels].values.astype('float64'))
cell_table_clusters['ecm_cluster'] = cell_classifications

no_ecm_mask_cell = cell_crops.ecm_fraction < 0.1

cell_table_clusters.loc[no_ecm_mask_cell.values, 'ecm_cluster'] = -1

# replace cluster integers with cluster names
replace_dict = {0: 'Fibronectin', 1: 'Collagen_only', 2: 'VIM', 3: 'Collagen_hot',
                -1: 'no_ecm'}

cell_table_clusters['ecm_cluster'] = cell_table_clusters['ecm_cluster'].replace(replace_dict)

# save the cell table
cell_table_clusters.to_csv(os.path.join(plot_dir, 'combined_cell_table_normalized_cell_labels_updated_ecm_clusters.csv'), index=False)

# group cells in each FOV according to their ECM cluster
cell_table_clusters = cell_table_clusters[cell_table_clusters.fov.isin(fov_subset)]
grouped_ecm_region = cell_table_clusters[['fov', 'ecm_cluster', 'cell_cluster_broad']].groupby(['fov', 'ecm_cluster']).value_counts(normalize=True)
grouped_ecm_region = grouped_ecm_region.unstack(level='cell_cluster_broad', fill_value=0).stack()

grouped_ecm_region = grouped_ecm_region.reset_index()
grouped_ecm_region.columns = ['fov',  'ecm_cluster', 'cell_cluster','count']


# plot the distribution of cell clusters in each ECM cluster
g = sns.FacetGrid(grouped_ecm_region, col='cell_cluster', col_wrap=3, hue='ecm_cluster',
                  palette=['Black'], sharey=False, aspect=2.5)
g.map(sns.violinplot, 'ecm_cluster', 'count',
      order=['Collagen_hot', 'Collagen_medium', 'Collagen_only',
       'Fibronectin_col', 'hot_low_collagen', 'no_ecm'])


# QC clustering results

# generate image with each crop set to the value of the cluster its assigned to
metadata_df = pd.read_csv(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/20230116/ecm_normalized_distribution/metadata_df.csv'))
img = 'TONIC_TMA2_R10C6'
cluster_crop_img = np.zeros((2048, 2048))

metadata_subset = tiled_crops[tiled_crops.fov == img]
for row_crop, col_crop, cluster in zip(metadata_subset.row_coord, metadata_subset.col_coord, metadata_subset.cluster):
    cluster_crop_img[row_crop:row_crop + crop_size, col_crop:col_crop + crop_size] = int(cluster)

io.imshow(cluster_crop_img)
