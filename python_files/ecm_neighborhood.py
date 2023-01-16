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

crop_size = 256

img_sums = []
metadata_list = []

# extract crops from each FOV
for fov in fov_subset:
    img_data = load_utils.load_imgs_from_tree(channel_dir,
                                                fovs=[fov],
                                                img_sub_folder='',
                                                channels=['Collagen1', 'Fibronectin',
                                                          'FAP', 'SMA', 'Vim'])
    ecm_mask = io.imread(os.path.join(mask_dir, fov, 'total_ecm.tiff'))
    for row_crop, col_crop in itertools.product(range(0, ecm_mask.shape[0], crop_size),
                                                range(0, ecm_mask.shape[1], crop_size)):

        # calculate percentage of image with ecm
        ecm_pix = np.sum(ecm_mask[row_crop:row_crop + crop_size, col_crop:col_crop + crop_size])
        area_prop = ecm_pix / crop_size**2

        # if less than 10% of the image is ecm, skip
        if area_prop < 0.1:
            continue

        # crop the image data
        crop_sums = img_data.values[0, row_crop:row_crop + crop_size,
                    col_crop:col_crop + crop_size, :].sum(axis=(0, 1))

        # normalize by area
        crop_sums = crop_sums / area_prop

        # append to list
        crop_metadata = [fov, row_crop, col_crop, area_prop]
        img_sums.append(crop_sums)
        metadata_list.append(crop_metadata)


img_df = pd.DataFrame(img_sums, columns=img_data.channels.values)
metadata_df = pd.DataFrame(metadata_list, columns=['fov', 'row', 'col', 'area_prop'])

# save dfs
img_df.to_csv(os.path.join(plot_dir, 'img_df_raw.csv'), index=False)
metadata_df.to_csv(os.path.join(plot_dir, 'metadata_df_raw.csv'), index=False)


# create a pipeline for normalization and clustering the data
kmeans_pipe = make_pipeline(preprocessing.PowerTransformer(method='yeo-johnson', standardize=True),
                            KMeans(n_clusters=5, random_state=0))

# fit the pipeline on the data
kmeans_pipe.fit(img_df.values)

# save the trained pipeline
pickle.dump(kmeans_pipe, open(os.path.join(plot_dir, 'kmeans_pipe.pkl'), 'wb'))


# load the model
kmeans_pipe = pickle.load(open(os.path.join(plot_dir, 'kmeans_pipe.pkl'), 'rb'))

kmeans_preds = kmeans_pipe.predict(img_df.values)

metadata_df['cluster'] = kmeans_preds.astype('str')
img_df['cluster'] = kmeans_preds.astype('str')

# generate average image for each cluster
cluster_means = img_df.groupby('cluster').mean()

# plot the average images
cluster_means_clustermap = sns.clustermap(cluster_means, cmap='Reds', figsize=(10, 10))
plt.savefig(os.path.join(plot_dir, 'cluster_means.png'), dpi=300)
plt.close()

# plot distribution of clusters in each fov
cluster_counts = metadata_df.groupby('fov').value_counts(['cluster'])
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
img_df.to_csv(os.path.join(plot_dir, 'img_df.csv'), index=False)
metadata_df.to_csv(os.path.join(plot_dir, 'metadata_df.csv'), index=False)


# create a stitched image with example images from each cluster
channels = cluster_means.columns[cluster_means_clustermap.dendrogram_col.reordered_ind]
n_examples = 15
for cluster in img_df.cluster.unique():
    cluster_data = metadata_df[metadata_df.cluster == cluster]
    cluster_data = cluster_data.sample(n=n_examples, random_state=0)

    stitched_img = np.zeros((crop_size * n_examples, crop_size * (len(channels) + 1)))
    for i in range(n_examples):
        fov_name = cluster_data.iloc[i]['fov']
        row_start = cluster_data.iloc[i]['row']
        col_start = cluster_data.iloc[i]['col']

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

    io.imsave(os.path.join(plot_dir, 'cluster_' + cluster + '.tiff'), stitched_img.astype('float32'),
                check_contrast=False)
