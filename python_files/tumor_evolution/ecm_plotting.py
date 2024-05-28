import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


ecm_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files/ecm'

tiled_crops = pd.read_csv(os.path.join(ecm_dir, 'tiled_crops.csv'))


# plot distribution of clusters in each fov
cluster_counts = tiled_crops.groupby('fov').value_counts(['tile_cluster'])
cluster_counts = cluster_counts.reset_index()
cluster_counts.columns = ['fov', 'tile_cluster', 'count']
cluster_counts = cluster_counts.pivot(index='fov', columns='tile_cluster', values='count')
cluster_counts = cluster_counts.fillna(0)
cluster_counts = cluster_counts.apply(lambda x: x / x.sum(), axis=1)

# plot the cluster counts
cluster_counts_clustermap = sns.clustermap(cluster_counts, cmap='Reds', figsize=(10, 10))
plt.savefig(os.path.join(ecm_dir, 'fov_cluster_counts.png'), dpi=300)
plt.close()


# use kmeans to cluster fovs
fov_kmeans_pipe = make_pipeline(KMeans(n_clusters=5, random_state=0))
fov_kmeans_pipe.fit(cluster_counts.values)

# save the trained pipeline
pickle.dump(fov_kmeans_pipe, open(os.path.join(ecm_dir, 'fov_classification_kmeans_pipe.pkl'), 'wb'))


# load the model
fov_kmeans_pipe = pickle.load(open(os.path.join(plot_dir, 'fov_classification_kmeans_pipe.pkl'), 'rb'))
kmeans_preds = fov_kmeans_pipe.predict(cluster_counts.values)


# get average cluster count in each kmeans cluster
cluster_counts['fov_cluster'] = kmeans_preds
cluster_means = cluster_counts.groupby('fov_cluster').mean()

cluster_means_clustermap = sns.clustermap(cluster_means, cmap='Reds', figsize=(10, 10))

rename_dict = {0: 'No_ECM', 1: 'Hot_Coll', 2: 'Cold_Coll', 3: 'Hot_Cold_Coll', 4: 'Hot_No_ECM'}
cluster_counts['fov_cluster'] = cluster_counts['fov_cluster'].replace(rename_dict)

# save the cluster counts
cluster_counts.to_csv(os.path.join(ecm_dir, 'fov_cluster_counts.csv'))



# correlate ECM subtypes with patient data using hierarchically clustered heatmap as index
test_fov = 'TONIC_TMA22_R7C1'
test_fov_idx = np.where(cluster_counts.index == test_fov)[0][0]

stop_idx = np.where(cluster_counts_clustermap.dendrogram_row.reordered_ind == test_fov_idx)[0][0]

cluster_counts_subset = cluster_counts.iloc[cluster_counts_clustermap.dendrogram_row.reordered_ind[:stop_idx + 1], :]

test_fov_2 = 'TONIC_TMA21_R9C5'
test_fov_idx_2 = np.where(cluster_counts.index == test_fov_2)[0][0]

stop_idx_2 = np.where(cluster_counts_clustermap.dendrogram_row.reordered_ind == test_fov_idx_2)[0][0]

cluster_counts_subset_2 = cluster_counts.iloc[cluster_counts_clustermap.dendrogram_row.reordered_ind[stop_idx:stop_idx_2 + 1], :]

cluster_counts_subset_3 = cluster_counts.iloc[cluster_counts_clustermap.dendrogram_row.reordered_ind[stop_idx_2:], :]

harmonized_metadata['ecm_cluster'] = 'inflamed'
harmonized_metadata.loc[harmonized_metadata.fov.isin(cluster_counts_subset.index), 'ecm_cluster'] = 'no_ecm'
harmonized_metadata.loc[harmonized_metadata.fov.isin(cluster_counts_subset_2.index), 'ecm_cluster'] = 'cold_collagen'

# plot the distribution of ECM subtypes in each patient by localization
g = sns.catplot(y='Localization', hue='ecm_cluster', data=harmonized_metadata,
                kind='count')
g.savefig(os.path.join(plot_dir, 'ecm_subtype_distribution.png'), dpi=300)
plt.close()

# plot marker expression in each ECM subtype
plot_df = core_df_func.merge(harmonized_metadata[['fov', 'ecm_cluster']], on='fov', how='left')
plot_df = plot_df[plot_df.Timepoint == 'primary_untreated']

# look at all fibroblasts
temp_df = plot_df[plot_df.subset == 'all']
temp_df = temp_df[temp_df.metric == 'cluster_broad_freq']
temp_df = temp_df[temp_df.cell_type == 'Stroma']
temp_df = temp_df[temp_df.functional_marker.isin(['PDL1', 'TIM3', 'IDO', 'HLADR', 'GLUT1'])]

g = sns.catplot(x='ecm_cluster', y='value', data=temp_df,
                kind='strip', col='functional_marker', sharey=False)
g.savefig(os.path.join(plot_dir, 'fibroblast_functional_status_by_ecm.png'), dpi=300)
plt.close()

# look at M2 macrophages
temp_df = plot_df[plot_df.subset == 'all']
temp_df = temp_df[temp_df.metric == 'cluster_freq']
temp_df = temp_df[temp_df.cell_type == 'M1_Mac']
temp_df = temp_df[temp_df.functional_marker.isin(['PDL1', 'TIM3', 'IDO', 'HLADR', 'GLUT1'])]

g = sns.catplot(x='ecm_cluster', y='value', data=temp_df,
                kind='box', col='functional_marker', sharey=False)
g.savefig(os.path.join(plot_dir, 'm1_mac_functional_status_by_ecm.png'), dpi=300)
plt.close()

# look at cell densities
plot_df = core_df_cluster.merge(harmonized_metadata[['fov', 'ecm_cluster']], on='fov', how='left')
plot_df = plot_df[plot_df.Timepoint == 'primary_untreated']

temp_df = plot_df[plot_df.subset == 'all']
temp_df = temp_df[temp_df.metric == 'cluster_broad_density']

g = sns.catplot(x='ecm_cluster', y='value', data=temp_df,
                kind='box', col='cell_type', sharey=False, col_wrap=4)
g.savefig(os.path.join(plot_dir, 'cell_density_primary_broad.png'), dpi=300)
plt.close()


# generate crops around cells to classify using the trained model
cell_table_clusters = pd.read_csv(os.path.join(analysis_dir, 'combined_cell_table_normalized_cell_labels_updated.csv'))
#cell_table_clusters = cell_table_clusters[cell_table_clusters.fov.isin(fov_subset)]
cell_table_clusters = cell_table_clusters[['fov', 'centroid-0', 'centroid-1', 'label']]

cell_crops = utils.generate_crop_sum_dfs(channel_dir=channel_dir,
                                         mask_dir=mask_dir,
                                         channels=channels,
                                         crop_size=crop_size, fovs=fov_subset,
                                         cell_table=cell_table_clusters)

# normalize based on ecm area
cell_crops = utils.normalize_by_ecm_area(crop_sums=cell_crops, crop_size=crop_size,
                                         channels=channels)

cell_classifications = kmeans_pipe.predict(cell_crops[channels].values.astype('float64'))
cell_crops['ecm_cluster'] = cell_classifications

no_ecm_mask_cell = cell_crops.ecm_fraction < 0.1

cell_crops.loc[no_ecm_mask_cell, 'ecm_cluster'] = -1

# replace cluster integers with cluster names
replace_dict = {0: 'Hot_Coll', 1: 'Fibro_Coll', 2: 'VIM_Fibro', 3: 'Cold_Coll',
                -1: 'no_ecm'}

cell_crops['ecm_cluster'] = cell_crops['ecm_cluster'].replace(replace_dict)
cell_crops.to_csv(os.path.join(ecm_dir, 'cell_crops.csv'), index=False)

# QC clustering results

# generate image with each crop set to the value of the cluster its assigned to
metadata_df = pd.read_csv(os.path.join(ecm_dir, 'metadata_df.csv'))
img = 'TONIC_TMA20_R5C3'
cluster_crop_img = np.zeros((2048, 2048))

metadata_subset = tiled_crops[tiled_crops.fov == img]
for row_crop, col_crop, cluster in zip(metadata_subset.row_coord, metadata_subset.col_coord, metadata_subset.cluster):
    if cluster == 'no_ecm':
        cluster = 0
    elif cluster == 'Hot_Coll':
        cluster = 1
    elif cluster == 'Cold_Coll':
        cluster = 2
    cluster_crop_img[row_crop:row_crop + crop_size, col_crop:col_crop + crop_size] = int(cluster)

io.imshow(cluster_crop_img)

# group cells in each FOV according to their ECM cluster
plot_cell_crops = cell_crops[['fov', 'id', 'ecm_cluster']]
plot_cell_crops = plot_cell_crops.rename(columns={'id': 'label'})

cell_table_clusters = pd.merge(cell_table_clusters, plot_cell_crops[['fov', 'label', 'ecm_cluster']],
                                 on=['fov', 'label'], how='left')

grouped_ecm_region = cell_table_clusters[['fov', 'ecm_cluster', 'cell_cluster_broad']].groupby(['fov', 'ecm_cluster']).value_counts(normalize=True)
grouped_ecm_region = grouped_ecm_region.unstack(level='cell_cluster_broad', fill_value=0).stack()

grouped_ecm_region = grouped_ecm_region.reset_index()
grouped_ecm_region.columns = ['fov',  'ecm_cluster', 'cell_cluster','count']


# plot the distribution of cell clusters in each ECM cluster
g = sns.FacetGrid(grouped_ecm_region, col='cell_cluster', col_wrap=3, hue='ecm_cluster',
                  palette=['Black'], sharey=False, aspect=2.5)
g.map(sns.violinplot, 'ecm_cluster', 'count',
      order=['Hot_Coll', 'VIM_Fibro', 'Fibro_Coll',
       'Cold_Coll', 'no_ecm'])

