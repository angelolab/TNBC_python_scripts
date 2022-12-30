import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from python_files.utils import create_long_df_by_functional, create_long_df_by_cluster

#
# This file creates plotting-ready data structures for cell prevalance and functional markers
#

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'

#
# Preprocess metadata to ensure all samples are present
#

# load relevant tables
core_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
timepoint_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint.csv'))
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'harmonized_metadata.csv'))
cell_table_clusters = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_clusters_only_kmeans_nh_mask.csv'))

# replace values in a column with other values
replace_dict = {1: 'cancer_vim_56', 2: 'immune_other', 3: 'cancer_sma', 4: 'cd4t', 5: 'cancer_ck17',
                6: 'CD8T', 7: 'FAP_fibro', 8: 'CD163_mac', 9: 'Bcells', 10: 'Cancer_Ecad',
                11: 'myeloid_party', 12: 'cancer_mono_cd56', 13: 'cancer_other_other', 14: 'vim_sma_cd31'}
cell_table_clusters['kmeans_labels'] = cell_table_clusters.kmeans_neighborhood
cell_table_clusters['kmeans_labels'].replace(replace_dict, inplace=True)


# check for FOVs present in imaged data that aren't in core metadata
missing_fovs = cell_table_clusters.loc[~cell_table_clusters.fov.isin(core_metadata.fov), 'fov'].unique()
cell_table_clusters = cell_table_clusters.loc[~cell_table_clusters.fov.isin(missing_fovs), :]

#
# Generate counts and proportions of cell clusters per FOV
#

# Create list to hold parameters for each df that will be produced
cluster_df_params = [['cluster_broad_freq', 'cell_cluster_broad', True],
                     ['cluster_broad_count', 'cell_cluster_broad', False],
                     ['cluster_freq', 'cell_cluster', True],
                     ['cluster_count', 'cell_cluster', False],
                     ['meta_cluster_freq', 'cell_meta_cluster', True],
                     ['meta_cluster_count', 'cell_meta_cluster', False],
                     ['kmeans_freq', 'kmeans_labels', True]]

cluster_dfs = []
for result_name, cluster_col_name, normalize in cluster_df_params:
    cluster_dfs.append(create_long_df_by_cluster(cell_table=cell_table_clusters,
                                                 result_name=result_name,
                                                 cluster_col_name=cluster_col_name,
                                                 subset_col='tumor_region',
                                                 normalize=normalize))


# create masks for dfs looking at only a subset of cells

# proportion of T cell subsets
tcell_mask = cell_table_clusters['cell_cluster'].isin(['Treg', 'CD8T', 'CD4T', 'T_Other'])

# proportion of immune cell subsets
immune_mask = cell_table_clusters['cell_cluster_broad'].isin(['Mono_Mac', 'T',
                                                              'Granulocyte', 'NK', 'B'])
immune_mask_2 = cell_table_clusters.cell_cluster == 'Immune_Other'
immune_mask = np.logical_or(immune_mask, immune_mask_2)

# proportion of stromal subsets
stroma_mask = cell_table_clusters['cell_cluster_broad'].isin(['Stroma'])

# proportion of cancer subsets
cancer_mask = cell_table_clusters['cell_cluster_broad'].isin(['Cancer'])

cluster_mask_params = [['tcell_freq', 'cell_cluster', True, tcell_mask],
                       ['immune_freq', 'cell_cluster', True, immune_mask],
                       ['stroma_freq', 'cell_meta_cluster', True, stroma_mask],
                       ['cancer_freq', 'cell_meta_cluster', True, cancer_mask]]

for result_name, cluster_col_name, normalize, mask in cluster_mask_params:
    cluster_dfs.append(create_long_df_by_cluster(cell_table=cell_table_clusters.loc[mask, :],
                                                 result_name=result_name,
                                                 cluster_col_name=cluster_col_name,
                                                 subset_col='tumor_region',
                                                 normalize=normalize))

# calculate total number of cells per image
grouped_cell_counts = cell_table_clusters[['fov']].groupby('fov').value_counts()
grouped_cell_counts = pd.DataFrame(grouped_cell_counts)
grouped_cell_counts.columns = ['value']
grouped_cell_counts.reset_index(inplace=True)
grouped_cell_counts['metric'] = 'cell_count'
grouped_cell_counts['cell_type'] = 'all'
grouped_cell_counts['subset'] = 'all'

#

# calculate total number of cells per region per image
grouped_cell_counts_region = cell_table_clusters[['fov', 'tumor_region']].groupby(['fov', 'tumor_region']).value_counts()
grouped_cell_counts_region = grouped_cell_counts_region.unstack(level='tumor_region', fill_value=0).stack()
grouped_cell_counts_region = pd.DataFrame(grouped_cell_counts_region)
grouped_cell_counts_region.columns = ['value']
grouped_cell_counts_region.reset_index(inplace=True)
grouped_cell_counts_region['metric'] = 'cell_count_tumor_region'
grouped_cell_counts_region.rename(columns={'tumor_region': 'subset'}, inplace=True)
grouped_cell_counts_region['cell_type'] = 'all'

# calculate proportions of cells per region per image
grouped_cell_freq_region = cell_table_clusters[['fov', 'tumor_region']].groupby(['fov'])
grouped_cell_freq_region = grouped_cell_freq_region['tumor_region'].value_counts(normalize=True)
grouped_cell_freq_region = grouped_cell_freq_region.unstack(level='tumor_region', fill_value=0).stack()
grouped_cell_freq_region = pd.DataFrame(grouped_cell_freq_region)
grouped_cell_freq_region.columns = ['value']
grouped_cell_freq_region.reset_index(inplace=True)
grouped_cell_freq_region['metric'] = 'cell_freq_tumor_region'
grouped_cell_freq_region.rename(columns={'tumor_region': 'subset'}, inplace=True)
grouped_cell_freq_region['cell_type'] = 'all'

# add manually defined dfs to overall list
cluster_dfs.extend([grouped_cell_counts,
                    grouped_cell_counts_region,
                    grouped_cell_freq_region])

# create single df with appropriate metadata
total_df = pd.concat(cluster_dfs, axis=0)


# check that all metadata from core_metadata succesfully transferred over
total_df = total_df.merge(harmonized_metadata, on='fov', how='inner')


# save annotated cluster counts
total_df.to_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'), index=False)

# create version aggregated by timepoint
total_df_grouped = total_df.groupby(['Tissue_ID', 'cell_type', 'metric', 'subset'])
total_df_timepoint = total_df_grouped['value'].agg([np.mean, np.std])
total_df_timepoint.reset_index(inplace=True)
total_df_timepoint = total_df_timepoint.merge(harmonized_metadata.drop('fov', axis=1).drop_duplicates(), on='Tissue_ID')

# save timepoint df
total_df_timepoint.to_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'), index=False)


#
# Create summary dataframe with proportions and counts of different functional marker populations
#

# load processed functional table
cell_table_func = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_functional_only_mask.csv'))
kmeans_data = cell_table_clusters[['kmeans_labels', 'fov', 'label']]
cell_table_func = cell_table_func.merge(kmeans_data, on=['fov', 'label'], how='inner')

# Columns which are not thresholded (such as ratios between markers) can only be calculated for
# dfs looking at normalized expression, and need to be dropped when calculating counts
count_drop_cols = ['H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio', 'kmeans_labels']

# Create list to hold parameters for each df that will be produced
func_df_params = [['cluster_broad_count', 'cell_cluster_broad', False],
                  ['cluster_broad_freq', 'cell_cluster_broad', True],
                  ['cluster_count', 'cell_cluster', False],
                  ['cluster_freq', 'cell_cluster', True],
                  ['meta_cluster_count', 'cell_meta_cluster', False],
                  ['meta_cluster_freq', 'cell_meta_cluster', True],
                  ['kmeans_freq', 'kmeans_labels', True]]

func_dfs = []
for result_name, cluster_col_name, normalize in func_df_params:
    # columns which are not functional markers need to be dropped from the df
    drop_cols = ['label']
    if not normalize:
        drop_cols.extend(count_drop_cols)

    # remove cluster_names except for the one specified for the df
    cluster_names = ['cell_meta_cluster', 'cell_cluster', 'cell_cluster_broad', 'kmeans_labels']
    cluster_names.remove(cluster_col_name)
    drop_cols.extend(cluster_names)

    # create df
    func_dfs.append(create_long_df_by_functional(func_table=cell_table_func,
                                                 result_name=result_name,
                                                 cluster_col_name=cluster_col_name,
                                                 drop_cols=drop_cols,
                                                 normalize=normalize,
                                                 subset_col='tumor_region'))

# create combined df
total_df_func = pd.concat(func_dfs, axis=0)

# check that all metadata from core_metadata succesfully transferred over
total_df_func = total_df_func.merge(harmonized_metadata, on='fov', how='inner')

# save combined df
total_df_func.to_csv(os.path.join(data_dir, 'functional_df_per_core.csv'), index=False)


# create version aggregated by timepoint
total_df_grouped_func = total_df_func.groupby(['Tissue_ID', 'cell_type', 'functional_marker', 'metric', 'subset'])
total_df_timepoint_func = total_df_grouped_func['value'].agg([np.mean, np.std])
total_df_timepoint_func.reset_index(inplace=True)
total_df_timepoint_func = total_df_timepoint_func.merge(harmonized_metadata.drop('fov', axis=1).drop_duplicates(), on='Tissue_ID')

# save timepoint df
total_df_timepoint_func.to_csv(os.path.join(data_dir, 'functional_df_per_timepoint.csv'), index=False)


# create histogram of number of cells per cluster per image
plot_df = total_df[total_df['metric'] == 'cluster_count']

# created fascetted histogram with seaborn
g = sns.FacetGrid(plot_df, col="cell_type", col_wrap=5, height=2.5, aspect=1.5)
g.map(sns.histplot, "value", bins=range(0, 400, 10))
g.savefig(os.path.join(plot_dir, 'cell_count_per_cluster.png'))
plt.close()

# create histogram of number of cells per image
plot_df = total_df[total_df['metric'] == 'cluster_count']
grouped = plot_df[['fov', 'value', 'cell_type']].groupby('fov').sum()

sns.histplot(grouped['value'], bins=range(0, 4000, 100))
freq_fovs = grouped[grouped['value'] > 500].index
total_df_filtered = total_df[total_df['fov'].isin(freq_fovs)]

# save annotated cluster counts
total_df_filtered.to_csv(os.path.join(data_dir, 'cluster_df_per_core_filtered.csv'), index=False)

# create version aggregated by timepoint
total_df_grouped_filtered = total_df_filtered.groupby(['Tissue_ID', 'cell_type', 'metric'])
total_df_timepoint_filtered = total_df_grouped_filtered['value'].agg([np.mean, np.std])
total_df_timepoint_filtered.reset_index(inplace=True)
total_df_timepoint_filtered = total_df_timepoint_filtered.merge(harmonized_metadata.drop('fov', axis=1).drop_duplicates(), on='Tissue_ID')

# save timepoint df
total_df_timepoint_filtered.to_csv(os.path.join(data_dir, 'cluster_df_per_timepoint_filtered.csv'), index=False)


# filter out low frequency clusters
cells = func_df_mean_cluster.cell_type.unique()

bad_rows = np.repeat(False, len(func_df_mean_cluster))

for cell in cells:
    cell_df = cluster_count_df[cluster_count_df['cell_type'] == cell]
    keep_fovs = cell_df[cell_df['value'] > 25].fov.unique()

    # find intersection between keep_fovs and freq_fovs
    keep_fovs = np.intersect1d(keep_fovs, freq_fovs)

    # remove rows specified by remove_mask
    bad_rows = bad_rows | ((func_df_mean_cluster['cell_type'] == cell) & (~func_df_mean_cluster['fov'].isin(keep_fovs)))

func_df_mean_cluster_filtered = func_df_mean_cluster[~bad_rows]
func_df_mean_cluster_filtered = func_df_mean_cluster_filtered.merge(harmonized_metadata, on='fov', how='inner')

total_df_grouped_func = func_df_mean_cluster_filtered.groupby(['Tissue_ID', 'cell_type', 'functional_marker', 'metric'])
total_df_timepoint_func = total_df_grouped_func['value'].agg([np.mean, np.std])
total_df_timepoint_func.reset_index(inplace=True)
total_df_timepoint_func = total_df_timepoint_func.merge(harmonized_metadata.drop('fov', axis=1).drop_duplicates(), on='Tissue_ID')

