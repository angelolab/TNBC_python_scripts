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


# proportion of cells in cell_cluster_broad per patient
cluster_broad_freq_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                                  result_name='cluster_broad_freq',
                                                  cluster_col_name='cell_cluster_broad',
                                                  subset_col='tumor_region', normalize=True)

# number of cells in cell_cluster_broad per patient
cluster_broad_count_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                                   result_name='cluster_broad_count',
                                                   cluster_col_name='cell_cluster_broad',
                                                   subset_col='tumor_region', normalize=False)

# proportion of cells in cell_cluster per patient
cluster_freq_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                            result_name='cluster_freq',
                                            cluster_col_name='cell_cluster',
                                            subset_col='tumor_region', normalize=True)

# number of cells in cell_cluster per patient
cluster_count_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                             result_name='cluster_count',
                                             cluster_col_name='cell_cluster',
                                             subset_col='tumor_region', normalize=False)

# proportion of cells in cell_meta_cluster per patient
cluster_meta_freq_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                                 result_name='meta_cluster_freq',
                                                 cluster_col_name='cell_meta_cluster',
                                                 subset_col='tumor_region', normalize=True)

# number of cells in cell_meta_cluster per patient
cluster_meta_count_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                                  result_name='meta_cluster_count',
                                                  cluster_col_name='cell_meta_cluster',
                                                  subset_col='tumor_region', normalize=False)


# proportion of T cell subsets
tcell_mask = cell_table_clusters['cell_cluster'].isin(['Treg', 'CD8T', 'CD4T', 'T_Other'])
tcell_freq_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[tcell_mask, :],
                                          result_name='tcell_freq',
                                          cluster_col_name='cell_cluster',
                                          subset_col='tumor_region', normalize=True)


# proportion of immune cell subsets
immune_mask = cell_table_clusters['cell_cluster_broad'].isin(['Mono_Mac', 'T',
                                                              'Granulocyte', 'NK', 'B'])
immune_mask_2 = cell_table_clusters.cell_cluster == 'Immune_Other'
immune_mask = np.logical_or(immune_mask, immune_mask_2)
immune_freq_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[immune_mask, :],
                                           result_name='immune_freq',
                                           cluster_col_name='cell_cluster',
                                           subset_col='tumor_region', normalize=True)

# proportion of stromal subsets
stroma_mask = cell_table_clusters['cell_cluster_broad'].isin(['Stroma'])
stroma_freq_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[stroma_mask, :],
                                           result_name='stroma_freq',
                                           cluster_col_name='cell_meta_cluster',
                                           subset_col='tumor_region', normalize=True)

# proportion of cancer subsets
cancer_mask = cell_table_clusters['cell_cluster_broad'].isin(['Cancer'])
cancer_freq_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[cancer_mask, :],
                                           result_name='cancer_freq',
                                           cluster_col_name='cell_meta_cluster',
                                           subset_col='tumor_region', normalize=True)

# distribution of neighborhoods
kmeans_freq_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                           result_name='kmeans_freq',
                                           cluster_col_name='kmeans_labels',
                                           subset_col='tumor_region', normalize=True)

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

# calculate frequency of cell types per region per image
grouped_cell_freq_region = cell_table_clusters[['fov', 'tumor_region']].groupby(['fov'])
grouped_cell_freq_region = grouped_cell_freq_region['tumor_region'].value_counts(normalize=True)
grouped_cell_freq_region = grouped_cell_freq_region.unstack(level='tumor_region', fill_value=0).stack()
grouped_cell_freq_region = pd.DataFrame(grouped_cell_freq_region)
grouped_cell_freq_region.columns = ['value']
grouped_cell_freq_region.reset_index(inplace=True)
grouped_cell_freq_region['metric'] = 'cell_freq_tumor_region'
grouped_cell_freq_region.rename(columns={'tumor_region': 'subset'}, inplace=True)
grouped_cell_freq_region['cell_type'] = 'all'


# create single df with appropriate metadata
total_df = pd.concat([cluster_broad_freq_df, cluster_freq_df, cluster_meta_freq_df,
                      tcell_freq_df, immune_freq_df, stroma_freq_df, cancer_freq_df,
                      kmeans_freq_df, cluster_broad_count_df, cluster_count_df,
                      cluster_meta_count_df, grouped_cell_counts, grouped_cell_counts_region,
                      grouped_cell_freq_region], axis=0)


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

# Total number of cells positive for each functional marker in cell_cluster_broad per image
func_df_counts_broad = create_long_df_by_functional(func_table=cell_table_func,
                                                    cluster_col_name='cell_cluster_broad',
                                                    drop_cols=['cell_meta_cluster', 'cell_cluster', 'label', 'H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio', 'kmeans_labels'],
                                                    normalize=False,
                                                    result_name='cluster_broad_count',
                                                    subset_col='tumor_region')

# Proportion of cells positive for each functional marker in cell_cluster_broad per image
func_df_mean_broad = create_long_df_by_functional(func_table=cell_table_func,
                                                  cluster_col_name='cell_cluster_broad',
                                                  drop_cols=['cell_meta_cluster', 'cell_cluster', 'label', 'kmeans_labels'],
                                                  normalize=True,
                                                  result_name='cluster_broad_freq',
                                                  subset_col='tumor_region')

# Total number of cells positive for each functional marker in cell_cluster per image
func_df_counts_cluster = create_long_df_by_functional(func_table=cell_table_func,
                                                      cluster_col_name='cell_cluster',
                                                      drop_cols=['cell_meta_cluster',
                                                                 'cell_cluster_broad', 'label', 'H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio', 'kmeans_labels'],
                                                      normalize=False,
                                                      result_name='cluster_count',
                                                      subset_col='tumor_region')

# Proportion of cells positive for each functional marker in cell_cluster_broad per image
func_df_mean_cluster = create_long_df_by_functional(func_table=cell_table_func,
                                                    cluster_col_name='cell_cluster',
                                                    drop_cols=['cell_meta_cluster',
                                                               'cell_cluster_broad', 'label', 'kmeans_labels'],
                                                    normalize=True,
                                                    result_name='cluster_freq',
                                                    subset_col='tumor_region')

# Total number of cells positive for each functional marker in cell_meta_cluster per image
func_df_counts_meta = create_long_df_by_functional(func_table=cell_table_func,
                                                   cluster_col_name='cell_meta_cluster',
                                                   drop_cols=['cell_cluster', 'cell_cluster_broad',
                                                              'label', 'H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio', 'kmeans_labels'],
                                                   normalize=False,
                                                   result_name='meta_cluster_count',
                                                   subset_col='tumor_region')

# Proportion of cells positive for each functional marker in cell_meta_cluster per image
func_df_mean_meta = create_long_df_by_functional(func_table=cell_table_func,
                                                 cluster_col_name='cell_meta_cluster',
                                                 drop_cols=['cell_cluster', 'cell_cluster_broad', 'label', 'kmeans_labels'],
                                                 normalize=True, result_name='meta_cluster_freq',
                                                 subset_col='tumor_region')


# Proportion of cells positive for each functional marker in kmeans neighborhood per image
func_df_mean_kmeans = create_long_df_by_functional(func_table=cell_table_func,
                                                   cluster_col_name='kmeans_labels',
                                                   drop_cols=['cell_cluster', 'cell_cluster_broad',
                                                              'label', 'cell_meta_cluster'],
                                                   normalize=True,
                                                   result_name='kmeans_freq',
                                                   subset_col='tumor_region')


# create combined df
total_df_func = pd.concat([func_df_counts_broad, func_df_mean_broad, func_df_counts_cluster,
                           func_df_mean_cluster, func_df_counts_meta, func_df_mean_meta, func_df_mean_kmeans])

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

