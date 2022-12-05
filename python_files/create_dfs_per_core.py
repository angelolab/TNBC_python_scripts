import os

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
cell_table_clusters = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_clusters_only_kmeans_nh.csv'))

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
cluster_broad_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                             result_name='cluster_broad_freq',
                                             cluster_col_name='cell_cluster_broad',
                                             normalize='index')

# proportion of cells in cell_cluster per patient
cluster_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                       result_name='cluster_freq',
                                       cluster_col_name='cell_cluster',
                                       normalize='index')


# proportion of cells in cell_meta_cluster per patient
cluster_meta_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                            result_name='meta_cluster_freq',
                                            cluster_col_name='cell_meta_cluster',
                                            normalize='index')

# proportion of T cell subsets
tcell_mask = cell_table_clusters['cell_cluster'].isin(['Treg', 'CD8T', 'CD4T', 'T_Other'])
tcell_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[tcell_mask, :],
                                     result_name='tcell_freq',
                                     cluster_col_name='cell_cluster',
                                     normalize='index')


# proportion of immune cell subsets
immune_mask = cell_table_clusters['cell_cluster_broad'].isin(['Mono_Mac', 'T',
                                                              'Granulocyte', 'NK', 'B'])
immune_mask_2 = cell_table_clusters.cell_cluster == 'Immune_Other'
immune_mask = np.logical_or(immune_mask, immune_mask_2)
immune_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[immune_mask, :],
                                      result_name='immune_freq',
                                      cluster_col_name='cell_cluster',
                                      normalize='index')

# proportion of stromal subsets
stroma_mask = cell_table_clusters['cell_cluster_broad'].isin(['Stroma'])
stroma_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[stroma_mask, :],
                                      result_name='stroma_freq',
                                      cluster_col_name='cell_meta_cluster',
                                      normalize='index')

# proportion of cancer subsets
cancer_mask = cell_table_clusters['cell_cluster_broad'].isin(['Cancer'])
cancer_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[cancer_mask, :],
                                      result_name='cancer_freq',
                                      cluster_col_name='cell_meta_cluster',
                                      normalize='index')

# distribution of neighborhoods
kmeans_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                      result_name='kmeans_freq',
                                      cluster_col_name='kmeans_labels',
                                      normalize='index')

# create single df with appropriate metadata
total_df = pd.concat([cluster_broad_df, cluster_df, cluster_meta_df, tcell_df, immune_df,
                      stroma_df, cancer_df, kmeans_df], axis=0)

# check that all metadata from core_metadata succesfully transferred over
total_df = total_df.merge(harmonized_metadata, on='fov', how='inner')


# save annotated cluster counts
total_df.to_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'), index=False)

# create version aggregated by timepoint
total_df_grouped = total_df.groupby(['Tissue_ID', 'cell_type', 'metric'])
total_df_timepoint = total_df_grouped['value'].agg([np.mean, np.std])
total_df_timepoint.reset_index(inplace=True)
total_df_timepoint = total_df_timepoint.merge(harmonized_metadata.drop('fov', axis=1).drop_duplicates(), on='Tissue_ID')

# save timepoint df
total_df_timepoint.to_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'), index=False)


#
# Create summary dataframe with proportions and counts of different functional marker populations
#

# load processed functional table
cell_table_func = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'))
kmeans_data = cell_table_clusters[['kmeans_labels', 'fov', 'label']]
cell_table_func = cell_table_func.merge(kmeans_data, on=['fov', 'label'], how='inner')

# Total number of cells positive for each functional marker in cell_cluster_broad per image
func_df_counts_broad = create_long_df_by_functional(func_table=cell_table_func,
                                                    cluster_col_name='cell_cluster_broad',
                                                    drop_cols=['cell_meta_cluster', 'cell_cluster', 'label', 'H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio', 'kmeans_labels'],
                                                    transform_func=np.sum,
                                                    result_name='cluster_broad_count')

# Proportion of cells positive for each functional marker in cell_cluster_broad per image
func_df_mean_broad = create_long_df_by_functional(func_table=cell_table_func,
                                                  cluster_col_name='cell_cluster_broad',
                                                  drop_cols=['cell_meta_cluster', 'cell_cluster', 'label', 'kmeans_labels'],
                                                  transform_func=np.mean,
                                                  result_name='cluster_broad_freq')

# Total number of cells positive for each functional marker in cell_cluster per image
func_df_counts_cluster = create_long_df_by_functional(func_table=cell_table_func,
                                                      cluster_col_name='cell_cluster',
                                                      drop_cols=['cell_meta_cluster',
                                                                 'cell_cluster_broad', 'label', 'H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio', 'kmeans_labels'],
                                                      transform_func=np.sum,
                                                      result_name='cluster_count')

# Proportion of cells positive for each functional marker in cell_cluster_broad per image
func_df_mean_cluster = create_long_df_by_functional(func_table=cell_table_func,
                                                    cluster_col_name='cell_cluster',
                                                    drop_cols=['cell_meta_cluster',
                                                               'cell_cluster_broad', 'label', 'kmeans_labels'],
                                                    transform_func=np.mean,
                                                    result_name='cluster_freq')

# Total number of cells positive for each functional marker in cell_meta_cluster per image
func_df_counts_meta = create_long_df_by_functional(func_table=cell_table_func,
                                                   cluster_col_name='cell_meta_cluster',
                                                   drop_cols=['cell_cluster', 'cell_cluster_broad',
                                                              'label', 'H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio', 'kmeans_labels'],
                                                   transform_func=np.sum,
                                                   result_name='meta_cluster_count')

# Proportion of cells positive for each functional marker in cell_meta_cluster per image
func_df_mean_meta = create_long_df_by_functional(func_table=cell_table_func,
                                                 cluster_col_name='cell_meta_cluster',
                                                 drop_cols=['cell_cluster', 'cell_cluster_broad', 'label', 'kmeans_labels'],
                                                 transform_func=np.mean, result_name='meta_cluster_freq')


# Proportion of cells positive for each functional marker in kmeans neighborhood per image
func_df_mean_kmeans = create_long_df_by_functional(func_table=cell_table_func,
                                                 cluster_col_name='kmeans_labels',
                                                 drop_cols=['cell_cluster', 'cell_cluster_broad', 'label', 'cell_meta_cluster'],
                                                 transform_func=np.mean, result_name='kmeans_freq')


# create combined df
total_df_func = pd.concat([func_df_counts_broad, func_df_mean_broad, func_df_counts_cluster,
                           func_df_mean_cluster, func_df_counts_meta, func_df_mean_meta, func_df_mean_kmeans])

# check that all metadata from core_metadata succesfully transferred over
total_df_func = total_df_func.merge(harmonized_metadata, on='fov', how='inner')

# save combined df
total_df_func.to_csv(os.path.join(data_dir, 'functional_df_per_core.csv'), index=False)


# create version aggregated by timepoint
total_df_grouped_func = total_df_func.groupby(['Tissue_ID', 'cell_type', 'functional_marker', 'metric'])
total_df_timepoint_func = total_df_grouped_func['value'].agg([np.mean, np.std])
total_df_timepoint_func.reset_index(inplace=True)
total_df_timepoint_func = total_df_timepoint_func.merge(harmonized_metadata.drop('fov', axis=1).drop_duplicates(), on='Tissue_ID')

# save timepoint df
total_df_timepoint_func.to_csv(os.path.join(data_dir, 'functional_df_per_timepoint.csv'), index=False)


