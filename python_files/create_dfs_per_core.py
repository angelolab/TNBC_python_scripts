import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from python_files.utils import create_long_df_by_functional, create_long_df_by_cluster

#
# This file creates plotting-ready data structures to enumerate the frequency, count, and density
# of cell populations. It also creates data structures with the frequency and count of functional
# marker positivity. Each of these dfs can be created across multiple levels of clustering
# granularity. For example, a broad classification might include T, B, Myeloid, Stroma, and Cancer,
# whereas a more granular clustering scheme would separate out CD4T, CD8T, Tregs, etc.
#

local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data'

#
# Preprocess metadata to ensure all samples are present
#

# load relevant tables
core_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'TONIC_data_per_core.csv'))
timepoint_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'TONIC_data_per_timepoint.csv'))
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'harmonized_metadata.csv'))
cell_table_clusters = pd.read_csv(os.path.join(data_dir, 'post_processing', 'cell_table_clusters.csv'))
cell_table_func = pd.read_csv(os.path.join(data_dir, 'post_processing', 'cell_table_func_all.csv'))
area_df = pd.read_csv(os.path.join(data_dir, 'post_processing', 'fov_annotation_mask_area.csv'))
annotations_by_mask = pd.read_csv(os.path.join(data_dir, 'post_processing', 'cell_annotation_mask.csv'))

# merge cell-level annotations
harmonized_annotations = annotations_by_mask
harmonized_annotations = harmonized_annotations.rename(columns={'mask_name': 'tumor_region'})
assert len(harmonized_annotations) == len(cell_table_clusters)

cell_table_clusters = cell_table_clusters.merge(harmonized_annotations, on=['fov', 'label'], how='left')
cell_table_func = cell_table_func.merge(harmonized_annotations, on=['fov', 'label'], how='left')

# check for FOVs present in imaged data that aren't in core metadata
missing_fovs = cell_table_clusters.loc[~cell_table_clusters.fov.isin(core_metadata.fov), 'fov'].unique()

# TONIC_TMA24_R11C2 and TONIC_TMA24_R11C3 are wrong tissue
cell_table_clusters = cell_table_clusters.loc[~cell_table_clusters.fov.isin(missing_fovs), :]
cell_table_func = cell_table_func.loc[~cell_table_func.fov.isin(missing_fovs), :]


#
# Generate counts and proportions of cell clusters per FOV
#

# Specify the types of cluster dfs to produce. Each row corresponds to a different way of
# summarizing the data. The first item in each list is the name for the df, the second is the
# name of the column to use to for the cluster labels, and the third is a boolean flag controlling
# whether normalized frequencies or total counts will be returned

cluster_df_params = [['cluster_broad_freq', 'cell_cluster_broad', True],
                     ['cluster_broad_count', 'cell_cluster_broad', False],
                     ['cluster_freq', 'cell_cluster', True],
                     ['cluster_count', 'cell_cluster', False],
                     ['meta_cluster_freq', 'cell_meta_cluster', True],
                     ['meta_cluster_count', 'cell_meta_cluster', False]]
                     #['kmeans_freq', 'kmeans_labels', True]]

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
grouped_cell_counts['metric'] = 'total_cell_count'
grouped_cell_counts['cell_type'] = 'all'
grouped_cell_counts['subset'] = 'all'

#

# calculate total number of cells per region per image
grouped_cell_counts_region = cell_table_clusters[['fov', 'tumor_region']].groupby(['fov', 'tumor_region']).value_counts()
grouped_cell_counts_region = pd.DataFrame(grouped_cell_counts_region)
grouped_cell_counts_region.columns = ['value']
grouped_cell_counts_region.reset_index(inplace=True)
grouped_cell_counts_region['metric'] = 'total_cell_count'
grouped_cell_counts_region.rename(columns={'tumor_region': 'subset'}, inplace=True)
grouped_cell_counts_region['cell_type'] = 'all'

# calculate proportions of cells per region per image
grouped_cell_freq_region = cell_table_clusters[['fov', 'tumor_region']].groupby(['fov'])
grouped_cell_freq_region = grouped_cell_freq_region['tumor_region'].value_counts(normalize=True)
grouped_cell_freq_region = pd.DataFrame(grouped_cell_freq_region)
grouped_cell_freq_region.columns = ['value']
grouped_cell_freq_region.reset_index(inplace=True)
grouped_cell_freq_region['metric'] = 'total_cell_freq'
grouped_cell_freq_region.rename(columns={'tumor_region': 'subset'}, inplace=True)
grouped_cell_freq_region['cell_type'] = 'all'

# add manually defined dfs to overall list
cluster_dfs.extend([grouped_cell_counts,
                    grouped_cell_counts_region,
                    grouped_cell_freq_region])

# create single df with appropriate metadata
total_df = pd.concat(cluster_dfs, axis=0)

# compute density of cells for counts-based metrics
count_metrics = total_df.metric.unique()
count_metrics = [x for x in count_metrics if 'count' in x]

count_df = total_df.loc[total_df.metric.isin(count_metrics), :]
area_df = area_df.rename(columns={'compartment': 'subset'})
count_df = count_df.merge(area_df, on=['fov', 'subset'], how='left')
count_df['value'] = count_df['value'] / count_df['area']
count_df['value'] = count_df['value'] * 1000

# rename metric from count to density
count_df['metric'] = count_df['metric'].str.replace('count', 'density')
count_df = count_df.drop(columns=['area'])
total_df = pd.concat([total_df, count_df], axis=0)

# check that all metadata from core_metadata succesfully transferred over
len_total_df = len(total_df)
total_df = total_df.merge(harmonized_metadata, on='fov', how='left')
assert len_total_df == len(total_df)


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


# Columns which are not thresholded (such as ratios between markers) can only be calculated for
# dfs looking at normalized expression, and need to be dropped when calculating counts
count_drop_cols = ['H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio']  # , 'kmeans_labels']

# Create list to hold parameters for each df that will be produced
func_df_params = [['cluster_broad_count', 'cell_cluster_broad', False],
                  ['cluster_broad_freq', 'cell_cluster_broad', True],
                  ['cluster_count', 'cell_cluster', False],
                  ['cluster_freq', 'cell_cluster', True],
                  ['meta_cluster_count', 'cell_meta_cluster', False],
                  ['meta_cluster_freq', 'cell_meta_cluster', True]]
                  #['kmeans_freq', 'kmeans_labels', True]]

func_dfs = []
for result_name, cluster_col_name, normalize in func_df_params:
    # columns which are not functional markers need to be dropped from the df
    drop_cols = ['label']
    if not normalize:
        drop_cols.extend(count_drop_cols)

    # remove cluster_names except for the one specified for the df
    cluster_names = ['cell_meta_cluster', 'cell_cluster', 'cell_cluster_broad'] # , 'kmeans_labels']
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

#
# Filter functional markers
#

# filter functional markers to only include FOVs with at least the specified number of cells
total_df = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
min_cells = 5

filtered_dfs = []
metrics = [['cluster_broad_count', 'cluster_broad_freq'],
           ['cluster_count', 'cluster_freq'],
           ['meta_cluster_count', 'meta_cluster_freq']]

for metric in metrics:
    # subset count df to include cells at the relevant clustering resolution
    for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
        count_df = total_df[total_df.metric == metric[0]]
        count_df = count_df[count_df.subset == compartment]

        # subset functional df to only include functional markers at this resolution
        func_df = total_df_func[total_df_func.metric.isin(metric)]
        func_df = func_df[func_df.subset == compartment]

        # for each cell type, determine which FOVs have high enough counts to be included
        for cell_type in count_df.cell_type.unique():
            keep_df = count_df[count_df.cell_type == cell_type]
            keep_df = keep_df[keep_df.value >= min_cells]
            keep_fovs = keep_df.fov.unique()

            # subset functional df to only include FOVs with high enough counts
            keep_markers = func_df[func_df.cell_type == cell_type]
            keep_markers = keep_markers[keep_markers.fov.isin(keep_fovs)]

            # append to list of filtered dfs
            filtered_dfs.append(keep_markers)

filtered_func_df = pd.concat(filtered_dfs)

# # identify combinations of markers and cell types to include in analysis based on threshold
# mean_percent_positive = 0.05
# sp_markers = [x for x in filtered_func_df.functional_marker.unique() if '__' not in x]
# broad_df = filtered_func_df[filtered_func_df.metric == 'cluster_broad_freq']
# broad_df = broad_df[broad_df.functional_marker.isin(sp_markers)]
# broad_df = broad_df[broad_df.subset == 'all']
# broad_df = broad_df[broad_df.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
# broad_df_agg = broad_df[['fov', 'functional_marker', 'cell_type', 'value']].groupby(['cell_type', 'functional_marker']).agg(np.mean)
#
# broad_df_agg = broad_df_agg.reset_index()
# broad_df = broad_df_agg.pivot(index='cell_type', columns='functional_marker', values='value')
# broad_df_include = broad_df > mean_percent_positive
#
# # include for all cells
# general_markers = ['Ki67', 'HLA1', 'Vim', 'H3K9ac_H3K27me3_ratio']
# broad_df_include[[general_markers]] = True
#
# # CD45 isoform ratios
# double_pos = np.logical_and(broad_df_include['CD45RO'], broad_df_include['CD45RB'])
# broad_df_include['CD45RO_CD45RB_ratio'] = double_pos
#
# # Cancer expression
# broad_df_include.loc['Cancer', ['HLADR', 'CD57']] = True
#
# broad_df_include.to_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_broad.csv'))
#
# # apply thresholds to medium level clustering
# assignment_dict_med = {'Cancer': ['Cancer', 'Cancer_EMT', 'Cancer_Other'],
#                      'Mono_Mac': ['M1_Mac', 'M2_Mac', 'Mac_Other', 'Monocyte', 'APC'],
#                      'B': ['B'],
#                      'T': ['CD4T', 'CD8T', 'Treg', 'T_Other', 'Immune_Other'],
#                      'Granulocyte': ['Neutrophil', 'Mast'],
#                      'Stroma': ['Endothelium', 'Fibroblast', 'Stroma'],
#                      'NK': ['NK'],
#                      'Other': ['Other']}
#
# # get a list of all cell types
# cell_types = np.concatenate([assignment_dict_med[key] for key in assignment_dict_med.keys()])
# cell_types.sort()
#
# med_df_include = pd.DataFrame(index=cell_types, columns=broad_df.columns)
#
# for key in assignment_dict_med.keys():
#     values = assignment_dict_med[key]
#     med_df_include.loc[values] = broad_df_include.loc[key].values
#
# # check if assignment makes sense
# med_df = filtered_func_df[filtered_func_df.metric == 'cluster_freq']
# med_df = med_df[med_df.functional_marker.isin(sp_markers)]
# med_df = med_df[med_df.subset == 'all']
# med_df = med_df[med_df.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
# med_df_agg = med_df[['fov', 'functional_marker', 'cell_type', 'value']].groupby(['cell_type', 'functional_marker']).agg(np.mean)
#
# med_df_agg.reset_index(inplace=True)
# med_df = med_df_agg.pivot(index='cell_type', columns='functional_marker', values='value')
# med_df_bin = med_df > mean_percent_positive
#
# # add CD38 signal
# med_df_include.loc[['Endothelium', 'Immune_Other'], 'CD38'] = True
#
# # add IDO signal
# med_df_include.loc[['APC'], 'IDO'] = True
#
# # compare to see where assignments disagree, to see if any others need to be added
# new_includes = (med_df_bin == True) & (med_df_include == False)
#
# med_df_include.to_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_med.csv'))
#
# # do the same for the fine-grained clustering
# assignment_dict_meta = {'Cancer': ['Cancer_CD56', 'Cancer_CK17', 'Cancer_Ecad'],
#                    'Cancer_EMT': ['Cancer_SMA', 'Cancer_Vim'],
#                    'Cancer_Other': ['Cancer_Other', 'Cancer_Mono'],
#                    'M1_Mac': ['CD68'],
#                    'M2_Mac': ['CD163'],
#                    'Mac_Other': ['CD68_CD163_DP'],
#                    'Monocyte': ['CD4_Mono', 'CD14'],
#                    'APC': ['CD11c_HLADR'],
#                    'B':  ['CD20'],
#                    'Endothelium': ['CD31', 'CD31_VIM'],
#                    'Fibroblast': ['FAP', 'FAP_SMA', 'SMA'],
#                    'Stroma': ['Stroma_Collagen', 'Stroma_Fibronectin', 'VIM'],
#                    'NK': ['CD56'],
#                    'Neutrophil': ['Neutrophil'],
#                    'Mast': ['Mast'],
#                    'CD4T': ['CD4T','CD4T_HLADR'],
#                    'CD8T': ['CD8T'],
#                    'Treg': ['Treg'],
#                    'T_Other': ['CD3_DN','CD4T_CD8T_DP'],
#                    'Immune_Other': ['Immune_Other'],
#                    'Other': ['Other']}
#
# # get a list of all cell types
# cell_types = np.concatenate([assignment_dict_meta[key] for key in assignment_dict_meta.keys()])
# cell_types.sort()
#
# meta_df_include = pd.DataFrame(index=cell_types, columns=broad_df.columns)
#
# for key in assignment_dict_meta.keys():
#     values = assignment_dict_meta[key]
#     meta_df_include.loc[values] = med_df_include.loc[key].values
#
# # check if assignment makes sense
# meta_df = filtered_func_df[filtered_func_df.metric == 'meta_cluster_freq']
# meta_df = meta_df[meta_df.functional_marker.isin(sp_markers)]
# meta_df = meta_df[meta_df.subset == 'all']
# meta_df = meta_df[meta_df.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
# meta_df_agg = meta_df[['fov', 'functional_marker', 'cell_type', 'value']].groupby(['cell_type', 'functional_marker']).agg(np.mean)
#
# meta_df_agg.reset_index(inplace=True)
# meta_df = meta_df_agg.pivot(index='cell_type', columns='functional_marker', values='value')
# meta_df_bin = meta_df > mean_percent_positive
#
# # compare to see where assignments disagree
# new_includes = (meta_df_bin == True) & (meta_df_include == False)
#
# meta_df_include.to_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_meta.csv'))

# process functional data so that only the specified cell type/marker combos are included

# load matrices
broad_df_include = pd.read_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_broad.csv'), index_col=0)
med_df_include = pd.read_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_med.csv'), index_col=0)
meta_df_include = pd.read_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_meta.csv'), index_col=0)

# identify metrics and dfs that will be filtered
filtering = [['cluster_broad_count', 'cluster_broad_freq', broad_df_include],
           ['cluster_count', 'cluster_freq', med_df_include],
           ['meta_cluster_count', 'meta_cluster_freq', meta_df_include]]

combo_dfs = []

for filters in filtering:
    # get variables
    metric_names = filters[:2]
    metric_df = filters[2]

    # subset functional df to only include functional markers at this resolution
    func_df = filtered_func_df[filtered_func_df.metric.isin(metric_names)]

    # loop over each cell type, and get the corresponding markers
    for cell_type in metric_df.index:
        markers = metric_df.columns[metric_df.loc[cell_type] == True]

        # subset functional df to only include this cell type
        func_df_cell = func_df[func_df.cell_type == cell_type]

        # subset functional df to only include markers for this cell type
        func_df_cell = func_df_cell[func_df_cell.functional_marker.isin(markers)]

        # append to list of dfs
        combo_dfs.append(func_df_cell)

# do the same thing for double positive cells

# # identify combinations of markers and cell types to include in analysis based on threshold
# dp_markers = [x for x in filtered_func_df.functional_marker.unique() if '__' in x]
# mean_percent_positive_dp = 0.05
# broad_df_dp = filtered_func_df[filtered_func_df.metric == 'cluster_broad_freq']
# broad_df_dp = broad_df_dp[broad_df_dp.subset == 'all']
# broad_df_dp = broad_df_dp[broad_df_dp.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
# broad_df_dp = broad_df_dp[broad_df_dp.functional_marker.isin(dp_markers)]
# broad_df_dp_agg = broad_df_dp[['fov', 'functional_marker', 'cell_type', 'value']].groupby(['cell_type', 'functional_marker']).agg(np.mean)
#
# broad_df_dp_agg = broad_df_dp_agg.reset_index()
# broad_df_dp = broad_df_dp_agg.pivot(index='cell_type', columns='functional_marker', values='value')
# broad_df_dp_include = broad_df_dp > mean_percent_positive_dp
#
# broad_df_dp_include.to_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_broad_dp.csv'))
#
# # do the same for medium-level clustering
# med_df_dp = filtered_func_df[filtered_func_df.metric == 'cluster_freq']
# med_df_dp = med_df_dp[med_df_dp.subset == 'all']
# med_df_dp = med_df_dp[med_df_dp.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
# med_df_dp = med_df_dp[med_df_dp.functional_marker.isin(dp_markers)]
# med_df_dp_agg = med_df_dp[['fov', 'functional_marker', 'cell_type', 'value']].groupby(['cell_type', 'functional_marker']).agg(np.mean)
#
# # create matrix of cell types and markers
# med_df_dp_agg = med_df_dp_agg.reset_index()
# med_df_dp = med_df_dp_agg.pivot(index='cell_type', columns='functional_marker', values='value')
# med_df_dp_include = med_df_dp > mean_percent_positive_dp
#
# med_df_dp_include.to_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_med_dp.csv'))
#
#
# # do the same for finest-level clustering
# meta_df_dp = filtered_func_df[filtered_func_df.metric == 'meta_cluster_freq']
# meta_df_dp = meta_df_dp[meta_df_dp.subset == 'all']
# meta_df_dp = meta_df_dp[meta_df_dp.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
# meta_df_dp = meta_df_dp[meta_df_dp.functional_marker.isin(dp_markers)]
# meta_df_dp_agg = meta_df_dp[['fov', 'functional_marker', 'cell_type', 'value']].groupby(['cell_type', 'functional_marker']).agg(np.mean)
#
# # create matrix of cell types and markers
# meta_df_dp_agg = meta_df_dp_agg.reset_index()
# meta_df_dp = meta_df_dp_agg.pivot(index='cell_type', columns='functional_marker', values='value')
# meta_df_dp_include = meta_df_dp > mean_percent_positive_dp
#
# meta_df_dp_include.to_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_meta_dp.csv'))

# load inclusion matrices
broad_df_include_dp = pd.read_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_broad_dp.csv'), index_col=0)
med_df_include_dp = pd.read_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_med_dp.csv'), index_col=0)
meta_df_include_dp = pd.read_csv(os.path.join(data_dir, 'post_processing', 'inclusion_matrix_meta_dp.csv'), index_col=0)

# identify metrics and dfs that will be filtered
filtering = [['cluster_broad_count', 'cluster_broad_freq', broad_df_include_dp],
           ['cluster_count', 'cluster_freq', med_df_include_dp],
           ['meta_cluster_count', 'meta_cluster_freq', meta_df_include_dp]]

for filters in filtering:
    # get variables
    metric_names = filters[:2]
    metric_df = filters[2]

    # subset functional df to only include functional markers at this resolution
    func_df = filtered_func_df[filtered_func_df.metric.isin(metric_names)]

    # loop over each cell type, and get the corresponding markers
    for cell_type in metric_df.index:
        markers = metric_df.columns[metric_df.loc[cell_type] == True]

        # subset functional df to only include this cell type
        func_df_cell = func_df[func_df.cell_type == cell_type]

        # subset functional df to only include markers for this cell type
        func_df_cell = func_df_cell[func_df_cell.functional_marker.isin(markers)]

        # append to list of dfs
        combo_dfs.append(func_df_cell)


# create manual df with total functional marker positivity across all cells in an image
func_table_small = cell_table_func.loc[:, ~cell_table_func.columns.isin(['cell_cluster', 'cell_cluster_broad', 'cell_meta_cluster', 'label', 'tumor_region'])]
func_table_small = func_table_small.loc[:, ~func_table_small.columns.isin(dp_markers)]

# group by specified columns
grouped_table = func_table_small.groupby('fov')
transformed = grouped_table.agg(np.mean)
transformed.reset_index(inplace=True)

# reshape to long df
long_df = pd.melt(transformed, id_vars=['fov'], var_name='functional_marker')
long_df['metric'] = 'total_freq'
long_df['cell_type'] = 'all'
long_df['subset'] = 'all'

long_df = long_df.merge(harmonized_metadata, on='fov', how='inner')

long_df.to_csv(os.path.join(data_dir, 'post_processing/total_func_per_core.csv'), index=False)

# append to list of dfs
combo_dfs.append(long_df)

# combine
combo_df = pd.concat(combo_dfs)
combo_df.to_csv(os.path.join(data_dir, 'functional_df_per_core_filtered.csv'), index=False)

# create version of filtered df aggregated by timepoint
combo_df_grouped_func = combo_df.groupby(['Tissue_ID', 'cell_type', 'functional_marker', 'metric', 'subset'])
combo_df_timepoint_func = combo_df_grouped_func['value'].agg([np.mean, np.std])
combo_df_timepoint_func.reset_index(inplace=True)
combo_df_timepoint_func = combo_df_timepoint_func.merge(harmonized_metadata.drop('fov', axis=1).drop_duplicates(), on='Tissue_ID')

# save timepoint df
combo_df_timepoint_func.to_csv(os.path.join(data_dir, 'functional_df_per_timepoint_filtered.csv'), index=False)


#
# Remove double positive functional markers that are highly correlated with single positive scores
#

# cluster_resolution = [['cluster_broad_freq', 'cluster_broad_count'],
#            ['cluster_freq', 'cluster_count'],
#            ['meta_cluster_freq', 'meta_cluster_count']]
#
# all_markers = combo_df.functional_marker.unique()
# dp_markers = [x for x in all_markers if '__' in x]
#
# exclude_lists = []
# for cluster in cluster_resolution:
#
#     # subset functional df to only include functional markers at this resolution
#     func_df = combo_df[combo_df.metric.isin(cluster)]
#
#     # add unique identifier for cell + marker combo
#     func_df['feature_name'] = func_df['cell_type'] + '__' + func_df['functional_marker']
#
#     # subset the df further to look at just frequency and just one compartment
#     func_df_subset = func_df[func_df.metric == cluster[0]]
#     func_df_subset = func_df_subset[func_df_subset.subset == 'all']
#
#     # loop over each cell type, and each double positive functional marker
#     exclude_markers = []
#     cell_types = func_df_subset.cell_type.unique()
#     for cell_type in cell_types:
#         for marker in dp_markers:
#             # get the two markers that make up the double positive marker
#             marker_1, marker_2 = marker.split('__')
#
#             # subset to only include this cell type and these markers
#             current_df = func_df_subset.loc[func_df_subset.cell_type == cell_type, :]
#             current_df = current_df.loc[current_df.functional_marker.isin([marker, marker_1, marker_2]), :]
#
#             # these markers are not present in this cell type
#             if len(current_df) == 0:
#                 continue
#
#             # this double positive marker is not present in this cell type
#             if marker not in current_df.functional_marker.unique():
#                 continue
#
#             current_df_wide = current_df.pivot(index='fov', columns='functional_marker', values='value')
#
#             # the double positive marker is present, but both single positives are not; exclude it
#             if len(current_df_wide.columns) != 3:
#                 exclude_markers.append(marker)
#                 continue
#
#             corr_1, _ = spearmanr(current_df_wide[marker_1].values, current_df_wide[marker].values)
#             corr_2, _ = spearmanr(current_df_wide[marker_2].values, current_df_wide[marker].values)
#
#             if (corr_1 > 0.7) | (corr_2 > 0.7):
#                 exclude_markers.append(cell_type + '__' + marker)
#
#     # add to list
#     exclude_lists.append(exclude_markers)
#
# # construct df to hold list of exlcuded cells
# exclude_df_cluster = ['cluster_broad_freq'] * len(exclude_lists[0]) + ['cluster_freq'] * len(exclude_lists[1]) + ['meta_cluster_freq'] * len(exclude_lists[2])
# exclude_df_name = exclude_lists[0] + exclude_lists[1] + exclude_lists[2]
#
# exclude_df = pd.DataFrame({'metric': exclude_df_cluster, 'feature_name': exclude_df_name})
# exclude_df.to_csv(os.path.join(data_dir, 'post_processing/exclude_double_positive_markers.csv'), index=False)

# use previously generated exclude list
exclude_df = pd.read_csv(os.path.join(data_dir, 'post_processing/exclude_double_positive_markers.csv'))

dedup_dfs = []

cluster_resolution = [['cluster_broad_freq', 'cluster_broad_count'],
           ['cluster_freq', 'cluster_count'],
           ['meta_cluster_freq', 'meta_cluster_count']]

for cluster in cluster_resolution:
    # subset functional df to only include functional markers at this resolution
    func_df = combo_df[combo_df.metric.isin(cluster)]

    # add unique identifier for cell + marker combo
    func_df['feature_name'] = func_df['cell_type'] + '__' + func_df['functional_marker']

    exclude_names = exclude_df.loc[exclude_df.metric == cluster[0], 'feature_name'].values

    # remove double positive markers that are highly correlated with single positive scores
    func_df = func_df[~func_df.feature_name.isin(exclude_names)]
    dedup_dfs.append(func_df)


dedup_dfs.append(long_df)
deduped_df = pd.concat(dedup_dfs)
deduped_df = deduped_df.drop('feature_name', axis=1)

# save deduped df
deduped_df.to_csv(os.path.join(data_dir, 'functional_df_per_core_filtered_deduped.csv'), index=False)

# create version aggregated by timepoint
deduped_df_grouped = deduped_df.groupby(['Tissue_ID', 'cell_type', 'functional_marker', 'metric', 'subset'])
deduped_df_timepoint = deduped_df_grouped['value'].agg([np.mean, np.std])
deduped_df_timepoint.reset_index(inplace=True)
deduped_df_timepoint = deduped_df_timepoint.merge(harmonized_metadata.drop('fov', axis=1).drop_duplicates(), on='Tissue_ID')

# save timepoint df
deduped_df_timepoint.to_csv(os.path.join(data_dir, 'functional_df_per_timepoint_filtered_deduped.csv'), index=False)

