import os

import pandas as pd
import numpy as np

from ark.utils.misc_utils import verify_same_elements, verify_in_list


#
# This file creates plotting-ready data structures for cell prevalance and functional markers
#

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'


def create_long_df_by_cluster(cell_table, result_name, col_name, normalize=False):
    """Creats a dataframe summarizing cell clusters across FOVs in long format

    Args:
        cell_table: the dataframe containing information on each cell
        result_name: the name to give the column which will contain the summarized information
        col_name: the name of the column in the cell_table to summarize cell clusters from
        normalize: whether to report the total or normalized counts in the result

    Returns:
        pd.DataFrame: long format dataframe containing the summarized data"""

    # create 2D summary table
    crosstab = pd.crosstab(index=cell_table['fov'], rownames=['fov'],
                           columns=cell_table[col_name], normalize=normalize)

    # convert to long format
    crosstab['fov'] = crosstab.index
    long_df = pd.melt(crosstab, id_vars=['fov'], var_name='cell_type')
    long_df['metric'] = result_name

    return long_df


def create_long_df_by_functional(func_table, cell_type_col, drop_cols,
                                 transform_func, result_name):
    """Function to summarize functional marker data by cell type"""

    verify_in_list(cell_type_col=cell_type_col, cell_table_columns=func_table.columns)
    verify_in_list(drop_cols=drop_cols, cell_table_columns=func_table.columns)

    # drop columns from table
    func_table_small = func_table.loc[:, ~func_table.columns.isin(drop_cols)]

    # group by specified columns
    grouped_table = func_table_small.groupby(['fov', cell_type_col])
    transformed = grouped_table.agg(transform_func)
    transformed.reset_index(inplace=True)

    # reshape to long df
    long_df = pd.melt(transformed, id_vars=['fov', cell_type_col], var_name='functional_marker')
    long_df['metric'] = result_name
    long_df = long_df.rename(columns={cell_type_col: 'cell_type'})

    return long_df


#
# Preprocess metadata to ensure all samples are present
#

# load relevant tables
core_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
timepoint_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint.csv'))
cell_table_clusters = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_clusters_only.csv'))


# handle NAs in Tissue_ID
core_missing = core_metadata.Tissue_ID.isnull()
imaged_cores = core_metadata.MIBI_data_generated
np.sum(np.logical_and(core_missing, imaged_cores))

# all of the missing cores were not imaged, can be dropped
core_metadata = core_metadata.loc[~core_missing, :]

# check for FOVs present in imaged data that aren't in core metadata
missing_fovs = cell_table_clusters.loc[~cell_table_clusters.fov.isin(core_metadata.fov), 'fov'].unique()
cell_table_clusters = cell_table_clusters.loc[~cell_table_clusters.fov.isin(missing_fovs), :]

# check timepoints
timepoint_missing = ~core_metadata.Tissue_ID.isin(timepoint_metadata.Tissue_ID)
timepoint_missing = core_metadata.Tissue_ID[timepoint_missing].unique()
print(timepoint_missing)

# get metadata on missing cores
core_metadata.loc[core_metadata.Tissue_ID.isin(timepoint_missing[3:]), :]

# remove missing cores
core_metadata = core_metadata.loc[~core_metadata.Tissue_ID.isin(timepoint_missing), :]

# subset for required columns to append
timepoint_metadata = timepoint_metadata.loc[:, ['Tissue_ID', 'TONIC_ID', 'Timepoint', 'Localization']]
core_metadata = core_metadata.loc[:, ['fov', 'Tissue_ID']]


# TODO: Create conserved metadata sheet that contains core metadata across all formats
#
# Generate counts and proportions of cell clusters per FOV
#


# proportion of cells in cell_cluster_broad per patient
cluster_broad_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                             result_name='cluster_broad_freq',
                                             col_name='cell_cluster_broad',
                                             normalize='index')

# proportion of cells in cell_cluster per patient
cluster_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                       result_name='cluster_freq',
                                       col_name='cell_cluster',
                                       normalize='index')


# proportion of cells in cell_meta_cluster per patient
cluster_meta_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                       result_name='meta_cluster_freq',
                                       col_name='cell_meta_cluster',
                                       normalize='index')

# proportion of T cell subsets
tcell_mask = cell_table_clusters['cell_cluster'].isin(['Treg', 'CD8T', 'CD4T', 'T_Other'])
tcell_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[tcell_mask, :],
                                     result_name='tcell_freq',
                                     col_name='cell_cluster',
                                     normalize='index')


# proportion of immune cell subsets
immune_mask = cell_table_clusters['cell_cluster_broad'].isin(['Mono_Mac', 'T',
                                                              'Granulocyte', 'NK', 'B'])
immune_mask_2 = cell_table_clusters.cell_cluster == 'Immune_Other'
immune_mask = np.logical_or(immune_mask, immune_mask_2)
immune_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[immune_mask, :],
                                      result_name='immune_freq',
                                      col_name='cell_cluster',
                                      normalize='index')

# proportion of stromal subsets
stromal_mask = cell_table_clusters['cell_cluster_broad'].isin(['Stroma'])
stromal_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[stromal_mask, :],
                                       result_name='stromal_freq',
                                       col_name='cell_meta_cluster',
                                       normalize='index')

# proportion of cancer subsets
cancer_mask = cell_table_clusters['cell_cluster_broad'].isin(['Cancer'])
cancer_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[cancer_mask, :],
                                      result_name='cancer_freq',
                                      col_name='cell_meta_cluster',
                                      normalize='index')


# create single df with appropriate metadata
total_df = pd.concat([cluster_broad_df, cluster_df, cluster_meta_df, tcell_df, immune_df,
                      stromal_df, cancer_df], axis=0)

# check that all metadata from core_metadata succesfully transferred over
total_df = total_df.merge(core_metadata, on='fov', how='left')
assert np.sum(total_df.Tissue_ID.isnull()) == 0

bad_metadata = total_df.loc[total_df.Tissue_ID.isnull(), 'fov'].unique()

# check that all metadata from timepoint metadata succesfully transferred over
total_df = total_df.merge(timepoint_metadata, on='Tissue_ID', how='inner')
assert np.sum(total_df.TONIC_ID.isnull()) == 0

# save annotated cluster counts
total_df.to_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'), index=False)

# create version aggregated by timepoint
total_df_grouped = total_df.groupby(['Tissue_ID', 'cell_type', 'metric'])
total_df_timepoint = total_df_grouped['value'].agg([np.mean, np.std])
total_df_timepoint.reset_index(inplace=True)
total_df_timepoint = total_df_timepoint.merge(timepoint_metadata, on='Tissue_ID')

# save timepoint df
total_df_timepoint.to_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'), index=False)


#
# Create summary dataframe with proportions and counts of different functional marker populations
#

# load processed functional table
cell_table_func = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'))


# TODO: Update names to match naming scheme of cell frequencies
# Total number of cells positive for each functional marker in cell_cluster_broad per image
func_df_counts_broad = create_long_df_by_functional(func_table=cell_table_func,
                                                    cell_type_col='cell_cluster_broad',
                                                    drop_cols=['cell_meta_cluster', 'cell_cluster'],
                                                    transform_func=np.sum,
                                                    result_name='counts_per_cluster_broad')

# Proportion of cells positive for each functional marker in cell_cluster_broad per image
func_df_mean_broad = create_long_df_by_functional(func_table=cell_table_func,
                                                  cell_type_col='cell_cluster_broad',
                                                  drop_cols=['cell_meta_cluster', 'cell_cluster'],
                                                  transform_func=np.mean,
                                                  result_name='avg_per_cluster_broad')

# Total number of cells positive for each functional marker in cell_cluster per image
func_df_counts_cluster = create_long_df_by_functional(func_table=cell_table_func,
                                                      cell_type_col='cell_cluster',
                                                      drop_cols=['cell_meta_cluster',
                                                                 'cell_cluster_broad'],
                                                      transform_func=np.sum,
                                                      result_name='counts_per_cluster')

# Proportion of cells positive for each functional marker in cell_cluster_broad per image
func_df_mean_cluster = create_long_df_by_functional(func_table=cell_table_func,
                                                    cell_type_col='cell_cluster',
                                                    drop_cols=['cell_meta_cluster',
                                                               'cell_cluster_broad'],
                                                    transform_func=np.mean,
                                                    result_name='avg_per_cluster')

# Total number of cells positive for each functional marker in cell_meta_cluster per image
func_df_counts_meta = create_long_df_by_functional(func_table=cell_table_func,
                                                   cell_type_col='cell_meta_cluster',
                                                   drop_cols=['cell_cluster', 'cell_cluster_broad'],
                                                   transform_func=np.sum,
                                                   result_name='counts_per_meta')

# Proportion of cells positive for each functional marker in cell_meta_cluster per image
func_df_mean_meta = create_long_df_by_functional(func_table=cell_table_func,
                                                 cell_type_col='cell_meta_cluster',
                                                 drop_cols=['cell_cluster', 'cell_cluster_broad'],
                                                 transform_func=np.mean, result_name='avg_per_meta')

# Ratio of histone markers in cell_cluster_broad per image


# create combined df
total_df_func = pd.concat([func_df_counts_broad, func_df_mean_broad, func_df_counts_cluster,
                           func_df_mean_cluster, func_df_counts_meta, func_df_mean_meta])

# check that all metadata from core_metadata succesfully transferred over
total_df_func = total_df_func.merge(core_metadata, on='fov', how='left')
assert np.sum(total_df_func.Tissue_ID.isnull()) == 0

bad_metadata = total_df_func.loc[total_df_func.Tissue_ID.isnull(), 'fov'].unique()

# check that all metadata from timepoint metadata succesfully transferred over
total_df_func = total_df_func.merge(timepoint_metadata, on='Tissue_ID', how='left')
assert np.sum(total_df_func.TONIC_ID.isnull()) == 0

# save combined df
total_df_func.to_csv(os.path.join(data_dir, 'functional_df_per_core.csv'), index=False)


# create version aggregated by timepoint
total_df_grouped_func = total_df_func.groupby(['Tissue_ID', 'cell_type', 'functional_marker', 'metric'])
total_df_timepoint_func = total_df_grouped_func['value'].agg([np.mean, np.std])
total_df_timepoint_func.reset_index(inplace=True)
total_df_timepoint_func = total_df_timepoint_func.merge(timepoint_metadata, on='Tissue_ID')

# save timepoint df
total_df_timepoint_func.to_csv(os.path.join(data_dir, 'functional_df_per_timepoint.csv'), index=False)


