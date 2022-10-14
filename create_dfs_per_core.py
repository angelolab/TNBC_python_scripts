import os

import pandas as pd
import numpy as np

from ark.utils.misc_utils import verify_same_elements, verify_in_list


#
# This file creates plotting-ready data structures for cell prevalance and functional markers
#

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'


def create_long_df_by_cluster(cell_table, result_name, col_name, metadata_df, metadata_cols,
                              normalize=False):
    """Creats a dataframe summarizing cell clusters across FOVs in long format

    Args:
        cell_table: the dataframe containing information on each cell
        result_name: the name to give the column which will contain the summarized information
        col_name: the name of the column in the cell_table to summarize cell clusters from
        metadata_df: dataframe containing metadata about each FOV
        metadata_cols: which columns from the metadata_df to include in the output table
        normalize: whether to report the total or normalized counts in the result

    Returns:
        pd.DataFrame: long format dataframe containing the summarized data"""

    # create 2D summary table
    crosstab = pd.crosstab(index=cell_table['fov'], rownames=['fov'],
                           columns=cell_table[col_name], normalize=normalize)

    # determine if any FOVs are missing from table as a result of zero counts
    crosstab['fov'] = crosstab.index
    missing_mask = ~metadata_df.fov.array.isin(crosstab.index)
    if np.any(missing_mask):
        print("the following FOVs are missing and will be dropped: {}".format(metadata_df.fov.array[missing_mask]))
        metadata_df = metadata_df.loc[~missing_mask, :]

    verify_same_elements(metadata_fovs=metadata_df.fov.array, data_fovs=crosstab.fov.array)
    crosstab = crosstab.reindex(metadata_df.fov.array)

    # append metadata
    for col in metadata_cols:
        crosstab[col] = metadata_df[col].array

    # convert to long format
    long_df = pd.melt(crosstab, id_vars=['fov'] + metadata_cols, var_name='cell_type')
    long_df['metric'] = result_name

    return long_df


def create_long_df_by_functional(func_table, cell_type_col, metadata_df, metadata_cols,
                                 drop_cols, transform_func, result_name):
    """Function to summarize functional marker data by cell type"""

    verify_in_list(cell_type_col=cell_type_col, cell_table_columns=func_table.columns)
    verify_in_list(drop_cols=drop_cols, cell_table_columns=func_table.columns)

    # drop columns from table
    func_table_small = func_table.loc[:, ~func_table.columns.isin(drop_cols)]

    # group by specified columns
    grouped_table = func_table_small.groupby(['fov', cell_type_col])
    transformed = grouped_table.agg(transform_func)
    transformed.reset_index(inplace=True)

    # add metadata
    verify_same_elements(functional_fovs=transformed.fov.array,
                         metadata_fovs=metadata_df.fov.array)
    transformed = transformed.merge(metadata_df[['fov'] + metadata_cols], on='fov')

    # reshape to long df
    long_df = pd.melt(transformed, id_vars=['fov', cell_type_col] + metadata_cols,
                      var_name='functional_marker')
    long_df['metric'] = result_name
    long_df = long_df.rename(columns={cell_type_col: 'cell_type'})

    return long_df


#
# Create summary dataframe with proportions and counts of different cell clusters per core
#

# load relevant tables
core_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
core_metadata = core_metadata.loc[core_metadata.MIBI_data_generated, :]
cell_table_clusters = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_clusters_only.csv'))

# subset for testing
# cell_table = cell_table.loc[cell_table.fov.isin(core_df.fov.array[:100]), :]
# core_df = core_df.loc[core_df.fov.isin(cell_table.fov.unique()), :]


# proportion of cells in cell_cluster_broad per patient
cluster_broad_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                             result_name='cluster_broad_freq',
                                             col_name='cell_cluster_broad',
                                             metadata_df=core_metadata,
                                             metadata_cols=['Tissue_ID'], normalize='index')

# proportion of cells in cell_cluster per patient
cluster_df = create_long_df_by_cluster(cell_table=cell_table_clusters,
                                       result_name='cluster_freq',
                                       col_name='cell_cluster',
                                       metadata_df=core_metadata,
                                       metadata_cols=['Tissue_ID'], normalize='index')

# proportion of T cell subsets
tcell_mask = cell_table_clusters['cell_cluster'].isin(['treg', 'CD8', 'CD4', 't_other'])
tcell_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[tcell_mask, :],
                                     result_name='tcell_freq',
                                     col_name='cell_cluster', metadata_df=core_metadata,
                                     metadata_cols=['Tissue_ID'], normalize='index')


# proportion of immune cell subsets
immune_mask = cell_table_clusters['cell_cluster_broad'].isin(['mono_macs', 't_cell', 'other',
                                                              'granulocyte', 'nk', 'b_cell'])
immune_df = create_long_df_by_cluster(cell_table=cell_table_clusters.loc[immune_mask, :],
                                      result_name='immune_freq',
                                      col_name='cell_cluster', metadata_df=core_metadata,
                                      metadata_cols=['Tissue_ID'], normalize='index')

# save total df
total_df = pd.concat([cluster_broad_df, cluster_df, tcell_df, immune_df])
total_df.to_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'), index=False)

# create version aggregated by timepoint
total_df_grouped = total_df.groupby(['Tissue_ID', 'cell_type', 'metric'])
total_df_timepoint = total_df_grouped['value'].agg([np.mean, np.std])
total_df_timepoint.reset_index(inplace=True)

# save timepoint df
total_df_timepoint.to_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'), index=False)


#
# Create summary dataframe with proportions and counts of different functional marker populations
#

# preprocess functional marker df to have specific combinations of markers
cell_table_func = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'))

# define marker combinations of interest
combinations = [['PD1', 'TCF1'], ['PD1', 'TIM3']]

# TODO: Make this compatible with negative gating in addition to positive gating

for combo in combinations:
    first_marker = combo[0]
    base_mask = cell_table_func[first_marker].array
    for marker in combo[1:]:
        base_mask = np.logical_and(base_mask, cell_table_func[marker].array)
    name = '_'.join(combo)
    cell_table_func[name] = base_mask

cell_table_func.to_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'), index=False)

# load processed functional table
cell_table_func = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'))


# Total number of cells positive for each functional marker in cell_cluster_broad per image
func_df_counts_broad = create_long_df_by_functional(func_table=cell_table_func,
                                                    cell_type_col='cell_cluster_broad',
                                                    metadata_df=core_metadata,
                                                    metadata_cols=['Tissue_ID'],
                                                    drop_cols=['cell_meta_cluster', 'cell_cluster'],
                                                    transform_func=np.sum,
                                                    result_name='counts_per_cluster_broad')

# Proportion of cells positive for each functional marker in cell_cluster_broad per image
func_df_mean_broad = create_long_df_by_functional(func_table=cell_table_func,
                                                  cell_type_col='cell_cluster_broad',
                                                  metadata_df=core_metadata,
                                                  metadata_cols=['Tissue_ID'],
                                                  drop_cols=['cell_meta_cluster', 'cell_cluster'],
                                                  transform_func=np.mean,
                                                  result_name='avg_per_cluster_broad')

# Total number of cells positive for each functional marker in cell_cluster per image
func_df_counts_cluster = create_long_df_by_functional(func_table=cell_table_func,
                                                      cell_type_col='cell_cluster',
                                                      metadata_df=core_metadata,
                                                      metadata_cols=['Tissue_ID'],
                                                      drop_cols=['cell_meta_cluster',
                                                                 'cell_cluster_broad'],
                                                      transform_func=np.sum,
                                                      result_name='counts_per_cluster')

# Proportion of cells positive for each functional marker in cell_cluster_broad per image
func_df_mean_cluster = create_long_df_by_functional(func_table=cell_table_func,
                                                    cell_type_col='cell_cluster',
                                                    metadata_df=core_metadata,
                                                    metadata_cols=['Tissue_ID'],
                                                    drop_cols=['cell_meta_cluster',
                                                               'cell_cluster_broad'],
                                                    transform_func=np.mean,
                                                    result_name='avg_per_cluster')

# Total number of cells positive for each functional marker in cell_meta_cluster per image
func_df_counts_meta = create_long_df_by_functional(func_table=cell_table_func,
                                                   cell_type_col='cell_meta_cluster',
                                                   metadata_df=core_metadata,
                                                   metadata_cols=['Tissue_ID'],
                                                   drop_cols=['cell_cluster', 'cell_cluster_broad'],
                                                   transform_func=np.sum,
                                                   result_name='counts_per_meta')

# Proportion of cells positive for each functional marker in cell_meta_cluster per image
func_df_mean_meta = create_long_df_by_functional(func_table=cell_table_func,
                                                 cell_type_col='cell_meta_cluster',
                                                 metadata_df=core_metadata,
                                                 metadata_cols=['Tissue_ID'],
                                                 drop_cols=['cell_cluster', 'cell_cluster_broad'],
                                                 transform_func=np.mean, result_name='avg_per_meta')

# save total df
total_df_func = pd.concat([func_df_counts_broad, func_df_mean_broad, func_df_counts_cluster,
                           func_df_mean_cluster, func_df_counts_meta, func_df_mean_meta])


total_df_func.to_csv(os.path.join(data_dir, 'functional_df_per_core.csv'), index=False)


# create version aggregated by timepoint
total_df_grouped_func = total_df_func.groupby(['Tissue_ID', 'cell_type', 'functional_marker', 'metric'])
total_df_timepoint_func = total_df_grouped_func['value'].agg([np.mean, np.std])
total_df_timepoint_func.reset_index(inplace=True)

# save timepoint df
total_df_timepoint_func.to_csv(os.path.join(data_dir, 'functional_df_per_timepoint.csv'), index=False)


