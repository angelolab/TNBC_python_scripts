import os

import ark.utils.misc_utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ark.utils.io_utils import list_folders
from ark.utils.misc_utils import verify_same_elements, verify_in_list

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'


def create_summary_df(cell_table, result_name, col_name, metadata_df, metadata_cols,
                      normalize=False):
    """Creats a dataframe summarizing the supplied col_name across FOVs in long format

    Args:
        cell_table: the dataframe containing information on each cell
        result_name: the name to give the column which will contain the summarized information
        col_name: the name of the column in the cell_table to summarize from
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

# load relevant tables
core_df = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_clusters_only.csv'))

# subset for testing
# cell_table = cell_table.loc[cell_table.fov.isin(core_df.fov.array[:100]), :]
# core_df = core_df.loc[core_df.fov.isin(cell_table.fov.unique()), :]


# create summary df with cell-level statistics
total_df = pd.DataFrame()

# proportion of cells in cell_cluster_broad per patient
cluster_broad_df = create_summary_df(cell_table=cell_table, result_name='cluster_broad_freq',
                                     col_name='cell_cluster_broad', metadata_df=core_df,
                                     metadata_cols=['Tissue_ID'], normalize='index')
total_df = pd.concat([total_df, cluster_broad_df])

# proportion of T cell subsets
tcell_mask = cell_table['cell_cluster'].isin(['treg', 'CD8', 'CD4', 't_other'])
tcell_df = create_summary_df(cell_table=cell_table.loc[tcell_mask, :],
                             result_name='tcell_freq',
                             col_name='cell_cluster', metadata_df=core_df,
                             metadata_cols=['Tissue_ID'], normalize='index')
total_df = pd.concat([total_df, tcell_df])

# save total df
total_df.to_csv(os.path.join(data_dir, 'summary_df_core.csv'), index=False)

# create version aggregated by timepoint
total_df_grouped = total_df.groupby(['Tissue_ID', 'cell_type', 'metric'])
total_df_timepoint = total_df_grouped['value'].agg([np.mean, np.std])
total_df_timepoint.reset_index(inplace=True)

# save timepoint df
total_df_timepoint.to_csv(os.path.join(data_dir, 'summary_df_timepoint.csv'), index=False)


## functional marker annotation and manipulation
cell_table_func = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'))

# define marker combinations of interest
combinations = [['PD1', 'TCF1'], ['PD1', 'TIM3']]

for combo in combinations:
    first_marker = combo[0]
    base_mask = cell_table_func[first_marker].array
    for marker in combo[1:]:
        base_mask = np.logical_and(base_mask, cell_table_func[marker].array)
    name = '_'.join(combo)
    cell_table_func[name] = base_mask


# subset for testing
cell_table_small = cell_table_func.loc[cell_table_func.fov.isin(cell_table_func.fov.unique()[:3])]


def create_summary_df_functional(func_table, cell_type_col, drop_cols, transform_func, result_name):
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


# create summary stats for different granularity levels for functional markers
func_df_counts_broad = create_summary_df_functional(func_table=cell_table_func, cell_type_col='cell_cluster_broad',
                                       drop_cols=['cell_meta_cluster', 'cell_cluster'],
                                       transform_func=np.sum, result_name='counts_per_cluster_broad')

func_df_mean_broad = create_summary_df_functional(func_table=cell_table_func, cell_type_col='cell_cluster_broad',
                                       drop_cols=['cell_meta_cluster', 'cell_cluster'],
                                       transform_func=np.mean, result_name='avg_per_cluster_broad')

func_df_counts_cluster = create_summary_df_functional(func_table=cell_table_func, cell_type_col='cell_cluster',
                                       drop_cols=['cell_meta_cluster', 'cell_cluster_broad'],
                                       transform_func=np.sum, result_name='counts_per_cluster')

func_df_mean_cluster = create_summary_df_functional(func_table=cell_table_func, cell_type_col='cell_cluster',
                                       drop_cols=['cell_meta_cluster', 'cell_cluster_broad'],
                                       transform_func=np.mean, result_name='avg_per_cluster')

func_df_counts_meta = create_summary_df_functional(func_table=cell_table_func, cell_type_col='cell_meta_cluster',
                                       drop_cols=['cell_cluster', 'cell_cluster_broad'],
                                       transform_func=np.sum, result_name='counts_per_meta')

func_df_mean_meta = create_summary_df_functional(func_table=cell_table_func, cell_type_col='cell_meta_cluster',
                                       drop_cols=['cell_cluster', 'cell_cluster_broad'],
                                       transform_func=np.mean, result_name='avg_per_meta')

# combine together into single df
total_df_func = pd.concat([func_df_counts_broad, func_df_mean_broad, func_df_counts_cluster,
                           func_df_mean_cluster, func_df_counts_meta, func_df_mean_meta])


total_df_func.to_csv(os.path.join(data_dir, 'functional_df_core.csv'), index=False)




