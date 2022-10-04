import os

import ark.utils.misc_utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ark.utils.io_utils import list_folders
from ark.utils.misc_utils import verify_same_elements

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


# create shortened cell table with only relevant columns
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated.csv'))

cell_table = cell_table.loc[:, ['fov', 'cell_meta_cluster', 'label', 'cell_cluster',
                             'cell_cluster_broad']]
cell_table.to_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_freqs.csv'),
                  index=False)


# add column denoting which images were acquired
all_fovs = list_folders('/Volumes/Big_Boy/TONIC_Cohort/image_data/samples')
core_df['MIBI_data'] = core_df['fov'].isin(all_fovs)
core_df.to_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))

# load relevant tables
core_df = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_freqs.csv'))

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
