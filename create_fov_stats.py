import os

import ark.utils.misc_utils
import seaborn as sns
import pandas as pd
import numpy as np

from ark.utils.io_utils import list_folders
from ark.utils.misc_utils import verify_same_elements


def create_summary_df(cell_table, result_name, col_name, metadata_df, metadata_cols,
                      normalize=False):
    # create 2D summary table
    crosstab = pd.crosstab(index=cell_table['fov'], rownames=['fov'],
                           columns=cell_table_small[col_name], normalize=normalize)

    # sort table in same order as metadata df
    crosstab['fov'] = crosstab.index
    verify_same_elements(metadata_fovs=metadata_df.fov.array, data_fovs=crosstab.fov.array)
    crosstab = crosstab.reindex(metadata_df.fov.array)

    # append metadata
    for col in metadata_cols:
        crosstab[col] = metadata_df[col].array

    # convert to long format
    long_df = pd.melt(crosstab, id_vars=['fov'] + metadata_cols, var_name='cell_type')
    long_df['metric'] = result_name

    return long_df


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'

# create shortened cell table with only relevant columns
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated.csv'))
cell_table = cell_table.loc[:, ['fov', 'cell_meta_cluster', 'label', 'cell_cluster',
                             'cell_cluster_broad']]
cell_table.to_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_freqs.csv'))


# add column denoting which images were acquired
all_fovs = list_folders('/Volumes/Big_Boy/TONIC_Cohort/image_data/samples')
core_df['MIBI_data'] = core_df['fov'].isin(all_fovs)
core_df.to_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))

# load relevant tables
core_df = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_freqs.csv'), index_col=[0])

# subset for testing
cell_table_small = cell_table.loc[:30000, :]
core_df_small = core_df.loc[core_df.fov.isin(cell_table_small.fov.unique()), :]


# create summary df with cell-level statistics
total_df = pd.DataFrame()

# proportion of cells in cell_cluster_broad per patient
cluster_broad_df = create_summary_df(cell_table=cell_table_small, result_name='cluster_broad_freq',
                                     col_name='cell_cluster_broad', metadata_df=core_df_small,
                                     metadata_cols=['Tissue_ID'], normalize='index')
total_df = pd.concat([total_df, cluster_broad_df])

# proportion of T cell subsets
tcell_mask = cell_table_small['cell_cluster'].isin(['treg', 'CD8', 'CD4', 't_other'])
tcell_df = create_summary_df(cell_table=cell_table_small.loc[tcell_mask, :],
                             result_name='tcell_freq',
                             col_name='cell_cluster', metadata_df=core_df_small,
                             metadata_cols=['Tissue_ID'], normalize='index')
total_df = pd.concat([total_df, tcell_df])

new_cross = pd.pivot(total_df, index='fov', columns='cell_type', values='value')
new_cross['fov'] = new_cross.index
new_cross.plot(x='fov', kind='bar', stacked=True)

sns.catplot(total_df.loc[total_df.metric == 'tcell_freq'], x='cell_type', y='value', hue='Tissue_ID')

sns.catplot(tcell_df, x='cell_type', y='value', hue='Tissue_ID')