import os

import seaborn as sns
import pandas as pd
import numpy as np

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'

# create shortened cell table with only relevant columns
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated.csv'))
cell_table = cell_table.loc[:, ['fov', 'cell_meta_cluster', 'label', 'cell_cluster',
                             'cell_cluster_broad']]
cell_table.to_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_freqs.csv'))

# load relevant tables
core_df = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_freqs.csv'))
fovs = cell_table.fov.array.unique()

# create wide format
for fov in cell_table.fov:
    subset_df = cell_table.loc[cell_table['fov'] == fov, :]

    # given a list of populations, compute the proportion of each population and append
    counts = subset_df.groupby('cell_cluster').size()
    props = counts / np.sum(counts)
    props_dict = dict(props)

    # for each population, append name and frequency to main table
    for key in props_dict:
        val = props_dict[key]
        name = key + '_prop_cell_cluster'
        core_df.loc[core_df.fov == fov, name] = val

core_df.to_csv(os.path.join(data_dir, 'TONIC_data_per_core_updated.csv'))


# create long format
fov = []
stat = []
category = []
value = []

cell_table_small = cell_table[:30000]

grouped_counts = cell_table_small.groupby(['fov', 'cell_cluster']).size().unstack(fill_value=0).stack()
sums = []
for i in range(len(cell_table_small.fov.unique())):
    sums.append(np.sum(grouped_counts[i * 19:(i + 1) * 19].array))

sums = np.repeat(sums, len(cell_table_small['cell_cluster'].unique()))
grouped_props = grouped_counts / sums

fovs = grouped_props.index.get_level_values(0)
category = grouped_props.index.get_level_values(1)
long_df = pd.DataFrame({'fov': fovs, 'category': category, 'value': grouped_props.array})

sns.catplot(data=long_df, x='category', y='value')

