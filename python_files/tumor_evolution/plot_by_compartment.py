import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dataset
core_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core_filtered.csv'))

plot_df = core_df_cluster.loc[core_df_cluster.subset.isin(['cancer_core', 'cancer_border',
                                                          'stroma_core', 'stroma_border', 'all'])]
#plot_df = plot_df.loc[plot_df.Timepoint == 'primary_untreated']
plot_df = plot_df.loc[plot_df.metric == 'cluster_density']

wide_df = plot_df[['fov', 'cell_type', 'value', 'subset']].pivot(index=['fov', 'cell_type'], columns='subset', values='value')
wide_df = wide_df.loc[wide_df['all'] > 0]
#wide_df = wide_df.divide(wide_df['all'], axis=0)
wide_df = wide_df.drop(columns=['all'])
wide_df = wide_df.divide(wide_df.sum(axis=1), axis=0)
wide_df = wide_df.reset_index()

wide_df_grouped = wide_df.groupby('cell_type').agg(np.mean)

# plot
sns.clustermap(wide_df_grouped, cmap='Reds', vmin=0, vmax=1, figsize=(10, 10))
plt.savefig(os.path.join(plot_dir, 'compartment_density_by_cell_type.png'))
plt.close()


# cluster the functional data
plot_df = core_df_func.loc[core_df_func.subset.isin(['cancer_core', 'cancer_border',
                                                            'stroma_core', 'stroma_border', 'all'])]
#plot_df = plot_df.loc[plot_df.Timepoint == 'primary_untreated']
plot_df = plot_df.loc[plot_df.metric == 'cluster_freq']

wide_df = plot_df[['fov', 'cell_type', 'value', 'subset', 'functional_marker']].pivot(index=['fov', 'cell_type', 'functional_marker'], columns='subset', values='value')
wide_df = wide_df.loc[wide_df['all'] > 0]
#wide_df = wide_df.divide(wide_df['all'], axis=0)
wide_df = wide_df.drop(columns=['all'])
#wide_df = wide_df.divide(wide_df.sum(axis=1), axis=0)
wide_df = wide_df.reset_index()

wide_df_grouped = wide_df.groupby(['cell_type', 'functional_marker']).agg(np.mean)

#remove nans
wide_df_grouped = wide_df_grouped.dropna()

# normalize
wide_df_grouped = wide_df_grouped.divide(wide_df_grouped.sum(axis=1), axis=0)

# subset based on the columns
colvals = wide_df_grouped.apply(np.var, axis=1)
col_cutoff = colvals.quantile(0.50)
keep_mask = colvals > col_cutoff
wide_df_grouped = wide_df_grouped.loc[keep_mask, :]


# plot
sns.clustermap(wide_df_grouped, cmap='Reds', vmin=0, vmax=1, figsize=(10, 10))
plt.savefig(os.path.join(plot_dir, 'compartment_functional_marker_positivity_by_cell_type.png'))
plt.close()

# TODO: figure out what normalization scheme makes the most sense
