import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dataset
core_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core.csv'))

plot_df = core_df_cluster.loc[core_df_cluster.subset.isin(['cancer_core', 'cancer_border',
                                                          'stroma_core', 'stroma_border', 'all'])]
plot_df = plot_df.loc[plot_df.Timepoint == 'primary_untreated']
plot_df = plot_df.loc[plot_df.metric == 'cluster_density']

wide_df = plot_df[['fov', 'cell_type', 'value', 'subset']].pivot(index=['fov', 'cell_type'], columns='subset', values='value')
wide_df = wide_df.loc[wide_df['all'] > 0]
wide_df = wide_df.divide(wide_df['all'], axis=0)
wide_df = wide_df.drop(columns=['all'])
wide_df = wide_df.reset_index()

wide_df_grouped = wide_df.groupby('cell_type').agg(np.mean)

# plot
sns.clustermap(wide_df_grouped, cmap='Reds', vmin=0, vmax=3, figsize=(10, 10))
plt.savefig(os.path.join(plot_dir, 'compartment_density_by_cell_type.png'))
plt.close()


# cluster the functional data
plot_df = core_df_func.loc[core_df_func.subset.isin(['cancer_core', 'cancer_border',
                                                            'stroma_core', 'stroma_border', 'all'])]
plot_df = plot_df.loc[plot_df.Timepoint == 'primary_untreated']
plot_df = plot_df.loc[plot_df.metric == 'cluster_broad_freq']

lymphocyte = ['B', 'CD4T', 'CD8T', 'Immune_Other', 'NK', 'T_Other', 'Treg']
cancer = ['Cancer', 'Cancer_EMT', 'Cancer_Other']
monocyte = ['APC', 'M1_Mac', 'M2_Mac', 'Mono_Mac', 'Monocyte', 'Mac_Other']
stroma = ['Fibroblast', 'Stroma', 'Endothelium']
granulocyte = ['Mast', 'Neutrophil']

keep_dict = {'CD38': ['B', 'Immune_other', 'NK', 'Endothelium'], 'CD45RB': lymphocyte, 'CD45RO': lymphocyte,
             'CD57': lymphocyte + cancer, 'CD69': lymphocyte,
             'GLUT1': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'HLA1': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'HLADR': lymphocyte + monocyte, 'IDO': ['APC', 'B'], 'Ki67': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'LAG3': ['B'], 'PD1': lymphocyte, 'PDL1_combined': lymphocyte + monocyte + granulocyte + cancer,
             'TBET': lymphocyte, 'TCF1': lymphocyte, 'TIM3': lymphocyte + monocyte + granulocyte}


keep_vector = np.zeros(plot_df.shape[0], dtype=bool)
for marker in keep_dict:
    marker_keep = (plot_df['cell_type'].isin(keep_dict[marker])) & \
                  (plot_df['functional_marker'] == marker)
    keep_vector = keep_vector | marker_keep

plot_df = plot_df[keep_vector]

wide_df = plot_df[['fov', 'cell_type', 'value', 'subset', 'functional_marker']].pivot(index=['fov', 'cell_type', 'functional_marker'], columns='subset', values='value')
wide_df = wide_df.loc[wide_df['all'] > 0]
wide_df = wide_df.divide(wide_df['all'], axis=0)
wide_df = wide_df.drop(columns=['all'])
wide_df = wide_df.reset_index()

wide_df_grouped = wide_df.groupby(['cell_type', 'functional_marker']).agg(np.mean)

#remove nans
wide_df_grouped = wide_df_grouped.dropna()
colvals = wide_df_grouped.apply(np.var, axis=1)

# subset based on the columns
col_cutoff = colvals.quantile(0.50)
keep_mask = colvals > col_cutoff
wide_df_grouped = wide_df_grouped.loc[keep_mask, :]


# plot
sns.clustermap(wide_df_grouped, cmap='Reds', vmin=0, vmax=3, figsize=(10, 10))
plt.savefig(os.path.join(plot_dir, 'compartment_functional_marker_positivity_by_cell_type.png'))
plt.close()