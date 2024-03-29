import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

#
# This script is for clustering patients across multiple FOV-level metrics to understand
# patterns of change
#

data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

fov_data_df = pd.read_csv(os.path.join(data_dir, 'fov_features_no_compartment.csv'))
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
fov_data_df = fov_data_df.merge(harmonized_metadata[['fov', 'Timepoint']].drop_duplicates(), on='fov', how='left')

# plot clustermap


# determine which timepoints to use
include_timepoints = ['primary_untreated', 'baseline', 'on_nivo', 'post_induction']
#include_timepoints = fov_data_df.Timepoint.unique()
fov_data_df_subset = fov_data_df[fov_data_df.Timepoint.isin(include_timepoints)]


data_wide = fov_data_df_subset.pivot(index='fov', columns='feature_name', values='normalized_value')


# replace Nan with 0
data_wide = data_wide.fillna(0)

# drop columns with a sum of zero
data_wide = data_wide.loc[:, (data_wide != 0).any(axis=0)]

# check for invalid values
data_wide = data_wide.replace([np.inf, -np.inf], np.nan)



clustergrid = sns.clustermap(data_wide, z_score=1, cmap='vlag', vmin=-3, vmax=3, figsize=(20, 20))
clustergrid.savefig(os.path.join(plot_dir, 'patient_by_feature_clustermap_primary_only_fov.png'), dpi=300)
plt.close()

# create correlation matrix using spearman correlation
corr_df = data_wide.corr(method='spearman')
clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))

clustergrid_columns = clustergrid.data2d.columns
clustergrid2 = sns.clustermap(corr_df.loc[clustergrid_columns[305:350], clustergrid_columns[305:350]], cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(plot_dir, 'spearman_correlation_clustermap.png'), dpi=300)
plt.close()


# subset the df based on high variance features
colvals = corr_df.apply(np.var, axis=0)

# subset based on the columns
col_cutoff = colvals.quantile(0.50)
keep_mask = colvals > col_cutoff
corr_df_subset = corr_df.loc[keep_mask, keep_mask]


# plot heatmap
clustergrid = sns.clustermap(corr_df_subset, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(plot_dir, 'spearman_correlation_heatmap_primary_subset_50.png'), dpi=300)
plt.close()


# plot correlations between features
keep_cols = corr_df_subset.columns[clustergrid.dendrogram_row.reordered_ind[82:90]]

plot_df = data_wide.loc[:, keep_cols]
g = sns.PairGrid(plot_df, diag_sharey=False)
g.map_lower(sns.regplot, scatter_kws={'s': 10, 'alpha': 0.5})
g.savefig(os.path.join(plot_dir, 'spearman_feature_paired_corelations_GLUT1.png'), dpi=300)
plt.close()

# create PCA of features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generate_pca(data, num_pcs):
    # scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # create PCA object
    pca = PCA(n_components=num_pcs)

    # fit PCA
    pca.fit(scaled_data)

    # transform data
    pca_data = pca.transform(scaled_data)

    return pca_data

# replace Nans with mean
data_wide_filled = data_wide.fillna(data_wide.mean())

# scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_wide_filled)

# create PCA object
pca = PCA(n_components=10)

# fit PCA
pca.fit(scaled_data)

# get explained variance
pca.explained_variance_ratio_

# transform data
pca_data = pca.transform(scaled_data)

pca_data_df = pd.DataFrame(pca_data, index=data_wide.index,
                            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])

pca_data_df = pca_data_df.melt(value_vars=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10'],
                                 var_name='PC', value_name='value', ignore_index=False).reset_index()
pca_data_df = pca_data_df.merge(harmonized_metadata[['Tissue_ID', 'Patient_ID', 'Timepoint', 'fov']], on='fov', how='left')
pca_data_df = pca_data_df.rename(columns={'PC': 'feature_name'})
pca_data_df.to_csv(os.path.join(data_dir, 'pca_data_df.csv'), index=False)

pca_data_df_grouped = pca_data_df.groupby(['Tissue_ID', 'feature_name', 'Timepoint', 'Patient_ID'])
pca_data_df_grouped = pca_data_df_grouped['value'].agg([np.mean, np.std])
pca_data_df_grouped = pca_data_df_grouped.reset_index()
pca_data_df_grouped.to_csv(os.path.join(data_dir, 'pca_data_df_grouped.csv'), index=False)
# get PCA loadings and create dataframe
pca_loadings = pd.DataFrame(pca.components_.T, index=data_wide.columns,
                            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                     'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])


# plot rows based on first two PCs
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1])


# select which PCs to plot
num_pcs = 4
pca_loadings_subset = pca_loadings.iloc[:, :num_pcs]

# set threshold for which features to include based on 75% quantile across all columns at once
pc_cutoff = pca_loadings_subset.abs().unstack().quantile(0.75)

# TODO: instead of looking at maximum 75%, look at difference between the features

# create mask for which features to include
pc_mask = pca_loadings_subset.abs() > pc_cutoff
keep_rows = pc_mask.any(axis=1)

# subset based on mask
pca_loadings_subset = pca_loadings_subset.loc[keep_rows, :]

# plot heatmap of loadings
clustergrid = sns.clustermap(pca_loadings_subset, cmap='vlag', figsize=(20, 20))
clustergrid.savefig(os.path.join(plot_dir, 'pca_loadings_heatmap.png'), dpi=300)
plt.close()

# get names of relevant features
relevant_features = pca_loadings_subset.index[clustergrid.dendrogram_row.reordered_ind[-10:]]
plotting_df = pd.DataFrame(scaled_data, columns=data_wide.columns, index=data_wide.index)
plotting_df = plotting_df.loc[:, relevant_features]

# get FOVs with highest PC1 value
pc2_cutoff = np.quantile(pca_data[:, 1], 0.95)
pca2_mask = pca_data[:, 1] > pc2_cutoff

plotting_df = plotting_df.loc[pca2_mask, :]

# plot values
clustergrid = sns.clustermap(plotting_df, cmap='vlag', vmin=-4, vmax=4, figsize=(20, 20))
clustergrid.savefig(os.path.join(plot_dir, 'high_PC2_FOVs.png'), dpi=300)
plt.close()


# NMF clustering of features
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler

# replace Nans with mean
data_wide_filled = data_wide.fillna(data_wide.mean())

# scale data
data_wide_filled = data_wide_filled.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=0)

# create NMF object
nmf = NMF(n_components=10)

# fit NMF
nmf.fit(data_wide_filled)

# transform data
nmf_data = nmf.transform(data_wide_filled)

nmf_data_df = pd.DataFrame(nmf_data, index=data_wide.index,
                            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])

sns.clustermap(nmf_data_df, cmap='Reds', figsize=(20, 20), vmin=0, vmax=1)

nmf_data_df = nmf_data_df.melt(value_vars=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                        'PC6', 'PC7', 'PC8', 'PC9', 'PC10'],
                                    var_name='feature_name', value_name='value', ignore_index=False).reset_index()
nmf_data_df = nmf_data_df.merge(harmonized_metadata[['Tissue_ID', 'Patient_ID', 'Timepoint', 'fov']], on='fov', how='left')
nmf_data_df.to_csv(os.path.join(data_dir, 'nmf_data_df.csv'), index=False)

nmf_data_df_grouped = nmf_data_df.groupby(['Tissue_ID', 'feature_name', 'Timepoint', 'Patient_ID'])
nmf_data_df_grouped = nmf_data_df_grouped['value'].agg([np.mean, np.std])
nmf_data_df_grouped = nmf_data_df_grouped.reset_index()
nmf_data_df_grouped.to_csv(os.path.join(data_dir, 'nmf_data_df_grouped.csv'), index=False)



# look at correlation between selected subsets of features
# plot correlations between features
keep_cols = ['Cancer_CD56_meta_cluster_prop','Cancer_CK17_meta_cluster_prop',
             'Cancer_Ecad_meta_cluster_prop', 'Cancer_Mono_meta_cluster_prop',
             'Cancer_Other_meta_cluster_prop',
             'Cancer_SMA_meta_cluster_prop','Cancer_Vim_meta_cluster_prop']

keep_cols = ['VIM_meta_cluster_prop', 'Stroma_Fibronectin_meta_cluster_prop',
             'Stroma_Collagen_meta_cluster_prop', 'SMA_meta_cluster_prop',
             'FAP_meta_cluster_prop', 'FAP_SMA_meta_cluster_prop',
             'CD31_meta_cluster_prop', 'CD31_VIM_meta_cluster_prop']

keep_cols = ['APC_cluster_prop', 'B_cluster_prop', 'CD4T_cluster_prop', 'CD8T_cluster_prop',
             'Endothelium_cluster_prop', 'Fibroblast_cluster_prop', 'Immune_Other_cluster_prop',
             'M1_Mac_cluster_prop', 'M2_Mac_cluster_prop', 'Mac_Other_cluster_prop', 'Mast_cluster_prop',
             'Monocyte_cluster_prop', 'NK_cluster_prop', 'Neutrophil_cluster_prop', 'Stroma_cluster_prop',
             'T_Other_cluster_prop', 'Treg_cluster_prop']

plot_df = data_wide.loc[:, keep_cols]
plot_df.columns = [name.split('_cluster')[0] for name in plot_df.columns]
g = sns.PairGrid(plot_df, diag_sharey=False)
g.map_lower(sns.regplot, scatter_kws={'s': 10, 'alpha': 0.5})
g.savefig(os.path.join(plot_dir, 'cancer_subtype_correlations.png'), dpi=300)
plt.close()


pca_data = generate_pca(plot_df, num_pcs=2)

# append to original dataframe for plotting
plot_df['PC1'] = pca_data[:, 0]
plot_df['PC2'] = pca_data[:, 1]
plot_df['Cancer_cat'] = plot_df.iloc[:, :-2].apply(lambda x: plot_df.columns.values[np.argmax(x)], axis=1)


# plot rows based on PCs
sns.scatterplot(x='PC1', y='PC2', hue='Fibroblast', data=plot_df)
plt.savefig(os.path.join(plot_dir, 'immune_subtype_pca_by_NK.png'), dpi=300)

sns.scatterplot(x='PC1', y='PC2', data=plot_df, hue='Cancer_cat', palette='Set1')
plt.savefig(os.path.join(plot_dir, 'fibroblast_subtype_pca_by_max.png'), dpi=300)
plt.close()

pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['fov'] = data_wide.index

pca_df = pd.melt(pca_df, id_vars='fov', value_vars=['PC1', 'PC2'])

pca_df.rename(columns={'variable': 'cell_type'}, inplace=True)
pca_df['metric'] = 'immune_PCA'

metadata_df = fov_data_df[['fov', 'Tissue_ID', 'Timepoint']]
metadata_df = metadata_df.drop_duplicates()
pca_df = pca_df.merge(metadata_df, on='fov')


