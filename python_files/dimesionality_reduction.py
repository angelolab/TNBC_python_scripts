import pandas as pd
import numpy as np
import os


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

fov_data_df = pd.read_csv(os.path.join(data_dir, 'fov_features.csv'))

# plot clustermap


# determine which timepoints to use
include_timepoints = ['primary_untreated']
fov_data_df_subset = fov_data_df[fov_data_df.Timepoint.isin(include_timepoints)]

# determine whether to use image-level or timepoint-level features
timepoint = False

if timepoint:
    # aggregate to timepoint level
    data_wide = fov_data_df_subset.groupby(['Tissue_ID', 'metric']).agg(np.mean)
    data_wide.reset_index(inplace=True)
    data_wide = data_wide.pivot(index='Tissue_ID', columns='metric', values='value')
else:
    # aggregate to image level
    data_wide = fov_data_df_subset.pivot(index='fov', columns='metric', values='value')


# replace Nan with 0
data_wide = data_wide.fillna(0)

# drop columns with a sum of zero
data_wide = data_wide.loc[:, (data_wide != 0).any(axis=0)]


sns.clustermap(data_wide, z_score=1, cmap='vlag', vmin=-3, vmax=3, figsize=(20, 20))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'patient_by_feature_clustermap.png'), dpi=300)
plt.close()

# create correlation matrix using spearman correlation
corr_df = data_wide.corr(method='spearman')
sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'spearman_correlation_clustermap.png'), dpi=300)
plt.close()

# create metrics to use for subsetting correlation matrix
# shifted_df = corr_df + 1
# shifted_df = shifted_df / shifted_df.sum(axis=0)
# colvals = shifted_df.apply(shannon_diversity, axis=0)

colvals = corr_df.apply(np.var, axis=0)

# subset based on the columns
col_cutoff = colvals.quantile(0.75)
keep_mask = colvals > col_cutoff
corr_df_subset = corr_df.loc[keep_mask, keep_mask]


# plot heatmap
clustergrid = sns.clustermap(corr_df_subset, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
#plt.tight_layout()
clustergrid.savefig(os.path.join(plot_dir, 'spearman_correlation_heatmap_primary_subset.png'), dpi=300)
plt.close()


# plot correlations between features
keep_cols = corr_df_subset.columns[clustergrid.dendrogram_row.reordered_ind[-6:]]

plot_df = data_wide.loc[:, keep_cols]
g = sns.PairGrid(plot_df, diag_sharey=False)
g.map_lower(sns.regplot, scatter_kws={'s': 10, 'alpha': 0.5})
g.savefig(os.path.join(plot_dir, 'spearman_feature_paired_corelations_2.png'), dpi=300)
plt.close()