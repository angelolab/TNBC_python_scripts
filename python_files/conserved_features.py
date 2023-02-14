import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr



data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

feature_df = pd.read_csv(os.path.join(data_dir, 'fov_features.csv'))
#harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'harmonized_metadata.csv'))

fov_metadata = harmonized_metadata[harmonized_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'on_nivo', 'post_induction'])]
fov_metadata = fov_metadata[fov_metadata.fov.isin(feature_df.fov.unique())]
keep_fov_1 = []
keep_fov_2 = []
for name, items in fov_metadata[['fov', 'Tissue_ID']].groupby('Tissue_ID'):
    if len(items) > 1:
        keep_fov_1.append(items.iloc[0].fov)
        keep_fov_2.append(items.iloc[1].fov)

# save fovs
fov_pairs = pd.DataFrame({'fov_1': keep_fov_1, 'fov_2': keep_fov_2})
fov_pairs.to_csv(os.path.join(data_dir, 'fov_pairs.csv'), index=False)

# load the fov pairs
fov_pairs = pd.read_csv(os.path.join(data_dir, 'fov_pairs.csv'))
keep_fov_1 = fov_pairs.fov_1
keep_fov_2 = fov_pairs.fov_2

# annotate the feature df
keep_fovs = pd.concat([fov_pairs.fov_1, fov_pairs.fov_2], axis=0)
conserved_df = feature_df[feature_df.fov.isin(keep_fovs)]
conserved_df['placeholder'] = conserved_df.fov.isin(keep_fov_1).astype(int)
conserved_df['placeholder'] = conserved_df['placeholder'].replace({0: 'fov1', 1: 'fov2'})

# pivot the df to wide format with matching fovs
index_cols = conserved_df.columns
index_cols = index_cols.drop(['fov', 'value', 'placeholder'])
conserved_df = conserved_df.pivot(index=index_cols, columns='placeholder', values='value')
conserved_df = conserved_df.reset_index()

p_vals = []
cors = []
names = []
for feature_name in conserved_df.feature_name.unique():
    values = conserved_df[(conserved_df.feature_name == feature_name)]
    values.dropna(inplace=True)
    #values = values[~values.isin([np.inf, -np.inf]).any(1)]

    if len(values) > 20:
        cor, p_val = spearmanr(values.fov1, values.fov2)
        p_vals.append(p_val)
        cors.append(cor)
        names.append(feature_name)

p_df = pd.DataFrame({'name': names, 'p_val': p_vals, 'cor': cors})
p_df['log_pval'] = -np.log10(p_df.p_val)
p_df.to_csv(os.path.join(data_dir, 'conserved_features.csv'), index=False)

p_df = pd.read_csv(os.path.join(data_dir, 'conserved_features.csv'))

sns.scatterplot(data=p_df, x='cor', y='log_pval')
plt.savefig(os.path.join(plot_dir, 'conserved_features_volcano.png'))
plt.close()

plot_subset = p_df[(p_df.log_pval >= 6) & (p_df.cor >= 0.5)]

# get a random row
row = plot_subset.sample(1).iloc[0]
name = row['name']
correlation = row['cor']
values = conserved_df[(conserved_df.feature_name == name)]
values.dropna(inplace=True)
sns.scatterplot(data=values, x='fov1', y='fov2')
plt.title(f'{name} correlation: {correlation}')
plt.savefig(os.path.join(plot_dir, 'conserved_features_1.png'))
plt.close()



# create cascading plot across p values and correlation values
fig, ax = plt.subplots(7, 4, figsize=(25, 20))
for p_idx, p_val in enumerate([3, 4, 5, 6, 7, 8, 9]):
    for c_idx, cor_val in enumerate([0.5, 0.55, 0.6, 0.65]):
        plot_subset = p_df[(p_df.log_pval > p_val - 0.25) & (p_df.log_pval < p_val + 0.25)]
        plot_subset = plot_subset[(plot_subset.cor > cor_val - 0.025) & (plot_subset.cor < cor_val + 0.025)]
        if len(plot_subset) > 0:
            # get a random row
            row = plot_subset.sample(1).iloc[0]
            feature_name = row['name']
            correlation = row['cor']
            values = conserved_df[conserved_df.feature_name == feature_name]
            values.dropna(inplace=True)
            sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax[p_idx, c_idx])
            ax[p_idx, c_idx].set_title(f'{p_val}__{cor_val}')

plt.savefig(os.path.join(plot_dir, 'conserved_features_thresholds_2.png'))
plt.close()
