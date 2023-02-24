import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

feature_df = pd.read_csv(os.path.join(data_dir, 'fov_features_no_compartment.csv'))
#harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'harmonized_metadata.csv'))

# fov_metadata = harmonized_metadata[harmonized_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'on_nivo', 'post_induction'])]
# fov_metadata = fov_metadata[fov_metadata.fov.isin(feature_df.fov.unique())]
# keep_fov_1 = []
# keep_fov_2 = []
# for name, items in fov_metadata[['fov', 'Tissue_ID']].groupby('Tissue_ID'):
#     if len(items) > 1:
#         keep_fov_1.append(items.iloc[0].fov)
#         keep_fov_2.append(items.iloc[1].fov)
#
# # save fovs
# fov_pairs = pd.DataFrame({'fov_1': keep_fov_1, 'fov_2': keep_fov_2})
# fov_pairs.to_csv(os.path.join(data_dir, 'fov_pairs.csv'), index=False)

# load the fov pairs
fov_pairs = pd.read_csv(os.path.join(data_dir, 'conserved_features/fov_pairs.csv'))
keep_fov_1 = fov_pairs.fov_1
keep_fov_2 = fov_pairs.fov_2

# annotate the feature df
keep_fovs = pd.concat([fov_pairs.fov_1, fov_pairs.fov_2], axis=0)
paired_df = feature_df[feature_df.fov.isin(keep_fovs)].copy()
paired_df['placeholder'] = paired_df.fov.isin(keep_fov_1).astype(int)
paired_df['placeholder'] = paired_df['placeholder'].replace({0: 'fov1', 1: 'fov2'})

# pivot the df to wide format with matching fovs
index_cols = paired_df.columns
index_cols = index_cols.drop(['fov', 'value', 'placeholder'])
paired_df = paired_df.pivot(index=index_cols, columns='placeholder', values='value')
paired_df = paired_df.reset_index()

p_vals = []
cors = []
names = []
for feature_name in paired_df.feature_name.unique():
    values = paired_df[(paired_df.feature_name == feature_name)].copy()
    values.dropna(inplace=True)
    #values = values[~values.isin([np.inf, -np.inf]).any(1)]

    if len(values) > 20:
        cor, p_val = spearmanr(values.fov1, values.fov2)
        p_vals.append(p_val)
        cors.append(cor)
        names.append(feature_name)

ranked_features = pd.DataFrame({'feature_name': names, 'p_val': p_vals, 'cor': cors})
ranked_features['log_pval'] = -np.log10(ranked_features.p_val)

# combine with feature metadata
ranked_features = ranked_features.merge(paired_df[['feature_name', 'compartment', 'cell_pop', 'feature_type']].drop_duplicates(), on='feature_name', how='left')
ranked_features.to_csv(os.path.join(data_dir, 'conserved_features/ranked_features_no_compartment.csv'), index=False)

ranked_features = pd.read_csv(os.path.join(data_dir, 'conserved_features/ranked_features_no_compartment.csv'))

sns.scatterplot(data=ranked_features, x='cor', y='log_pval')
plt.savefig(os.path.join(plot_dir, 'conserved_features_volcano.png'))
plt.close()

ranked_features['conserved'] = (ranked_features.log_pval >= 6) & (ranked_features.cor >= 0.5)

ranked_features['highly_conserved'] = (ranked_features.log_pval >= 10) & (ranked_features.cor >= 0.7)

# plot a specified row
row = 121
name = ranked_features.loc[row, 'feature_name']
correlation = ranked_features.loc[row, 'cor']
p_val = ranked_features.loc[row, 'p_val']
values = paired_df[(paired_df.feature_name == name)]
values.dropna(inplace=True)
fig, ax = plt.subplots()
sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax)
ax.text(0.05, 0.95, f'cor: {correlation:.2f}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top')
ax.text(0.65, 0.95, f'p: {p_val:.2e}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top')
plt.title(f'{name}')
plt.savefig(os.path.join(plot_dir, 'conserved_features_{}.png'.format(name)))
plt.close()

# drop rows where eitehr is greater than 0.2
values = values[(values.fov1 < 0.2) & (values.fov2 < 0.2)]

# plot a given feature across different compartments
feature_name = 'HLA1+__Cancer__'

# find all the rows where feature_name is a substring of the feature_name
names = ranked_features.feature_name[ranked_features.feature_name.str.contains(feature_name, regex=False)]

# plot a combined scatterplot with each compartment as a column
fig, ax = plt.subplots(1, len(names), figsize=(len(names) * 6, 4))
for i, name in enumerate(names):
    values = paired_df[(paired_df.feature_name == name)].copy()
    values.dropna(inplace=True)
    sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax[i])
    ax[i].set_title(name.split('__')[-1])
    cor = ranked_features[ranked_features.feature_name == name].cor.values[0]
    p_val = ranked_features[ranked_features.feature_name == name].p_val.values[0]
    ax[i].text(0.05, 0.95, f'cor: {cor:.2f}', transform=ax[i].transAxes, fontsize=10, verticalalignment='top')
    ax[i].text(0.55, 0.95, f'p: {p_val:.2e}', transform=ax[i].transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, feature_name + '.png'))
plt.close()

# create cascading plot across p values and correlation values
fig, ax = plt.subplots(7, 4, figsize=(20, 35))
for p_idx, p_val in enumerate([3, 4, 5, 6, 7, 8, 9]):
    for c_idx, cor_val in enumerate([0.5, 0.55, 0.6, 0.65]):
        plot_subset = p_df[(p_df.log_pval > p_val - 0.25) & (p_df.log_pval < p_val + 0.25)]
        plot_subset = plot_subset[(plot_subset.cor > cor_val - 0.025) & (plot_subset.cor < cor_val + 0.025)]
        if len(plot_subset) > 0:
            # get a random row
            row = plot_subset.sample(1).iloc[0]
            feature_name = row['name']
            correlation = row['cor']
            values = paired_df[paired_df.feature_name == feature_name]
            values.dropna(inplace=True)
            sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax[p_idx, c_idx])
            ax[p_idx, c_idx].set_title(f'{p_val}__{cor_val}')

plt.savefig(os.path.join(plot_dir, 'conserved_features_thresholds_4.png'))
plt.close()


# # get all of the multi-compartment features
# unique_names = conserved_df.feature_name[conserved_df.compartment == 'all'].unique()
# unique_names = [x.split('__all')[0] for x in unique_names]
#
# # for each unique feature, figure out which compartments are present
# for feature in unique_names:
#     matches = plot_subset.name[plot_subset.name.str.contains(feature + '__', regex=False)]
#     if len(matches) > 0:
#         # in all but one compartment, or all compartments, which makes it global
#         if len(matches) > 3:
#             compartment = 'all'
#         # only one compartment; either all, or the one its found in
#         elif len(matches) == 1:
#             comp = matches.values[0].split('__')[-1]
#
#             if comp == 'all':
#                 compartment = 'all'
#             else:
#                 compartment = comp.split('_')[0]
#
#         # two compartments
#         else:
#             suffs = [x.split('__')[-1] for x in matches]
#             if 'all' in suffs:
#                 compartment = 'all'
#             else:
#                 compartment = 'both'
#         plot_subset.loc[plot_subset.name.str.contains(feature + '__', regex=False), 'compartment'] = compartment

ranked_features_conserved = ranked_features[ranked_features.conserved]
p_df_subset[['feature_type', 'highly_conserved']].groupby('feature_type').mean().reset_index()
p_df_subset[['cell_pop', 'highly_conserved']].groupby('cell_pop').mean().reset_index()

# save only conserved features
feature_df_conserved = []
for feature_name, compartment in zip(ranked_features_conserved.feature_name,
                                     ranked_features_conserved.compartment):
    values = feature_df[(feature_df.feature_name == feature_name) & (feature_df.compartment == compartment)].copy()
    values.dropna(inplace=True)
    feature_df_conserved.append(values)

feature_df_conserved = pd.concat(feature_df_conserved)
feature_df_conserved.to_csv(os.path.join(data_dir, 'conserved_features/fov_features_conserved.csv'), index=False)

# do the same for timepoint level features
timepoint_df = pd.read_csv(os.path.join(data_dir, 'timepoint_features.csv'))

timepoint_df_conserved = []
for feature_name, compartment in zip(ranked_features_conserved.feature_name,
                                     ranked_features_conserved.compartment):
    values = timepoint_df[(timepoint_df.feature_name == feature_name) & (timepoint_df.compartment == compartment)].copy()
    values.dropna(inplace=True)
    timepoint_df_conserved.append(values)

timepoint_df_conserved = pd.concat(timepoint_df_conserved)
timepoint_df_conserved.to_csv(os.path.join(data_dir, 'conserved_features/timepoint_features_conserved.csv'), index=False)