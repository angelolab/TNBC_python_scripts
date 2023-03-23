import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr



data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

feature_df = pd.read_csv(os.path.join(data_dir, 'fov_features_no_compartment.csv'))
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))

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
paired_df.to_csv(os.path.join(data_dir, 'conserved_features/paired_df.csv'), index=False)

p_vals = []
cors = []
names = []
for feature_name in paired_df.feature_name.unique():
    values = paired_df[(paired_df.feature_name == feature_name)].copy()
    values.dropna(inplace=True)

    # remove rows where both values are 0
    zero_mask = (values.fov1 == 0) & (values.fov2 == 0)
    values = values[~zero_mask]

    if len(values) > 20:
        cor, p_val = spearmanr(values.fov1, values.fov2)
        p_vals.append(p_val)
        cors.append(cor)
        names.append(feature_name)

ranked_features = pd.DataFrame({'feature_name': names, 'p_val': p_vals, 'cor': cors})
ranked_features['log_pval'] = -np.log10(ranked_features.p_val)
ranked_features['log_log_pval'] = -np.emath.logn(1000, ranked_features.p_val)

# combine with feature metadata
ranked_features = ranked_features.merge(paired_df[['feature_name', 'feature_name_unique',
                                                   'compartment', 'cell_pop', 'cell_pop_level',
                                                   'feature_type']].drop_duplicates(), on='feature_name', how='left')
ranked_features['conserved'] = (ranked_features.log_pval >= 6) & (ranked_features.cor >= 0.5)
ranked_features['highly_conserved'] = (ranked_features.log_pval >= 10) & (ranked_features.cor >= 0.7)

ranked_features.to_csv(os.path.join(data_dir, 'conserved_features/ranked_features_no_compartment.csv'), index=False)

ranked_features = pd.read_csv(os.path.join(data_dir, 'conserved_features/ranked_features_no_compartment.csv'))

sns.scatterplot(data=ranked_features, x='cor', y='log_pval')
plt.savefig(os.path.join(plot_dir, 'conserved_features_volcano.png'))
plt.close()

# plot all compartments for each feature for density vs freq
feature_names = ranked_features.feature_name[ranked_features.feature_type.isin(['density', 'freq'])].unique()
feature_names = [x for x in feature_names if '__all' in x]
feature_names = [x.split('__all')[0] for x in feature_names]
feature_names = [x.split('_density')[0] for x in feature_names]
feature_names = [x.split('_freq')[0] for x in feature_names]
feature_names = list(set(feature_names))
feature_names = [x for x in feature_names if 'ratio' not in x]

for feature_name in feature_names:
    feature_mask = ranked_features.feature_name.str.startswith(feature_name)
    feature_df = ranked_features[feature_mask].copy()
    density_names = feature_df.feature_name[feature_df.feature_name.str.contains('density')].values
    freq_names = feature_df.feature_name[feature_df.feature_name.str.contains('freq')].values

    # just plot the density
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    for i, density_name in enumerate(density_names):
        values = paired_df[(paired_df.feature_name == density_name)].copy()
        values.dropna(inplace=True)
        sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax[i])
        ax[i].set_title(density_name)
        correlation, p_val = spearmanr(values.fov1, values.fov2)
        ax[i].set_xlabel('correlation: {:.2f}'.format(correlation))
        ax[i].set_ylabel('p_val: {:.2f}'.format(p_val))
    plt.savefig(os.path.join(plot_dir, 'Correlation_compartment_meta_{}.png'.format(feature_name)))
    plt.close()

    # fig, ax = plt.subplots(2, 5, figsize=(25, 10))
    # for i, density_name in enumerate(density_names):
    #     values = paired_df[(paired_df.feature_name == density_name)].copy()
    #     values.dropna(inplace=True)
    #     values.fov1 = np.log10(values.fov1.values)
    #     values.fov2 = np.log10(values.fov2.values)
    #     sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax[0, i])
    #     ax[0, i].set_title(density_name)
    #     correlation, p_val = spearmanr(values.fov1, values.fov2)
    #     ax[0, i].set_xlabel('correlation: {:.2f}'.format(correlation))
    #     ax[0, i].set_ylabel('p_val: {:.2f}'.format(p_val))

    # for i, freq_name in enumerate(freq_names):
    #     values = paired_df[(paired_df.feature_name == freq_name)].copy()
    #     values.dropna(inplace=True)
    #     values.fov1 = np.log10(values.fov1.values)
    #     values.fov2 = np.log10(values.fov2.values)
    #     sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax[1, i])
    #     ax[1, i].set_title(freq_name)
    #     correlation, p_val = \
    #     ranked_features[(ranked_features.feature_name == freq_name)][['cor', 'log_pval']].values[
    #         0]
    #     ax[1, i].set_xlabel('correlation: {:.2f}'.format(correlation))
    #     ax[1, i].set_ylabel('p_val: {:.2f}'.format(p_val))

    for i, freq_name in enumerate(density_names):
        values = paired_df[(paired_df.feature_name == freq_name)].copy()
        values.dropna(inplace=True)
        sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax[1, i])
        ax[1, i].set_title(freq_name)
        correlation, p_val = \
        ranked_features[(ranked_features.feature_name == freq_name)][['cor', 'log_pval']].values[
            0]
        ax[1, i].set_xlabel('correlation: {:.2f}'.format(correlation))
        ax[1, i].set_ylabel('p_val: {:.2f}'.format(p_val))

    plt.savefig(os.path.join(plot_dir, 'Correlation_compartment_log_{}.png'.format(feature_name)))
    plt.close()


# generate individual plots for selected features
for min_cor in [0.5, 0.6, 0.7, 0.8]:
    plot_features = ranked_features[(ranked_features.cor > min_cor) & (ranked_features.cor < min_cor + 0.1)].copy()

    # randomize the order of the features
    plot_features = plot_features.sample(frac=1)

    for i in range(min(len(plot_features), 30)):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        feature_name = plot_features.iloc[i].feature_name
        values = paired_df[(paired_df.feature_name == feature_name)].copy()
        values.dropna(inplace=True)

        # drop rows where either value is inf
        values = values[~values.isin([np.inf, -np.inf]).any(1)]

        sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax[0])
        correlation, p_val = spearmanr(values.fov1, values.fov2)
        ax[0].set_xlabel('untransformed')

        logged_values = values.copy()
        min_val = min(logged_values.fov1.min(), logged_values.fov2.min())
        if min_val > 0:
            min_val = 0
        logged_values.fov1 = np.log10(values.fov1.values + (min_val * -1.01))
        logged_values.fov2 = np.log10(values.fov2.values + (min_val * -1.01))
        sns.scatterplot(data=logged_values, x='fov1', y='fov2', ax=ax[1])
        ax[1].set_xlabel('log10 transformed')

        # plot transformed values


        # transformed_vals = PowerTransformer().fit_transform(values[['fov1', 'fov2']].values)
        # sns.scatterplot(transformed_vals[:, 0], transformed_vals[:, 1], ax=ax[2])
        # ax[2].set_xlabel('Power transformed')

        # set title for whole figure
        fig.suptitle(feature_name + ' correlation: {:.2f}'.format(correlation))

        # only save the first 2 digits of the correlation
        plt.savefig(os.path.join(plot_dir, 'Correlation_compartment_{}_{}.png'.format(feature_name, str(correlation)[:4])))
        plt.close()


# plot a specified row
row = 25
row = np.where(ranked_features.feature_name == 'TCF1+__Cancer__all')[0][0]
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
fig, ax = plt.subplots(7, 6, figsize=(20, 35))
for p_idx, p_val in enumerate([3, 4, 5, 6, 7, 8, 9]):
    for c_idx, cor_val in enumerate([0.4, 0.45, 0.5, 0.55, 0.6, 0.65]):
        plot_subset = ranked_features[(ranked_features.log_pval > p_val - 0.25) & (ranked_features.log_pval < p_val + 0.25)]
        plot_subset = plot_subset[(plot_subset.cor > cor_val - 0.025) & (plot_subset.cor < cor_val + 0.025)]
        if len(plot_subset) > 0:
            # get a random row
            row = plot_subset.sample(1).iloc[0]
            feature_name = row['feature_name']
            correlation = row['cor']
            values = paired_df[paired_df.feature_name == feature_name]
            values.dropna(inplace=True)

            logged_values = values.copy()
            min_val = min(logged_values.fov1.min(), logged_values.fov2.min())
            if min_val > 0:
                min_val = 0
            logged_values.fov1 = np.log10(values.fov1.values + (min_val * -1.01))
            logged_values.fov2 = np.log10(values.fov2.values + (min_val * -1.01))
            sns.scatterplot(data=logged_values, x='fov1', y='fov2', ax=ax[p_idx, c_idx])
            ax[p_idx, c_idx].set_title(f'{p_val}__{cor_val}')

plt.savefig(os.path.join(plot_dir, 'conserved_features_thresholds_logged.png'))
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
# feature_df_conserved = []
# for feature_name, compartment in zip(ranked_features_conserved.feature_name,
#                                      ranked_features_conserved.compartment):
#     values = feature_df[(feature_df.feature_name == feature_name) & (feature_df.compartment == compartment)].copy()
#     values.dropna(inplace=True)
#     feature_df_conserved.append(values)
#feature_df_conserved = pd.concat(feature_df_conserved)

feature_df_conserved = feature_df[feature_df.feature_name.isin(ranked_features_conserved.feature_name)].copy()
feature_df_conserved.to_csv(os.path.join(data_dir, 'conserved_features/fov_features_conserved.csv'), index=False)

# do the same for timepoint level features
timepoint_df = pd.read_csv(os.path.join(data_dir, 'timepoint_features.csv'))

# timepoint_df_conserved = []
# for feature_name, compartment in zip(ranked_features_conserved.feature_name,
#                                      ranked_features_conserved.compartment):
#     values = timepoint_df[(timepoint_df.feature_name == feature_name) & (timepoint_df.compartment == compartment)].copy()
#     values.dropna(inplace=True)
#     timepoint_df_conserved.append(values)
#
# timepoint_df_conserved = pd.concat(timepoint_df_conserved)
timepoint_df_conserved = timepoint_df[timepoint_df.feature_name.isin(ranked_features_conserved.feature_name)].copy()
timepoint_df_conserved.to_csv(os.path.join(data_dir, 'conserved_features/timepoint_features_conserved.csv'), index=False)

# look at correlation between features
correlation_df = feature_df[feature_df.fov.isin(keep_fovs)].copy()
correlation_df = correlation_df[correlation_df.feature_name.isin(ranked_features.feature_name)].copy()
data_wide = correlation_df.pivot(index='fov', columns='feature_name', values='value')

# create correlation matrix using spearman correlation
corr_df = data_wide.corr(method='spearman')
corr_df_abs = corr_df.abs()
# corr_df = corr_df.fillna(0)
# clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
# clustergrid.savefig(os.path.join(plot_dir, 'spearman_correlation_clustermap.png'), dpi=300)
# plt.close()

# get total correlation of each feature
corr_df_abs[corr_df_abs < 0.5] = 0
corr_sums = corr_df_abs.sum(axis=0)
ranked_features['correlation_sum'] = corr_sums.values
ranked_features['correlated_feature'] = ranked_features['correlation_sum'] > 7
non_nans = data_wide.count(axis=0)
sparse_features = non_nans < (len(data_wide) / 3)
ranked_features['sparse_feature'] = sparse_features.values


sns.catplot(data=corr_sums, x='conserve_type', y='correlation_sum', kind='swarm')
sns.scatterplot(data=ranked_features, x='cor', y='correlation_sum', hue='correlated_feature')
plt.savefig(os.path.join(plot_dir, 'correlation_sum_vs_cor_value_by_connection.png'))
plt.close()


# separately plot the highly correlated and poorly correlated features
data_wide_corr = data_wide.iloc[:, ~data_wide.columns.isin(corr_sums.feature_name[corr_sums.binarized_sum.values])]
corr_df = data_wide_corr.corr(method='spearman')
corr_df_abs = corr_df.abs()
corr_df = corr_df.fillna(0)
clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))


# get counts of each feature type for conserved vs non-conserved features
grouped_by_feature = ranked_features[['feature_type', 'correlated_feature', 'conserved', 'sparse_feature']].groupby('feature_type').mean().reset_index()

# plot as a barplot
g = sns.barplot(data=grouped_by_feature, x='feature_type', y='conserved', color='grey')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Feature Type')
g.set_ylabel('Proportion conserved')
g.set_title('Proportion of features conserved by feature type')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'proportion_conserved_by_feature_type.png'))
plt.close()

# plot as a barplot
g = sns.barplot(data=grouped_by_feature, x='feature_type', y='correlated_feature', color='grey')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Feature Type')
g.set_ylabel('Proportion correlated')
g.set_title('Proportion of features that are correlated by feature type')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'proportion_correlated_by_feature_type.png'))
plt.close()

# plot as a barplot
g = sns.barplot(data=grouped_by_feature, x='feature_type', y='sparse_feature', color='grey')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Feature Type')
g.set_ylabel('Proportion sparse')
g.set_title('Proportion of features that are sparse by feature type')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'proportion_sparse_by_feature_type.png'))
plt.close()

grouped_by_pop = ranked_features[['cell_pop', 'correlated_feature', 'conserved', 'sparse_feature']].groupby('cell_pop').mean().reset_index()

# plot as a barplot
g = sns.barplot(data=grouped_by_pop, x='cell_pop', y='conserved', color='grey')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Cell Population')
g.set_ylabel('Proportion conserved')
g.set_title('Proportion of features conserved by cell population')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'proportion_conserved_by_cell_pop.png'))
plt.close()

# plot as a barplot
g = sns.barplot(data=grouped_by_pop, x='cell_pop', y='correlated_feature', color='grey')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Cell Population')
g.set_ylabel('Proportion correlated')
g.set_title('Proportion of features that are correlated by cell population')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'proportion_correlated_by_cell_pop.png'))
plt.close()

# plot as a barplot
g = sns.barplot(data=grouped_by_pop, x='cell_pop', y='sparse_feature', color='grey')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Cell Population')
g.set_ylabel('Proportion sparse')
g.set_title('Proportion of features that are sparse by cell population')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'proportion_sparse_by_cell_pop.png'))
plt.close()

