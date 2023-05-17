import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from scipy.stats import spearmanr

# this file contains code for summarizing the data on an individual core level


def compute_euclidian_distance(input_df):
    # remove rows with any nans
    output_df = input_df.dropna(axis=0)

    # unit normalize each column
    output_df = output_df.apply(lambda x: (x / np.linalg.norm(x)), axis=0)

    # compute euclidian distance
    dist = np.linalg.norm(output_df.values[:, :1] - output_df.values[:, 1:])

    return dist

def compute_spearmanr(input_df):
    # remove rows with any nans
    output_df = input_df.dropna(axis=0)

    # compute spearmanr
    rho, pval = spearmanr(output_df.values[:, :1], output_df.values[:, 1:])

    return rho


def compute_pairwise_distances(input_df, metadata_name):
    euc_distances = []
    spearmanr_distances = []
    for col_1, col_2 in itertools.combinations(input_df.columns, 2):
        subset_df = input_df[[col_1, col_2]]
        euc_distances.append(compute_euclidian_distance(subset_df))
        spearmanr_distances.append(compute_spearmanr(subset_df))

    # create dataframe with distances and metadata
    return_df = pd.DataFrame({'euc_distance': [np.mean(euc_distances)],
                                'spearmanr_distance': [np.mean(spearmanr_distances)],
                              'metadata': [metadata_name]})

    return return_df


def compute_pairwise_differences(input_df, metadata_name):
    col_differences = []

    for col_1, col_2 in itertools.combinations(input_df.columns, 2):
        subset_df = input_df[[col_1, col_2]]
        col_differences.append(np.abs(subset_df[col_1] - subset_df[col_2]))

    # combine differences into a single dataframe
    col_differences = pd.concat(col_differences, axis=1)
    col_means = col_differences.mean(axis=1)

    # create dataframe with distances and metadata
    return_df = pd.DataFrame({'col_mean': col_means,
                                'metadata': metadata_name})

    return return_df


def generate_grouped_distances(sample, group_by, harmonized_metadata, data_df):
    grouped_distances = []
    grouped = harmonized_metadata[[group_by, sample]].groupby(group_by)
    value = 'normalized_value' if sample == 'fov' else 'normalized_mean'
    for group_name, group_members in grouped:
        if len(group_members) > 1:
            group_df = data_df.loc[data_df[sample].isin(group_members[sample].values), :]
            wide_df = pd.pivot(group_df, index='feature_name_unique', columns=sample, values=value)
            grouped_distances.append(compute_pairwise_distances(input_df=wide_df,
                                                                metadata_name=group_name))
    grouped_distances = pd.concat(grouped_distances)

    return grouped_distances


def generate_grouped_differences(sample, group_by, harmonized_metadata, data_df):
    grouped_differences = []
    grouped = harmonized_metadata[[group_by, sample]].groupby(group_by)
    value = 'normalized_value' if sample == 'fov' else 'normalized_mean'

    for group_name, group_members in grouped:
        if len(group_members) > 1:
            group_df = data_df.loc[data_df[sample].isin(group_members[sample].values), :]
            wide_df = pd.pivot(group_df, index='feature_name_unique', columns=sample, values=value)
            grouped_differences.append(compute_pairwise_differences(input_df=wide_df,
                                                                    metadata_name=group_name))

    grouped_differences = pd.concat(grouped_differences)

    return grouped_differences





local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

# load dataset
fov_df = pd.read_csv(os.path.join(data_dir, 'fov_features_no_compartment.csv'))
timepoint_df = pd.read_csv(os.path.join(data_dir, 'timepoint_features_no_compartment.csv'))
ranked_features = pd.read_csv(os.path.join(data_dir, 'conserved_features/ranked_features_no_compartment.csv'))
fov_pairs = pd.read_csv(os.path.join(data_dir, 'conserved_features/fov_pairs.csv'))
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
harmonized_metadata = harmonized_metadata.loc[harmonized_metadata.MIBI_data_generated, :]

# add randomized categories
harmonized_metadata['Tissue_ID_random'] = harmonized_metadata['Tissue_ID'].sample(frac=1).values
harmonized_metadata['TONIC_ID_random'] = harmonized_metadata['TONIC_ID'].sample(frac=1).values


# compute distances
fov_distances = generate_grouped_distances(sample='fov', group_by='Tissue_ID', harmonized_metadata=harmonized_metadata,
                                           data_df=feature_df)



fov_distances['type'] = 'paired_fovs'
fov_distances_random = generate_grouped_distances(sample='fov', group_by='Tissue_ID_random',
                                                  harmonized_metadata=harmonized_metadata, data_df=fov_df)
fov_distances_random['type'] = 'randomized_fovs'

timepoint_distances = generate_grouped_distances(sample='Tissue_ID', group_by='TONIC_ID',
                                                 data_df=timepoint_df, harmonized_metadata=harmonized_metadata)
timepoint_distances['type'] = 'paired_timepoints'

timepoint_distances_random = generate_grouped_distances(sample='Tissue_ID', group_by='TONIC_ID_random',
                                                        data_df=timepoint_df, harmonized_metadata=harmonized_metadata)
timepoint_distances_random['type'] = 'randomized_timepoints'

combined_distances = pd.concat([fov_distances, fov_distances_random, timepoint_distances, timepoint_distances_random])


g = sns.catplot(combined_distances, x='type', y='distance', kind='box', height=5, aspect=2, color='grey',
                order=['paired_fovs', 'paired_timepoints', 'randomized_fovs', 'randomized_timepoints'])
g.set(ylim=(0, 2))
g.fig.suptitle('Pairwise distances samples')
g.savefig(os.path.join(plot_dir, 'pairwise_distances_samples.png'), dpi=300, bbox_inches='tight')
plt.close()


# calculate distance using different features
worst_150 = ranked_features.feature_name_unique[ranked_features.combined_rank > 250].values
best_150 = ranked_features.feature_name_unique[ranked_features.combined_rank < 150].values
best_25 = ranked_features.feature_name_unique[ranked_features.combined_rank < 25].values
best_75 = ranked_features.feature_name_unique[ranked_features.combined_rank < 75].values

worst_150_subset = fov_df.loc[fov_df.feature_name_unique.isin(worst_150), :]
best_150_subset = fov_df.loc[fov_df.feature_name_unique.isin(best_150), :]
best_25_subset = fov_df.loc[fov_df.feature_name_unique.isin(best_25), :]
best_75_subset = fov_df.loc[fov_df.feature_name_unique.isin(best_75), :]


worst_150_distances = generate_grouped_distances(sample='fov', group_by='Tissue_ID',
                                                    data_df=worst_150_subset, harmonized_metadata=harmonized_metadata)
worst_150_distances['type'] = 'worst_150'

best_150_distances = generate_grouped_distances(sample='fov', group_by='Tissue_ID',
                                                    data_df=best_150_subset, harmonized_metadata=harmonized_metadata)
best_150_distances['type'] = 'best_150'

best_25_distances = generate_grouped_distances(sample='fov', group_by='Tissue_ID',
                                                    data_df=best_25_subset, harmonized_metadata=harmonized_metadata)
best_25_distances['type'] = 'best_25'


best_75_distances = generate_grouped_distances(sample='fov', group_by='Tissue_ID',
                                                    data_df=best_75_subset, harmonized_metadata=harmonized_metadata)
best_75_distances['type'] = 'best_75'

cutoff_distances = pd.concat([fov_distances, worst_150_distances, best_150_distances, fov_distances_random,
                              best_25_distances, best_75_distances])

g = sns.catplot(cutoff_distances, x='type', y='distance', kind='box', height=5, aspect=2, color='grey',
                order=['best_25', 'best_75', 'best_150', 'paired_fovs', 'worst_150', 'randomized_fovs'])
g.set(ylim=(0, 2))
g.fig.suptitle('Pairwise distances across FOVs based on feature set')
g.savefig(os.path.join(plot_dir, 'pairwise_distances_cutoffs.png'), dpi=300, bbox_inches='tight')
plt.close()

# get characteristics of FOVs with most divergence from one another
# fov_distances.rename(columns={'metadata': 'Tissue_ID'}, inplace=True)
# fov_distances = pd.merge(fov_distances, harmonized_metadata[['TONIC_ID', 'Tissue_ID', 'Timepoint']].drop_duplicates(), on='Tissue_ID')
#
fov_distances = fov_distances.sort_values(by=['euc_distance'], ascending=False)


# compare different timepoints together
timepoint_pairs = [['primary_untreated', 'baseline'], ['baseline', 'on_nivo'], ['baseline', 'post_induction'],
                   ['post_induction', 'on_nivo']]
# TODO: subset for eahc pair, compute distances, then summarize
timepoint_list = []
for pair in timepoint_pairs:
    current_df = timepoint_df.loc[timepoint_df.Timepoint.isin(pair), :]
timepoint_distances = generate_grouped_distances(sample='Tissue_ID', group_by='TONIC_ID',
                                                    data_df=timepoint_df)
timepoint_distances['type'] = 'paired_timepoints'

timepoint_distances_random = generate_grouped_distances(sample='Tissue_ID', group_by='TONIC_ID_random',
                                                    data_df=timepoint_df)
timepoint_distances_random['type'] = 'randomized_timepoints'

combined_distances = pd.concat([fov_distances, fov_distances_random, timepoint_distances, timepoint_distances_random])


g = sns.catplot(combined_distances, x='type', y='distance', kind='box', height=5, aspect=2, color='grey',
                order=['paired_fovs', 'paired_timepoints', 'randomized_fovs', 'randomized_timepoints'])
g.set(ylim=(0, 2))
g.fig.suptitle('Pairwise distances samples')
g.savefig(os.path.join(plot_dir, 'pairwise_distances_samples.png'), dpi=300, bbox_inches='tight')
plt.close()


# evaluate correlates of intra-sample heterogeneity
fov_distances = fov_distances.rename(columns={'metadata': 'Tissue_ID'})
fov_distances = fov_distances.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint', 'Localization']].drop_duplicates(),
                                    on='Tissue_ID')

g = sns.catplot(fov_distances, x='Localization', y='euc_distance', kind='box', color='grey')
g.set_xticklabels(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'localization_heterogeneity.png'), dpi=300, bbox_inches='tight')
plt.close()

g = sns.catplot(fov_distances, x='Timepoint', y='euc_distance', kind='box', color='grey')
g.set_xticklabels(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'timepoint_heterogeneity.png'), dpi=300, bbox_inches='tight')
plt.close()


# compare differences between paired FOVs
fov_differences = generate_grouped_differences(sample='fov', group_by='Tissue_ID', harmonized_metadata=harmonized_metadata,
                                           data_df=feature_df)
fov_differences = fov_differences.reset_index()
fov_differences_wide = pd.pivot(fov_differences, index='feature_name_unique', columns='metadata', values='col_mean')

fov_differences_wide['nan_count'] = fov_differences_wide.isna().sum(axis=1)
fov_differences_wide = fov_differences_wide.loc[fov_differences_wide.nan_count < 150, :]

fov_differences_wide['total'] = fov_differences_wide.mean(axis=1)
fov_differences_wide = fov_differences_wide.reset_index()


# compare consistency ranking with mean difference
combined_data = pd.merge(ranked_features[['feature_name_unique', 'consistency_score']], fov_differences_wide[['total', 'feature_name_unique']], on='feature_name_unique')
sns.scatterplot(data=combined_data, x='consistency_score', y='total')

subset = combined_data.loc[(combined_data.consistency_score > 0.8) & (combined_data.total > 1.25) , :]

fov_distances['ranking'] = fov_distances['euc_distance'].rank(ascending=True)
fov_distances['ranking'] /= fov_distances['ranking'].max()

fov_distances['distance_cat'] = 'med'
fov_distances.loc[fov_distances.ranking > 0.8, 'distance_cat'] = 'high'
fov_distances.loc[fov_distances.ranking < 0.2, 'distance_cat'] = 'low'

fov_differences.rename(columns={'metadata': 'Tissue_ID'}, inplace=True)

fov_differences = fov_differences.merge(fov_distances[['Tissue_ID', 'distance_cat']], on='Tissue_ID')

# get the average differences across each distance_cat for each feature
fov_differences_grouped = fov_differences.groupby(['feature_name_unique', 'distance_cat']).mean().reset_index()
fov_differences_grouped_wide = pd.pivot(fov_differences_grouped, index='feature_name_unique', columns='distance_cat', values='col_mean')

fov_differences_grouped_wide['diff'] = fov_differences_grouped_wide['high'] - fov_differences_grouped_wide['low']

fov_differences_grouped_wide.sort_values(by='diff', ascending=False, inplace=True)

# plot histogram of differences in each category
fig, ax = plt.subplots(1,3, figsize=(10, 5))
sns.histplot(data=fov_differences_grouped_wide, x='low', ax=ax[0], color='grey', bins=50, binrange=(0, 1.5))
sns.histplot(data=fov_differences_grouped_wide, x='med', ax=ax[1], color='grey', bins=50,  binrange=(0, 1.5))
sns.histplot(data=fov_differences_grouped_wide, x='high', ax=ax[2], color='grey', bins=50, binrange=(0, 1.5))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'difference_histograms_by_fov_category.png'), dpi=300, bbox_inches='tight')
plt.close()

