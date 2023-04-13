import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr, ttest_ind

from python_files.utils import find_conserved_features

local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'


# combine overlays together into a single image for easier viewing of what changes are happening over time
cluster_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay'
overlay_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/baseline_nivo_overlay'
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))

#patients = harmonized_metadata.loc[harmonized_metadata.primary_baseline == True, 'TONIC_ID'].unique()
patients = harmonized_metadata.loc[harmonized_metadata.baseline_on_nivo == True, 'TONIC_ID'].unique()
fov_df = harmonized_metadata.loc[harmonized_metadata.TONIC_ID.isin(patients), ['TONIC_ID', 'fov', 'Timepoint']]
fov_df = fov_df.loc[fov_df.fov.isin(cell_table_clusters.fov.unique())]
for patient in patients:

    # get all primary samples
    timepoint_1 = fov_df.loc[(fov_df.TONIC_ID == patient) & (fov_df.Timepoint == 'baseline'), 'fov'].unique()
    timepoint_2 = fov_df.loc[(fov_df.TONIC_ID == patient) & (fov_df.Timepoint == 'on_nivo'), 'fov'].unique()

    max_len = max(len(timepoint_1), len(timepoint_2))

    fig, axes = plt.subplots(2, max_len, figsize=(max_len*5, 10))
    for i in range(len(timepoint_1)):
        try:
            axes[0, i].imshow(plt.imread(os.path.join(cluster_dir, timepoint_1[i] + '.png')))
            axes[0, i].axis('off')
            axes[0, i].set_title('Baseline')
        except:
            print('No primary image for {}'.format(patient))

    for i in range(len(timepoint_2)):
        try:
            axes[1, i].imshow(plt.imread(os.path.join(cluster_dir, timepoint_2[i] + '.png')))
            axes[1, i].axis('off')
            axes[1, i].set_title('On Nivo')
        except:
            print('No baseline image for {}'.format(patient))

    plt.tight_layout()
    plt.savefig(os.path.join(overlay_dir, 'TONIC_{}.png'.format(patient)), dpi=300)
    plt.close()

# identify features that are conserved over time
timepoint_features = pd.read_csv(os.path.join(data_dir, 'conserved_features/timepoint_features_conserved.csv'))
keep_ids = harmonized_metadata.Tissue_ID[harmonized_metadata.baseline_on_nivo == True].unique()
timepoint_features = timepoint_features.loc[timepoint_features.Tissue_ID.isin(keep_ids)]
timepoint_features = timepoint_features.merge(harmonized_metadata[['TONIC_ID', 'Tissue_ID', 'Timepoint']].drop_duplicates(), on='Tissue_ID')
timepoint_features = timepoint_features.loc[timepoint_features.Timepoint.isin(['baseline', 'on_nivo'])]

# generate paired df by patient
paired_df = timepoint_features.pivot(index=['feature_name', 'TONIC_ID'], columns='Timepoint', values='mean')
paired_df = paired_df.reset_index()

p_vals = []
cors = []
names = []
for feature_name in paired_df.feature_name.unique():
    values = paired_df[(paired_df.feature_name == feature_name)].copy()
    values.dropna(inplace=True)
    #values = values[~values.isin([np.inf, -np.inf]).any(1)]

    if len(values) > 20:
        cor, p_val = spearmanr(values.on_nivo, values.baseline)
        p_vals.append(p_val)
        cors.append(cor)
        names.append(feature_name)

ranked_features = pd.DataFrame({'feature_name': names, 'p_val': p_vals, 'cor': cors})
ranked_features['log_pval'] = -np.log10(ranked_features.p_val)

# combine with feature metadata
ranked_features = ranked_features.merge(paired_df[['feature_name', 'compartment', 'cell_pop', 'feature_type']].drop_duplicates(), on='feature_name', how='left')

sns.scatterplot(data=ranked_features, x='cor', y='log_pval')
plt.savefig(os.path.join(plot_dir, 'conserved_features_volcano.png'))
plt.close()

ranked_features['conserved'] = (ranked_features.log_pval >= 6) & (ranked_features.cor >= 0.5)

ranked_features['highly_conserved'] = (ranked_features.log_pval >= 10) & (ranked_features.cor >= 0.7)

row = 49
name = ranked_features.loc[row, 'feature_name']
correlation = ranked_features.loc[row, 'cor']
p_val = ranked_features.loc[row, 'p_val']
values = paired_df[(paired_df.feature_name == name)]
values.dropna(inplace=True)
fig, ax = plt.subplots()
sns.scatterplot(data=values, x='on_nivo', y='baseline', ax=ax)
ax.text(0.05, 0.95, f'cor: {correlation:.2f}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top')
ax.text(0.65, 0.95, f'p: {p_val:.2e}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top')
plt.title(f'{name}')
plt.savefig(os.path.join(plot_dir, 'evolution_baseline_nivo_features_{}.png'.format(name)))
plt.close()

# generate distance metric between each pair of patients to cluster based on similarity

wide_timepoint_features = timepoint_features.pivot(index=['Tissue_ID', 'TONIC_ID', 'Timepoint'], columns='feature_name', values='mean')

# z score features in each column
wide_timepoint_features_norm = wide_timepoint_features.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
wide_timepoint_features_norm = wide_timepoint_features_norm.reset_index()

from scipy.spatial.distance import pdist, squareform

distances = []
ids = []

# calculate distances
for tonic_id, data in wide_timepoint_features_norm.groupby('TONIC_ID'):
    data = data.drop(['TONIC_ID', 'Tissue_ID'], axis=1)
    if len(data) == 1:
        print('Only one timepoint for {}'.format(tonic_id))
        continue

    # drop columns with any Nans
    data = data.dropna(axis=1, how='any')

    data = data.drop(['Timepoint'], axis=1)

    # calculate distance between each pair of patients
    dist = pdist(data.values, metric='euclidean')[0]

    # add to list
    distances.append(dist)
    ids.append(tonic_id)

# convert to df
distances = pd.DataFrame({'distances': distances, 'TONIC_ID': ids})

g = sns.catplot(data=distances, y='distances', kind='swarm')
g.set(ylim=(0, 30))
g.savefig(os.path.join(plot_dir, 'paired_distances_primary_met.png'))
plt.close()

# compute randomized distances
randomized_distances = []
randomized_ids = []

indices = np.arange(len(wide_timepoint_features_norm) - 1)
np.random.shuffle(indices)

for i in np.arange(0, len(indices), 2):
    # get two random patients
    patient1 = wide_timepoint_features_norm.iloc[indices[i]]
    patient2 = wide_timepoint_features_norm.iloc[indices[i+1]]

    # calculate distance between each pair of patients
    combined = pd.concat([patient1, patient2], axis=1).T
    combined = combined.drop(['TONIC_ID', 'Tissue_ID', 'Timepoint'], axis=1)

    # drop columns with any Nans
    combined = combined.dropna(axis=1, how='any')

    dist = pdist(combined.values.astype('float'), metric='euclidean')[0]

    # add to list
    randomized_distances.append(dist)
    randomized_ids.append(i)

# convert to df
randomized_distances = pd.DataFrame({'distances': randomized_distances, 'TONIC_ID': randomized_ids})

g = sns.catplot(data=randomized_distances, y='distances', kind='swarm', )
g.set(ylim=(0, 30))
g.savefig(os.path.join(plot_dir, 'randomized_distances_primary_met.png'))
plt.close()

# generate heatmap and volcano plot with differences between timepoints
timepoint_features = pd.read_csv(os.path.join(data_dir, 'timepoint_features_no_compartment.csv'))
timepoint_features = timepoint_features.merge(harmonized_metadata[['Tissue_ID', 'Timepoint', 'Localization']].drop_duplicates(), on='Tissue_ID', how='left')


def compute_timepoint_enrichment(feature_df, timepoint_1_name, timepoint_1_list,
                                 timepoint_2_name, timepoint_2_list,):
    """Compute enrichment of a feature between two timepoints.

    Args:
        feature_df (pd.DataFrame): dataframe containing features
        timepoint_1 (list): list of timepoints to compare to timepoint_2
        timepoint_2 (list): list of timepoints to compare to timepoint_1
    """
    # get unique features
    features = feature_df.feature_name.unique()

    feature_names = []
    timepoint_1_means = []
    timepoint_1_norm_means = []
    timepoint_2_means = []
    timepoint_2_norm_means = []
    log_pvals = []

    analysis_df = timepoint_features.loc[(timepoint_features.Timepoint.isin(timepoint_1_list + timepoint_2_list)), :]

    for feature_name in features:
        values = analysis_df.loc[(analysis_df.feature_name == feature_name), :]
        tp_1_vals = values.loc[values.Timepoint.isin(timepoint_1_list), 'raw_mean'].values
        tp_1_norm_vals = values.loc[values.Timepoint.isin(timepoint_1_list), 'normalized_mean'].values
        tp_2_vals = values.loc[values.Timepoint.isin(timepoint_2_list), 'raw_mean'].values
        tp_2_norm_vals = values.loc[values.Timepoint.isin(timepoint_2_list), 'normalized_mean'].values
        timepoint_1_means.append(tp_1_vals.mean())
        timepoint_1_norm_means.append(tp_1_norm_vals.mean())
        timepoint_2_means.append(tp_2_vals.mean())
        timepoint_2_norm_means.append(tp_2_norm_vals.mean())

        # compute t-test for difference between timepoints
        t, p = ttest_ind(tp_1_norm_vals, tp_2_norm_vals)
        log_pvals.append(-np.log10(p))

    means_df = pd.DataFrame({timepoint_1_name + '_mean': timepoint_1_means,
                             timepoint_2_name + '_mean': timepoint_2_means,
                             timepoint_1_name + '_norm_mean': timepoint_1_norm_means,
                             timepoint_2_name + '_norm_mean': timepoint_2_norm_means,
                             'log_pval': log_pvals}, index=features)
    # calculate difference
    means_df['mean_diff'] = means_df[timepoint_1_name + '_norm_mean'].values - means_df[timepoint_2_name + '_norm_mean'].values

    return means_df


means_df = compute_timepoint_enrichment(feature_df=timepoint_features, timepoint_1_name='primary_untreated',
                                        timepoint_1_list=['primary_untreated'], timepoint_2_name='baseline_met',
                                        timepoint_2_list=['baseline'])
means_df = means_df.reset_index().rename(columns={'index': 'feature_name'})
#means_df['fold_change'] = means_df.timepoint_1_mean.values / means_df.timepoint_2_mean.values
#means_df['log_fold_change'] = np.log2(means_df.fold_change.values)
# timepoint1_sign = np.sign(means_df.timepoint_1_mean.values)
# timepoint2_sign = np.sign(means_df.timepoint_2_mean.values)
# double_neg = (timepoint1_sign == -1) & (timepoint2_sign == -1)
# means_df.loc[double_neg, 'log_fold_change'] = -1 * means_df.loc[double_neg, 'log_fold_change']
ranked_features = pd.read_csv(os.path.join(data_dir, 'conserved_features/ranked_features_no_compartment.csv'))

means_df = means_df.merge(ranked_features[['feature_name', 'combined_rank', 'conserved']], on='feature_name', how='left')


sns.clustermap(means_df_filtered[['primary_untreated_norm_mean', 'baseline_met_norm_mean']], cmap='RdBu_r', center=0, col_cluster=False)
plt.savefig(os.path.join(plot_dir, 'clustermap_primary_baseline.png'))
plt.close()

sns.scatterplot(data=means_df, x='mean_diff', y='log_pval', hue='conserved', palette='Reds')
plt.savefig(os.path.join(plot_dir, 'volcano_plot_primary_baseline_conserved.png'))
plt.close()




means_df_filtered = means_df.loc[(means_df.log_pval > 2) & (np.abs(means_df.mean_diff) > 0.3), :]

means_df_filtered = means_df_filtered.sort_values('mean_diff', ascending=False)
means_df_filtered.to_csv(os.path.join(plot_dir, 'primary_baseline_log2fc_features.csv'))

for idx, feature in enumerate(means_df_filtered.index):
    feature_subset = timepoint_features.loc[(timepoint_features.feature_name == feature), :]
    feature_subset = feature_subset.loc[(feature_subset.Timepoint.isin(['primary_untreated', 'baseline'])), :]

    g = sns.catplot(data=feature_subset, x='Timepoint', y='raw_mean', kind='strip')
    # add a title
    g.fig.suptitle(feature)
    g.savefig(os.path.join(plot_dir, 'primary_baseline_{}_{}.png'.format(idx, feature)))
    plt.close()
