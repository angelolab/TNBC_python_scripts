import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr, ttest_ind, ttest_rel

from python_files.utils import find_conserved_features, compare_timepoints, compare_populations
from python_files.utils import summarize_population_enrichment, summarize_timepoint_enrichment

local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))

# # combine overlays together into a single image for easier viewing of what changes are happening over time
# cluster_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay'
# overlay_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/baseline_nivo_overlay'

# #patients = harmonized_metadata.loc[harmonized_metadata.primary_baseline == True, 'TONIC_ID'].unique()
# patients = harmonized_metadata.loc[harmonized_metadata.baseline_on_nivo == True, 'TONIC_ID'].unique()
# fov_df = harmonized_metadata.loc[harmonized_metadata.TONIC_ID.isin(patients), ['TONIC_ID', 'fov', 'Timepoint']]
# fov_df = fov_df.loc[fov_df.fov.isin(cell_table_clusters.fov.unique())]
# for patient in patients:
#
#     # get all primary samples
#     timepoint_1 = fov_df.loc[(fov_df.TONIC_ID == patient) & (fov_df.Timepoint == 'baseline'), 'fov'].unique()
#     timepoint_2 = fov_df.loc[(fov_df.TONIC_ID == patient) & (fov_df.Timepoint == 'on_nivo'), 'fov'].unique()
#
#     max_len = max(len(timepoint_1), len(timepoint_2))
#
#     fig, axes = plt.subplots(2, max_len, figsize=(max_len*5, 10))
#     for i in range(len(timepoint_1)):
#         try:
#             axes[0, i].imshow(plt.imread(os.path.join(cluster_dir, timepoint_1[i] + '.png')))
#             axes[0, i].axis('off')
#             axes[0, i].set_title('Baseline')
#         except:
#             print('No primary image for {}'.format(patient))
#
#     for i in range(len(timepoint_2)):
#         try:
#             axes[1, i].imshow(plt.imread(os.path.join(cluster_dir, timepoint_2[i] + '.png')))
#             axes[1, i].axis('off')
#             axes[1, i].set_title('On Nivo')
#         except:
#             print('No baseline image for {}'.format(patient))
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(overlay_dir, 'TONIC_{}.png'.format(patient)), dpi=300)
#     plt.close()

# identify features that are conserved from primary to met
timepoint_features = pd.read_csv(os.path.join(data_dir, 'timepoint_features_filtered.csv'))
timepoint_features = timepoint_features.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint', 'primary__baseline']].drop_duplicates(), on='Tissue_ID')

feature_subset = timepoint_features.loc[(timepoint_features.primary__baseline) & (timepoint_features.Timepoint.isin(['primary_untreated', 'baseline'])), :]

index_cols = feature_subset.columns
index_cols = index_cols.drop(['Timepoint', 'raw_mean', 'normalized_mean', 'raw_std', 'normalized_std', 'primary__baseline', 'Tissue_ID'])
paired_df = feature_subset.pivot(index=index_cols, columns='Timepoint', values=['raw_mean', 'normalized_mean'])
paired_df = paired_df.reset_index()

# clean up columns
paired_df.columns = ['_'.join(col).strip() for col in paired_df.columns.values]
paired_df.columns = [x[:-1] if x.endswith('_') else x for x in paired_df.columns]


ranked_features = find_conserved_features(paired_df=paired_df, sample_name_1='raw_mean_primary_untreated',
                                              sample_name_2='raw_mean_baseline', min_samples=10)
# combine with feature metadata
ranked_features = ranked_features.merge(paired_df[['feature_name', 'compartment', 'cell_pop', 'feature_type']].drop_duplicates(), on='feature_name', how='left')

sns.scatterplot(data=ranked_features, x='cor', y='log_pval')
plt.savefig(os.path.join(plot_dir, 'conserved_features_volcano.png'))
plt.close()



# generate plot for best ranked features
max_rank = 30
plot_features = ranked_features.loc[ranked_features.combined_rank <= max_rank, :]
# sort by combined rank
plot_features.sort_values(by='combined_rank', inplace=True)

for i in range(len(plot_features)):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    feature_name = plot_features.iloc[i].feature_name
    #feature_name = plot_features[i]
    values = paired_df[(paired_df.feature_name == feature_name)].copy()
    values.dropna(inplace=True)

    sns.scatterplot(data=values, x='primary_untreated', y='baseline', ax=ax[0])
    #sns.scatterplot(data=values, x='raw_value_fov1', y='raw_value_fov2', ax=ax[0])
    #sns.scatterplot(data=values, x='value_fov1', y='value_fov2', ax=ax[0])
    correlation, p_val = spearmanr(values.primary_untreated, values.baseline)
    ax[0].set_xlabel('untransformed')

    logged_values = values.copy()
    min_val = min(values.primary_untreated.min(), values.baseline.min())
    if min_val > 0:
        increment = 0
    elif min_val == 0:
        increment = 0.01
    else:
        increment = min_val * -1.01
    logged_values.normalized_value_fov1 = np.log10(values.primary_untreated.values + increment)
    logged_values.normalized_value_fov2 = np.log10(values.baseline.values + increment)
    sns.scatterplot(data=logged_values, x='primary_untreated', y='baseline', ax=ax[1])
    ax[1].set_xlabel('log10 transformed')

    # set title for whole figure
    fig.suptitle(feature_name + ' correlation: {:.2f} rank: {}'.format(correlation, plot_features.iloc[i].combined_rank))

    # convert int rank to string with two leading zeros
    int_rank = int(plot_features.iloc[i].combined_rank)
    str_rank = str(int_rank)
    if int_rank < 10:
        str_rank = '0' + str_rank

    plt.savefig(os.path.join(plot_dir, 'Correlation_{}_{}.png'.format(str_rank, feature_name)))
    plt.close()

# create df to look at coordinated shifts
paired_df['shift'] = paired_df.normalized_mean_baseline - paired_df.normalized_mean_primary_untreated
paired_df = paired_df.dropna(axis=0)

wide_df = paired_df.pivot(index='feature_name', columns='Patient_ID', values='shift')
wide_df = wide_df.fillna(0)

sns.clustermap(wide_df, cmap='RdBu_r', center=0, figsize=(10, 100), vmin=-1, vmax=1)
total_diff = wide_df.apply(lambda x: np.sum(np.abs(x)), axis=1)
med = np.median(total_diff)
wide_df_small = wide_df.loc[total_diff > med, :]

sns.clustermap(wide_df_small, cmap='RdBu_r', center=0, figsize=(10, 100), vmin=-1, vmax=1)
plt.savefig(os.path.join(plot_dir, 'clustermap_evolution_primary_met.png'))
plt.close()


# compare features prevalence in primary vs metastatic
primary_met_means = compare_timepoints(feature_df=timepoint_features, timepoint_1_name='primary', timepoint_1_list=['primary_untreated'],
                     timepoint_2_name='metastatic', timepoint_2_list=['baseline'], paired='primary__baseline',
                   feature_suff='mean')


summarize_timepoint_enrichment(input_df=primary_met_means, feature_df=timepoint_features, timepoints=['baseline', 'primary_untreated'],
                                 pval_thresh=2, diff_thresh=0.3, output_dir=plot_dir + '/primary_baseline_evolution')


# plot evolution of features
feature_name = 'H3K9ac_H3K27me3_ratio+__all'
feature_name = 'TBET+__CD8T'
patients = harmonized_metadata.Patient_ID[harmonized_metadata.primary__baseline].unique()
timepoints = ['primary_untreated', 'baseline']
plot_df = timepoint_features.loc[(timepoint_features.feature_name == feature_name) & (timepoint_features.Patient_ID.isin(patients)) & (timepoint_features.Timepoint.isin(timepoints)), :]
plot_df['Timepoint'] = pd.Categorical(plot_df.Timepoint, categories=timepoints, ordered=True)
sns.lineplot(data=plot_df, x='Timepoint', y='raw_mean', units='Patient_ID', estimator=None, color='grey', alpha=0.5, marker='o')
plt.title(feature_name)
plt.savefig(os.path.join(plot_dir, 'primary_met_evolution_{}.png'.format(feature_name)))
plt.close()

# create categories
primary_met_means['category'] = 'other'
primary_met_means.loc[primary_met_means.feature_name.str.contains('H3K9ac_H3K27me3_ratio'), 'category'] = 'Histone'
primary_met_means.loc[primary_met_means.feature_name.str.contains('Vim+'), 'category'] = 'Vimentin'

sns.scatterplot(data=primary_met_means, x='mean_diff', y='log_pval', hue='category')
plt.savefig(os.path.join(plot_dir, 'primary_met_evolution_scatter.png'))
plt.close()



all_dfs = []
for name in ['all', 'ln_met', 'other_met']:
    if name == 'all':
        current_df = timepoint_features.copy()
    elif name == 'ln_met':
        current_df = timepoint_features.loc[timepoint_features.Localization.isin(['Lymphnode', 'Breast']), :]
    else:
        current_df = timepoint_features.loc[timepoint_features.Localization != 'Lymphnode', :]

    # find enriched features
    means_df = compute_timepoint_enrichment(feature_df=current_df, timepoint_1_name='primary_untreated',
                                            timepoint_1_list=['primary_untreated'], timepoint_2_name='baseline_met',
                                            timepoint_2_list=['baseline'])
    means_df['comparison'] = name
    all_dfs.append(means_df)

