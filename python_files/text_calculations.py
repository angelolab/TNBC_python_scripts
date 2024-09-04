import os

import numpy as np
import pandas as pd

base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
metadata_dir = os.path.join(base_dir, 'intermediate_files/metadata')
sequence_dir = os.path.join(base_dir, 'sequencing_data')


harmonized_metadata = pd.read_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'))
study_fovs = harmonized_metadata.loc[harmonized_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), 'fov'].values
ranked_features_all = pd.read_csv(os.path.join(base_dir, 'analysis_files/feature_ranking.csv'))
ranked_features = ranked_features_all.loc[ranked_features_all.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo'])]
feature_metadata = pd.read_csv(os.path.join(base_dir, 'analysis_files/feature_metadata.csv'))
feature_metadata['compartment_binary'] = feature_metadata.compartment.apply(lambda x: 'compartment' if x != 'all' else 'all')
multivariate_dir = os.path.join(base_dir, 'multivariate_lasso')


rna_metadata = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_tissue_rna_id.tsv'), sep='\t')
rna_metadata = rna_metadata.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID']].drop_duplicates(), on='Tissue_ID', how='left')

print(len(rna_metadata.Patient_ID.unique()))
print(len(harmonized_metadata.Tissue_ID.unique()))

cell_table = pd.read_csv(os.path.join(base_dir, 'analysis_files/cell_table_clusters.csv'))
cell_table = cell_table.loc[cell_table.fov.isin(study_fovs), :]

cluster_counts = np.unique(cell_table.fov, return_counts=True)[1]
print(np.min(cluster_counts))
print(np.max(cluster_counts))

combined_df = pd.read_csv(os.path.join(base_dir, 'analysis_files/timepoint_combined_features.csv'))
combined_df = combined_df.loc[combined_df.Timepoint.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo']), :]
combined_df = combined_df.loc[combined_df.feature_name_unique == 'Ki67+__Treg', :]
print(combined_df.raw_mean.mean())

cluster_df_core = pd.read_csv(os.path.join(base_dir, 'output_files/cluster_df_per_core.csv'))
cluster_df_core_int = cluster_df_core.loc[cluster_df_core.cell_type.isin(['Structural', 'Fibroblast', 'CAF', 'Endothelium', 'Smooth_Muscle']), :]
cluster_df_core_int = cluster_df_core_int.loc[cluster_df_core_int.metric.isin(['cluster_broad_count', 'cluster_count']), :]
cluster_df_core_int = cluster_df_core_int.loc[cluster_df_core_int.subset == 'all', :]
cluster_df_core_int = cluster_df_core_int.loc[cluster_df_core_int.fov.isin(study_fovs), :]

cluster_df_core_wide = pd.pivot(cluster_df_core_int, index='fov', columns='cell_type', values='value')

# divide all columns by last column
cluster_df_core_wide = cluster_df_core_wide.div(cluster_df_core_wide['Structural'], axis=0)

print(np.mean(cluster_df_core_wide['Endothelium']))
print(np.mean(cluster_df_core_wide['Smooth_Muscle']))
print(np.mean(cluster_df_core_wide['CAF']))
print(np.mean(cluster_df_core_wide['Fibroblast']))

cluster_df_core_int2 = cluster_df_core.loc[cluster_df_core.cell_type.isin(['Other', 'Immune_Other']), :]
cluster_df_core_int2 = cluster_df_core_int2.loc[cluster_df_core_int2.metric.isin(['cluster_freq']), :]
cluster_df_core_int2 = cluster_df_core_int2.loc[cluster_df_core_int2.subset == 'all', :]
cluster_df_core_int2 = cluster_df_core_int2.loc[cluster_df_core_int2.fov.isin(study_fovs), :]

print(np.mean(cluster_df_core_int2.loc[cluster_df_core_int2.cell_type == 'Other', 'value']))
print(np.mean(cluster_df_core_int2.loc[cluster_df_core_int2.cell_type == 'Immune_Other', 'value']))

# t test for ratios
from scipy.stats import ttest_ind
ttest_ind(comparison_df.density_score.values, comparison_df.ratio_score.values)

# proportion test for compartments
from statsmodels.stats.proportion import proportions_ztest

ranked_features['compartment_binary'] = ranked_features.compartment.apply(lambda x: 'compartment' if x != 'all' else 'all')
feature_metadata['compartment_binary'] = feature_metadata.compartment.apply(lambda x: 'compartment' if x != 'all' else 'all')
top_counts = ranked_features.iloc[:100, :].groupby('compartment_binary').count().iloc[:, 0]
total_counts = feature_metadata.groupby('compartment_binary').count().iloc[:, 0]

stat, pval = proportions_ztest(top_counts, total_counts)

# proportion test for spatial
spatial_features = ['mixing_score', 'cell_diversity', 'compartment_area_ratio', 'pixie_ecm',
                    'compartment_area', 'fiber', 'linear_distance', 'ecm_fraction', 'ecm_cluster']
spatial_mask = np.logical_or(ranked_features.feature_type.isin(spatial_features), ranked_features.compartment != 'all')
ranked_features['spatial_feature'] = spatial_mask

spatial_mask_metadata = np.logical_or(feature_metadata.feature_type.isin(spatial_features), feature_metadata.compartment != 'all')
feature_metadata['spatial_feature'] = spatial_mask_metadata

# calculate proportion of spatial features in top 100 vs all features
top_count_spatial = ranked_features.iloc[:100, :].groupby('spatial_feature').count().iloc[:, 0]
total_counts_spatial = feature_metadata.groupby('spatial_feature').count().iloc[:, 0]

stat, pval = proportions_ztest(top_count_spatial, total_counts_spatial)

# PDL1+ macs
combined_df = pd.read_csv(os.path.join(base_dir, 'analysis_files/timepoint_combined_features_with_outcomes.csv'))

feature_name = 'PDL1+__CD68_Mac'
timepoint = 'on_nivo'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                                    (combined_df.Timepoint == timepoint), :]


ttest_ind(plot_df.loc[plot_df.Clinical_benefit == 'Yes', 'raw_mean'].values,
            plot_df.loc[plot_df.Clinical_benefit == 'No', 'raw_mean'].values)


# shared/non-shared features over time
ranked_features = pd.read_csv(os.path.join(base_dir, 'analysis_files/feature_ranking.csv'))
top_features = ranked_features.loc[ranked_features.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo']), :]
top_feature_names = top_features.feature_name_unique[:100].unique()
top_features = top_features.loc[top_features.feature_name_unique.isin(top_feature_names), :]
top_features_wide = pd.pivot(top_features, index='feature_name_unique', columns='comparison', values='feature_rank_global')
top_features_wide['top_100_count'] = top_features_wide.apply(lambda x: np.sum(x[:4] <= 100), axis=1)
top_features_wide['top_200_count'] = top_features_wide.apply(lambda x: np.sum(x[:4] <= 200), axis=1)
top_features_wide['top_350_count'] = top_features_wide.apply(lambda x: np.sum(x[:4] <= 350), axis=1)

# look at top 100 repeats
len([x for x in top_features_wide.loc[top_features_wide.top_100_count > 1, ].index if 'cancer_border' in x])

np.sum(top_features_wide.top_350_count > 1)
len(top_features_wide.loc[top_features_wide.top_350_count > 2, ].index)

shared_3 = top_features_wide.loc[top_features_wide.top_350_count > 2, ].index
len(shared_3)
len([x for x in shared_3 if 'diversity' in x])
len([x for x in shared_3 if 'ratio' in x])

# lasso model accuracies
cv_scores = pd.read_csv(os.path.join(base_dir, 'multivariate_lasso', 'formatted_cv_scores.csv'))
print(np.mean(cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'primary'), 'value']))
print(np.mean(cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'baseline'), 'value']))
print(np.mean(cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'pre_nivo'), 'value']))

ttest_ind(cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'primary'), 'value'],
            cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'baseline'), 'value'])

ttest_ind(cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'primary'), 'value'],
            cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'pre_nivo'), 'value'])

print(np.mean(cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'on_nivo'), 'value']))

# RNA
print(np.mean(cv_scores.loc[np.logical_and(cv_scores.assay == 'RNA', cv_scores.variable == 'on_nivo'), 'value']))
print(np.mean(cv_scores.loc[np.logical_and(cv_scores.assay == 'RNA', cv_scores.variable == 'pre_nivo'), 'value']))

ttest_ind(cv_scores.loc[np.logical_and(cv_scores.assay == 'RNA', cv_scores.variable == 'on_nivo'), 'value'],
            cv_scores.loc[np.logical_and(cv_scores.assay == 'RNA', cv_scores.variable == 'pre_nivo'), 'value'])

print(np.mean(cv_scores.loc[np.logical_and(cv_scores.assay == 'RNA', cv_scores.variable == 'baseline'), 'value']))

ttest_ind(cv_scores.loc[np.logical_and(cv_scores.assay == 'RNA', cv_scores.variable == 'on_nivo'), 'value'],
            cv_scores.loc[np.logical_and(cv_scores.assay == 'RNA', cv_scores.variable == 'baseline'), 'value'])

# ttest_ind(cv_scores.loc[np.logical_and(cv_scores.assay == 'RNA', cv_scores.variable == 'on_nivo'), 'value'],
#             cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'on_nivo'), 'value'])

ttest_ind(cv_scores.loc[np.logical_and(cv_scores.assay == 'RNA', cv_scores.variable == 'baseline'), 'value'],
            cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'baseline'), 'value'])

ttest_ind(cv_scores.loc[np.logical_and(cv_scores.assay == 'RNA', cv_scores.variable == 'pre_nivo'), 'value'],
            cv_scores.loc[np.logical_and(cv_scores.assay == 'MIBI', cv_scores.variable == 'pre_nivo'), 'value'])

print(np.mean(cv_scores.loc[np.logical_and(cv_scores.assay == 'DNA', cv_scores.variable == 'baseline'), 'value']))