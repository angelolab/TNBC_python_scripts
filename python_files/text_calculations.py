import os

import numpy as np
import pandas as pd

base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
metadata_dir = os.path.join(base_dir, 'intermediate_files/metadata')
sequence_dir = os.path.join(base_dir, 'sequencing_data')


harmonized_metadata = pd.read_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'))
clinical_data = pd.read_csv(os.path.join(base_dir, 'intermediate_files/metadata/patient_clinical_data.csv'))

# RNA
harmonized_metadata = harmonized_metadata[harmonized_metadata.Timepoint.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo'])]
rna_metadata = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_tissue_rna_id.tsv'), sep='\t')
rna_metadata = rna_metadata.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID']].drop_duplicates(), on='Tissue_ID', how='left')
rna_metadata = rna_metadata[['Patient_ID', 'Tissue_ID']].merge(clinical_data, on='Patient_ID').drop_duplicates()
rna_metadata = rna_metadata[rna_metadata.Clinical_benefit.isin(['Yes', 'No'])]

# MIBI
harmonized_metadata = harmonized_metadata.loc[harmonized_metadata.MIBI_data_generated, :]
harmonized_metadata = harmonized_metadata[['Patient_ID', 'Timepoint', 'Tissue_ID', 'fov']].merge(clinical_data, on='Patient_ID')
harmonized_metadata = harmonized_metadata[harmonized_metadata.Clinical_benefit.isin(['Yes', 'No'])]

study_fovs = harmonized_metadata.loc[harmonized_metadata.Timepoint.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo']), 'fov'].values
ranked_features_all = pd.read_csv(os.path.join(base_dir, 'analysis_files/feature_ranking.csv'))
ranked_features = ranked_features_all.loc[ranked_features_all.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo'])]
feature_metadata = pd.read_csv(os.path.join(base_dir, 'analysis_files/feature_metadata.csv'))
feature_metadata['compartment_binary'] = feature_metadata.compartment.apply(lambda x: 'compartment' if x != 'all' else 'all')
multivariate_dir = os.path.join(base_dir, 'multivariate_lasso')

# DNA
wes_metadata = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_WES_meta_table.tsv'), sep='\t')
wes_metadata = wes_metadata.rename(columns={'Individual.ID': 'Patient_ID'})
wes_metadata = wes_metadata.loc[wes_metadata.Clinical_benefit.isin(['Yes', 'No']), :]

print(len(rna_metadata.Patient_ID.unique()))
print(len(rna_metadata.Tissue_ID.unique()))
print(len(harmonized_metadata.Tissue_ID.unique()))
print(len(harmonized_metadata))
print(len(wes_metadata.Patient_ID.unique()))


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
                    'compartment_area', 'fiber', 'linear_distance', 'ecm_fraction', 'ecm_cluster', 'kmeans_cluster']
spatial_mask = np.logical_or(ranked_features.feature_type.isin(spatial_features), ranked_features.compartment != 'all')
ranked_features['spatial_feature'] = spatial_mask

spatial_mask_metadata = np.logical_or(feature_metadata.feature_type.isin(spatial_features), feature_metadata.compartment != 'all')
feature_metadata['spatial_feature'] = spatial_mask_metadata

# calculate proportion of spatial features in top 100 vs all features
top_count_spatial = ranked_features.iloc[:100, :].groupby('spatial_feature').count().iloc[:, 0]
total_counts_spatial = feature_metadata.groupby('spatial_feature').count().iloc[:, 0]

stat, pval = proportions_ztest(top_count_spatial, total_counts_spatial)

# PDL1+ macs
combined_df = pd.read_csv(os.path.join(base_dir, 'analysis_files/timepoint_combined_features_outcome_labels.csv'))

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

# feature type summary
feature_metadata = pd.read_csv(os.path.join(base_dir, 'analysis_files/feature_metadata.csv'))
feature_classes = {'cell_abundance': ['density', 'density_ratio', 'density_proportion'],
                     'diversity': ['cell_diversity', 'region_diversity'],
                     'cell_phenotype': ['functional_marker', 'morphology', ],
                     'cell_interactions': ['mixing_score', 'linear_distance', 'kmeans_cluster'],
                   'structure': ['compartment_area_ratio', 'compartment_area', 'ecm_cluster', 'ecm_fraction', 'pixie_ecm', 'fiber']}

# label with appropriate high-level summary category
for feature_class in feature_classes.keys():
    feature_metadata.loc[feature_metadata.feature_type.isin(feature_classes[feature_class]), 'feature_class'] = feature_class

# add extra column to make stacked bar plotting work easily
feature_metadata_stacked = feature_metadata.copy()
feature_metadata_stacked['count'] = 1
feature_metadata_stacked = feature_metadata_stacked[['feature_class', 'count']].groupby(['feature_class']).sum().reset_index()
print(feature_metadata_stacked.sum())
print(feature_metadata_stacked)

cell_table_adj = pd.read_csv(os.path.join(base_dir, 'supplementary_figs/review_figures/Cancer_reclustering/reassigned_cell_table.csv'))
counts_table = cell_table_adj[cell_table_adj.cell_meta_cluster.isin(['Other', 'Stroma_Collagen', 'Stroma_Fibronectin', 'SMA', 'VIM'])][['cell_meta_cluster', 'cell_meta_cluster_new']]
counts_table = counts_table.groupby(by=['cell_meta_cluster', 'cell_meta_cluster_new'], observed=True).value_counts().reset_index()
counts_table.loc[counts_table.cell_meta_cluster_new!='Cancer_new', 'cell_meta_cluster_new'] = 'Same'
counts_table = counts_table.pivot(index='cell_meta_cluster', columns='cell_meta_cluster_new', values='count').reset_index()
row_sums = counts_table.select_dtypes(include='number').sum(axis=1)
counts_table.iloc[:, 1:] = counts_table.iloc[:, 1:].div(row_sums, axis=0)
cancer_new_count = len(cell_table_adj[cell_table_adj.cell_cluster_broad_new=='Cancer_new'])

print(counts_table.Cancer_new.mean())
print(cancer_new_count/len(cell_table_adj))
