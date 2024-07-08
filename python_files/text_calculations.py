import os

import numpy as np
import pandas as pd

base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
metadata_dir = os.path.join(base_dir, 'intermediate_files/metadata')
sequence_dir = os.path.join(base_dir, 'sequencing_data')


harmonized_metadata = pd.read_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'))
study_fovs = harmonized_metadata.loc[harmonized_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), 'fov'].values


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