import os
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.lines as mlines
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

from SpaceCat.preprocess import preprocess_table
from SpaceCat.features import SpaceCat
from ark.utils.plot_utils import color_segmentation_by_stat

import random
random.seed(13)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ANALYSIS_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files"
INTERMEDIATE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files"
OUTPUT_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/output_files"
SUPPLEMENTARY_FIG_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs/review_figures"
SPACECAT_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/TONIC_SpaceCat"


## 2.5 & 4.3 SpaceCat prediction comparison ##
NT_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, 'NTPublic')

# read in data
NT_DIR = '/Volumes/Shared/Noah Greenwald/NTPublic'
cell_table = pd.read_csv(os.path.join(NT_DIR, 'analysis_files/cell_table_final.csv'))
clinical_data = pd.read_csv(os.path.join(NT_DIR, 'analysis_files/clinical.csv'))
clinical_data = clinical_data.rename(columns={'PatientID': 'Patient_ID'})
cell_table = cell_table.merge(clinical_data, on=['Patient_ID', 'BiopsyPhase'])
cell_table_adj = cell_table[cell_table.isPerProtocol]
cell_table_adj = cell_table_adj[cell_table_adj.cellAnnotation.isin(['TME', 'invasive'])]
cell_table_adj = cell_table_adj[cell_table_adj.BiopsyPhase != 'Post-treatment']

assignments = pd.read_csv(os.path.join(NT_DIR, 'intermediate_files/mask_dir/individual_masks_all/cell_annotation_mask.csv'))
areas = pd.read_csv(os.path.join(NT_DIR, 'intermediate_files/mask_dir/individual_masks_all/fov_annotation_mask_area.csv'))
areas = areas.rename(columns={'area': 'compartment_area'})
cell_table_new = cell_table_adj.merge(assignments, on=['fov', 'label']).rename(columns={'mask_name': 'compartment'})
cell_table_new = cell_table_new.merge(areas, on=['fov', 'compartment'])

# convert cell table to anndata
markers = ['Ki67', 'pH2AX', 'Helios', 'OX40', 'GATA3', 'TOX', 'IDO', 'AR', 'ICOS', 'TCF1', 'PDGFRB',
           'GZMB', 'CD79a', 'Vimentin', 'Calponin', 'MPO', 'T-bet', 'PD-L1 (73-10)', 'PD-L1 (SP142)',
           'c-PARP', 'HLA-ABC', 'Caveolin-1', 'PD-1', 'HLA-DR']
centroid_cols = ['centroid-0', 'centroid-1']
cell_data_cols = ['fov', 'label', 'centroid-0', 'centroid-1', 'cell_meta_cluster',
                  'cell_cluster', 'cell_cluster_broad', 'cell_size', 'compartment', 'compartment_area',
                  'BiopsyPhase', 'Patient_ID',
                  'cell_meta_cluster_id', 'isEpithelial', 'Colour', 'PrintOrder', 'cellAnnotation',
                  'isPerProtocol', 'pCR', 'Arm']

# create anndata from table subsetted for marker info, which will be stored in adata.X
adata = anndata.AnnData(cell_table_new.loc[:, markers])
adata.obs = cell_table_new.loc[:, cell_data_cols, ]
adata.obs_names = [str(i) for i in adata.obs_names]
adata.obsm['spatial'] = cell_table_new.loc[:, centroid_cols].values
os.makedirs(os.path.join(NT_DIR, 'adata'), exist_ok=True)
adata.write_h5ad(os.path.join(NT_DIR, 'adata', 'adata.h5ad'))

# preprocess data
markers = ['Ki67', 'pH2AX', 'Helios', 'OX40',
       'GATA3', 'TOX', 'IDO', 'AR', 'ICOS', 'TCF1', 'PDGFRB', 'GZMB', 'CD79a',
       'Vimentin', 'Calponin', 'MPO', 'T-bet', 'PD-L1 (73-10)',
       'PD-L1 (SP142)', 'c-PARP', 'HLA-ABC', 'Caveolin-1', 'PD-1', 'HLA-DR']

functional_marker_thresholds = [[marker, 0.5] for marker in markers]
adata_processed = preprocess_table(adata, functional_marker_thresholds, image_key='fov',
                                   seg_label_key='label', seg_dir=None, mask_dir=None)
adata_processed.write_h5ad(os.path.join(NT_DIR, 'adata', 'adata_preprocessed.h5ad'))

# Initialize the class
adata_processed = anndata.read_h5ad(os.path.join(NT_DIR, 'adata', 'adata_preprocessed.h5ad'))

features = SpaceCat(adata_processed, image_key='fov', seg_label_key='label', cell_area_key='area',
                    cluster_key=['cell_cluster', 'cell_cluster_broad'],
                    compartment_key='compartment', compartment_area_key='compartment_area')

mixing_df = pd.read_csv(os.path.join(NT_DIR, 'intermediate_files/spatial_analysis/mixing_score/homogeneous_mixing_scores-ratio5_count200.csv'))
cols = mixing_df.columns.tolist()
keep_cols = [col for col in cols if 'mixing_score' in col]
mixing_df = mixing_df[['fov'] + keep_cols]
mixing_df = mixing_df[mixing_df.fov.isin(adata_processed.obs.fov.unique())]

# specify cell type pairs to compute a ratio for
ratio_pairings = [('CD8T', 'CD4T'), ('CD4T', 'Treg'), ('CD8T', 'Treg')]
# specify addtional per cell and per image stats
per_cell_stats = [
    ['morphology', 'cell_cluster', ['cell_size']]
]
per_img_stats = [
    ['mixing_score', mixing_df]
]

# Generate features and save anndata
adata_processed = features.run_spacecat(functional_feature_level='cell_cluster', diversity_feature_level='cell_cluster',
                                        pixel_radius=25,
                                        specified_ratios_cluster_key='cell_cluster', specified_ratios=ratio_pairings,
                                        per_cell_stats=per_cell_stats, per_img_stats=per_img_stats,
                                        correlation_filtering_thresh=0.95)
adata_processed.write_h5ad(os.path.join(NT_DIR, 'adata', 'adata_processed.h5ad'))

# Save finalized tables to csv
folder = 'SpaceCat'
os.makedirs(os.path.join(NT_DIR, folder), exist_ok=True)
adata_processed.uns['combined_feature_data'].to_csv(os.path.join(NT_DIR, folder, 'combined_feature_data.csv'), index=False)
adata_processed.uns['combined_feature_data_filtered'].to_csv(os.path.join(NT_DIR, folder, 'combined_feature_data_filtered.csv'), index=False)
adata_processed.uns['feature_metadata'].to_csv(os.path.join(NT_DIR, folder, 'feature_metadata.csv'), index=False)
adata_processed.uns['excluded_features'].to_csv(os.path.join(NT_DIR, folder, 'excluded_features.csv'), index=False)

pred_chemo = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat/prediction_model/chemotherapy/patient_outcomes/all_timepoints_results.csv'))
pred_chemo_immuno = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat/prediction_model/immunotherapy+chemotherapy/patient_outcomes/all_timepoints_results.csv'))
pred_chemo = pred_chemo.rename(columns={'auc_baseline_list': 'baseline_C', 'auc_on_treatment_list': 'on_treatment_C',
                                        'auc_baseline__on_treatment_list': 'both_C'})
pred_chemo_immuno = pred_chemo_immuno.rename(columns={'auc_baseline_list': 'baseline_C&I', 'auc_on_treatment_list': 'on_treatment_C&I',
                                                      'auc_baseline__on_treatment_list': 'both_C&I'})
NT_preds = pd.concat([pred_chemo, pred_chemo_immuno], axis=1)
NT_preds = NT_preds[['baseline_C', 'baseline_C&I', 'on_treatment_C', 'on_treatment_C&I', 'both_C', 'both_C&I']]
NT_preds = NT_preds.rename(columns={'baseline_C': 'Baseline (C)', 'baseline_C&I': 'Baseline (C&I)', 'on_treatment_C': 'On-treatment (C)',
                                    'on_treatment_C&I': 'On-treatment (C&I)', 'both_C': 'Both (C)', 'both_C&I': 'Both (C&I)'})

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.stripplot(data=NT_preds.melt(), x='variable', y='value', ax=ax, dodge=True, color='#1f77b4')
sns.boxplot(data=NT_preds.melt(), x='variable', y='value', ax=ax, showfliers=False, color='#1f77b4', width=0.6)
ax.scatter(list(range(0, 6)), [0.48, 0.77, 0.69, 0.77, 0.55, 0.82], color='gold', marker='o', s=16)
plt.xticks(rotation=45)
plt.ylim(0.45, 0.85)
plt.title('SpaceCat Prediction on NT data')
plt.ylabel('AUC')
plt.xlabel('')

# Create custom legend handles
blue_line = mlines.Line2D([], [], color="#1f77b4", marker="o", label="SpaceCat", linestyle='None')
orange_line = mlines.Line2D([], [], color="gold", marker="o", label="Wang et al.", linestyle='None')
plt.legend(handles=[blue_line, orange_line], loc='lower right')
sns.despine()
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'NTPublic', 'NT_prediction_comparison.pdf'), bbox_inches='tight', dpi=300)


## 3.2 Low cellularity ##

low_cellularity_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, 'low_cellularity')
os.makedirs(low_cellularity_viz_dir, exist_ok=True)

###### change this dir and upload finalized tonic spacecat files
adata = anndata.read_h5ad(os.path.join(SPACECAT_DIR, 'adata', 'adata_preprocessed.h5ad'))
cell_table = adata.obs

cluster_counts = np.unique(cell_table.fov, return_counts=True)[1]
plt.figure(figsize=(8, 6))
g = sns.histplot(data=cluster_counts, bins=32)
y = cell_table.groupby(by='fov', observed=True).count().sort_values(by='label').label

plt.axvline(x=y.iloc[int(round(len(y)*.05, 0))], color='pink', linestyle='--')
plt.axvline(x=y.iloc[int(round(len(y)*.1, 0))], color='red', linestyle='--')
plt.axvline(x=y.iloc[int(round(len(y)*.15, 0))], color='purple', linestyle='--')
plt.axvline(x=y.iloc[int(round(len(y)*.2, 0))], color='black', linestyle='--')
sns.despine()
plt.title("Histogram of Cell Counts per Image")
plt.xlabel("Number of Cells in an Image")

# Create custom legend handles
line_5 = mlines.Line2D([], [], color="pink", marker='_', label="5% of images", linestyle='None')
line_10 = mlines.Line2D([], [], color="red", marker='_', label="10% of images", linestyle='None')
line_15 = mlines.Line2D([], [], color="purple", marker='_', label="15% of images", linestyle='None')
line_20 = mlines.Line2D([], [], color="black", marker='_', label="20% of images", linestyle='None')
plt.legend(handles=[line_5, line_10, line_15, line_20], loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(low_cellularity_viz_dir, 'low_cell_counts_thresholded.pdf'), bbox_inches='tight', dpi=300)

# generate predictions with various low cellularity images dropped
for drop_perc in [0.0, 0.05, 0.10, 0.15, 0.20]:
    adata = anndata.read_h5ad(os.path.join(SPACECAT_DIR, 'adata', 'adata_preprocessed.h5ad'))
    cell_counts = cell_table.groupby(by='fov').count().sort_values(by='label').label
    keep_fovs = cell_counts.iloc[int(round(len(y) * drop_perc, 0)):].index
    adata = adata[adata.obs.fov.isin(keep_fovs)]

    # read in image level dataframes
    fiber_df = pd.read_csv(os.path.join(SPACECAT_DIR, 'fiber_stats_table.csv'))
    fiber_df = fiber_df[fiber_df.fov.isin(keep_fovs)]

    fiber_tile_df = pd.read_csv(os.path.join(SPACECAT_DIR, 'fiber_stats_per_tile.csv'))
    fiber_tile_df = fiber_tile_df[fiber_tile_df.fov.isin(keep_fovs)]

    mixing_df = pd.read_csv(os.path.join(SPACECAT_DIR, 'formatted_mixing_scores.csv'))
    mixing_df = mixing_df[mixing_df.fov.isin(keep_fovs)]

    kmeans_img_proportions = pd.read_csv(os.path.join(SPACECAT_DIR, 'neighborhood_image_proportions.csv'))
    kmeans_img_proportions = kmeans_img_proportions[kmeans_img_proportions.fov.isin(keep_fovs)]

    kmeans_compartment_proportions = pd.read_csv(os.path.join(SPACECAT_DIR, 'neighborhood_compartment_proportions.csv'))
    kmeans_compartment_proportions = kmeans_compartment_proportions[kmeans_compartment_proportions.fov.isin(keep_fovs)]

    pixie_ecm = pd.read_csv(os.path.join(SPACECAT_DIR, 'pixie_ecm_stats.csv'))
    pixie_ecm = pixie_ecm[pixie_ecm.fov.isin(keep_fovs)]

    ecm_frac = pd.read_csv(os.path.join(SPACECAT_DIR, 'ecm_fraction_stats.csv'))
    ecm_frac = ecm_frac[ecm_frac.fov.isin(keep_fovs)]

    ecm_clusters = pd.read_csv(os.path.join(SPACECAT_DIR, 'ecm_cluster_stats.csv'))
    ecm_clusters = ecm_clusters[ecm_clusters.fov.isin(keep_fovs)]

    # specify cell type pairs to compute a ratio for
    ratio_pairings = [('CD8T', 'CD4T'), ('CD4T', 'Treg'), ('CD8T', 'Treg'), ('CD68_Mac', 'CD163_Mac')]
    # specify addtional per cell and per image stats
    per_cell_stats = [
        ['morphology', 'cell_cluster', ['area', 'major_axis_length']],
        ['linear_distance', 'cell_cluster_broad',
         ['distance_to__B', 'distance_to__Cancer', 'distance_to__Granulocyte', 'distance_to__Mono_Mac',
          'distance_to__NK', 'distance_to__Other', 'distance_to__Structural', 'distance_to__T']]

    ]
    per_img_stats = [
        ['fiber', fiber_df],
        ['fiber', fiber_tile_df],
        ['mixing_score', mixing_df],
        ['kmeans_cluster', kmeans_img_proportions],
        ['kmeans_cluster', kmeans_compartment_proportions],
        ['pixie_ecm', pixie_ecm],
        ['ecm_fraction', ecm_frac],
        ['ecm_cluster', ecm_clusters]
    ]

    features = SpaceCat(adata, image_key='fov', seg_label_key='label', cell_area_key='area',
                        cluster_key=['cell_cluster', 'cell_cluster_broad'],
                        compartment_key='compartment', compartment_area_key='compartment_area')

    # Generate features and save anndata
    adata_processed = features.run_spacecat(functional_feature_level='cell_cluster',
                                            diversity_feature_level='cell_cluster', pixel_radius=50,
                                            specified_ratios_cluster_key='cell_cluster',
                                            specified_ratios=ratio_pairings,
                                            per_cell_stats=per_cell_stats, per_img_stats=per_img_stats)
    os.makedirs(os.path.join(SPACECAT_DIR, 'low_cellularity/adata'), exist_ok=True)
    adata_processed.write_h5ad(os.path.join(SPACECAT_DIR, 'low_cellularity/adata', f'adata_processed_{str(drop_perc * 100)}.h5ad'))

    # Save finalized tables to csv
    output_dir = os.path.join(SPACECAT_DIR, f'low_cellularity/SpaceCat_{str(drop_perc * 100)}')
    os.makedirs(output_dir, exist_ok=True)
    adata_processed.uns['combined_feature_data'].to_csv(os.path.join(output_dir, 'combined_feature_data.csv'), index=False)
    adata_processed.uns['combined_feature_data_filtered'].to_csv(os.path.join(output_dir, 'combined_feature_data_filtered.csv'), index=False)
    adata_processed.uns['feature_metadata'].to_csv(os.path.join(output_dir, 'feature_metadata.csv'), index=False)
    adata_processed.uns['excluded_features'].to_csv(os.path.join(output_dir, 'excluded_features.csv'), index=False)

# run postprocessing and prediction scripts
# read in result
df_0 = pd.read_csv(os.path.join(SPACECAT_DIR, 'low_cellularity/SpaceCat_0.0/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
df_5 = pd.read_csv(os.path.join(SPACECAT_DIR, 'low_cellularity/SpaceCat_5.0/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
df_10 = pd.read_csv(os.path.join(SPACECAT_DIR, 'low_cellularity/SpaceCat_10.0/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
df_15 = pd.read_csv(os.path.join(SPACECAT_DIR, 'low_cellularity/SpaceCat_15.0/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
df_20 = pd.read_csv(os.path.join(SPACECAT_DIR, 'low_cellularity/SpaceCat_20.0/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))

df_0 = df_0.mean()
df_5 = df_5.mean()
df_10 = df_10.mean()
df_15 = df_15.mean()
df_20 = df_20.mean()
df = pd.concat([df_0, df_5, df_10, df_15, df_20], axis=1)
df = df.rename(columns={0: 'All images', 1: 'Drop 5% of images', 2: 'Drop 10% of images',
                        3: 'Drop 15% of images', 4: 'Drop 20% of images'})
df = df.reset_index()
df.replace('auc_on_nivo_list', 'On nivo', inplace=True)
df.replace('auc_post_induction_list', 'Pre nivo', inplace=True)
df.replace('auc_primary_list', 'Primary', inplace=True)
df.replace('auc_baseline_list', 'Baseline', inplace=True)
df['order'] = df['index'].replace({'Primary':0, 'Baseline':1, 'Pre nivo':2, 'On nivo': 3})
df = df.sort_values(by='order')
df = df.drop(columns=['order'])

df = df.rename(columns={'index':'Timepoint'})
df = pd.melt(df, ['Timepoint'])

sns.scatterplot(data=df[df.variable=='All images'], x='Timepoint', y='value', hue='variable', palette=sns.color_palette(['black']), edgecolors='black')
sns.scatterplot(data=df[df.variable!='All images'], x='Timepoint', y='value', hue='variable', palette=sns.color_palette(['dimgrey', 'darkgrey', 'lightgrey', 'whitesmoke']), edgecolors='black')
plt.xticks(rotation=30)
plt.ylabel('Mean AUC')
plt.title('Prediction when dropping low cellularity images')
sns.despine()
plt.gca().legend().set_title('')
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'low_cellularity', 'low_cellularity_prediction_comparisons.pdf'), bbox_inches = 'tight', dpi =300)


## 3.7 Fiber feature usefulness ##

# color fibers by alignment & length stats
fiber_table = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'fiber_segmentation_processed_data/fiber_object_table.csv'))
fibseg_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, 'fiber_features/fiber_masks')

feature_fovs = {
    'alignment_score': ['TONIC_TMA14_R8C3', 'TONIC_TMA4_R10C1'],
    'major_axis_length': ['TONIC_TMA7_R9C4', 'TONIC_TMA13_R8C1']
}
for metric in feature_fovs:
    fov_list = feature_fovs[metric]
    fiber_table_sub = fiber_table[fiber_table.fov.isin(fov_list)]
    fiber_table_sub[f'{metric}_norm'] = np.log(fiber_table_sub[metric])
    if metric == 'alignment_score':
        fiber_table_sub[f'{metric}_norm'] = fiber_table_sub[f'{metric}_norm'] + 0.5
    metric = f'{metric}_norm'

    prop_dir = os.path.join(fibseg_dir, f'colored_{metric}')
    color_segmentation_by_stat(
        fovs=fiber_table_sub.fov.unique(), data_table=fiber_table_sub, seg_dir=fibseg_dir, save_dir=prop_dir,
        stat_name=metric,
        cmap="Blues", seg_suffix="_fiber_labels.tiff", erode=True)

#  ECM stats
ecm_cluster_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, 'ECM_clusters')
os.makedirs(ecm_cluster_viz_dir, exist_ok=True)

ecm_dir = os.path.join(INTERMEDIATE_DIR, 'ecm')
tiled_crops = pd.read_csv(os.path.join(ecm_dir, 'tiled_crops.csv'))
harmonized_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'harmonized_metadata.csv'))

# plot distribution of clusters in each fov
cluster_counts = tiled_crops.groupby('fov').value_counts(['tile_cluster'])
cluster_counts = cluster_counts.reset_index()
cluster_counts.columns = ['fov', 'tile_cluster', 'count']
cluster_counts = cluster_counts.pivot(index='fov', columns='tile_cluster', values='count')
cluster_counts = cluster_counts.fillna(0)
cluster_counts = cluster_counts.apply(lambda x: x / x.sum(), axis=1)

# plot the cluster counts
cluster_counts_clustermap = sns.clustermap(cluster_counts, cmap='Reds', figsize=(10, 10))
plt.close()

# use kmeans to cluster fovs
fov_kmeans_pipe = make_pipeline(KMeans(n_clusters=5, random_state=0))
fov_kmeans_pipe.fit(cluster_counts.values)
kmeans_preds = fov_kmeans_pipe.predict(cluster_counts.values)

# get average cluster count in each kmeans cluster
cluster_counts['fov_cluster'] = kmeans_preds
cluster_means = cluster_counts.groupby('fov_cluster').mean()
cluster_means_clustermap = sns.clustermap(cluster_means, cmap='Reds', figsize=(10, 10))
plt.close()

rename_dict = {0: 'No_ECM', 1: 'Hot_Coll', 2: 'Cold_Coll', 3: 'Hot_Cold_Coll', 4: 'Hot_No_ECM'}
cluster_counts['fov_cluster'] = cluster_counts['fov_cluster'].replace(rename_dict)

# correlate ECM subtypes with patient data using hierarchically clustered heatmap as index
test_fov = 'TONIC_TMA10_R10C6'
test_fov_idx = np.where(cluster_counts.index == test_fov)[0][0]
stop_idx = np.where(cluster_counts_clustermap.dendrogram_row.reordered_ind == test_fov_idx)[0][0]
cluster_counts_subset = cluster_counts.iloc[cluster_counts_clustermap.dendrogram_row.reordered_ind[:stop_idx + 1], :]

test_fov_2 = 'TONIC_TMA11_R10C6'
test_fov_idx_2 = np.where(cluster_counts.index == test_fov_2)[0][0]
stop_idx_2 = np.where(cluster_counts_clustermap.dendrogram_row.reordered_ind == test_fov_idx_2)[0][0]

cluster_counts_subset_2 = cluster_counts.iloc[cluster_counts_clustermap.dendrogram_row.reordered_ind[stop_idx:stop_idx_2 + 1], :]
cluster_counts_subset_3 = cluster_counts.iloc[cluster_counts_clustermap.dendrogram_row.reordered_ind[stop_idx_2:], :]

harmonized_metadata['ecm_cluster'] = 'inflamed'
harmonized_metadata.loc[harmonized_metadata.fov.isin(cluster_counts_subset.index), 'ecm_cluster'] = 'no_ecm'
harmonized_metadata.loc[harmonized_metadata.fov.isin(cluster_counts_subset_2.index), 'ecm_cluster'] = 'cold_collagen'
df = harmonized_metadata[['Tissue_ID', 'Localization', 'ecm_cluster']].groupby(by=['Localization', 'ecm_cluster']).count().reset_index()
df['Tissue_ID'] = df['Tissue_ID'] / df.groupby('Localization')['Tissue_ID'].transform('sum')

# plot the distribution of ECM subtypes in each patient by localization
g = sns.barplot(x='Localization', y='Tissue_ID', hue='ecm_cluster', data=df)
plt.xticks(rotation=45)
plt.title('ECM cluster by localization')
plt.ylim(0, 1)
plt.ylabel('Proportion')
sns.despine()
plt.savefig(os.path.join(ecm_cluster_viz_dir, 'ECM_cluster_localization.pdf'), bbox_inches='tight', dpi=300)

# plot marker expression in each ECM subtype
core_df_func = pd.read_csv(os.path.join(OUTPUT_DIR, 'functional_df_per_core_filtered_deduped.csv'))
plot_df = core_df_func.merge(harmonized_metadata[['fov', 'ecm_cluster']], on='fov', how='left')
plot_df = plot_df[plot_df.Timepoint == 'primary']

# look at all fibroblasts
temp_df = plot_df[plot_df.subset == 'all']
temp_df = temp_df[temp_df.metric == 'cluster_broad_freq']
temp_df = temp_df[temp_df.cell_type == 'Structural']
temp_df = temp_df[temp_df.functional_marker.isin(['PDL1', 'TIM3', 'IDO', 'HLADR', 'GLUT1'])]

g = sns.catplot(x='ecm_cluster', y='value', data=temp_df,
                kind='box', col='functional_marker', sharey=False, color='grey', showfliers=False)
g.map_dataframe(sns.stripplot, x="ecm_cluster", y="value", data=temp_df, color='black', jitter=0.3)
plt.suptitle('Functional expression in Structural Cells')
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'ECM_clusters', 'Functional_expression_Structural.pdf'), bbox_inches='tight', dpi=300)

# look at M2 macrophages
temp_df = plot_df[plot_df.subset == 'all']
temp_df = temp_df[temp_df.metric == 'cluster_freq']
temp_df = temp_df[temp_df.cell_type == 'CD68_Mac']
temp_df = temp_df[temp_df.functional_marker.isin(['PDL1', 'TIM3', 'IDO', 'HLADR', 'GLUT1'])]

g = sns.catplot(x='ecm_cluster', y='value', data=temp_df,
                kind='box', col='functional_marker', sharey=False, color='grey', showfliers=False)
g.map_dataframe(sns.stripplot, x="ecm_cluster", y="value", data=temp_df, color='black', jitter=0.3)
plt.suptitle('Functional expression in CD68 Macrophages')
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'ECM_clusters', 'Functional_expression_CD68_Mac.pdf'), bbox_inches='tight', dpi=300)


## 3.9 Location bias for features associated with immune cells ##

###### change this dir and upload finalized tonic spacecat files
ranked_features_all = pd.read_csv(os.path.join(SPACECAT_DIR, 'SpaceCat/feature_ranking.csv'))
ranked_features = ranked_features_all.loc[
    ranked_features_all.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo'])]
feature_metadata = pd.read_csv(os.path.join(SPACECAT_DIR, 'SpaceCat/feature_metadata.csv'))

immune_cells = ['Immune', 'Mono_Mac', 'B', 'T', 'Granulocyte', 'NK'] + ['CD68_Mac', 'CD163_Mac', 'Mac_Other',
                                                                        'Monocyte', 'APC'] + ['CD4T', 'CD8T', 'Treg',
                                                                                              'T_Other'] + [
                   'Neutrophil', 'Mast']
immmune_feature_names = []
for cell in (immune_cells):
    for feat in ranked_features.feature_name.unique():
        if cell in feat:
            immmune_feature_names.append(feat)

immune_features = ranked_features[ranked_features.feature_name.isin(immmune_feature_names)]
immune_features_comp = immune_features[immune_features.compartment != 'all']

fig, axs = plt.subplots(2, 2, figsize=(8, 8), layout='constrained')
fig.suptitle('Changes in compartment features when dropping immune specific features')
fig2, axs2 = plt.subplots(2, 2, figsize=(8, 8), layout='constrained')
fig2.suptitle('Changes in compartment features when dropping immune specific features')

for immune_drop, coords in [[0, (0, 0)], [0.10, (0, 1)], [0.25, (1, 0)], [0.50, (1, 1)]]:
    i, j = coords
    feature_df = immune_features_comp
    if immune_drop == 0.0:
        feature_df = ranked_features
    # subset features
    idx_list = list(feature_df.index)
    sample_perc = int(len(idx_list) * immune_drop)
    sub_idx_list = random.sample(idx_list, sample_perc)
    sub_df = ranked_features[~ranked_features.index.isin(sub_idx_list)]
    sub_df_comp = ranked_features[~ranked_features.index.isin(sub_idx_list)]
    sub_df_comp = sub_df_comp[sub_df_comp.compartment != 'all']

    # calculate abundance of each compartment in the top 100 and across all features
    total_counts = sub_df_comp.groupby('compartment').count().iloc[:, 0]
    sub_prop = total_counts / np.sum(total_counts)

    # create df
    ratio_df = pd.DataFrame({'compartment': sub_prop.index, 'ratio': sub_prop.values})
    ratio_df = ratio_df.sort_values(by='ratio', ascending=False)
    ratio_df.loc[ratio_df.compartment == 'all', 'color_order'] = 5
    ratio_df.loc[ratio_df.compartment == 'cancer_core', 'color_order'] = 1
    ratio_df.loc[ratio_df.compartment == 'cancer_border', 'color_order'] = 2
    ratio_df.loc[ratio_df.compartment == 'stroma_border', 'color_order'] = 3
    ratio_df.loc[ratio_df.compartment == 'stroma_core', 'color_order'] = 4

    cmap = ['blue', 'deepskyblue', 'lightcoral', 'firebrick', 'grey']
    sns.barplot(data=ratio_df, x='compartment', y='ratio', hue='color_order', palette=cmap, ax=axs[i][j])
    sns.despine()
    axs[i][j].set_ylim(0, 0.5)
    axs[i][j].set_ylabel('')
    axs[i][j].tick_params(axis='x', labelrotation=60)
    axs[i][j].get_legend().remove()
    axs[i][j].set_title(f'Drop {int(immune_drop * 100)}%')
    if immune_drop == 0.0:
        axs[i][j].set_title(f'All features')

    # look at enrichment by compartment
    top_counts = sub_df.iloc[:100, :].groupby('compartment').count().iloc[:, 0]
    total_counts = sub_df.groupby('compartment').count().iloc[:, 0]

    # calculate abundance of each compartment in the top 100 and across all features
    top_prop = top_counts / np.sum(top_counts)
    total_prop = total_counts / np.sum(total_counts)
    top_ratio = top_prop / total_prop
    top_ratio = np.log2(top_ratio)

    # create df
    ratio_df = pd.DataFrame({'compartment': top_ratio.index, 'ratio': top_ratio.values})
    ratio_df = ratio_df.sort_values(by='ratio', ascending=False)

    ratio_df.loc[ratio_df.compartment == 'all', 'color_order'] = 1
    ratio_df.loc[ratio_df.compartment == 'cancer_core', 'color_order'] = 2
    ratio_df.loc[ratio_df.compartment == 'cancer_border', 'color_order'] = 3
    ratio_df.loc[ratio_df.compartment == 'stroma_border', 'color_order'] = 4
    ratio_df.loc[ratio_df.compartment == 'stroma_core', 'color_order'] = 5

    cmap = ['grey', 'blue', 'deepskyblue', 'lightcoral', 'firebrick']

    sns.barplot(data=ratio_df, x='compartment', y='ratio', hue='color_order', palette=cmap, ax=axs2[i][j])
    sns.despine()
    axs2[i][j].set_ylim(-0.8, 1.3)
    axs2[i][j].set_ylabel('')
    axs2[i][j].tick_params(axis='x', labelrotation=60)
    axs2[i][j].get_legend().remove()
    axs2[i][j].set_title(f'Drop {int(immune_drop * 100)}%')
    if immune_drop == 0.0:
        axs2[i][j].set_title(f'All features')

os.makedirs(os.path.join(SUPPLEMENTARY_FIG_DIR, 'location_bias'), exist_ok=True)
fig.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'location_bias', 'location_bias_compartment_features.pdf'),
            bbox_inches='tight', dpi=300)
fig2.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'location_bias', 'location_bias_all_features.pdf'),
             bbox_inches='tight', dpi=300)


## 4.1 Wang et al. cell interactions ##

NT_interactions_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, 'NT_interaction_features')
os.makedirs(NT_interactions_viz_dir, exist_ok=True)

adata = anndata.read_h5ad(os.path.join(SPACECAT_DIR, 'SpaceCat/adata/adata_preprocessed.h5ad'))
neighborhood_mat = adata.obsm['neighbors_counts_cell_cluster_radius50']
neighborhood_mat = pd.concat([adata.obs.loc[:, ['fov', 'label', 'cell_cluster']], neighborhood_mat], axis=1)
interactions_df = neighborhood_mat[['fov', 'cell_cluster']]

# define epithelial and TME cell clusters
ep_cells = ['Cancer_1', 'Cancer_2', 'Cancer_3']
tme_cells = ['CD68_Mac', 'CD163_Mac', 'Mac_Other', 'Monocyte', 'APC', 'B',
             'CD4T', 'CD8T', 'Treg', 'T_Other', 'Neutrophil', 'Mast',
             'Endothelium', 'CAF', 'Fibroblast', 'Smooth_Muscle', 'NK', 'Immune_Other', 'Other']

interactions_df['Epi'] = neighborhood_mat[ep_cells].sum(axis=1)
interactions_df['TME'] = neighborhood_mat[tme_cells].sum(axis=1)
interactions_df = interactions_df.groupby(by=['fov', 'cell_cluster']).sum(numeric_only=True).reset_index()

interactions_df = interactions_df.merge(adata.obs[['fov', 'area']].groupby(by='fov').count().reset_index(), on='fov')
interactions_df = interactions_df.rename(columns={'area': 'total_cells'})
interactions_df['Epi'] = interactions_df['Epi'] / interactions_df['total_cells']
interactions_df['TME'] = interactions_df['TME'] / interactions_df['total_cells']
interactions_df['isEpi'] = interactions_df['cell_cluster'].isin(ep_cells)
interactions_df['isEpi'] = ['Epi' if check else 'TME' for check in interactions_df['isEpi']]

# sum total homotypic and heterotypic interactions per cell type
interactions_df = pd.melt(interactions_df, id_vars=['fov', 'cell_cluster', 'isEpi'], value_vars=['Epi', 'TME'])
interactions_df['feature_type'] = [cell + 'Hom' if cell == interactions_df['variable'][i] else cell + 'Het' for i, cell in enumerate(interactions_df.isEpi)]
interactions_df['feature'] = interactions_df["cell_cluster"].astype(str) + '__' + interactions_df["feature_type"].astype(str)
interactions_df = interactions_df.pivot(index='fov', columns='feature', values='value').reset_index()
interactions_df.to_csv(os.path.join(NT_interactions_viz_dir, 'Epi_TME_interactions.csv'), index=False)

# run SpaceCat with interaction features included
adata = anndata.read_h5ad(os.path.join(SPACECAT_DIR, 'adata', 'adata.h5ad'))

# read in image level dataframes
fiber_df = pd.read_csv(os.path.join(SPACECAT_DIR, 'fiber_stats_table.csv'))
fiber_tile_df = pd.read_csv(os.path.join(SPACECAT_DIR, 'fiber_stats_per_tile.csv'))
mixing_df = pd.read_csv(os.path.join(SPACECAT_DIR, 'formatted_mixing_scores.csv'))
kmeans_img_proportions = pd.read_csv(os.path.join(SPACECAT_DIR, 'neighborhood_image_proportions.csv'))
kmeans_compartment_proportions = pd.read_csv(os.path.join(SPACECAT_DIR, 'neighborhood_compartment_proportions.csv'))
pixie_ecm = pd.read_csv(os.path.join(SPACECAT_DIR, 'pixie_ecm_stats.csv'))
ecm_frac = pd.read_csv(os.path.join(SPACECAT_DIR, 'ecm_fraction_stats.csv'))
ecm_clusters = pd.read_csv(os.path.join(SPACECAT_DIR, 'ecm_cluster_stats.csv'))
cell_interactions = pd.read_csv(os.path.join(NT_interactions_viz_dir, 'Epi_TME_interactions.csv'))

# specify cell type pairs to compute a ratio for
ratio_pairings = [('CD8T', 'CD4T'), ('CD4T', 'Treg'), ('CD8T', 'Treg'), ('CD68_Mac', 'CD163_Mac')]
# specify addtional per cell and per image stats
per_cell_stats=[
    ['morphology', 'cell_cluster', ['area', 'major_axis_length']],
    ['linear_distance', 'cell_cluster_broad', ['distance_to__B', 'distance_to__Cancer', 'distance_to__Granulocyte', 'distance_to__Mono_Mac',
                                               'distance_to__NK', 'distance_to__Other', 'distance_to__Structural', 'distance_to__T']]

]
per_img_stats=[
    ['fiber', fiber_df],
    ['fiber', fiber_tile_df],
    ['mixing_score', mixing_df],
    ['kmeans_cluster', kmeans_img_proportions],
    ['kmeans_cluster', kmeans_compartment_proportions],
    ['pixie_ecm', pixie_ecm],
    ['ecm_fraction', ecm_frac],
    ['ecm_cluster', ecm_clusters],
    ['cell_interactions', cell_interactions]
]

features = SpaceCat(adata, image_key='fov', seg_label_key='label', cell_area_key='area',
                    cluster_key=['cell_cluster', 'cell_cluster_broad'],
                    compartment_key='compartment', compartment_area_key='compartment_area')
# Generate features and save anndata
adata_processed = features.run_spacecat(functional_feature_level='cell_cluster', diversity_feature_level='cell_cluster', pixel_radius=50,
                                        specified_ratios_cluster_key='cell_cluster', specified_ratios=ratio_pairings,
                                        per_cell_stats=per_cell_stats, per_img_stats=per_img_stats)
os.makedirs(os.path.join(NT_interactions_viz_dir, 'adata'), exist_ok=True)
adata_processed.write_h5ad(os.path.join(NT_interactions_viz_dir, 'adata', 'adata_processed.h5ad'))

# plot prediction comparison
preds = pd.read_csv(os.path.join(NT_interactions_viz_dir, 'prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
preds = preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
preds = preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                              'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
preds = preds.melt()
preds['Analysis'] = 'excluding'

adj_preds = pd.read_csv(os.path.join(NT_interactions_viz_dir, 'prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
adj_preds = adj_preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
adj_preds = adj_preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                                      'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
adj_preds = adj_preds.melt()
adj_preds['Analysis'] = 'including'
all_preds = pd.concat([preds, adj_preds])

fig, ax = plt.subplots()
sns.boxplot(data=all_preds, x='variable', y='value', ax=ax, width=0.6, hue='Analysis',
           palette=sns.color_palette(["#1f77b4", "gold"]), showfliers=False)
sns.stripplot(data=all_preds, x='variable', y='value', ax=ax, hue='Analysis',
           palette=sns.color_palette(["#1f77b4", "gold"]), dodge=True, jitter=0.2)

plt.xticks(rotation=45)
plt.title('TONIC prediction including and excluding interaction features')
plt.ylabel('AUC')
plt.xlabel('')
plt.legend(loc='lower right').set_title('')
sns.despine()
blue_line = mlines.Line2D([], [], color="#1f77b4", marker="o", label="excluding", linestyle='None')
yellow_line = mlines.Line2D([], [], color="gold", marker="o", label="including", linestyle='None')
plt.legend(handles=[blue_line, yellow_line], loc='lower right')
plt.savefig(os.path.join(NT_interactions_viz_dir, 'cell_interaction_prediction_comparison.pdf'), bbox_inches='tight', dpi=300)

## correlation plots for interaction features
fov_data_df = pd.read_csv(os.path.join(os.path.join(NT_interactions_viz_dir, 'SpaceCat/combined_feature_data_filtered.csv')))
fov_data_wide = fov_data_df.pivot(index='fov', columns='feature_name_unique', values='normalized_value')
corr_df = fov_data_wide.corr(method='spearman')
corr_df = corr_df.fillna(0)
corr_df = corr_df[[col for col in corr_df.columns if np.logical_or('Het' in col, 'Hom' in col)]]
corr_df = corr_df[np.logical_and(~corr_df.index.str.contains('Het'), ~corr_df.index.str.contains('Hom'))]

# heatmap for all features
clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(NT_interactions_viz_dir, 'interaction_feature_clustermap_filtered.pdf'), dpi=300)
pd.DataFrame(clustergrid.data2d.index).to_csv(os.path.join(NT_interactions_viz_dir, 'interaction_feature_clustermap_order.csv'))
plt.close()

# heatmap for >0.7 correlated features
sub_df = corr_df[corr_df.max(axis=1)>0.7]
clustergrid = sns.clustermap(sub_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(NT_interactions_viz_dir, 'interaction_feature_clustermap_filtered_corr70.pdf'), dpi=300)
pd.DataFrame(clustergrid.data2d.index).to_csv(os.path.join(NT_interactions_viz_dir, 'interaction_feature_clustermap_order_corr70.csv'))
plt.close()

# heatmap for >0.9 correlated features
sub_df = corr_df[corr_df.max(axis=1)>0.9]
clustergrid = sns.clustermap(sub_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'NT_interaction_features/interaction_feature_clustermap_filtered_corr90.pdf'), dpi=300)
pd.DataFrame(clustergrid.data2d.index).to_csv(os.path.join(SUPPLEMENTARY_FIG_DIR, 'NT_interaction_features/interaction_feature_clustermap_order_corr90.csv'))
plt.close()

## correlation plots for mixing score features
corr_df = fov_data_wide.corr(method='spearman')
corr_df = corr_df.fillna(0)
corr_df = corr_df[[col for col in corr_df.columns if 'mixing_score' in col]]
corr_df = corr_df[np.logical_and(~corr_df.index.str.contains('Het'), ~corr_df.index.str.contains('Hom'))]
corr_df = corr_df[~corr_df.index.str.contains('mixing_score')]

# heatmap for all features
clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'NT_interaction_features/mixing_score_feature_clustermap_filtered.pdf'), dpi=300)
pd.DataFrame(clustergrid.data2d.index).to_csv(os.path.join(SUPPLEMENTARY_FIG_DIR, 'NT_interaction_features/mixing_score_feature_clustermap_order.csv'))
plt.close()

# heatmap for >0.7 correlated features
sub_df = corr_df[corr_df.max(axis=1)>0.7]
clustergrid = sns.clustermap(sub_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'NT_interaction_features/mixing_score_feature_clustermap_filtered_corr70.pdf'), dpi=300)
pd.DataFrame(clustergrid.data2d.index).to_csv(os.path.join(SUPPLEMENTARY_FIG_DIR, 'NT_interaction_features/mixing_score_feature_clustermap_order_corr70.csv'))
plt.close()


## 4.6.1 immune_agg features ##

immune_agg_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "immune_agg_features")
os.makedirs(immune_agg_viz_dir, exist_ok=True)

# generate violin plots for top immune_agg features
ranked_features_df = pd.read_csv(os.path.join(immune_agg_viz_dir, 'feature_ranking.csv'))
combined_df = pd.read_csv(os.path.join(immune_agg_viz_dir, 'timepoint_combined_features_outcome_labels.csv'))
feat_list = list(ranked_features_df[ranked_features_df.compartment == 'immune_agg'].feature_name_unique.drop_duplicates()[:11]) + ['immune_agg__proportion']
input_df_filtered = ranked_features_df[np.isin(ranked_features_df['feature_name_unique'], feat_list)].copy()
timepoints = ['primary', 'baseline', 'pre_nivo', 'on_nivo']

# plot the results
_, axes = plt.subplots(4, 3, figsize = (4.5*4, 4.5*3), gridspec_kw={'hspace': 0.6, 'wspace': 0.5, 'bottom':0.05})
for idx, feature in enumerate(input_df_filtered.feature_name_unique.unique()):
    feature_subset = combined_df.loc[(combined_df.feature_name_unique == feature), :]
    feature_subset = feature_subset.loc[(feature_subset.Timepoint.isin(timepoints)), :]
    if len(feature_subset['Clinical_benefit'].unique()) != 2:
        continue
    g = sns.violinplot(data=feature_subset, x='Clinical_benefit', y='raw_mean', linewidth=1, width=0.6,
                       palette=['#377eb8', '#e41a1c'], order=['No', 'Yes'], ax=axes.flat[idx], hue='Clinical_benefit', legend=False)
    g = sns.stripplot(data=feature_subset, x='Clinical_benefit', y='raw_mean', linewidth=0.8, size=5, edgecolor="black",
                      dodge=False, palette=['#377eb8', '#e41a1c'], order=['No', 'Yes'], ax=axes.flat[idx], hue='Clinical_benefit', legend=False)
    g.set_title(feature, fontsize=14)
    g.tick_params(labelsize=14)
    g.set_xlabel('Clinical_benefit', fontsize=14)
    g.set_ylabel('Frequency', fontsize=14)
    fdr_pval = input_df_filtered[np.isin(input_df_filtered['feature_name_unique'], feature)]['fdr_pval'].values[0]
    plt.text(.3, .9, "fdr_pval ={:.2f}".format(fdr_pval), transform=g.transAxes, fontsize=14)
plt.savefig(os.path.join(immune_agg_viz_dir, 'immune_agg_feature_plots.pdf'), bbox_inches='tight')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ranked_features_all = pd.read_csv(os.path.join(SUPPLEMENTARY_FIG_DIR, 'immune_agg_features', 'feature_ranking.csv'))
ranked_features = ranked_features_all.loc[ranked_features_all.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo'])]

# make volcano plots
for subset in ['all', 'immune_agg']:
    plot_title = 'All Features'
    if subset=='immune_agg':
        ranked_features = ranked_features[ranked_features.compartment=='immune_agg']
        plot_title = 'Immune Aggregate Features'


    # plot total volcano
    fig, ax = plt.subplots(figsize=(3,3))
    sns.scatterplot(data=ranked_features, x='med_diff', y='log_pval', alpha=1, hue='importance_score',
                    palette=sns.color_palette("icefire", as_cmap=True), s=2.5, edgecolor='none', ax=ax)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 8)
    sns.despine()

    # add gradient legend
    norm = plt.Normalize(ranked_features.importance_score.min(), ranked_features.importance_score.max())
    sm = plt.cm.ScalarMappable(cmap="icefire", norm=norm)
    ax.get_legend().remove()
    ax.figure.colorbar(sm, ax=ax)
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(os.path.join(immune_agg_viz_dir, f'Figure3a_volcano-{subset}.pdf'))

# generate prediction comparison boxplot
preds = pd.read_csv(os.path.join(SUPPLEMENTARY_FIG_DIR, 'immune_agg_features', 'all_timepoints_results_MIBI-immune_agg.csv'))
preds = preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
preds = preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                              'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
preds = preds.melt()
preds['immune_agg'] = 'include'

old_preds = pd.read_csv(os.path.join(SUPPLEMENTARY_FIG_DIR, 'immune_agg_features', 'all_timepoints_results_MIBI.csv'))
old_preds = old_preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
old_preds = old_preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                                      'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
old_preds = old_preds.melt()
old_preds['immune_agg'] = 'exclude'
all_preds = pd.concat([preds, old_preds])

fig, ax = plt.subplots()
sns.boxplot(data=all_preds, x='variable', y='value', ax=ax, width=0.6, hue='immune_agg',
            palette=sns.color_palette(["#1f77b4", "darkseagreen"]))
sns.stripplot(data=all_preds, x='variable', y='value', ax=ax, hue='immune_agg',
              palette=sns.color_palette(["#1f77b4", "darkseagreen"]), dodge=True, jitter=0.2)
plt.xticks(rotation=45)
plt.title('TONIC prediction including and excluding immune aggregate features')
plt.ylabel('AUC')
plt.xlabel('')
# Add the custom legend
blue_line = mlines.Line2D([], [], color="#1f77b4", marker="o", label="including", linestyle='None')
green_line = mlines.Line2D([], [], color="darkseagreen", marker="o", label="excluding", linestyle='None')
plt.legend(handles=[blue_line, green_line], loc='lower right')
sns.despine()

plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'immune_agg_features', 'immune_agg_prediction_comparison.pdf'),
            bbox_inches='tight', dpi=300)


## 4.10.3 functional marker thresholding in SpaceCat ##

functional_marker_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "functional_marker_auto_thresholds")
os.makedirs(functional_marker_viz_dir, exist_ok=True)

# Functional marker thresholding
cell_table = pd.read_csv(
    os.path.join(ANALYSIS_DIR, "combined_cell_table_normalized_cell_labels_updated.csv")
)
functional_marker_thresholds = [['Ki67', 0.002], ['CD38', 0.004], ['CD45RB', 0.001], ['CD45RO', 0.002],
                                ['CD57', 0.002], ['CD69', 0.002], ['GLUT1', 0.002], ['IDO', 0.001],
                                ['LAG3', 0.002], ['PD1', 0.0005], ['PDL1', 0.001],
                                ['HLA1', 0.001], ['HLADR', 0.001], ['TBET', 0.0015], ['TCF1', 0.001],
                                ['TIM3', 0.001], ['Vim', 0.002], ['Fe', 0.1]]
functional_marker_thresholds = pd.DataFrame(functional_marker_thresholds)
functional_marker_thresholds = functional_marker_thresholds.rename(columns={0: 'marker', 1: 'manual_thresh'})

auto_thresh = []
for marker in functional_marker_thresholds.marker:
    auto_thresh.append(cell_table[marker].values.mean() + cell_table[marker].values.std())
functional_marker_thresholds['auto_thresh'] = auto_thresh

fig, ax = plt.subplots()
x = functional_marker_thresholds.manual_thresh
y = functional_marker_thresholds.auto_thresh

ax.scatter(x=x, y=y, s=20)
ax.plot(x, x, 'red')
ax.set_xlabel("Manual Threshold")
ax.set_ylabel("Automatic Threshold")
ax.set_ylim(0, 0.01)
ax.set_xlim(0, 0.005)
fig.set_figheight(5)
fig.set_figwidth(6)
plt.title("Functional Marker Thresholding")
for i, txt in enumerate(functional_marker_thresholds.marker):
    ax.annotate(txt, (x[i] + 0.00005, y[i] + 0.00005), fontsize=12)

plt.savefig(os.path.join(functional_marker_viz_dir, 'functional_marker_auto_thresholds.pdf'), bbox_inches='tight',
            dpi=300)


## 4.8 NT pre-treatment features ##
