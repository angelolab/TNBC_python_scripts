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
from venny4py.venny4py import venny4py
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests

from SpaceCat.preprocess import preprocess_table
from SpaceCat.features import SpaceCat
from ark.utils.plot_utils import color_segmentation_by_stat
from python_files.utils import compare_populations, compare_timepoints, summarize_timepoint_enrichment

import random
random.seed(13)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ANALYSIS_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files"
INTERMEDIATE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files"
OUTPUT_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/output_files"
REVIEW_FIG_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs/review_figures"
SPACECAT_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/TONIC_SpaceCat"


## 2.5 & 4.3 SpaceCat prediction comparison ##
## 4.1 Wang et al. cell interactions ##

# NT DATA COMPARISON
NT_viz_dir = os.path.join(REVIEW_FIG_DIR, 'NTPublic')

# read in data
NT_DIR = '/Volumes/Shared/Noah Greenwald/NTPublic'
cell_table = pd.read_csv(os.path.join(NT_DIR, 'analysis_files/cell_table.csv'))
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

# NT cell interactions
adata = anndata.read_h5ad(os.path.join(NT_DIR, 'adata', 'adata_preprocessed.h5ad'))
neighborhood_mat = adata.obsm['neighbors_counts_cell_cluster_radius25']
neighborhood_mat = pd.concat([adata.obs.loc[:, ['fov', 'label', 'cell_cluster']], neighborhood_mat], axis=1)
interactions_df = neighborhood_mat[['fov', 'cell_cluster']]

# define epithelial and TME cell clusters
ep_cells = ['Epithelial_1', 'Epithelial_3', 'Epithelial_2']
tme_cells = ['APC', 'B', 'CD4T', 'CD8T', 'DC', 'Endothelial',
       'Fibroblast', 'M2_Mac', 'NK', 'Neutrophil', 'PDPN', 'Plasma',
       'Treg']
interactions_df['Epi'] = neighborhood_mat[ep_cells].sum(axis=1)
interactions_df['TME'] = neighborhood_mat[tme_cells].sum(axis=1)
interactions_df = interactions_df.groupby(by=['fov', 'cell_cluster']).sum(numeric_only=True).reset_index()
interactions_df = interactions_df.merge(adata.obs[['fov', 'cell_size']].groupby(by='fov').count().reset_index(), on='fov')
interactions_df = interactions_df.rename(columns={'cell_size': 'total_cells'})
interactions_df['Epi'] = interactions_df['Epi'] / interactions_df['total_cells']
interactions_df['TME'] = interactions_df['TME'] / interactions_df['total_cells']
interactions_df['isEpi'] = interactions_df['cell_cluster'].isin(ep_cells)
interactions_df['isEpi'] = ['Epi' if check else 'TME' for check in interactions_df['isEpi']]

# sum total homotypic and heterotypic interactions per cell type
interactions_df = pd.melt(interactions_df, id_vars=['fov', 'cell_cluster', 'isEpi'], value_vars=['Epi', 'TME'])
interactions_df['feature_type'] = [cell + 'Hom' if cell == interactions_df['variable'][i] else cell + 'Het' for i, cell in enumerate(interactions_df.isEpi)]
interactions_df['feature'] = interactions_df["cell_cluster"].astype(str) + '__' + interactions_df["feature_type"].astype(str)
interactions_df = interactions_df.pivot(index='fov', columns='feature', values='value').reset_index()
interactions_df.to_csv(os.path.join(NT_DIR, 'SpaceCat', 'Epi_TME_interactions.csv'), index=False)

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
    ['mixing_score', mixing_df],
    ['cell_interactions', interactions_df]
]

# Generate features and save anndata
adata_processed = features.run_spacecat(functional_feature_level='cell_cluster', diversity_feature_level='cell_cluster',
                                        pixel_radius=25,
                                        specified_ratios_cluster_key='cell_cluster', specified_ratios=ratio_pairings,
                                        per_cell_stats=per_cell_stats, per_img_stats=per_img_stats,
                                        correlation_filtering_thresh=0.95)

# Save finalized tables to csv
folder = 'SpaceCat_NT_combined'
os.makedirs(os.path.join(NT_DIR, folder), exist_ok=True)
adata_processed.write_h5ad(os.path.join(NT_DIR, folder, 'adata_processed.h5ad'))
adata_processed.uns['combined_feature_data'].to_csv(os.path.join(NT_DIR, folder, 'combined_feature_data.csv'), index=False)
adata_processed.uns['combined_feature_data_filtered'].to_csv(os.path.join(NT_DIR, folder, 'combined_feature_data_filtered.csv'), index=False)
adata_processed.uns['feature_metadata'].to_csv(os.path.join(NT_DIR, folder, 'feature_metadata.csv'), index=False)
adata_processed.uns['excluded_features'].to_csv(os.path.join(NT_DIR, folder, 'excluded_features.csv'), index=False)

# Save finalized tables to csv
combined_feature_data = adata_processed.uns['combined_feature_data']
combined_feature_data = combined_feature_data[combined_feature_data.feature_type != 'cell_interactions']
combined_feature_data_filtered = adata_processed.uns['combined_feature_data_filtered']
combined_feature_data_filtered = combined_feature_data_filtered[combined_feature_data_filtered.feature_type != 'cell_interactions']
folder = 'SpaceCat'
os.makedirs(os.path.join(NT_DIR, folder), exist_ok=True)
adata_processed.write_h5ad(os.path.join(NT_DIR, folder, 'adata_processed.h5ad'))
combined_feature_data.to_csv(os.path.join(NT_DIR, folder, 'combined_feature_data.csv'), index=False)
combined_feature_data_filtered.to_csv(os.path.join(NT_DIR, folder, 'combined_feature_data_filtered.csv'), index=False)

NT_DIR = '/Volumes/Shared/Noah Greenwald/NTPublic'
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
NT_preds = NT_preds.melt()
NT_preds['cancer_revised'] = 1

pred_chemo = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat_NT_combined/prediction_model/chemotherapy/patient_outcomes/all_timepoints_results.csv'))
pred_chemo_immuno = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat_NT_combined/prediction_model/immunotherapy+chemotherapy/patient_outcomes/all_timepoints_results.csv'))
pred_chemo = pred_chemo.rename(columns={'auc_baseline_list': 'baseline_C', 'auc_on_treatment_list': 'on_treatment_C',
                                        'auc_baseline__on_treatment_list': 'both_C'})
pred_chemo_immuno = pred_chemo_immuno.rename(columns={'auc_baseline_list': 'baseline_C&I', 'auc_on_treatment_list': 'on_treatment_C&I',
                                                      'auc_baseline__on_treatment_list': 'both_C&I'})
combo_preds = pd.concat([pred_chemo, pred_chemo_immuno], axis=1)
combo_preds = combo_preds[['baseline_C', 'baseline_C&I', 'on_treatment_C', 'on_treatment_C&I', 'both_C', 'both_C&I']]
combo_preds = combo_preds.rename(columns={'baseline_C': 'Baseline (C)', 'baseline_C&I': 'Baseline (C&I)', 'on_treatment_C': 'On-treatment (C)',
                                          'on_treatment_C&I': 'On-treatment (C&I)', 'both_C': 'Both (C)', 'both_C&I': 'Both (C&I)'})
combo_preds = combo_preds.melt()
combo_preds['cancer_revised'] = 2

og_preds = pd.read_csv(os.path.join(NT_DIR, 'NT_preds.csv'))
og_preds = og_preds.replace('Base&On', 'Both')
og_preds['variable'] = og_preds['Timepoint'] + ' (' + og_preds['Arm'] + ')'
og_preds = og_preds[['Fold', 'LassoAUC', 'variable']]
og_preds = og_preds.pivot(index='Fold', columns='variable')
og_preds.columns = og_preds.columns.droplevel(0)
og_preds = og_preds.melt()
og_preds['cancer_revised'] = 0
all_preds = pd.concat([NT_preds, og_preds, combo_preds])

fig, ax = plt.subplots()
sns.boxplot(data=all_preds, x='variable', y='value', ax=ax, width=0.6, hue='cancer_revised',
            palette=sns.color_palette(["gold", "#1f77b4", "darkseagreen"]), showfliers=False)
sns.stripplot(data=all_preds, x='variable', y='value', ax=ax, hue='cancer_revised',
              palette=sns.color_palette(["gold", "#1f77b4", "darkseagreen"]), dodge=True, jitter=0.2)
fig.set_figheight(4)
fig.set_figwidth(8)
plt.xticks(rotation=45)
plt.title('Wang et al. dataset predictions')
plt.ylabel('AUC')
plt.xlabel('')
plt.ylim((0, 1))

# Add the custom legend
yellow_line = mlines.Line2D([], [], color="gold", marker="o", label="Wang et al. predictions", linestyle='None')
blue_line = mlines.Line2D([], [], color="#1f77b4", marker="o", label="SpaceCat predictions", linestyle='None')
green_line = mlines.Line2D([], [], color="darkseagreen", marker="o", label="SpaceCat & Wang et al. features", linestyle='None')
plt.legend(handles=[yellow_line, blue_line, green_line], loc='lower right')
sns.despine()
plt.savefig(os.path.join(NT_viz_dir, 'NT_prediction_comparison.pdf'), bbox_inches='tight', dpi=300)

# TONIC DATA COMPARISON
BASE_DIR = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
SpaceCat_dir = os.path.join(BASE_DIR, 'TONIC_SpaceCat')

# add interaction features
adata = anndata.read_h5ad(os.path.join(SpaceCat_dir, 'SpaceCat', 'adata_processed.h5ad'))
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

interactions_df = interactions_df.merge(adata.obs[['fov', 'cell_size']].groupby(by='fov').count().reset_index(), on='fov')
interactions_df = interactions_df.rename(columns={'cell_size': 'total_cells'})
interactions_df['Epi'] = interactions_df['Epi'] / interactions_df['total_cells']
interactions_df['TME'] = interactions_df['TME'] / interactions_df['total_cells']
interactions_df['isEpi'] = interactions_df['cell_cluster'].isin(ep_cells)
interactions_df['isEpi'] = ['Epi' if check else 'TME' for check in interactions_df['isEpi']]

# sum total homotypic and heterotypic interactions per cell type
interactions_df = pd.melt(interactions_df, id_vars=['fov', 'cell_cluster', 'isEpi'], value_vars=['Epi', 'TME'])
interactions_df['feature_type'] = [cell + 'Hom' if cell == interactions_df['variable'][i] else cell + 'Het' for i, cell in enumerate(interactions_df.isEpi)]
interactions_df['feature'] = interactions_df["cell_cluster"].astype(str) + '__' + interactions_df["feature_type"].astype(str)
interactions_df = interactions_df.pivot(index='fov', columns='feature', values='value').reset_index()
interactions_df.to_csv(os.path.join(SpaceCat_dir, 'Epi_TME_interactions.csv'), index=False)

# run SpaceCat
adata = anndata.read_h5ad(os.path.join(SpaceCat_dir, 'adata', 'adata.h5ad'))

# read in image level dataframes
fiber_df = pd.read_csv(os.path.join(SpaceCat_dir, 'fiber_stats_table.csv'))
fiber_tile_df = pd.read_csv(os.path.join(SpaceCat_dir, 'fiber_stats_per_tile.csv'))
mixing_df = pd.read_csv(os.path.join(SpaceCat_dir, 'formatted_mixing_scores.csv'))
kmeans_img_proportions = pd.read_csv(os.path.join(SpaceCat_dir, 'neighborhood_image_proportions.csv'))
kmeans_compartment_proportions = pd.read_csv(os.path.join(SpaceCat_dir, 'neighborhood_compartment_proportions.csv'))
pixie_ecm = pd.read_csv(os.path.join(SpaceCat_dir, 'pixie_ecm_stats.csv'))
ecm_frac = pd.read_csv(os.path.join(SpaceCat_dir, 'ecm_fraction_stats.csv'))
ecm_clusters = pd.read_csv(os.path.join(SpaceCat_dir, 'ecm_cluster_stats.csv'))
cell_interactions = pd.read_csv(os.path.join(SpaceCat_dir, 'Epi_TME_interactions.csv'))

# specify cell type pairs to compute a ratio for
ratio_pairings = [('CD8T', 'CD4T'), ('CD4T', 'Treg'), ('CD8T', 'Treg'), ('CD68_Mac', 'CD163_Mac')]

# specify addtional per cell and per image stats
per_cell_stats = [
    ['morphology', 'cell_cluster', ['area', 'major_axis_length']],
    ['linear_distance', 'cell_cluster_broad', ['distance_to__B', 'distance_to__Cancer', 'distance_to__Granulocyte', 'distance_to__Mono_Mac',
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

# Save finalized tables to csv
folder = 'SpaceCat_NT_combined'
os.makedirs(os.path.join(SpaceCat_dir, folder), exist_ok=True)
adata_processed.write_h5ad(os.path.join(SpaceCat_dir, folder, 'adata_processed.h5ad'))
adata_processed.uns['combined_feature_data'].to_csv(os.path.join(SpaceCat_dir, folder, 'combined_feature_data.csv'), index=False)
adata_processed.uns['combined_feature_data_filtered'].to_csv(os.path.join(SpaceCat_dir, folder, 'combined_feature_data_filtered.csv'), index=False)
adata_processed.uns['feature_metadata'].to_csv(os.path.join(SpaceCat_dir, folder, 'feature_metadata.csv'), index=False)
adata_processed.uns['excluded_features'].to_csv(os.path.join(SpaceCat_dir, folder, 'excluded_features.csv'), index=False)

# Save finalized tables to csv
combined_feature_data = adata_processed.uns['combined_feature_data']
combined_feature_data = combined_feature_data[combined_feature_data.feature_type != 'cell_interactions']
combined_feature_data_filtered = adata_processed.uns['combined_feature_data_filtered']
combined_feature_data_filtered = combined_feature_data_filtered[combined_feature_data_filtered.feature_type != 'cell_interactions']
folder = 'SpaceCat'
os.makedirs(os.path.join(NT_DIR, folder), exist_ok=True)
adata_processed.write_h5ad(os.path.join(NT_DIR, folder, 'adata_processed.h5ad'))
combined_feature_data.to_csv(os.path.join(NT_DIR, folder, 'combined_feature_data.csv'), index=False)
combined_feature_data_filtered.to_csv(os.path.join(NT_DIR, folder, 'combined_feature_data_filtered.csv'), index=False)

## NT features only on TONIC
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/TONIC_SpaceCat/NT_features_only'
os.makedirs(os.path.join(data_dir, 'analysis_files'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'output_files'), exist_ok=True)
harmonized_metadata = pd.read_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files/harmonized_metadata.csv')

fov_data_df = pd.read_csv(os.path.join(data_dir, 'combined_feature_data.csv'))
fov_data_df_sub = fov_data_df[fov_data_df.feature_type.isin(['density', 'cell_interactions'])]
fov_data_df_sub = fov_data_df_sub[fov_data_df_sub.compartment == 'all']

fov_data_df_Ki67 = fov_data_df[fov_data_df.compartment == 'all']
fov_data_df_Ki67 = fov_data_df[fov_data_df.feature_name.str.contains('Ki67+')]
fov_data_df = pd.concat([fov_data_df_sub, fov_data_df_Ki67])
fov_data_df.loc[fov_data_df['feature_type'] == 'cell_interactions', 'cell_pop_level'] = 'nan'
fov_data_df = pd.merge(fov_data_df, harmonized_metadata[['Tissue_ID', 'fov']], on='fov', how='left')
grouped = fov_data_df.groupby(['Tissue_ID', 'feature_name', 'feature_name_unique', 'compartment',
                               'cell_pop_level', 'feature_type']).agg({'raw_value': ['mean', 'std'],
                                                                       'normalized_value': ['mean', 'std']})
grouped.columns = ['raw_mean', 'raw_std', 'normalized_mean', 'normalized_std']
grouped = grouped.reset_index()
grouped.to_csv(os.path.join(data_dir, 'analysis_files', 'timepoint_features.csv'), index=False)

NT_viz_dir = os.path.join(REVIEW_FIG_DIR, 'NTPublic')

preds = pd.read_csv(os.path.join(SpaceCat_dir, 'SpaceCat/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
preds = preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
preds = preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                              'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
preds = preds.melt()
preds['Analysis'] = 0

adj_preds = pd.read_csv(os.path.join(SpaceCat_dir, 'SpaceCat_NT_combined/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
adj_preds = adj_preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
adj_preds = adj_preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                                      'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
adj_preds = adj_preds.melt()
adj_preds['Analysis'] = 2

nt_feats_preds = pd.read_csv(os.path.join(SpaceCat_dir, 'NT_features_only/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
nt_feats_preds = nt_feats_preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
nt_feats_preds = nt_feats_preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                                                'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
nt_feats_preds = nt_feats_preds.melt()
nt_feats_preds['Analysis'] = 1
all_preds = pd.concat([preds, adj_preds, nt_feats_preds])

fig, ax = plt.subplots()
sns.boxplot(data=all_preds, x='variable', y='value', ax=ax, width=0.6, hue='Analysis',
            palette=sns.color_palette(["#1f77b4", 'gold', "darkseagreen"]), showfliers=False)
sns.stripplot(data=all_preds, x='variable', y='value', ax=ax, hue='Analysis',
              palette=sns.color_palette(["#1f77b4", 'gold', "darkseagreen"]), dodge=True, jitter=0.2)

fig.set_figheight(4)
fig.set_figwidth(8)
plt.xticks(rotation=45)
plt.title('TONIC dataset prediction')
plt.ylabel('AUC')
plt.xlabel('')
plt.ylim((0, 1))
plt.legend(loc='lower right').set_title('')
sns.despine()

blue_line = mlines.Line2D([], [], color="#1f77b4", marker="o", label="SpaceCat features", linestyle='None')
yellow_line = mlines.Line2D([], [], color="gold", marker="o", label="Wang et al. features", linestyle='None')
green_line = mlines.Line2D([], [], color="darkseagreen", marker="o", label="SpaceCat & Wang et al. features", linestyle='None')
plt.legend(handles=[blue_line, yellow_line, green_line], loc='lower right')
plt.savefig(os.path.join(NT_viz_dir, 'TONIC_prediction_comparison.pdf'), bbox_inches='tight', dpi=300)

## correlation plots for interaction features
fov_data_df = pd.read_csv(os.path.join(os.path.join(NT_DIR, 'SpaceCat_NT_combined/combined_feature_data_filtered.csv')))
fov_data_wide = fov_data_df.pivot(index='fov', columns='feature_name_unique', values='normalized_value')
corr_df = fov_data_wide.corr(method='spearman')
corr_df = corr_df.fillna(0)
corr_df = corr_df[[col for col in corr_df.columns if np.logical_or('Het' in col, 'Hom' in col)]]
corr_df = corr_df[np.logical_and(~corr_df.index.str.contains('Het'), ~corr_df.index.str.contains('Hom'))]

# heatmap for all features
clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(NT_viz_dir, 'interaction_feature_clustermap_filtered.pdf'), dpi=300)
pd.DataFrame(clustergrid.data2d.index).to_csv(os.path.join(NT_viz_dir, 'interaction_feature_clustermap_order.csv'))
plt.close()

# heatmap for >0.7 correlated features
sub_df = corr_df[corr_df.max(axis=1) > 0.7]
clustergrid = sns.clustermap(sub_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(NT_viz_dir, 'interaction_feature_clustermap_filtered_corr70.pdf'), dpi=300)
pd.DataFrame(clustergrid.data2d.index).to_csv(os.path.join(NT_viz_dir, 'interaction_feature_clustermap_order_corr70.csv'))
plt.close()

# heatmap for >0.9 correlated features
sub_df = corr_df[corr_df.max(axis=1) > 0.9]
clustergrid = sns.clustermap(sub_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(NT_viz_dir, 'interaction_feature_clustermap_filtered_corr90.pdf'), dpi=300)
pd.DataFrame(clustergrid.data2d.index).to_csv(os.path.join(NT_viz_dir, 'interaction_feature_clustermap_order_corr90.csv'))
plt.close()

## correlation plots for mixing score features
corr_df = fov_data_wide.corr(method='spearman')
corr_df = corr_df.fillna(0)
corr_df = corr_df[[col for col in corr_df.columns if 'mixing_score' in col]]
corr_df = corr_df[np.logical_and(~corr_df.index.str.contains('Het'), ~corr_df.index.str.contains('Hom'))]
corr_df = corr_df[~corr_df.index.str.contains('mixing_score')]

# heatmap for all features
clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(NT_viz_dir, 'mixing_score_feature_clustermap_filtered.pdf'), dpi=300)
pd.DataFrame(clustergrid.data2d.index).to_csv(os.path.join(NT_viz_dir, 'mixing_score_feature_clustermap_order.csv'))
plt.close()

# heatmap for >0.7 correlated features
sub_df = corr_df[corr_df.max(axis=1)>0.7]
clustergrid = sns.clustermap(sub_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(NT_viz_dir, 'mixing_score_feature_clustermap_filtered_corr70.pdf'), dpi=300)
pd.DataFrame(clustergrid.data2d.index).to_csv(os.path.join(NT_viz_dir, 'mixing_score_feature_clustermap_order_corr70.csv'))
plt.close()


## 2.8 / 4.8  Pre-treatment and On-treatment NT vs TONIC comparisons
# Original NT features
file_path = os.path.join(NT_DIR, '/data/41586_2023_6498_MOESM3_ESM.xlsx')
NT_features = pd.read_excel(file_path, sheet_name=None)
cell_table = pd.read_csv(os.path.join(NT_DIR, 'analysis_files/cell_table.csv'))
cell_table = cell_table.replace({'{': '', '}': ''}, regex=True)
cell_table['cell_meta_cluster'] = [cell.replace('^', '')for cell in cell_table['cell_meta_cluster']]
cell_dict = dict(zip(cell_table.cell_meta_cluster, cell_table.cell_cluster))

density_pvals = NT_features['Table 8 Densities p values'][['Time point', 'Cell phenotype', 'p.value', 'Arm']]
density_pvals = density_pvals.rename(columns={'Cell phenotype': 'feature_name_unique'})
density_pvals = density_pvals.replace(cell_dict)
density_pvals['feature_name_unique'] = density_pvals['feature_name_unique'] + '__cell_cluster_density'

ki67_pvals = NT_features['Table 11 Ki67 p values'][['Time point', 'Cell phenotype', 'p.value', 'Unnamed: 10']]
ki67_pvals = ki67_pvals.rename(columns={'Unnamed: 10': 'Arm'})
ki67_pvals = ki67_pvals.rename(columns={'Cell phenotype': 'feature_name_unique'})
ki67_pvals = ki67_pvals.replace(cell_dict)
ki67_pvals['feature_name_unique'] = 'Ki67+__' + ki67_pvals['feature_name_unique']

interaction_pvals = NT_features['Table 10 Interaction p values'][['Time point', 'from', 'to cell phenotype', 'p.value', 'Arm']]
ep_cells = cell_table[cell_table.isEpithelial == 1].cell_meta_cluster.unique()
tme_cells = cell_table[cell_table.isEpithelial == 0].cell_meta_cluster.unique()
interaction_pvals['from'] = interaction_pvals['from'].replace('Epithelial', 'Epi')
interaction_pvals['isEpi'] = ['Epi' if cell in ep_cells else 'TME' for cell in interaction_pvals['to cell phenotype']]
interaction_pvals['interaction'] = ['Hom' if f == Epi else 'Het' for f, Epi in zip(interaction_pvals['from'], interaction_pvals['isEpi'])]
interaction_pvals = interaction_pvals.replace(cell_dict)
interaction_pvals['feature_name_unique'] = interaction_pvals['to cell phenotype'] + '__' + interaction_pvals['from'] + interaction_pvals['interaction']
interaction_pvals = interaction_pvals[['Time point', 'feature_name_unique', 'p.value', 'Arm']]

features = pd.concat([density_pvals, ki67_pvals, interaction_pvals])
features = features[features['p.value']<=0.05]
sig_features = features.replace({'Cancer_1__TMEHet': 'Cancer_Immune_mixing_score',
                                 'Cancer_2__TMEHet': 'Cancer_Immune_mixing_score',
                                 'Cancer_3__TMEHet': 'Cancer_Immune_mixing_score',
                                 'CD4T__TMEHom': 'Structural_T_mixing_score',
                                 'CD8T__TMEHom': 'Structural_T_mixing_score',
                                 'Treg__TMEHom': 'Structural_T_mixing_score'})
pre_treatment_features = sig_features[sig_features['Time point'] == 'Baseline']
on_treatment_features = sig_features[sig_features['Time point'] == 'On-treatment']

tonic_features = pd.read_csv(os.path.join(SPACECAT_DIR, '/SpaceCat/analysis_files/feature_ranking.csv'))
tonic_features = tonic_features[tonic_features['fdr_pval'] <= 0.05]
tonic__sig_features = tonic_features[tonic_features.compartment == 'all']
tonic__sig_features = tonic__sig_features[~tonic__sig_features.feature_name_unique.str.contains('core')]
tonic__sig_features = tonic__sig_features[~tonic__sig_features.feature_name_unique.str.contains('border')]
tonic_pre_treatment_features = tonic__sig_features[tonic__sig_features.comparison.isin(['baseline', 'pre_nivo'])]
tonic_pre_treatment_features = tonic_pre_treatment_features[['feature_name_unique', 'pval', 'comparison']]
tonic_on_treatment_features = tonic__sig_features[tonic__sig_features.comparison == 'on_nivo']
tonic_on_treatment_features = tonic_on_treatment_features[['feature_name_unique', 'pval', 'comparison']]

NT_feats = set(pre_treatment_features.feature_name_unique.unique())
TONIC_feats = set(tonic_pre_treatment_features.feature_name_unique.unique())
sets = {'Wang et al.': NT_feats, 'TONIC': TONIC_feats}
venny4py(sets=sets, colors="yb")
plt.title("Pre-treatment Features")
plt.savefig(os.path.join(NT_viz_dir, 'Pre_treatment_features.pdf'), bbox_inches='tight', dpi=300)

NT_feats = set(on_treatment_features.feature_name_unique.unique())
TONIC_feats = set(tonic_on_treatment_features.feature_name_unique.unique())
sets = {'Wang et al.': NT_feats, 'TONIC': TONIC_feats}
venny4py(sets=sets, colors="yb")
plt.title("On-treatment Features")
plt.savefig(os.path.join(NT_viz_dir, 'On_treatment_features.pdf'), bbox_inches='tight', dpi=300)

# compare SpaceCat features
NT_features = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat/analysis_files/feature_ranking_immunotherapy+chemotherapy.csv'))
NT_features = NT_features[NT_features['fdr_pval'] <= 0.05]
NT__sig_features = NT_features[NT_features.compartment == 'all']
for feature in NT__sig_features.feature_name_unique.unique():
    if 'Epithelial' in feature:
        feature_new = feature.replace('Epithelial', 'Cancer')
        NT__sig_features = NT__sig_features.replace({feature: feature_new})
NT__sig_features = NT__sig_features[~NT__sig_features.feature_name_unique.str.contains('core')]
NT__sig_features = NT__sig_features[~NT__sig_features.feature_name_unique.str.contains('border')]
NT_pre_treatment_features = NT__sig_features[NT__sig_features.comparison == 'Baseline']
NT_pre_treatment_features = NT_pre_treatment_features[['feature_name_unique', 'pval', 'comparison']]
NT_on_treatment_features = NT__sig_features[NT__sig_features.comparison == 'On-treatment']
NT_on_treatment_features = NT_on_treatment_features[['feature_name_unique', 'pval', 'comparison']]

NT_feats = set(NT_pre_treatment_features.feature_name_unique.unique())
TONIC_feats = set(tonic_pre_treatment_features.feature_name_unique.unique())
sets = {'Wang et al.': NT_feats, 'TONIC': TONIC_feats}
venny4py(sets=sets, colors="yb")
plt.title("Pre-treatment Features")
plt.savefig(os.path.join(NT_viz_dir, 'SpaceCat_pre_treatment_features.pdf'), bbox_inches='tight', dpi=300)

NT_feats = set(NT_on_treatment_features.feature_name_unique.unique())
TONIC_feats = set(tonic_on_treatment_features.feature_name_unique.unique())
sets = {'Wang et al.': NT_feats, 'TONIC': TONIC_feats}
venny4py(sets=sets, colors="yb")
plt.title("On-treatment Features")
plt.savefig(os.path.join(NT_viz_dir, 'SpaceCat_on_treatment_features.pdf'), bbox_inches='tight', dpi=300)


## 3.2 Low cellularity ##
low_cellularity_viz_dir = os.path.join(REVIEW_FIG_DIR, 'low_cellularity')
os.makedirs(low_cellularity_viz_dir, exist_ok=True)

# EXPLORATORY ANALYSIS
adata = anndata.read_h5ad(os.path.join(SpaceCat_dir, 'adata', 'adata.h5ad'))
harmonized_metadata = pd.read_csv(ANALYSIS_DIR, 'harmonized_metadata.csv')

cell_table = adata.obs
cellularity_df = cell_table.groupby(by='fov', observed=True).count().sort_values(by='label').label.reset_index()
cellularity_df['low_cellularity'] = 'No'
cellularity_df.iloc[:int(round(len(cellularity_df)*.1, 0)), -1] = 'Yes'
cellularity_df = cellularity_df.drop(columns='label')
cellularity_df = cellularity_df.merge(harmonized_metadata[['fov', 'Patient_ID']], on='fov')
cellularity_df.to_csv(os.path.join(low_cellularity_viz_dir, 'low_cellularity_images.csv'), index=False)

# low cellularity by patient
patient_counts = cellularity_df.groupby(by=['Patient_ID', 'low_cellularity']).count().reset_index()
patient_counts_total = cellularity_df[['fov', 'Patient_ID']].groupby(by='Patient_ID').count().reset_index()

bar1 = sns.barplot(x="Patient_ID",  y="fov", data=patient_counts_total, color='lightblue')
norm_cell = patient_counts[patient_counts.low_cellularity == 'No']
bar2 = sns.barplot(x="Patient_ID", y="fov", data=norm_cell, estimator=sum, ci=None,  color='darkblue')
top_bar = mpatches.Patch(color='lightblue', label='Low cellularity images')
bottom_bar = mpatches.Patch(color='darkblue', label='Regular images')
plt.legend(handles=[top_bar, bottom_bar])
plt.xticks([])
plt.title("Low Cellularity by Patient")
plt.ylabel("# of Images")
plt.savefig(os.path.join(low_cellularity_viz_dir, 'Low_cellularity_by_patient.pdf'), bbox_inches='tight', dpi=300)

# low cellularity by timepoint
cellularity_df = pd.read_csv(os.path.join(REVIEW_FIG_DIR, 'low_cellularity', 'low_cellularity_images.csv'))
cellularity_df = cellularity_df.merge(harmonized_metadata[['fov', 'Timepoint']], on='fov')
timepoint_counts = cellularity_df.groupby(by=['Timepoint', 'low_cellularity']).count().reset_index()
timepoint_counts_total = cellularity_df[['fov', 'Timepoint']].groupby(by='Timepoint').count().reset_index()
sort_dict = {'baseline': 0, 'primary': 1, 'pre_nivo': 2, 'on_nivo': 3}
timepoint_counts_total = timepoint_counts_total.iloc[timepoint_counts_total['Timepoint'].map(sort_dict).sort_values().index]

bar1 = sns.barplot(x="Timepoint",  y="fov", data=timepoint_counts_total, color='lightblue')
norm_cell = timepoint_counts[timepoint_counts.low_cellularity == 'No']
bar2 = sns.barplot(x="Timepoint", y="fov", data=norm_cell, estimator=sum, ci=None,  color='darkblue')
top_bar = mpatches.Patch(color='lightblue', label='Low cellularity images')
bottom_bar = mpatches.Patch(color='darkblue', label='Regular images')
plt.legend(handles=[top_bar, bottom_bar])
plt.xticks(rotation=45)
plt.title("Low Cellularity by Timepoint")
plt.ylabel("# of Images")
plt.savefig(os.path.join(low_cellularity_viz_dir, 'Low_cellularity_by_timepoint.pdf'), bbox_inches='tight', dpi=300)

# low cellularity vs regular image features
harmonized_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'analysis_files/harmonized_metadata.csv'))
feature_metadata = pd.read_csv(os.path.join(SpaceCat_dir, 'feature_metadata.csv'))
cellularity_data = pd.read_csv(os.path.join(REVIEW_FIG_DIR, 'low_cellularity', 'low_cellularity_images.csv'))
combined_df = pd.read_csv(os.path.join(SpaceCat_dir, 'combined_feature_data_filtered.csv'))
combined_df = combined_df.merge(cellularity_data, on='fov')
combined_df = combined_df.merge(harmonized_metadata[['fov', 'Timepoint']], on='fov')
combined_df.to_csv(os.path.join(low_cellularity_viz_dir, 'combined_feature_data_filtered.csv'), index=False)

# generate a single set of top hits across all comparisons
method = 'ttest'
total_dfs = []
for comparison in combined_df.Timepoint.unique():
    population_df = compare_populations(feature_df=combined_df, pop_col='low_cellularity',
                                        timepoints=[comparison], pop_1='No', pop_2='Yes', method=method,
                                        feature_suff='value')
    if np.sum(~population_df.log_pval.isna()) == 0:
        continue
    long_df = population_df[['feature_name_unique', 'log_pval', 'mean_diff', 'med_diff']]
    long_df['comparison'] = comparison
    long_df = long_df.dropna()
    long_df['pval'] = 10 ** (-long_df.log_pval)
    long_df['fdr_pval'] = multipletests(long_df.pval, method='fdr_bh')[1]
    total_dfs.append(long_df)

ranked_features_df = pd.concat(total_dfs)
ranked_features_df['log10_qval'] = -np.log10(ranked_features_df.fdr_pval)
ranked_features_df['pval_rank'] = ranked_features_df.fdr_pval.rank(ascending=True)
ranked_features_df['cor_rank'] = ranked_features_df.med_diff.abs().rank(ascending=False)
ranked_features_df['combined_rank'] = (ranked_features_df.pval_rank.values + ranked_features_df.cor_rank.values) / 2

max_rank = len(~ranked_features_df.med_diff.isna())
normalized_rank = ranked_features_df.combined_rank / max_rank
ranked_features_df['importance_score'] = 1 - normalized_rank
ranked_features_df = ranked_features_df.sort_values('importance_score', ascending=False)
ranked_features_df['signed_importance_score'] = ranked_features_df.importance_score * np.sign(
    ranked_features_df.med_diff)
ranked_features_df = ranked_features_df.merge(feature_metadata, on='feature_name_unique', how='left')

feature_type_dict = {'functional_marker': 'phenotype', 'linear_distance': 'interactions',
                     'density': 'density', 'cell_diversity': 'diversity', 'density_ratio': 'density',
                     'mixing_score': 'interactions', 'region_diversity': 'diversity',
                     'compartment_area_ratio': 'compartment', 'density_proportion': 'density',
                     'morphology': 'phenotype', 'pixie_ecm': 'ecm', 'fiber': 'ecm', 'ecm_cluster': 'ecm',
                     'compartment_area': 'compartment', 'ecm_fraction': 'ecm'}
ranked_features_df['feature_type_broad'] = ranked_features_df.feature_type.map(feature_type_dict)
ranked_features_df['feature_rank_global_evolution'] = ranked_features_df.importance_score.rank(ascending=False)
ranked_features_no_evo = ranked_features_df.loc[
                         ranked_features_df.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo']), :]
ranked_features_no_evo['feature_rank_global'] = ranked_features_no_evo.importance_score.rank(ascending=False)
ranked_features_df = ranked_features_df.merge(
    ranked_features_no_evo.loc[:, ['feature_name_unique', 'comparison', 'feature_rank_global']],
    on=['feature_name_unique', 'comparison'], how='left')

# get ranking for each comparison
ranked_features_df['feature_rank_comparison'] = np.nan
for comparison in ranked_features_df.comparison.unique():
    ranked_features_comp = ranked_features_df.loc[ranked_features_df.comparison == comparison, :]
    ranked_features_comp['temp_comparison'] = ranked_features_comp.importance_score.rank(ascending=False)
    ranked_features_df = ranked_features_df.merge(
        ranked_features_comp.loc[:, ['feature_name_unique', 'comparison', 'temp_comparison']],
        on=['feature_name_unique', 'comparison'], how='left')
    ranked_features_df['feature_rank_comparison'] = ranked_features_df['temp_comparison'].fillna(
        ranked_features_df['feature_rank_comparison'])
    ranked_features_df.drop(columns='temp_comparison', inplace=True)
# saved formatted df
ranked_features_df.to_csv(os.path.join(low_cellularity_viz_dir, 'cellularity_feature_ranking.csv'), index=False)

# top features by type
top_fts = ranked_features_df[:100]
top_fts_type = top_fts[['feature_name_unique', 'feature_type']].groupby(by='feature_type').count().reset_index()
top_fts_type = top_fts_type.sort_values(by='feature_name_unique')
sns.barplot(top_fts_type, y='feature_type', x='feature_name_unique')
plt.ylabel("Feature Type")
plt.xlabel("Count")
plt.title("Top Features Differing Between Low Cellularity and Regular Images")
plt.savefig(os.path.join(low_cellularity_viz_dir, 'Low_cellularity_features_by_type.pdf'), bbox_inches='tight', dpi=300)

# top features by timepoint
top_fts_tp = top_fts[['feature_name_unique', 'comparison']].groupby(by='comparison').count().reset_index()
top_fts_tp = top_fts_tp.sort_values(by='feature_name_unique')
sns.barplot(top_fts_tp, y='comparison', x='feature_name_unique')
plt.ylabel("Timepoint")
plt.xlabel("Count")
plt.title("Top Features Differing Between Low Cellularity and Regular Images")
plt.savefig(os.path.join(low_cellularity_viz_dir, 'Low_cellularity_features_by_timepoint.pdf'), bbox_inches='tight', dpi=300)


# DROPPING LOW CELL IMAGES
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
plt.savefig(os.path.join(REVIEW_FIG_DIR, 'low_cellularity', 'low_cellularity_prediction_comparisons.pdf'), bbox_inches = 'tight', dpi =300)


## 3.5 Baseline to On-nivo feature evolution ##

baseline_on_nivo_viz_dir = os.path.join(REVIEW_FIG_DIR, 'baseline_on_nivo_evolution')
harmonized_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'harmonized_metadata.csv'))
timepoint_features = pd.read_csv(os.path.join(SPACECAT_DIR, 'analysis_files/timepoint_features_filtered.csv'))
timepoint_features = timepoint_features.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint', 'baseline__on_nivo']].drop_duplicates(), on='Tissue_ID')
feature_subset = timepoint_features.loc[(timepoint_features.baseline__on_nivo) & (timepoint_features.Timepoint.isin(['baseline', 'on_nivo'])), :]

for paired_status in ['baseline__on_nivo', None]:
    subdir_name = 'paired' if paired_status else 'unpaired'
    subdir_path = os.path.join(baseline_on_nivo_viz_dir, subdir_name)
    os.makedirs(subdir_path)

    primary_met_means = compare_timepoints(
        feature_df=timepoint_features, timepoint_1_name='baseline', timepoint_1_list=['baseline'],
        timepoint_2_name='on_nivo', timepoint_2_list=['on_nivo'], paired=paired_status, feature_suff='mean')

    summarize_timepoint_enrichment(input_df=primary_met_means, feature_df=timepoint_features,
                                   timepoints=['baseline', 'on_nivo'],
                                   pval_thresh=2, diff_thresh=0.3, output_dir=os.path.join(baseline_on_nivo_viz_dir, subdir_name))


## 3.7 Fiber feature usefulness ##

# color fibers by alignment & length stats
fiber_table = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'fiber_segmentation_processed_data/fiber_object_table.csv'))
fibseg_dir = os.path.join(REVIEW_FIG_DIR, 'fiber_features/fiber_masks')

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
ecm_cluster_viz_dir = os.path.join(REVIEW_FIG_DIR, 'ECM_clusters')
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
plt.savefig(os.path.join(REVIEW_FIG_DIR, 'ECM_clusters', 'Functional_expression_Structural.pdf'), bbox_inches='tight', dpi=300)

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
plt.savefig(os.path.join(REVIEW_FIG_DIR, 'ECM_clusters', 'Functional_expression_CD68_Mac.pdf'), bbox_inches='tight', dpi=300)


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

os.makedirs(os.path.join(REVIEW_FIG_DIR, 'location_bias'), exist_ok=True)
fig.savefig(os.path.join(REVIEW_FIG_DIR, 'location_bias', 'location_bias_compartment_features.pdf'),
            bbox_inches='tight', dpi=300)
fig2.savefig(os.path.join(REVIEW_FIG_DIR, 'location_bias', 'location_bias_all_features.pdf'),
             bbox_inches='tight', dpi=300)


## 3.11 Evolution features ##

# longitudinal features
BASE_DIR = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/TONIC_SpaceCat/SpaceCat'
ranked_features = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/feature_ranking.csv'))

overlap_type_dict = {'global': [['primary', 'baseline', 'pre_nivo', 'on_nivo'],
                                ['primary__baseline', 'baseline__pre_nivo', 'baseline__on_nivo', 'pre_nivo__on_nivo']],
                     'primary': [['primary'], ['primary__baseline']],
                     'baseline': [['baseline'], ['primary__baseline', 'baseline__pre_nivo', 'baseline__on_nivo']],
                     'pre_nivo': [['pre_nivo'], ['baseline__pre_nivo', 'pre_nivo__on_nivo']],
                     'on_nivo': [['on_nivo'], ['baseline__on_nivo', 'pre_nivo__on_nivo']]}
overlap_results = {}
for overlap_type, comparisons in overlap_type_dict.items():
    static_comparisons, evolution_comparisons = comparisons

    overlap_top_features = ranked_features.copy()
    overlap_top_features = overlap_top_features.loc[overlap_top_features.comparison.isin(static_comparisons + evolution_comparisons)]
    overlap_top_features.loc[overlap_top_features.comparison.isin(static_comparisons), 'comparison'] = 'static'
    overlap_top_features.loc[overlap_top_features.comparison.isin(evolution_comparisons), 'comparison'] = 'evolution'
    overlap_top_features = overlap_top_features[['feature_name_unique', 'comparison']].drop_duplicates()
    overlap_top_features = overlap_top_features.iloc[:100, :]
    static_ids = overlap_top_features.loc[
        overlap_top_features.comparison == 'static', 'feature_name_unique'].unique()
    evolution_ids = overlap_top_features.loc[
        overlap_top_features.comparison == 'evolution', 'feature_name_unique'].unique()

    overlap_results[overlap_type] = {'static_ids': static_ids, 'evolution_ids': evolution_ids}

overlap_features = list(set(static_ids).intersection(set(evolution_ids)))
evolution_ids = [feature for feature in evolution_ids if feature not in overlap_features]
evolution_feature_data = ranked_features[ranked_features.feature_name_unique.isin(evolution_ids)]
evolution_feature_data = evolution_feature_data[evolution_feature_data.comparison.isin(['primary__baseline', 'baseline__pre_nivo', 'baseline__on_nivo', 'pre_nivo__on_nivo'])]
evolution_feature_data = evolution_feature_data[['feature_name_unique', 'log_pval', 'mean_diff', 'med_diff',
       'comparison', 'pval', 'fdr_pval', 'log10_qval', 'pval_rank', 'cor_rank',
       'combined_rank', 'importance_score', 'signed_importance_score',
       'feature_name', 'compartment', 'cell_pop_level', 'feature_type']]
evolution_feature_data.sort_values(by='feature_name_unique', inplace=True)
evolution_feature_data.to_csv(os.path.join(REVIEW_FIG_DIR, 'evolution_features_and_multimodal_modeling', 'evolution_features_table.csv'), index=False)

# multimodal modeling
# run all_timepoints_combined_modalities.R script
# multimodal prediction plots
multi_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/sequencing_data/multimodal_prediction'
baseline_results = pd.read_csv(os.path.join(multi_dir, 'baseline_results.csv'))
pre_nivo_results = pd.read_csv(os.path.join(multi_dir, 'pre_nivo_results.csv'))
on_nivo_results = pd.read_csv(os.path.join(multi_dir, 'on_nivo_results.csv'))

baseline_results['Timepoint'] = 'Baseline'
baseline_results = baseline_results.rename(columns={'auc_baseline_list': 'Combined', 'auc_rna_baseline_list': 'RNA', 'auc_protein_baseline_list': 'MIBI'})
pre_nivo_results['Timepoint'] = 'Pre nivo'
pre_nivo_results = pre_nivo_results.rename(columns={'auc_induction_list': 'Combined', 'auc_rna_induction_list': 'RNA', 'auc_protein_induction_list': 'MIBI'})
on_nivo_results['Timepoint'] = 'On nivo'
on_nivo_results = on_nivo_results.rename(columns={'auc_on_nivo_list': 'Combined', 'auc_rna_on_nivo_list': 'RNA', 'auc_protein_on_nivo_list': 'MIBI'})
all_results = pd.concat([baseline_results, pre_nivo_results, on_nivo_results])
all_results = all_results[['Timepoint', 'Combined']]

fig, ax = plt.subplots()
sns.boxplot(data=all_results, x='Timepoint', y='Combined', ax=ax, width=0.6,
            palette=sns.color_palette(["chocolate"]), showfliers=False)
sns.stripplot(data=all_results, x='Timepoint', y='Combined', ax=ax,
              palette=sns.color_palette(["chocolate"]), jitter=0.1)
fig.set_figheight(5)
fig.set_figwidth(6)
plt.xticks(rotation=45)
plt.title('Combined MIBI & RNA multivariate model accuracy')
plt.ylabel('AUC')
plt.xlabel('')
plt.ylim((0, 1))
sns.despine()
plt.savefig(os.path.join(REVIEW_FIG_DIR, 'evolution_features_and_multimodal_modeling', 'combined_model_prediction.pdf'), bbox_inches='tight', dpi=300)


## 4.4 Limit multivariate model features ##
limit_features_dir = os.path.join(REVIEW_FIG_DIR, 'limit_model_features')

preds_5 = pd.read_csv(os.path.join(limit_features_dir, 'prediction_model_5/patient_outcomes/all_timepoints_results_MIBI.csv'))
preds_10 = pd.read_csv(os.path.join(limit_features_dir, 'limit_model_features/prediction_model_10/patient_outcomes/all_timepoints_results_MIBI.csv'))
preds_15 = pd.read_csv(os.path.join(limit_features_dir, 'limit_model_features/prediction_model_15/patient_outcomes/all_timepoints_results_MIBI.csv'))
preds_20 = pd.read_csv(os.path.join(limit_features_dir, 'prediction_model_20/patient_outcomes/all_timepoints_results_MIBI.csv'))
preds_reg = pd.read_csv(os.path.join(SPACECAT_DIR, 'SpaceCat/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))

df_5 = preds_5.mean()
df_10 = preds_10.mean()
df_15 = preds_15.mean()
df_20 = preds_20.mean()
df_reg = preds_reg.mean()
df = pd.concat([df_reg, df_20, df_15, df_10, df_5], axis=1)
df = df.rename(columns={0: 'All features', 1: 'Top 20 features', 2: 'Top 15 features', 3: 'Top 10 features', 4: 'Top 5 features'})

df = df.reset_index()
df.replace('auc_on_nivo_list', 'On nivo', inplace=True)
df.replace('auc_post_induction_list', 'Pre nivo', inplace=True)
df.replace('auc_primary_list', 'Primary', inplace=True)
df.replace('auc_baseline_list', 'Baseline', inplace=True)
df['order'] = df['index'].replace({'Primary':0, 'Baseline':1, 'Pre nivo':2, 'On nivo': 3})
df = df.sort_values(by='order')
df = df.drop(columns=['order'])
df = df.rename(columns={'index': 'Timepoint'})
df = pd.melt(df, ['Timepoint'])

sns.scatterplot(data=df[df.variable == 'All features'], x='Timepoint', y='value', hue='variable', palette=sns.color_palette(['black']), edgecolors='black')
sns.scatterplot(data=df[df.variable != 'All features'], x='Timepoint', y='value', hue='variable', palette=sns.color_palette(['dimgrey', 'darkgrey', 'lightgrey', 'whitesmoke']), edgecolors='black')
plt.xticks(rotation=30)
plt.ylabel('Mean AUC')
plt.title('Model accuracy by feature amount')
sns.despine()
plt.gca().legend(loc='lower right').set_title('')
plt.savefig(os.path.join(limit_features_dir, 'feature_cap_prediction_comparisons.pdf'), bbox_inches='tight', dpi=300)


## 4.5 Other/Stroma_Collagen/Stroma_Fibronectin to Cancer reassignment ##

reclustering_dir = os.path.join(REVIEW_FIG_DIR, "Cancer_reclustering")
os.makedirs(reclustering_dir, exist_ok=True)

adata = anndata.read_h5ad(os.path.join(SPACECAT_DIR, 'adata', 'adata_preprocessed.h5ad'))
cell_table = adata.obs
cancer_cells = cell_table[cell_table.cell_cluster_broad == 'Cancer']
non_cancer_cells = cell_table[cell_table.cell_cluster_broad != 'Cancer']

# Area distribution of cancer vs noncancer cells
ax = sns.histplot(non_cancer_cells.area, color="blue", alpha=0, label="Non-cancer cells", binwidth=50)
sns.histplot(non_cancer_cells.area, color="blue", alpha=0.5, label="Non-cancer cells", binwidth=50)
sns.histplot(cancer_cells.area, color="red", alpha=0.5, label="Cancer cells", binwidth=50)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.xlim((0, 2500))
plt.legend()
labels = ["Non-cancer Cells", "Cancer cells", "Cancer cell median"]
handles, _ = ax.get_legend_handles_labels()
handles.append(plt.axvline(x=cancer_cells.area.median(), c='red'))
plt.title("Cell Area Distribution")
plt.xlabel('Area')
plt.savefig(os.path.join(reclustering_dir, 'cancer_noncancer_area_dist.pdf'), bbox_inches='tight')


# Cancer neighborhood proportions
neighbors_mat = adata.obsm['neighbors_freqs_cell_cluster_radius50']
neighborhood_mat = pd.concat([adata.obs.loc[:, ['fov', 'label', 'area', 'cell_cluster', 'cell_meta_cluster']],
                             neighbors_mat], axis=1)
def cancer_sum(row, cluster_level):
    row = row.drop(['fov', 'label', cluster_level])
    return row.sum()

cancer_neighbors = neighborhood_mat[neighborhood_mat.cell_cluster.isin(['Cancer_1', 'Cancer_2', 'Cancer_3'])]
cancer_neighbors = cancer_neighbors[['fov', 'label', 'area', 'cell_cluster', 'Cancer_1', 'Cancer_2', 'Cancer_3']]
cancer_neighbors['cancer_neighbors_prop'] = cancer_neighbors.apply(cancer_sum, cluster_level='cell_cluster', axis=1)

plt.hist(cancer_neighbors.cancer_neighbors_prop, bins=20, density=True, rwidth=0.9, edgecolor='black')
plt.title('Cancer cell proportion of Cancer cell neighborhoods')
plt.xlabel('Cancer proportion')
plt.ylabel('Density')
plt.savefig(os.path.join(reclustering_dir, 'cancer_neighborhood_proportion_cancer.pdf'), bbox_inches='tight')


neighborhood_counts = pd.read_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files/spatial_analysis/neighborhood_mats/neighborhood_counts-cell_meta_cluster_radius50.csv')
neighborhood_counts = neighborhood_counts[neighborhood_counts.fov.isin(cell_table.fov.unique())]
cancer_cell_types = ['Cancer_CD56', 'Cancer_CK17', 'Cancer_Ecad', 'Cancer_SMA', 'Cancer_Vim', 'Cancer_Other',
                     'Cancer_Mono']
neighborhood_counts_meta = neighborhood_counts.iloc[:, :3]
neighborhood_counts = neighborhood_counts.iloc[:, 3:]

tables = {}
for c in ['Other', 'Stroma_Collagen', 'Stroma_Fibronectin']:
    neighborhood_counts_c = neighborhood_counts.drop(columns=[c])
    neighborhood_freqs_c = neighborhood_counts_c.div(neighborhood_counts_c.sum(axis=1), axis=0)
    neighborhood_freqs_c = neighborhood_counts_meta.merge(neighborhood_freqs_c, left_index=True, right_index=True)

    c_neighbors = neighborhood_freqs_c[neighborhood_freqs_c.cell_meta_cluster == c]
    c_neighbors_cancer = c_neighbors[['fov', 'label', 'cell_meta_cluster'] + cancer_cell_types]
    c_neighbors_cancer['cancer_neighbors_prop'] = c_neighbors_cancer.apply(cancer_sum, cluster_level='cell_meta_cluster', axis=1)
    c_neighbors_cancer_sub = c_neighbors_cancer
    tables[c] = c_neighbors_cancer_sub[['fov', 'label', 'cell_meta_cluster', 'cancer_neighbors_prop']]
new_cancer_cells = pd.concat(tables.values())
new_cancer_cells['cancer_test'] = False
new_cancer_cells.loc[new_cancer_cells['cancer_neighbors_prop'] >= 0.7, 'cancer_test'] = True

cell_table_adj = cell_table.copy()
cell_table_adj = cell_table_adj.merge(new_cancer_cells, on=['fov', 'label', 'cell_meta_cluster'], how='left')
cell_table_adj['cell_cluster_broad_new'] = cell_table_adj['cell_cluster_broad'].copy()
cell_table_adj['cell_meta_cluster_new'] = cell_table_adj['cell_meta_cluster'].copy()
cell_table_adj['cell_cluster_new'] = cell_table_adj['cell_cluster'].copy()
cell_table_adj.loc[cell_table_adj['cancer_test']==True, 'cell_cluster_broad_new'] = 'Cancer'
cell_table_adj.loc[cell_table_adj['cancer_test']==True, 'cell_meta_cluster_new'] = 'Cancer_new'
cell_table_adj['cell_cluster_new'] = cell_table_adj['cell_cluster_new'].cat.add_categories('Cancer_new')
cell_table_adj.loc[cell_table_adj['cancer_test']==True, 'cell_cluster_new'] = 'Cancer_new'

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
fig.suptitle('Cancer cell proportion of cell neighborhoods')
fig.supxlabel('Cancer proportion')
fig.supylabel('Density')
for c, ax in zip(['Other', 'Stroma_Collagen', 'Stroma_Fibronectin'], axes.flat):
    cluster_table_sub = cell_table_adj[cell_table_adj.cell_meta_cluster.isin([c, 'Cancer_new'])]
    ax.hist(cluster_table_sub.cancer_neighbors_prop, bins=20, density=True, rwidth=0.9, edgecolor='black')
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 10))
    ax.set_title(f'{c} neighborhoods')
plt.savefig(os.path.join(reclustering_dir, 'misc_neighborhood_proportion_cancer.pdf'), bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
cancer_table = cell_table_adj[cell_table_adj.cell_cluster_broad_new=='Cancer'][['fov', 'cell_cluster_new']]
cancer_table_counts = cancer_table.groupby(by=['cell_cluster_new'], observed=True).count().reset_index()
cancer_table_counts = cancer_table_counts.sort_values(by='fov', ascending=False)
total_count = cancer_table_counts[['fov']].sum()
cancer_table_counts['cancer_prop'] = cancer_table_counts['fov'].div(total_count.values[0])
plt.bar(cancer_table_counts.cell_cluster_new, cancer_table_counts.cancer_prop, color=['lightgrey']*(len(cancer_table_counts)-1)+['green'])
plt.xticks(rotation=45)
plt.title('Cancer cell cluster proportions')
green_patch = mpatches.Patch(color='green', label='New Cancer cells')
grey_patch = mpatches.Patch(color='lightgrey', label='Original Cancer cells')
plt.legend(handles=[green_patch, grey_patch])
plt.savefig(os.path.join(reclustering_dir, 'Cancer_cell_new_proportions_broad.pdf'), bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
cancer_table = cell_table_adj[cell_table_adj.cell_cluster_broad_new=='Cancer'][['fov', 'cell_meta_cluster_new']]
cancer_table_counts = cancer_table.groupby(by=['cell_meta_cluster_new'], observed=True).count().reset_index()
cancer_table_counts = cancer_table_counts.sort_values(by='fov', ascending=False)
total_count = cancer_table_counts[['fov']].sum()
cancer_table_counts['cancer_prop'] = cancer_table_counts['fov'].div(total_count.values[0])
plt.bar(cancer_table_counts.cell_meta_cluster_new, cancer_table_counts.cancer_prop, color=['lightgrey']*(len(cancer_table_counts)-1)+['green'])
plt.xticks(rotation=45)
plt.title('Cancer cell cluster proportions')
green_patch = mpatches.Patch(color='green', label='New Cancer cells')
grey_patch = mpatches.Patch(color='lightgrey', label='Original Cancer cells')
plt.legend(handles=[green_patch, grey_patch])
plt.savefig(os.path.join(reclustering_dir, 'Cancer_cell_new_proportions.pdf'), bbox_inches='tight')

counts_table = cell_table_adj[cell_table_adj.cell_meta_cluster.isin(['Other', 'Stroma_Collagen', 'Stroma_Fibronectin'])][['cell_meta_cluster', 'cell_meta_cluster_new']]
counts_table = counts_table.groupby(by=['cell_meta_cluster', 'cell_meta_cluster_new'], observed=True).value_counts().reset_index()
counts_table.loc[counts_table.cell_meta_cluster_new!='Cancer_new', 'cell_meta_cluster_new'] = 'Same'
counts_table = counts_table.pivot(index='cell_meta_cluster', columns='cell_meta_cluster_new', values='count').reset_index()
row_sums = counts_table.select_dtypes(include='number').sum(axis=1)
counts_table.iloc[:, 1:] = counts_table.iloc[:, 1:].div(row_sums, axis=0)

counts_table.plot(x='cell_meta_cluster', kind='barh', stacked=True, title='Proportion of reassigned Cancer cells',
                  mark_right=True, color=['green', 'lightgrey'])
green_patch = mpatches.Patch(color='green', label='New Cancer cells')
grey_patch = mpatches.Patch(color='lightgrey', label='Non-cancer')
plt.legend(handles=[green_patch, grey_patch])
plt.ylabel('Cell Type')
plt.savefig(os.path.join(reclustering_dir, 'reassigned_cancer_cells.pdf'), bbox_inches='tight')


## 4.6.1 immune_agg features ##

immune_agg_viz_dir = os.path.join(REVIEW_FIG_DIR, "immune_agg_features")
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

ranked_features_all = pd.read_csv(os.path.join(REVIEW_FIG_DIR, 'immune_agg_features', 'feature_ranking.csv'))
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
preds = pd.read_csv(os.path.join(REVIEW_FIG_DIR, 'immune_agg_features', 'all_timepoints_results_MIBI-immune_agg.csv'))
preds = preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
preds = preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                              'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
preds = preds.melt()
preds['immune_agg'] = 'include'

old_preds = pd.read_csv(os.path.join(REVIEW_FIG_DIR, 'immune_agg_features', 'all_timepoints_results_MIBI.csv'))
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

plt.savefig(os.path.join(REVIEW_FIG_DIR, 'immune_agg_features', 'immune_agg_prediction_comparison.pdf'),
            bbox_inches='tight', dpi=300)


## 4.10.3 functional marker thresholding in SpaceCat ##

functional_marker_viz_dir = os.path.join(REVIEW_FIG_DIR, "functional_marker_auto_thresholds")
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
