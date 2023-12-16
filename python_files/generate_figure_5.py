import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
from ark.utils.plot_utils import cohort_cluster_plot
import ark.settings as settings
import skimage.io as io


base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
metadata_dir = os.path.join(base_dir, 'intermediate_files/metadata')
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/'
harmonized_metadata = pd.read_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'))
seg_dir = os.path.join(base_dir, 'segmentation_data/deepcell_output')
image_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/'

study_fovs = harmonized_metadata.loc[harmonized_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), 'fov'].values


ranked_features_all = pd.read_csv(os.path.join(base_dir, 'analysis_files/feature_ranking.csv'))
ranked_features = ranked_features_all.loc[ranked_features_all.comparison.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]

top_features = ranked_features.loc[ranked_features.comparison.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
top_features = top_features.iloc[:100, :]


# summarize distribution of top features
top_features_by_comparison = top_features[['feature_name_unique', 'comparison']].groupby('comparison').count().reset_index()
top_features_by_comparison.columns = ['comparison', 'num_features']
top_features_by_comparison = top_features_by_comparison.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_comparison, x='comparison', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Figure5_num_features_per_comparison.pdf'))
plt.close()


# summarize overlap of top features
top_features_by_feature = top_features[['feature_name_unique', 'comparison']].groupby('feature_name_unique').count().reset_index()
feature_counts = top_features_by_feature.groupby('comparison').count().reset_index()
feature_counts.columns = ['num_comparisons', 'num_features']

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=feature_counts, x='num_comparisons', y='num_features', color='grey', ax=ax)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Figure5_num_comparisons_per_feature.pdf'))
plt.close()



# get overlap between static and evolution top features
overlap_type_dict = {'global': [['primary_untreated', 'baseline', 'post_induction', 'on_nivo'],
                                ['primary__baseline', 'baseline__post_induction', 'baseline__on_nivo', 'post_induction__on_nivo']],
                     'primary': [['primary_untreated'], ['primary__baseline']],
                     'baseline': [['baseline'], ['primary__baseline', 'baseline__post_induction', 'baseline__on_nivo']],
                     'post_induction': [['post_induction'], ['baseline__post_induction', 'post_induction__on_nivo']],
                     'on_nivo': [['on_nivo'], ['baseline__on_nivo', 'post_induction__on_nivo']]}

overlap_results = {}
for overlap_type, comparisons in overlap_type_dict.items():
    static_comparisons, evolution_comparisons = comparisons

    overlap_top_features = ranked_features.copy()
    overlap_top_features = overlap_top_features.loc[overlap_top_features.comparison.isin(static_comparisons + evolution_comparisons)]
    overlap_top_features.loc[overlap_top_features.comparison.isin(static_comparisons), 'comparison'] = 'static'
    overlap_top_features.loc[overlap_top_features.comparison.isin(evolution_comparisons), 'comparison'] = 'evolution'
    overlap_top_features = overlap_top_features[['feature_name_unique', 'comparison']].drop_duplicates()
    overlap_top_features = overlap_top_features.iloc[:100, :]
    # keep_features = overlap_top_features.feature_name_unique.unique()[:100]
    # overlap_top_features = overlap_top_features.loc[overlap_top_features.feature_name_unique.isin(keep_features), :]
    # len(overlap_top_features.feature_name_unique.unique())
    static_ids = overlap_top_features.loc[
        overlap_top_features.comparison == 'static', 'feature_name_unique'].unique()
    evolution_ids = overlap_top_features.loc[
        overlap_top_features.comparison == 'evolution', 'feature_name_unique'].unique()

    overlap_results[overlap_type] = {'static_ids': static_ids, 'evolution_ids': evolution_ids}


# get counts of features in each category
for overlap_type, results in overlap_results.items():
    static_ids = results['static_ids']
    evolution_ids = results['evolution_ids']
    venn2([set(static_ids), set(evolution_ids)], set_labels=('Static', 'Evolution'))
    plt.title(overlap_type)
    plt.savefig(os.path.join(plot_dir, 'Figure6_top_features_overlap_{}.pdf'.format(overlap_type)))
    plt.close()



# plot same features across multiple timepoints
comparison_features = ['Cancer__T__ratio__cancer_border', 'Cancer__T__ratio__cancer_core', 'PDL1+__APC',
                       'Stroma__T__ratio__cancer_core', 'CD8T__cluster_density__cancer_border', 'CD45RO+__all', 'Cancer_Other__proportion_of__Cancer']

for feature_name in comparison_features:
    plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name), :]

    for comparison in ['primary_untreated', 'baseline', 'post_induction', 'on_nivo']:
        # plot
        comparison_df = plot_df.loc[plot_df.Timepoint == comparison, :]
        sns.stripplot(data=comparison_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                    color='grey')
        plt.title(feature_name)
        plt.savefig(os.path.join(plot_dir, 'comparison_features', f'{feature_name}_{comparison}.png'))
        plt.close()

# features to keep: CAncer_T_ratio_cancer_border, CD8T_density_cancer_border, Cancer_3_proprtion of cancer
comparison_features = ['Cancer__T__ratio__cancer_border',  'Cancer_Other__proportion_of__Cancer'] # 'CD8T__cluster_density__cancer_border',
plotting_ranges = [[0, 15],  [0, 1]] # [0, .4],
for feature_name, plotting_range in zip(comparison_features, plotting_ranges):
    plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name), :]

    for comparison in ['primary_untreated', 'baseline', 'post_induction', 'on_nivo']:
        # plot
        comparison_df = plot_df.loc[plot_df.Timepoint == comparison, :]
        # plot feature across timepoints
        fig, ax = plt.subplots(figsize=(4, 5))
        sns.stripplot(data=comparison_df, x='Clinical_benefit', y='raw_mean', ax=ax, order=['Yes', 'No'], color='black')
        sns.boxplot(data=comparison_df, x='Clinical_benefit', y='raw_mean', color='grey', order=['Yes', 'No'], ax=ax, showfliers=False, width=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylim(plotting_range)

        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{feature_name}_{comparison}.pdf'))
        plt.close()


# longitudinal T / Cancer ratios
combined_df = pd.read_csv(os.path.join(base_dir, 'analysis_files/timepoint_combined_features.csv'))

# check for longitudinal patients
longitudinal_patients = combined_df.loc[combined_df.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo',]), :]
longitudinal_patients = longitudinal_patients.loc[longitudinal_patients.Clinical_benefit == 'Yes', :]
longitudinal_patients = longitudinal_patients.loc[longitudinal_patients.feature_name_unique == 'T__Cancer__ratio__cancer_border', :]

longitudinal_wide = longitudinal_patients.pivot(index=['Patient_ID'], columns='Timepoint', values='raw_mean')

feature_name = 'T__Cancer__ratio__cancer_border'

# corresponding overlays
cell_table_clusters = pd.read_csv(os.path.join(base_dir, 'analysis_files/cell_table_clusters.csv'))
annotations_by_mask = pd.read_csv(os.path.join(base_dir, 'intermediate_files/mask_dir/individual_masks-no_tagg_tls/cell_annotation_mask.csv'))
annotations_by_mask = annotations_by_mask.rename(columns={'mask_name': 'tumor_region'})
cell_table_clusters = cell_table_clusters.merge(annotations_by_mask, on=['fov', 'label'], how='left')

tc_colormap = pd.DataFrame({'T_C_ratio': ['T', 'Cancer', 'Other_region', 'Other_cells'],
                         'color': ['navajowhite','white', 'grey', 'grey']})

## check for nivo FOVs
timepoint = 'on_nivo'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                                    (combined_df.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(2, 4))
sns.stripplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='grey', ax=ax, showfliers=False, width=0.3)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([-15, 0])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure5_feature_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()


# pick patients for visualization
subset = plot_df.loc[plot_df.raw_mean > -4, :]

pats = [26, 33, 59, 62, 64, 65, 115, 118]
fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID.isin(pats) & harmonized_metadata.MIBI_data_generated.values), 'fov'].unique()

# add column for T in cancer border, T elsewhere, and others
cell_table_subset = cell_table_clusters.loc[(cell_table_clusters.fov.isin(fovs)), :]
cell_table_subset['T_C_ratio'] = cell_table_subset.cell_cluster_broad
cell_table_subset.loc[~cell_table_subset.T_C_ratio.isin(['T', 'Cancer']), 'T_C_ratio'] = 'Other_cells'
cell_table_subset.loc[cell_table_subset.tumor_region != 'cancer_border', 'T_C_ratio'] = 'Other_region'

#outside_t = (cell_table_subset.cell_cluster_broad == 'T') & (cell_table_subset.tumor_region != 'cancer_border')
#outside_cancer = (cell_table_subset.cell_cluster_broad == 'Cancer') & (cell_table_subset.tumor_region != 'cancer_border')
#cell_table_subset.loc[outside_t, 'T_C_ratio'] = 'T_outside'
#cell_table_subset.loc[outside_cancer, 'T_C_ratio'] = 'Cancer_outside'

tc_nivo_plot_dir = os.path.join(plot_dir, 'Figure5_tc_overlays_nivo')
if not os.path.exists(tc_nivo_plot_dir):
    os.mkdir(tc_nivo_plot_dir)


for pat in pats:
    pat_fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID == pat) & (harmonized_metadata.MIBI_data_generated.values) & (harmonized_metadata.Timepoint == timepoint), 'fov'].unique()
    pat_df = cell_table_subset.loc[cell_table_subset.fov.isin(pat_fovs), :]

    pat_dir = os.path.join(tc_nivo_plot_dir, 'patient_{}'.format(pat))
    if not os.path.exists(pat_dir):
        os.mkdir(pat_dir)

    cohort_cluster_plot(
        fovs=pat_fovs,
        seg_dir=seg_dir,
        save_dir=pat_dir,
        cell_data=pat_df,
        erode=True,
        fov_col=settings.FOV_ID,
        label_col=settings.CELL_LABEL,
        cluster_col='T_C_ratio',
        seg_suffix="_whole_cell.tiff",
        cmap=tc_colormap,
        display_fig=False,
    )


## check for induction FOVs
timepoint = 'post_induction'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                                    (combined_df.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(2, 4))
sns.stripplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='grey', ax=ax, showfliers=False, width=0.3)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([-15, 0])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure5_feature_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()


# pick patients for visualization
subset = plot_df.loc[plot_df.raw_mean > -6, :]

pats = [26, 4, 5, 11, 40, 37, 46, 56, 62, 64, 65, 102]
fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID.isin(pats) & harmonized_metadata.MIBI_data_generated.values), 'fov'].unique()

# add column for T in cancer border, T elsewhere, and others
cell_table_subset = cell_table_clusters.loc[(cell_table_clusters.fov.isin(fovs)), :]
cell_table_subset['T_C_ratio'] = cell_table_subset.cell_cluster_broad
cell_table_subset.loc[~cell_table_subset.T_C_ratio.isin(['T', 'Cancer']), 'T_C_ratio'] = 'Other_cells'
cell_table_subset.loc[cell_table_subset.tumor_region != 'cancer_border', 'T_C_ratio'] = 'Other_region'

#outside_t = (cell_table_subset.cell_cluster_broad == 'T') & (cell_table_subset.tumor_region != 'cancer_border')
#outside_cancer = (cell_table_subset.cell_cluster_broad == 'Cancer') & (cell_table_subset.tumor_region != 'cancer_border')
#cell_table_subset.loc[outside_t, 'T_C_ratio'] = 'T_outside'
#cell_table_subset.loc[outside_cancer, 'T_C_ratio'] = 'Cancer_outside'

tc_induction_plot_dir = os.path.join(plot_dir, 'Figure5_tc_overlays_induction')
if not os.path.exists(tc_induction_plot_dir):
    os.mkdir(tc_induction_plot_dir)


for pat in pats:
    pat_fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID == pat) & (harmonized_metadata.MIBI_data_generated.values) & (harmonized_metadata.Timepoint == timepoint), 'fov'].unique()
    pat_df = cell_table_subset.loc[cell_table_subset.fov.isin(pat_fovs), :]

    pat_dir = os.path.join(tc_induction_plot_dir, 'patient_{}'.format(pat))
    if not os.path.exists(pat_dir):
        os.mkdir(pat_dir)

    cohort_cluster_plot(
        fovs=pat_fovs,
        seg_dir=seg_dir,
        save_dir=pat_dir,
        cell_data=pat_df,
        erode=True,
        fov_col=settings.FOV_ID,
        label_col=settings.CELL_LABEL,
        cluster_col='T_C_ratio',
        seg_suffix="_whole_cell.tiff",
        cmap=tc_colormap,
        display_fig=False,
    )

## check for baseline FOVs
timepoint = 'baseline'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                                    (combined_df.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(2, 4))
sns.stripplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='grey', ax=ax, showfliers=False, width=0.3)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([-15, 0])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure5_feature_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()


# pick patients for visualization
subset = plot_df.loc[plot_df.raw_mean > -6, :]

pats = [26, 5, 11, 56, 64, 65, 84, 100, 102, 115, 118]
fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID.isin(pats) & harmonized_metadata.MIBI_data_generated.values), 'fov'].unique()

# add column for T in cancer border, T elsewhere, and others
cell_table_subset = cell_table_clusters.loc[(cell_table_clusters.fov.isin(fovs)), :]
cell_table_subset['T_C_ratio'] = cell_table_subset.cell_cluster_broad
cell_table_subset.loc[~cell_table_subset.T_C_ratio.isin(['T', 'Cancer']), 'T_C_ratio'] = 'Other_cells'
cell_table_subset.loc[cell_table_subset.tumor_region != 'cancer_border', 'T_C_ratio'] = 'Other_region'

#outside_t = (cell_table_subset.cell_cluster_broad == 'T') & (cell_table_subset.tumor_region != 'cancer_border')
#outside_cancer = (cell_table_subset.cell_cluster_broad == 'Cancer') & (cell_table_subset.tumor_region != 'cancer_border')
#cell_table_subset.loc[outside_t, 'T_C_ratio'] = 'T_outside'
#cell_table_subset.loc[outside_cancer, 'T_C_ratio'] = 'Cancer_outside'

tc_baseline_plot_dir = os.path.join(plot_dir, 'Figure5_tc_overlays_baseline')
if not os.path.exists(tc_baseline_plot_dir):
    os.mkdir(tc_baseline_plot_dir)


for pat in pats:
    pat_fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID == pat) & (harmonized_metadata.MIBI_data_generated.values) & (harmonized_metadata.Timepoint == timepoint), 'fov'].unique()
    pat_df = cell_table_subset.loc[cell_table_subset.fov.isin(pat_fovs), :]

    pat_dir = os.path.join(tc_baseline_plot_dir, 'patient_{}'.format(pat))
    if not os.path.exists(pat_dir):
        os.mkdir(pat_dir)

    cohort_cluster_plot(
        fovs=pat_fovs,
        seg_dir=seg_dir,
        save_dir=pat_dir,
        cell_data=pat_df,
        erode=True,
        fov_col=settings.FOV_ID,
        label_col=settings.CELL_LABEL,
        cluster_col='T_C_ratio',
        seg_suffix="_whole_cell.tiff",
        cmap=tc_colormap,
        display_fig=False,
    )

## check for primary FOVs
timepoint = 'primary_untreated'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                                    (combined_df.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(2, 4))
sns.stripplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='grey', ax=ax, showfliers=False, width=0.3)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([-15, 0])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure5_feature_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()


# pick patients for visualization
subset = plot_df.loc[plot_df.Clinical_benefit == "Yes", :]

pats = [26, 59, 105, 4, 26, 11, 37, 14, 46, 62, 121, 85]
fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID.isin(pats) & harmonized_metadata.MIBI_data_generated.values), 'fov'].unique()

# add column for T in cancer border, T elsewhere, and others
cell_table_subset = cell_table_clusters.loc[(cell_table_clusters.fov.isin(fovs)), :]
cell_table_subset['T_C_ratio'] = cell_table_subset.cell_cluster_broad
cell_table_subset.loc[~cell_table_subset.T_C_ratio.isin(['T', 'Cancer']), 'T_C_ratio'] = 'Other_cells'
cell_table_subset.loc[cell_table_subset.tumor_region != 'cancer_border', 'T_C_ratio'] = 'Other_region'

#outside_t = (cell_table_subset.cell_cluster_broad == 'T') & (cell_table_subset.tumor_region != 'cancer_border')
#outside_cancer = (cell_table_subset.cell_cluster_broad == 'Cancer') & (cell_table_subset.tumor_region != 'cancer_border')
#cell_table_subset.loc[outside_t, 'T_C_ratio'] = 'T_outside'
#cell_table_subset.loc[outside_cancer, 'T_C_ratio'] = 'Cancer_outside'

tc_primary_plot_dir = os.path.join(plot_dir, 'Figure5_tc_overlays_primary')
if not os.path.exists(tc_primary_plot_dir):
    os.mkdir(tc_primary_plot_dir)


for pat in pats:
    pat_fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID == pat) & (harmonized_metadata.MIBI_data_generated.values) & (harmonized_metadata.Timepoint == timepoint), 'fov'].unique()
    pat_df = cell_table_subset.loc[cell_table_subset.fov.isin(pat_fovs), :]

    pat_dir = os.path.join(tc_primary_plot_dir, 'patient_{}'.format(pat))
    if not os.path.exists(pat_dir):
        os.mkdir(pat_dir)

    cohort_cluster_plot(
        fovs=pat_fovs,
        seg_dir=seg_dir,
        save_dir=pat_dir,
        cell_data=pat_df,
        erode=True,
        fov_col=settings.FOV_ID,
        label_col=settings.CELL_LABEL,
        cluster_col='T_C_ratio',
        seg_suffix="_whole_cell.tiff",
        cmap=tc_colormap,
        display_fig=False,
    )


# generate crops for selected FOVs
# nivo: 33, 65, 115
# induction: 37
# baseline: 26
# primary: 4, 11, 37
fovs = ['TONIC_TMA12_R5C6', 'TONIC_TMA7_R3C6', 'TONIC_TMA5_R5C2', 'TONIC_TMA2_R8C4'] # 65 (nivo), 37 (induction), 26 (baseline), 4 (primary)

subset_dir = os.path.join(plot_dir, 'Figure5_tc_overlays_selected')
if not os.path.exists(subset_dir):
    os.mkdir(subset_dir)

cohort_cluster_plot(
    fovs=fovs,
    seg_dir=seg_dir,
    save_dir=subset_dir,
    cell_data=cell_table_subset,
    erode=True,
    fov_col=settings.FOV_ID,
    label_col=settings.CELL_LABEL,
    cluster_col='T_C_ratio',
    seg_suffix="_whole_cell.tiff",
    cmap=tc_colormap,
    display_fig=False,
)


# same thing for compartment masks
compartment_colormap = pd.DataFrame({'tumor_region': ['cancer_core', 'cancer_border', 'stroma_border', 'stroma_core'],
                         'color': ['blue', 'deepskyblue', 'lightcoral', 'firebrick']})
subset_mask_dir = os.path.join(plot_dir, 'Figure5_tc_overlays_selected_masks')
if not os.path.exists(subset_mask_dir):
    os.mkdir(subset_mask_dir)

cohort_cluster_plot(
    fovs=fovs,
    seg_dir=seg_dir,
    save_dir=subset_mask_dir,
    cell_data=cell_table_subset,
    erode=True,
    fov_col=settings.FOV_ID,
    label_col=settings.CELL_LABEL,
    cluster_col='tumor_region',
    seg_suffix="_whole_cell.tiff",
    cmap=compartment_colormap,
    display_fig=False,
)


# crop overlays
fov1 = fovs[0]
row_start, col_start = 1350, 200
row_len, col_len = 600, 900

for dir in [subset_dir, subset_mask_dir]:
    fov1_image = io.imread(os.path.join(dir, 'cluster_masks_colored', fov1 + '.tiff'))
    fov1_image = fov1_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
    io.imsave(os.path.join(dir, 'cluster_masks_colored', fov1 + '_crop.tiff'), fov1_image)

fov2 = fovs[1]
row_start, col_start = 424, 0
row_len, col_len = 600, 900

for dir in [subset_dir, subset_mask_dir]:
    fov2_image = io.imread(os.path.join(dir, 'cluster_masks_colored', fov2 + '.tiff'))
    fov2_image = fov2_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
    io.imsave(os.path.join(dir, 'cluster_masks_colored', fov2 + '_crop.tiff'), fov2_image)

fov3 = fovs[2]
row_start, col_start = 700, 450
row_len, col_len = 900, 600

for dir in [subset_dir, subset_mask_dir]:
    fov3_image = io.imread(os.path.join(dir, 'cluster_masks_colored', fov3 + '.tiff'))
    fov3_image = fov3_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
    io.imsave(os.path.join(dir, 'cluster_masks_colored', fov3 + '_crop.tiff'), fov3_image)

fov4 = fovs[3]
row_start, col_start = 1200, 750
row_len, col_len = 600, 900

for dir in [subset_dir, subset_mask_dir]:
    fov4_image = io.imread(os.path.join(dir, 'cluster_masks_colored', fov4 + '.tiff'))
    fov4_image = fov4_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
    io.imsave(os.path.join(dir, 'cluster_masks_colored', fov4 + '_crop.tiff'), fov4_image)



