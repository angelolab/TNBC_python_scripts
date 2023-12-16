import os
import shutil

import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn2
from ark.utils.plot_utils import cohort_cluster_plot, color_segmentation_by_stat
import ark.settings as settings
import skimage.io as io


base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
metadata_dir = os.path.join(base_dir, 'intermediate_files/metadata')
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/'
harmonized_metadata = pd.read_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'))
seg_dir = os.path.join(base_dir, 'segmentation_data/deepcell_output')
image_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/'

study_fovs = harmonized_metadata.loc[harmonized_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), 'fov'].values



#
# Figure 5
#

# additional features:
# Cancer / T ratio cancer border in baseline.
# Cancer / Immune mixing score baseline to nivo
# Cancer / T ratio cance core on nivo
# PDL1+ M1_Mac on Nivo
# Diversity Cell Cluster Cancer On nivo
# cluster broad diversity cancer border on nivo



# plot specific top features
combined_df = pd.read_csv(os.path.join(base_dir, 'analysis_files/timepoint_combined_features.csv'))

# PDL1+__APC in induction
feature_name = 'PDL1+__APC'
timepoint = 'post_induction'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                                    (combined_df.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='grey', ax=ax, showfliers=False)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([0, 1])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure5_feature_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()

cell_table_func = pd.read_csv(os.path.join(base_dir, 'analysis_files/cell_table_func_single_positive.csv'))

# corresponding overlays
subset = plot_df.loc[plot_df.raw_mean < 0.1, :]

pats = [5, 26, 40, 37, 33, 62, 102]
pats = [16, 15, 17, 27, 32, 34, 39, 48, 57]
fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID.isin(pats) & harmonized_metadata.MIBI_data_generated.values), 'fov'].unique()

cell_table_subset = cell_table_func.loc[(cell_table_func.fov.isin(fovs)), :]
cell_table_subset['APC_plot'] = cell_table_subset.cell_cluster
cell_table_subset.loc[cell_table_subset.cell_cluster != 'APC', 'APC_plot'] = 'Other'
cell_table_subset.loc[(cell_table_subset.cell_cluster == 'APC') & (cell_table_subset.PDL1.values), 'APC_plot'] = 'APC_PDL1+'

apc_colormap = pd.DataFrame({'APC_plot': ['APC', 'Other', 'APC_PDL1+'],
                         'color': ['blue','grey', 'lightsteelblue']})
apc_plot_dir = os.path.join(plot_dir, 'Figure5_APC_overlays_new')
if not os.path.exists(apc_plot_dir):
    os.mkdir(apc_plot_dir)


for pat in pats:
    pat_fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID == pat) & (harmonized_metadata.MIBI_data_generated.values) & (harmonized_metadata.Timepoint == 'post_induction'), 'fov'].unique()
    pat_df = cell_table_subset.loc[cell_table_subset.fov.isin(pat_fovs), :]

    pat_dir = os.path.join(apc_plot_dir, 'patient_{}'.format(pat))
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
        cluster_col='APC_plot',
        seg_suffix="_whole_cell.tiff",
        cmap=apc_colormap,
        display_fig=False,
    )


# create crops for selected FOVs
subset_fovs = ['TONIC_TMA2_R11C6', 'TONIC_TMA11_R8C1', 'TONIC_TMA4_R5C6', 'TONIC_TMA6_R5C3'] # 5, 62, 15, 32

subset_dir = os.path.join(apc_plot_dir, 'selected_fovs')
if not os.path.exists(subset_dir):
    os.mkdir(subset_dir)


cohort_cluster_plot(
    fovs=subset_fovs,
    seg_dir=seg_dir,
    save_dir=subset_dir,
    cell_data=cell_table_subset,
    erode=True,
    fov_col=settings.FOV_ID,
    label_col=settings.CELL_LABEL,
    cluster_col='APC_plot',
    seg_suffix="_whole_cell.tiff",
    cmap=apc_colormap,
    display_fig=False,
)


# select crops for visualization
fov1 = subset_fovs[0]
row_start, col_start = 1100, 1000
row_len, col_len = 500, 1000

fov1_image = io.imread(os.path.join(subset_dir, 'cluster_masks_colored', fov1 + '.tiff'))
fov1_image = fov1_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
io.imsave(os.path.join(subset_dir, 'cluster_masks_colored', fov1 + '_crop.tiff'), fov1_image)


fov2 = subset_fovs[1]
row_start, col_start = 20, 1500
row_len, col_len = 1000, 500

fov2_image = io.imread(os.path.join(subset_dir, 'cluster_masks_colored', fov2 + '.tiff'))
fov2_image = fov2_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
io.imsave(os.path.join(subset_dir, 'cluster_masks_colored', fov2 + '_crop.tiff'), fov2_image)



fov3 = subset_fovs[2]
row_start, col_start = 0, 0
row_len, col_len = 500, 1000

fov3_image = io.imread(os.path.join(subset_dir, 'cluster_masks_colored', fov3 + '.tiff'))
fov3_image = fov3_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
io.imsave(os.path.join(subset_dir, 'cluster_masks_colored', fov3 + '_crop.tiff'), fov3_image)


fov4 = subset_fovs[3]
row_start, col_start = 1000, 100
row_len, col_len = 1000, 500

fov4_image = io.imread(os.path.join(subset_dir, 'cluster_masks_colored', fov4 + '.tiff'))
fov4_image = fov4_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
io.imsave(os.path.join(subset_dir, 'cluster_masks_colored', fov4 + '_crop.tiff'), fov4_image)


# diversity of cancer border at induction
feature_name = 'cluster_broad_diversity_cancer_border'
timepoint = 'post_induction'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                                    (combined_df.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='grey', ax=ax, showfliers=False)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([0, 2])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure5_feature_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()



# corresponding overlays
cell_table_clusters = pd.read_csv(os.path.join(data_dir, 'post_processing/cell_table_clusters.csv'))
annotations_by_mask = pd.read_csv(os.path.join(data_dir, 'post_processing', 'cell_annotation_mask.csv'))
annotations_by_mask = annotations_by_mask.rename(columns={'mask_name': 'tumor_region'})
cell_table_clusters = cell_table_clusters.merge(annotations_by_mask, on=['fov', 'label'], how='left')

subset = plot_df.loc[plot_df.raw_mean < 0.3, :]

pats = [5, 40, 37, 46, 56, 33, 62, 64]
pats = [8, 16, 17, 27, 25, 31, 34]
fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID.isin(pats) & harmonized_metadata.MIBI_data_generated.values), 'fov'].unique()

# 25, 62 previously included

# add column for CD8T in cancer border, CD8T elsewhere, and others
cell_table_subset = cell_table_clusters.loc[(cell_table_clusters.fov.isin(fovs)), :]
cell_table_subset['border_plot'] = cell_table_subset.cell_cluster_broad
cell_table_subset.loc[cell_table_subset.tumor_region != 'cancer_border', 'border_plot'] = 'Other_region'

figure_dir = os.path.join(plot_dir, 'Figure5_border_diversity')
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

diversity_colormap = pd.DataFrame({'border_plot': ['Cancer', 'Stroma', 'Granulocyte', 'T', 'B', 'Mono_Mac', 'Other', 'NK', 'Other_region'],
                         'color': ['white', 'lightcoral', 'sandybrown', 'lightgreen', 'aqua', 'dodgerblue', 'darkviolet', 'crimson', 'gray']})


for pat in pats:
    pat_dir = os.path.join(figure_dir, 'patient_{}'.format(pat))
    if not os.path.exists(pat_dir):
        os.mkdir(pat_dir)
    pat_fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID == pat) & (harmonized_metadata.MIBI_data_generated.values) & (harmonized_metadata.Timepoint == 'post_induction'), 'fov'].unique()
    pat_df = cell_table_subset.loc[cell_table_subset.fov.isin(pat_fovs), :]

    cohort_cluster_plot(
        fovs=pat_fovs,
        seg_dir=seg_dir,
        save_dir=pat_dir,
        cell_data=pat_df,
        erode=True,
        fov_col=settings.FOV_ID,
        label_col=settings.CELL_LABEL,
        cluster_col='border_plot',
        seg_suffix="_whole_cell.tiff",
        cmap=diversity_colormap,
        display_fig=False,
    )

subset_fovs = ['TONIC_TMA2_R11C6', 'TONIC_TMA5_R4C4'] # 5, 25

subset_dir = os.path.join(figure_dir, 'selected_fovs')
if not os.path.exists(subset_dir):
    os.mkdir(subset_dir)

cohort_cluster_plot(
    fovs=subset_fovs,
    seg_dir=seg_dir,
    save_dir=subset_dir,
    cell_data=cell_table_subset,
    erode=True,
    fov_col=settings.FOV_ID,
    label_col=settings.CELL_LABEL,
    cluster_col='border_plot',
    seg_suffix="_whole_cell.tiff",
    cmap=diversity_colormap,
    display_fig=False,
)


# same thing for compartment masks
compartment_colormap = pd.DataFrame({'tumor_region': ['cancer_core', 'cancer_border', 'stroma_border', 'stroma_core'],
                         'color': ['blue', 'deepskyblue', 'lightcoral', 'firebrick']})
subset_mask_dir = os.path.join(figure_dir, 'selected_fovs_masks')
if not os.path.exists(subset_mask_dir):
    os.mkdir(subset_mask_dir)

cohort_cluster_plot(
    fovs=subset_fovs,
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
fov1 = subset_fovs[0]
row_start, col_start = 1200, 300
row_len, col_len = 500, 1000

for dir in [subset_dir, subset_mask_dir]:
    fov1_image = io.imread(os.path.join(dir, 'cluster_masks_colored', fov1 + '.tiff'))
    fov1_image = fov1_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
    io.imsave(os.path.join(dir, 'cluster_masks_colored', fov1 + '_crop.tiff'), fov1_image)

fov2 = subset_fovs[1]
row_start, col_start = 400, 20
row_len, col_len = 500, 1000

for dir in [subset_dir, subset_mask_dir]:
    fov2_image = io.imread(os.path.join(dir, 'cluster_masks_colored', fov2 + '.tiff'))
    fov2_image = fov2_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
    io.imsave(os.path.join(dir, 'cluster_masks_colored', fov2 + '_crop.tiff'), fov2_image)



# change in CD8T density in cancer border
feature_name = 'CD8T__cluster_density__cancer_border'
timepoint = 'post_induction__on_nivo'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                            (combined_df.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='grey', ax=ax, showfliers=False)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([-.05, .2])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure5_feature_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()


# corresponding overlays
cell_table_clusters = pd.read_csv(os.path.join(data_dir, 'post_processing/cell_table_clusters.csv'))
annotations_by_mask = pd.read_csv(os.path.join(data_dir, 'post_processing', 'cell_annotation_mask.csv'))
annotations_by_mask = annotations_by_mask.rename(columns={'mask_name': 'tumor_region'})
cell_table_clusters = cell_table_clusters.merge(annotations_by_mask, on=['fov', 'label'], how='left')

pats = [62, 65, 26, 117, 2]
fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID.isin(pats) & harmonized_metadata.MIBI_data_generated.values), 'fov'].unique()

# add column for CD8T in cancer border, CD8T elsewhere, and others
cell_table_subset = cell_table_clusters.loc[(cell_table_clusters.fov.isin(fovs)), :]
cell_table_subset['CD8T_plot'] = cell_table_subset.tumor_region
cell_table_subset.loc[cell_table_subset.cell_cluster == 'CD8T', 'CD8T_plot'] = 'CD8T'
cell_table_subset.loc[(cell_table_subset.cell_cluster == 'CD8T') & (cell_table_subset.tumor_region == 'cancer_border'), 'CD8T_plot'] = 'border_CD8T'
cell_table_subset.loc[cell_table_subset.CD8T_plot.isin(['stroma_core', 'stroma_border', 'tls', 'tagg']), 'CD8T_plot'] = 'stroma'
cell_table_subset.loc[cell_table_subset.CD8T_plot.isin(['cancer_core', 'cancer_border']), 'CD8T_plot'] = 'cancer'

figure_dir = os.path.join(plot_dir, 'Figure5_CD8T_density')
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

CD8_colormap = pd.DataFrame({'CD8T_plot': ['stroma', 'cancer', 'CD8T', 'border_CD8T'],
                         'color': ['skyblue', 'wheat', 'coral', 'maroon']})

for pat in pats:
    pat_dir = os.path.join(figure_dir, 'Patient_{}'.format(pat))
    if not os.path.exists(pat_dir):
        os.mkdir(pat_dir)
    for timepoint in ['post_induction', 'on_nivo']:
        pat_fovs = harmonized_metadata.loc[(harmonized_metadata.Patient_ID == pat) & (harmonized_metadata.MIBI_data_generated.values) & (harmonized_metadata.Timepoint == timepoint), 'fov'].unique()
        pat_df = cell_table_subset.loc[cell_table_subset.fov.isin(pat_fovs), :]

        tp_dir = os.path.join(pat_dir, timepoint)
        if not os.path.exists(tp_dir):
            os.mkdir(tp_dir)

        # create_cell_overlay(cell_table=pat_df, seg_folder='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output',
        #                     fovs=pat_fovs, cluster_col='CD8T_plot', plot_dir=tp_dir,
        #                     save_names=['{}.png'.format(x) for x in pat_fovs])

        cohort_cluster_plot(
            fovs=pat_fovs,
            seg_dir=seg_dir,
            save_dir=tp_dir,
            cell_data=pat_df,
            erode=True,
            fov_col=settings.FOV_ID,
            label_col=settings.CELL_LABEL,
            cluster_col='CD8T_plot',
            seg_suffix="_whole_cell.tiff",
            cmap=CD8_colormap,
            display_fig=False,
        )

# generate crops for selected FOVs
subset_fovs = ['TONIC_TMA2_R4C4', 'TONIC_TMA2_R4C6', 'TONIC_TMA12_R5C6', 'TONIC_TMA12_R6C2']

subset_dir = os.path.join(figure_dir, 'selected_fovs')
if not os.path.exists(subset_dir):
    os.mkdir(subset_dir)

cohort_cluster_plot(
    fovs=subset_fovs,
    seg_dir=seg_dir,
    save_dir=subset_dir,
    cell_data=cell_table_subset,
    erode=True,
    fov_col=settings.FOV_ID,
    label_col=settings.CELL_LABEL,
    cluster_col='CD8T_plot',
    seg_suffix="_whole_cell.tiff",
    cmap=CD8_colormap,
    display_fig=False,
)

fov1 = 'TONIC_TMA2_R4C4'
row_start, col_start = 400, 1100
row_len, col_len = 700, 500

fov1_image = io.imread(os.path.join(subset_dir, 'cluster_masks_colored', fov1 + '.tiff'))
fov1_image = fov1_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
io.imsave(os.path.join(subset_dir, 'cluster_masks_colored', fov1 + '_crop.tiff'), fov1_image)

fov2 = 'TONIC_TMA2_R4C6'
row_start, col_start = 900, 0
row_len, col_len = 700, 500

fov2_image = io.imread(os.path.join(subset_dir, 'cluster_masks_colored', fov2 + '.tiff'))
fov2_image = fov2_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
io.imsave(os.path.join(subset_dir, 'cluster_masks_colored', fov2 + '_crop.tiff'), fov2_image)

fov3 = 'TONIC_TMA12_R5C6'
row_start, col_start = 800, 600
row_len, col_len = 500, 700

fov3_image = io.imread(os.path.join(subset_dir, 'cluster_masks_colored', fov3 + '.tiff'))
fov3_image = fov3_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
io.imsave(os.path.join(subset_dir, 'cluster_masks_colored', fov3 + '_crop.tiff'), fov3_image)

fov4 = 'TONIC_TMA12_R6C2'
row_start, col_start = 300, 600
row_len, col_len = 700, 500

fov4_image = io.imread(os.path.join(subset_dir, 'cluster_masks_colored', fov4 + '.tiff'))
fov4_image = fov4_image[row_start:row_start + row_len, col_start:col_start + col_len, :]
io.imsave(os.path.join(subset_dir, 'cluster_masks_colored', fov4 + '_crop.tiff'), fov4_image)



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


#
# Figure 6
#

# plot top features
top_features = ranked_features.loc[ranked_features.comparison.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
top_features = top_features.iloc[:100, :]

for idx, (feature_name, comparison) in enumerate(zip(top_features.feature_name_unique, top_features.comparison)):
    plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                              (combined_df.Timepoint == comparison), :]

    # plot
    sns.stripplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                color='grey')
    plt.title(feature_name + ' in ' + comparison)
    plt.savefig(os.path.join(plot_dir, 'top_features', f'{idx}_{feature_name}.png'))
    plt.close()


# summarize distribution of top features
top_features_by_comparison = top_features[['feature_name_unique', 'comparison']].groupby('comparison').count().reset_index()
top_features_by_comparison.columns = ['comparison', 'num_features']
top_features_by_comparison = top_features_by_comparison.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_comparison, x='comparison', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Figure6_num_features_per_comparison.pdf'))
plt.close()


# summarize overlap of top features
top_features_by_feature = top_features[['feature_name_unique', 'comparison']].groupby('feature_name_unique').count().reset_index()
feature_counts = top_features_by_feature.groupby('comparison').count().reset_index()
feature_counts.columns = ['num_comparisons', 'num_features']

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=feature_counts, x='num_comparisons', y='num_features', color='grey', ax=ax)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Figure6_num_comparisons_per_feature.pdf'))
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

# def get_top_x_features_by_list(df, detail_names, x=5, plot_val='importance_score', ascending=False):
#     scores, names = [], []
#     for feature in detail_names:
#         keep_idx = np.logical_or(df.feature_type_detail == feature, df.feature_type_detail_2 == feature)
#         plot_df = df.loc[keep_idx, :]
#         plot_df = plot_df.sort_values(by=plot_val, ascending=ascending)
#         temp_scores = plot_df.iloc[:x, :][plot_val].values
#         scores.append(temp_scores)
#         names.append([feature] * len(temp_scores))
#
#     score_df = pd.DataFrame({'score': np.concatenate(scores), 'name': np.concatenate(names)})
#     return score_df
#
#
# # get importance score of top 5 examples for cell-based features
# cell_type_list, cell_prop_list, comparison_list = [], [], []
#
# for groupings in [[['nivo'], ['post_induction__on_nivo', 'on_nivo', 'baseline__on_nivo']],
#                   [['baseline'], ['baseline']],
#                   [['induction'], ['baseline__post_induction', 'post_induction']]]:
#
#     # score of top 5 features
#     name, comparisons = groupings
#     # cell_type_features = get_top_x_features_by_list(df=ranked_features.loc[ranked_features.comparison.isin(comparisons)],
#     #                                                 detail_names=cell_ordering + ['T', 'Mono_Mac'], x=5, plot_val='combined_rank',
#     #                                                 ascending=True)
#     #
#     # meds = cell_type_features.groupby('name').median().sort_values(by='score', ascending=True)
#     #
#     # fig, ax = plt.subplots(figsize=(4, 6))
#     # sns.stripplot(data=cell_type_features, x='name', y='score', ax=ax, order=meds.index, color='black')
#     # sns.boxplot(data=cell_type_features, x='name', y='score', order=meds.index, color='grey', ax=ax, showfliers=False, width=0.5)
#     # ax.set_title('Densities Ranking')
#     # #ax.set_ylim([0, 1])
#     # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#     # sns.despine()
#     # plt.tight_layout()
#     # plt.savefig(os.path.join(plot_dir, 'Figure5_cell_type_rank_{}.pdf'.format(name)))
#     # plt.close()
#
#     # proportion of features belonging to each cell type
#
#     current_comparison_features = top_features.loc[top_features.comparison.isin(comparisons), :]
#     for cell_type in cell_ordering + ['T', 'Mono_Mac']:
#         cell_idx = np.logical_or(current_comparison_features.feature_type_detail == cell_type,
#                                     current_comparison_features.feature_type_detail_2 == cell_type)
#         cell_type_list.append(cell_type)
#         cell_prop_list.append(np.sum(cell_idx) / len(current_comparison_features))
#         comparison_list.append(name[0])
#
# proportion_df = pd.DataFrame({'cell_type': cell_type_list, 'proportion': cell_prop_list, 'comparison': comparison_list})
#
# fig, ax = plt.subplots(figsize=(4, 4))
# sns.barplot(data=proportion_df, x='cell_type', y='proportion', hue='comparison', hue_order=['nivo', 'baseline', 'induction'], ax=ax)
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, 'Figure5_cell_type_proportion.pdf'))
# plt.close()

# plot top featurse across all comparisons
all_top_features = ranked_features.copy()
all_top_features['feature_cat'] = 0
all_top_features.iloc[:100, -1] = 1
all_top_features.iloc[100:300, -1] = 0.5
all_top_features = all_top_features.loc[all_top_features.feature_name_unique.isin(all_top_features.feature_name_unique.values[:100])]
all_top_features = all_top_features.pivot(index='feature_name_unique', columns='comparison', values='feature_cat')
all_top_features = all_top_features.fillna(0)

#sns.clustermap(data=all_top_features, cmap='RdBu_r', vmin=-1, vmax=1, figsize=(10, 10))
sns.clustermap(data=all_top_features, cmap='YlOrBr', figsize=(10, 30))
plt.savefig(os.path.join(plot_dir, 'top_features_clustermap_all.pdf'))
plt.close()



# plot top features excluding evolution timepoints
all_top_features = ranked_features.copy()
all_top_features = all_top_features.loc[all_top_features.comparison.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
all_top_features['feature_cat'] = 0
all_top_features.iloc[:100, -1] = 1
all_top_features.iloc[100:300, -1] = 0.5
all_top_features = all_top_features.loc[all_top_features.feature_name_unique.isin(all_top_features.feature_name_unique.values[:100])]
all_top_features = all_top_features.pivot(index='feature_name_unique', columns='comparison', values='feature_cat')
all_top_features = all_top_features.fillna(0)

sns.clustermap(data=all_top_features, cmap='YlOrBr', figsize=(10, 15))
plt.savefig(os.path.join(plot_dir, 'top_features_clustermap_no_evolution_scaled.pdf'))
plt.close()


# same as above, but excluding values for signed importance score that are below threshold
all_top_features = ranked_features.copy()
all_top_features = all_top_features.loc[all_top_features.comparison.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
all_top_features.loc[all_top_features.importance_score < 0.85, 'signed_importance_score'] = 0
all_top_features = all_top_features.loc[all_top_features.feature_name_unique.isin(all_top_features.feature_name_unique.values[:100]), :]

all_top_features = all_top_features.pivot(index='feature_name_unique', columns='comparison', values='signed_importance_score')
all_top_features = all_top_features.fillna(0)


sns.clustermap(data=all_top_features, cmap='RdBu_r', figsize=(10, 15))
plt.savefig(os.path.join(plot_dir, 'top_features_clustermap_no_evolution_scaled_signed.pdf'))
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



# TODO: redo this analysis for specific pairs of static and evolution features: i.e. just looking at primary, just looking at on-nivo, etc
# identify features with opposite effects at different timepoints
opposite_features = []
induction_peak_features = []
for feature in all_top_features.index:
    feature_vals = all_top_features.loc[feature, :]

    # get sign of the feature with the max absolute value
    max_idx = np.argmax(np.abs(feature_vals))
    max_sign = np.sign(feature_vals[max_idx])

    # determine if any features of opposite sign have absolute value > 0.85
    opposite_idx = np.logical_and(np.abs(feature_vals) > 0.8, np.sign(feature_vals) != max_sign)
    if np.sum(opposite_idx) > 0:
        opposite_features.append(feature)

    # determine which features have a peak at induction
    opposite_indices = set(np.where(opposite_idx)[0]).union(set([max_idx]))

    # check if 4 and 5 are in the set
    if set([4, 5]).issubset(opposite_indices):
        induction_peak_features.append(feature)


#opposite_features = all_top_features.loc[opposite_features, :]

# create connected lineplots for features with opposite effects

# select patients with data at all timepoints
pats = harmonized_metadata.loc[harmonized_metadata.baseline__on_nivo, 'Patient_ID'].unique().tolist()
pats2 = harmonized_metadata.loc[harmonized_metadata.post_induction__on_nivo, 'Patient_ID'].unique().tolist()
#pats = set(pats).intersection(set(pats2))
#pats = set(pats).union(set(pats2))
pats = pats2

for feature in opposite_features:
    plot_df = combined_df.loc[(combined_df.feature_name_unique == feature) &
                                        (combined_df.Timepoint.isin(['baseline', 'post_induction', 'on_nivo'])) &
                                        (combined_df.Patient_ID.isin(pats)), :]

    plot_df_wide = plot_df.pivot(index=['Patient_ID', 'Clinical_benefit'], columns='Timepoint', values='raw_mean')
    #plot_df_wide.dropna(inplace=True)
    # divide each row by the baseline value
    #plot_df_wide = plot_df_wide.divide(plot_df_wide.loc[:, 'baseline'], axis=0)
    #plot_df_wide = plot_df_wide.subtract(plot_df_wide.loc[:, 'baseline'], axis=0)
    plot_df_wide = plot_df_wide.reset_index()

    plot_df_norm = pd.melt(plot_df_wide, id_vars=['Patient_ID', 'Clinical_benefit'], value_vars=['post_induction', 'on_nivo'])

    plot_df_1 = plot_df_norm.loc[plot_df_norm.Clinical_benefit == 'No', :]
    plot_df_2 = plot_df_norm.loc[plot_df_norm.Clinical_benefit == 'Yes', :]
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    sns.lineplot(data=plot_df_1, x='Timepoint', y='value', units='Patient_ID', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[0])
    sns.lineplot(data=plot_df_2, x='Timepoint', y='value', units='Patient_ID', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[1])
    sns.lineplot(data=plot_df_norm, x='Timepoint', y='value', units='Patient_ID',  hue='Clinical_benefit', estimator=None, alpha=0.5, marker='o', ax=ax[2])

    # set ylimits
    # ax[0].set_ylim([-0.6, 0.6])
    # ax[1].set_ylim([-0.6, 0.6])
    # ax[2].set_ylim([-0.6, 0.6])

    # add responder and non-responder titles
    ax[0].set_title('non-responders')
    ax[1].set_title('responders')
    ax[2].set_title('combined')

    # set figure title
    fig.suptitle(feature)
    plt.savefig(os.path.join(plot_dir, 'longitudinal_response_raw_{}.png'.format(feature)))
    plt.close()

all_opp_features = ranked_features.loc[ranked_features.feature_name_unique.isin(opposite_features), :]
#all_opp_features = all_opp_features.loc[all_opp_features.comparison.isin(['post_induction', 'post_induction__on_nivo']), :]
all_opp_features = all_opp_features.pivot(index='feature_name_unique', columns='comparison', values='signed_importance_score')
all_opp_features = all_opp_features.fillna(0)

sns.clustermap(data=all_opp_features, cmap='RdBu_r', vmin=-1, vmax=1, figsize=(10, 10))
plt.savefig(os.path.join(plot_dir, 'opposite_clustermap.pdf'))
plt.close()

# plot induction peak features
induction_peak_df = ranked_features.loc[ranked_features.feature_name_unique.isin(induction_peak_features), :]
induction_peak_df = induction_peak_df.pivot(index='feature_name_unique', columns='comparison', values='signed_importance_score')
induction_peak_df = induction_peak_df.fillna(0)

induction_peak_df = induction_peak_df[['baseline', 'baseline__on_nivo', 'on_nivo', 'baseline__post_induction',
       'post_induction', 'post_induction__on_nivo']]

sns.clustermap(data=induction_peak_df, cmap='RdBu_r', vmin=-1, vmax=1, figsize=(10, 10), col_cluster=False)
plt.savefig(os.path.join(plot_dir, 'induction_peak_clustermap.pdf'))
plt.close()


# create averaged lineplot for induction peak
baseline_induction_pats = harmonized_metadata.loc[harmonized_metadata.baseline__post_induction, 'Patient_ID'].unique().tolist()
induction_nivo_pats = harmonized_metadata.loc[harmonized_metadata.post_induction__on_nivo, 'Patient_ID'].unique().tolist()
combined_pats = set(baseline_induction_pats).intersection(set(induction_nivo_pats))

plot_df = combined_df.loc[(combined_df.Timepoint.isin(['baseline', 'post_induction', 'on_nivo'])) &
                                    (combined_df.Patient_ID.isin(combined_pats)), :]
plot_df = plot_df.loc[plot_df.feature_name_unique.isin(induction_peak_features), :]

plot_df_wide = plot_df.pivot(index=['Patient_ID', 'iRECIST_response', 'feature_name_unique'], columns='Timepoint', values='raw_mean')
plot_df_wide.dropna(inplace=True)
plot_df_wide = plot_df_wide.reset_index()
plot_df_wide['unique_id'] = np.arange(0, plot_df_wide.shape[0], 1)

induction_peak_features = ['PDL1+__APC', 'CD45RO+__Immune_Other', 'PDL1+__M2_Mac', 'Ki67+__T_Other', 'CD45RO__CD69+__NK', 'TIM3+__CD4T', 'CD69+__CD4T', 'PDL1+__CD4T']
# # divide each row by the baseline value
# #plot_df_wide = plot_df_wide.divide(plot_df_wide.loc[:, 'baseline'], axis=0)
# #plot_df_wide = plot_df_wide.subtract(plot_df_wide.loc[:, 'baseline'], axis=0)
#
plot_df_norm = pd.melt(plot_df_wide, id_vars=['Patient_ID', 'iRECIST_response', 'feature_name_unique', 'unique_id'], value_vars=['baseline', 'post_induction', 'on_nivo'])

plot_df_test = plot_df_norm.loc[plot_df_norm.feature_name_unique == 'PDL1+__APC', :]
plot_df_1 = plot_df_test.loc[plot_df_test.iRECIST_response == 'non-responders', :]
plot_df_2 = plot_df_test.loc[plot_df_test.iRECIST_response == 'responders', :]

plot_df_1 = plot_df_norm.loc[plot_df_norm.iRECIST_response == 'non-responders', :]
plot_df_2 = plot_df_norm.loc[plot_df_norm.iRECIST_response == 'responders', :]
fig, ax = plt.subplots(1, 4, figsize=(15, 10))
sns.lineplot(data=plot_df_1, x='Timepoint', y='value', units='unique_id', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[0])
sns.lineplot(data=plot_df_2, x='Timepoint', y='value', units='unique_id', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[1])
sns.lineplot(data=plot_df_test, x='Timepoint', y='value', units='unique_id',  hue='iRECIST_response', estimator=None, alpha=0.5, marker='o', ax=ax[2])
sns.lineplot(data=plot_df_test, x='Timepoint', y='value', hue='iRECIST_response', estimator='median', alpha=0.5, marker='o', ax=ax[3])

plt.savefig(os.path.join(plot_dir, 'triple_induction_peak_PDL1_APC.png'))
plt.close()

# # set ylimits
# # ax[0].set_ylim([-0.6, 0.6])
# # ax[1].set_ylim([-0.6, 0.6])
# # ax[2].set_ylim([-0.6, 0.6])
#
# # add responder and non-responder titles
# ax[0].set_title('non-responders')
# ax[1].set_title('responders')
# ax[2].set_title('combined')
#
# # set figure title
# fig.suptitle(feature)
# plt.savefig(os.path.join(plot_dir, 'longitudinal_response_raw_{}.png'.format(feature)))
# plt.close()

# make facceted seaborn correlation plot

plot_df = combined_df.loc[(combined_df.Timepoint.isin(['baseline'])) &
                                    (combined_df.feature_name_unique.isin(induction_peak_features)), :]

plot_df_wide = plot_df.pivot(index=['Patient_ID', 'iRECIST_response'], columns='feature_name_unique', values='raw_mean')
plot_df_wide = plot_df_wide.reset_index()
plot_df_wide = plot_df_wide.drop(['Patient_ID'], axis=1)

sns.pairplot(plot_df_wide, hue='iRECIST_response', diag_kind='kde', plot_kws={'alpha': 0.5, 's': 80})
plt.savefig(os.path.join(plot_dir, 'induction_peak_pairplot.png'))
plt.close()

# compute correlations by timepoint
corrs = []
timepoints = []

for timepoint in combined_df.Timepoint.unique():
    timepoint_df = combined_df.loc[(combined_df.Timepoint == timepoint) &
                                    (~combined_df.feature_name_unique.isin(induction_peak_features)), :]

    timepoint_df_wide = timepoint_df.pivot(index=['Patient_ID', 'iRECIST_response'], columns='feature_name_unique', values='raw_mean')
    timepoint_df_wide = timepoint_df_wide.reset_index()
    timepoint_df_wide = timepoint_df_wide.drop(['Patient_ID', 'iRECIST_response'], axis=1)

    corr_vals = timepoint_df_wide.corr(method='spearman').values.flatten()
    corr_vals = corr_vals[corr_vals != 1]
    corrs.extend(corr_vals.tolist())
    timepoints.extend(['others'] * len(corr_vals))

plot_df = pd.DataFrame({'correlation': corrs, 'timepoint': timepoints})
sns.boxplot(data=plot_df, x='timepoint', y='correlation')

switch_patients = patient_metadata.loc[~patient_metadata.survival_diff, 'Patient_ID'].tolist()
# plot patients that switched from non-responders to responders
for feature in induction_peak_features:
    plot_df = combined_df.loc[(combined_df.feature_name_unique == feature) &
                                        (combined_df.Timepoint.isin(['baseline', 'post_induction', 'on_nivo'])) &
                                        (combined_df.Patient_ID.isin(pats)), :]

    plot_df_wide = plot_df.pivot(index=['Patient_ID', 'Clinical_benefit'], columns='Timepoint', values='raw_mean')
    #plot_df_wide.dropna(inplace=True)
    # divide each row by the baseline value
    #plot_df_wide = plot_df_wide.divide(plot_df_wide.loc[:, 'baseline'], axis=0)
    #plot_df_wide = plot_df_wide.subtract(plot_df_wide.loc[:, 'baseline'], axis=0)
    plot_df_wide = plot_df_wide.reset_index()

    plot_df_norm = pd.melt(plot_df_wide, id_vars=['Patient_ID', 'Clinical_benefit'], value_vars=['baseline', 'post_induction', 'on_nivo'])
    plot_df_norm['patient_switch'] = plot_df_norm.Patient_ID.isin(switch_patients)

    plot_df_1 = plot_df_norm.loc[plot_df_norm.Clinical_benefit == 'No', :]
    plot_df_2 = plot_df_norm.loc[plot_df_norm.Clinical_benefit == 'Yes', :]
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    sns.lineplot(data=plot_df_1, x='Timepoint', y='value', units='Patient_ID', hue='patient_switch', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[0])
    sns.lineplot(data=plot_df_2, x='Timepoint', y='value', units='Patient_ID', hue='patient_switch', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[1])
    sns.lineplot(data=plot_df_norm, x='Timepoint', y='value', units='Patient_ID',  hue='Clinical_benefit', estimator=None, alpha=0.5, marker='o', ax=ax[2])

    # set ylimits
    # ax[0].set_ylim([-0.6, 0.6])
    # ax[1].set_ylim([-0.6, 0.6])
    # ax[2].set_ylim([-0.6, 0.6])

    # add responder and non-responder titles
    ax[0].set_title('non-responders')
    ax[1].set_title('responders')
    ax[2].set_title('combined')

    # set figure title
    fig.suptitle(feature)
    plt.savefig(os.path.join(plot_dir, 'longitudinal_response_raw_{}.png'.format(feature)))
    plt.close()


# Figure 7
cv_scores = pd.read_csv(os.path.join(base_dir, 'multivariate_lasso', 'results_1112_cv.csv'))
cv_scores['fold'] = len(cv_scores)

cv_scores_long = pd.melt(cv_scores, id_vars=['fold'], value_vars=cv_scores.columns)


fig, ax = plt.subplots(1, 1, figsize=(3, 4))
order = ['primary', 'post_induction', 'baseline', 'on_nivo']
         #'primary_AND_baseline', 'primary_AND_post_induction', 'baseline_AND_post_induction']
sns.stripplot(data=cv_scores_long, x='variable', y='value', order=order,
                color='black', ax=ax)
sns.boxplot(data=cv_scores_long, x='variable', y='value', order=order,
                color='grey', ax=ax, showfliers=False)

ax.set_title('AUC')
ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure6_AUC.pdf'))
plt.close()



# testing args
# width = 0.8, linewidth=None, gap=0,
fig, ax = plt.subplots(1, 1, figsize=(3, 4))
order = ['primary', 'post_induction', 'baseline', 'on_nivo']
sns.boxplot(data=cv_scores_long, x='variable', y='value', order=order,
                color='grey', ax=ax, showfliers=False, width=0.3)
