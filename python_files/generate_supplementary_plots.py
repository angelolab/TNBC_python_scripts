# File with code for generating supplementary plots
import itertools
import os
import random

import numpy as np
import pandas as pd
import matplotlib
from ark.utils.plot_utils import cohort_cluster_plot
from toffy import qc_comp, qc_metrics_plots
from alpineer import io_utils


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, Normalize

import supplementary_plot_helpers

ANALYSIS_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files"
INTERMEDIATE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files"
OUTPUT_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/output_files"
SUPPLEMENTARY_FIG_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"


# Panel validation
panel_validation_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "panel_validation")
if not os.path.exists(panel_validation_viz_dir):
    os.makedirs(panel_validation_viz_dir)

controls_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/controls"
controls_fov = "TONIC_TMA1_colon_bottom"
supplementary_plot_helpers.validate_panel(
    controls_dir, controls_fov, panel_validation_viz_dir, font_size=180
)

samples_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples"
samples_fov = "TONIC_TMA2_R5C4"
samples_channels = sorted(io_utils.remove_file_extensions(
    io_utils.list_files(os.path.join(samples_dir, samples_fov), substrs=".tiff")
))
exclude_chans = ["CD11c_nuc_exclude", "CK17_smoothed", "ECAD_smoothed", "FOXP3_nuc_include",
                 "chan_39", "chan_45", "chan_48", "chan_115", "chan_141"]
for ec in exclude_chans:
    if ec in samples_channels:
        samples_channels.remove(ec)
supplementary_plot_helpers.validate_panel(
    samples_dir, samples_fov, panel_validation_viz_dir, channels=samples_channels, font_size=320
)


# ROI selection


# QC
qc_metrics = ["Non-zero mean intensity"]
channel_exclude = ["chan_39", "chan_45", "CD11c_nuc_exclude", "CD11c_nuc_exclude_update",
                   "FOXP3_nuc_include", "FOXP3_nuc_include_update", "CK17_smoothed",
                   "FOXP3_nuc_exclude_update", "chan_48", "chan_141", "chan_115", "LAG3"]

## FOV spatial location
cohort_path = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples"
qc_tma_metrics_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/qc_metrics/qc_tma_metrics"
if not os.path.exists(qc_tma_metrics_dir):
    os.makedirs(qc_tma_metrics_dir)

fovs = io_utils.list_folders(cohort_path)
tmas = list(set([fov.split('_R')[0] for fov in fovs]))

qc_tmas = qc_comp.QCTMA(
    qc_metrics=qc_metrics,
    cohort_path=cohort_path,
    metrics_dir=qc_tma_metrics_dir,
)

qc_tmas.compute_qc_tma_metrics(tmas=tmas)
qc_tmas.qc_tma_metrics_zscore(tmas=tmas, channel_exclude=channel_exclude)
qc_metrics_plots.qc_tmas_metrics_plot(qc_tmas=qc_tmas, tmas=tmas, save_figure=True, dpi=300)

## longitudinal controls
control_path = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/controls"
qc_control_metrics_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/qc_metrics/qc_longitudinal_control"
if not os.path.exists(qc_control_metrics_dir):
    os.makedirs(qc_control_metrics_dir)

folders = io_utils.list_folders(control_path, "TMA3_")
control_substrs = [name.split("_")[2] + '_' + name.split("_")[3] if len(name.split("_")) == 4
                   else name.split("_")[2] + '_' + name.split("_")[3]+'_' + name.split("_")[4]
                   for name in folders]

all_folders = io_utils.list_folders(control_path)
for i, control in enumerate(control_substrs):
    control_sample_name = control
    print(control)
    if control == 'tonsil_bottom':
        fovs = [folder for folder in all_folders if control in folder and len(folder) <= 25]
    else:
        fovs = [folder for folder in all_folders if control in folder]

    qc_control = qc_comp.QCControlMetrics(
        qc_metrics=qc_metrics,
        cohort_path=control_path,
        metrics_dir=qc_control_metrics_dir,
    )

    qc_control.compute_control_qc_metrics(
        control_sample_name=control_sample_name,
        fovs=fovs,
        channel_exclude=channel_exclude,
        channel_include=None,
    )

    qc_metrics_plots.longitudinal_control_heatmap(
        qc_control=qc_control, control_sample_name=control_sample_name, save_figure=True, dpi=300
    )

dfs = []
for control in control_substrs:
    df = pd.read_csv(os.path.join(qc_control_metrics_dir, f"{control}_combined_nonzero_mean_stats.csv"))
    df['fov'] = [i.replace('_' + control, '') for i in list(df['fov'])]
    log2_norm_df: pd.DataFrame = df.pivot(
        index="channel", columns="fov", values="Non-zero mean intensity"
    ).transform(func=lambda row: np.log2(row / row.mean()), axis=1)
    if control != 'tonsil_bottom_duplicate1':
        dup_col = [col for col in log2_norm_df.columns if 'duplicate1' in col]
        log2_norm_df = log2_norm_df.drop(columns=dup_col) if dup_col else log2_norm_df

    mean_t_df: pd.DataFrame = (
        log2_norm_df.mean(axis=0)
        .to_frame(name="mean")
        .transpose()
        .sort_values(by="mean", axis=1)
    )
    transformed_df: pd.DataFrame = pd.concat(
        objs=[log2_norm_df, mean_t_df]
    ).sort_values(by="mean", axis=1, inplace=False)
    transformed_df.rename_axis("channel", axis=0, inplace=True)
    transformed_df.rename_axis("fov", axis=1, inplace=True)

    dfs.append(transformed_df)
all_data = pd.concat(dfs).replace([np.inf, -np.inf], 0, inplace=True)
all_data = all_data.groupby(['channel']).mean()
all_data = all_data.sort_values(by="mean", axis=1, inplace=False).round(2)


fig = plt.figure(figsize=(12,12), dpi=300)
fig.set_layout_engine(layout="constrained")
gs = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, height_ratios=[len(all_data.index) - 1, 1])
_norm = Normalize(vmin=-1, vmax=1)
_cmap = sns.color_palette("vlag", as_cmap=True)
fig.suptitle(f"Average per TMA - QC: Non-zero Mean Intensity ")

annotation_kws = {
    "horizontalalignment": "center",
    "verticalalignment": "center",
    "fontsize": 8,
}

ax_heatmap = fig.add_subplot(gs[0, 0])
sns.heatmap(
    data=all_data[~all_data.index.isin(["mean"])],
    ax=ax_heatmap,
    linewidths=1,
    linecolor="black",
    cbar_kws={"shrink": 0.5},
    annot=True,
    annot_kws=annotation_kws,
    xticklabels=False,
    norm=_norm,
    cmap=_cmap,
)

ax_heatmap.collections[0].colorbar.ax.set_title(r"$\log_2(QC)$")
ax_heatmap.set_yticks(
    ticks=ax_heatmap.get_yticks(),
    labels=ax_heatmap.get_yticklabels(),
    rotation=0,
)
ax_heatmap.set_xlabel(None)

ax_avg = fig.add_subplot(gs[1, 0])
sns.heatmap(
    data=all_data[all_data.index.isin(["mean"])],
    ax=ax_avg,
    linewidths=1,
    linecolor="black",
    annot=True,
    annot_kws=annotation_kws,
    fmt=".2f",
    cmap=ListedColormap(["white"]),
    cbar=False,
)
ax_avg.set_yticks(
    ticks=ax_avg.get_yticks(),
    labels=["Mean"],
    rotation=0,
)
ax_avg.set_xticks(
    ticks=ax_avg.get_xticks(),
    labels=ax_avg.get_xticklabels(),
    rotation=45,
    ha="right",
    rotation_mode="anchor",
)
ax_heatmap.set_ylabel("Channel")
ax_avg.set_xlabel("FOV")

fig.savefig(fname=os.path.join(qc_control_metrics_dir, "figures/log2_avgs.png"), dpi=300,
            bbox_inches="tight")


# Image processing
## show a run with images stitched in acquisition order pre- and post-normalization
acquisition_order_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "acquisition_order")
if not os.path.exists(acquisition_order_viz_dir):
    os.makedirs(acquisition_order_viz_dir)

run_name = "2022-01-14_TONIC_TMA2_run1"
pre_norm_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/rosetta"
post_norm_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/normalized"
save_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"

# NOTE: image not scaled up, this happens in Photoshop
supplementary_plot_helpers.stitch_before_after_norm(
    pre_norm_dir, post_norm_dir, acquisition_order_viz_dir, run_name,
    "H3K9ac", pre_norm_subdir="normalized", padding=0, step=1
)


# Cell identification and classification
cell_table = pd.read_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files/cell_table_clusters.csv')
cluster_order = {'Cancer': 0, 'Cancer_EMT': 1, 'Cancer_Other': 2, 'CD4T': 3, 'CD8T': 4, 'Treg': 5,
                 'T_Other': 6, 'B': 7, 'NK': 8, 'M1_Mac': 9, 'M2_Mac': 10, 'Mac_Other': 11,
                 'Monocyte': 12, 'APC': 13, 'Mast': 14, 'Neutrophil': 15, 'Fibroblast': 16,
                 'Stroma': 17, 'Endothelium': 18, 'Other': 19, 'Immune_Other': 20}
cell_table = cell_table.sort_values(by=['cell_cluster'], key=lambda x: x.map(cluster_order))

save_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs'

## cell cluster counts
sns.histplot(data=cell_table, x="cell_cluster")
sns.despine()
plt.title("Cell Cluster Counts")
plt.xlabel("Cell Cluster")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "cells_per_cluster.png"), dpi=300)

## fov cell counts
cluster_counts = np.unique(cell_table.fov, return_counts=True)[1]
plt.figure(figsize=(8, 6))
g = sns.histplot(data=cluster_counts, kde=True)
sns.despine()
plt.title("Histogram of Cell Counts per Image")
plt.xlabel("Number of Cells in an Image")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "cells_per_fov.png"), dpi=300)

## cell type composition by tissue location of met and timepoint
meta_data = pd.read_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files/harmonized_metadata.csv')
meta_data = meta_data[['fov', 'Patient_ID', 'Timepoint', 'Localization']]

all_data = cell_table.merge(meta_data, on=['fov'], how='left')

for metric in ['Localization', 'Timepoint']:
    data = all_data[all_data.Timepoint == 'baseline'] if metric == 'Localization' else all_data

    groups = np.unique(data.Localization) if metric == 'Localization' else \
        ['primary', 'baseline', 'post_induction', 'on_nivo']
    dfs = []
    for group in groups:
        sub_data = data[data[metric] == group]

        df = sub_data.groupby("cell_cluster_broad").count().reset_index()
        df = df.set_index('cell_cluster_broad').transpose()
        sub_df = df.iloc[:1].reset_index(drop=True)
        sub_df.insert(0, metric, [group])
        sub_df[metric] = sub_df[metric].map(str)
        sub_df = sub_df.set_index(metric)

        dfs.append(sub_df)
    prop_data = pd.concat(dfs).transform(func=lambda row: row / row.sum(), axis=1)

    color_map = {'Cancer': 'dimgrey', 'Stroma': 'darksalmon', 'T': 'navajowhite',
                 'Mono_Mac': 'red', 'B': 'darkviolet', 'Other': 'yellowgreen',
                 'Granulocyte': 'aqua', 'NK': 'dodgerblue'}

    means = prop_data.mean(axis=0).reset_index()
    means = means.sort_values(by=[0], ascending=False)
    prop_data = prop_data[means.cell_cluster_broad]

    colors = [color_map[cluster] for cluster in means.cell_cluster_broad]
    prop_data.plot(kind='bar', stacked=True, color=colors)
    sns.despine()
    plt.ticklabel_format(style='plain', useOffset=False, axis='y')
    plt.gca().set_ylabel("Cell Proportions")
    xlabel = "Tissue Location" if metric == 'Localization' else "Timepoint"
    plt.gca().set_xlabel(xlabel)
    plt.xticks(rotation=30)
    plt.title(f"Cell Type Composition by {xlabel}")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1],
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.tight_layout()
    plot_name = "cell_props_by_tissue_loc.png" if metric == 'Localization' else "cell_props_by_timepoint.png"
    plt.savefig(os.path.join(save_dir, plot_name), dpi=300)

## colored cell cluster masks from random subset of 20 FOVs
random.seed(13)
seg_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output'

all_fovs = list(cell_table['fov'].unique())
fovs = random.sample(all_fovs, 20)
cell_table_subset = cell_table[cell_table.fov.isin(fovs)]

cohort_cluster_plot(
    fovs=fovs,
    seg_dir=seg_dir,
    save_dir=save_dir,
    cell_data=cell_table_subset,
    erode=True,
    fov_col='fov',
    label_col='label',
    cluster_col='cell_cluster_broad',
    seg_suffix="_whole_cell.tiff",
    cmap=color_map,
    display_fig=False,
)

# Functional marker thresholding
cell_table = pd.read_csv(
    os.path.join(ANALYSIS_DIR, "combined_cell_table_normalized_cell_labels_updated.csv")
)
functional_marker_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "functional_marker_dist_thresholds_test")
if not os.path.exists(functional_marker_viz_dir):
    os.makedirs(functional_marker_viz_dir)

marker_info = {
    "Ki67": {
        "populations": ["Cancer", "Mast"],
        "threshold": 0.002,
        "x_range": (0, 0.012),
        "x_ticks": np.array([0, 0.004, 0.008, 0.012]),
        "x_tick_labels": np.array([0, 0.004, 0.008, 0.012]),
    },
    "CD38": {
        "populations": ["Endothelium", "Cancer_EMT"],
        "threshold": 0.004,
        "x_range": (0, 0.02),
        "x_ticks": np.array([0, 0.005, 0.01, 0.015, 0.02]),
        "x_tick_labels": np.array([0, 0.005, 0.01, 0.015, 0.02]),
    },
    "CD45RB": {
        "populations": ["CD4T", "Stroma"],
        "threshold": 0.001,
        "x_range": (0, 0.015),
        "x_ticks": np.array([0, 0.005, 0.010, 0.015]),
        "x_tick_labels": np.array([0, 0.005, 0.010, 0.015])
    },
    "CD45RO": {
        "populations": ["CD4T", "Fibroblast"],
        "threshold": 0.002,
        "x_range": (0, 0.02),
        "x_ticks": np.array([0, 0.005, 0.01, 0.015, 0.02]),
        "x_tick_labels": np.array([0, 0.005, 0.01, 0.015, 0.02])
    },
    "CD57": {
        "populations": ["CD8T", "B"],
        "threshold": 0.002,
        "x_range": (0, 0.006),
        "x_ticks": np.array([0, 0.002, 0.004, 0.006]),
        "x_tick_labels": np.array([0, 0.002, 0.004, 0.006])
    },
    "CD69": {
        "populations": ["Treg", "Cancer"],
        "threshold": 0.002,
        "x_range": (0, 0.008),
        "x_ticks": np.array([0, 0.002, 0.004, 0.006, 0.008]),
        "x_tick_labels": np.array([0, 0.002, 0.004, 0.006, 0.008])
    },
    "GLUT1": {
        "populations": ["Cancer_EMT", "M2_Mac"],
        "threshold": 0.002,
        "x_range": (0, 0.02),
        "x_ticks": np.array([0, 0.005, 0.01, 0.015, 0.02]),
        "x_tick_labels": np.array([0, 0.005, 0.01, 0.015, 0.02])
    },
    "IDO": {
        "populations": ["APC", "M1_Mac"],
        "threshold": 0.001,
        "x_range": (0, 0.003),
        "x_ticks": np.array([0, 0.001, 0.002, 0.003]),
        "x_tick_labels": np.array([0, 0.001, 0.002, 0.003])
    },
    "PD1": {
        "populations": ["CD8T", "Stroma"],
        "threshold": 0.0005,
        "x_range": (0, 0.002),
        "x_ticks": np.array([0, 0.0005, 0.001, 0.0015, 0.002]),
        "x_tick_labels": np.array([0, 0.0005, 0.001, 0.0015, 0.002])
    },
    "PDL1": {
        "populations": ["Cancer", "Stroma"],
        "threshold": 0.001,
        "x_range": (0, 0.003),
        "x_ticks": np.array([0, 0.001, 0.002, 0.003]),
        "x_tick_labels": np.array([0, 0.001, 0.002, 0.003]),
    },
    "HLA1": {
        "populations": ["APC", "Stroma"],
        "threshold": 0.001,
        "x_range": (0, 0.025),
        "x_ticks": np.array([0, 0.0125, 0.025]),
        "x_tick_labels": np.array([0, 0.0125, 0.025])
    },
    "HLADR": {
        "populations": ["APC", "Neutrophil"],
        "threshold": 0.001,
        "x_range": (0, 0.025),
        "x_ticks": np.array([0, 0.0125, 0.025]),
        "x_tick_labels": np.array([0, 0.0125, 0.025])
    },
    "TBET": {
        "populations": ["NK", "B"],
        "threshold": 0.0015,
        "x_range": (0, 0.0045),
        "x_ticks": np.array([0, 0.0015, 0.003, 0.0045]),
        "x_tick_labels": np.array([0, 0.0015, 0.003, 0.0045])
    },
    "TCF1": {
        "populations": ["CD4T", "M1_Mac"],
        "threshold": 0.001,
        "x_range": (0, 0.003),
        "x_ticks": np.array([0, 0.001, 0.002, 0.003]),
        "x_tick_labels": np.array([0, 0.001, 0.002, 0.003])
    },
    "TIM3": {
        "populations": ["Monocyte", "Endothelium"],
        "threshold": 0.001,
        "x_range": (0, 0.004),
        "x_ticks": np.array([0, 0.001, 0.002, 0.003, 0.004]),
        "x_tick_labels": np.array([0, 0.001, 0.002, 0.003, 0.004])
    },
    "Vim": {
        "populations": ["Endothelium", "B"],
        "threshold": 0.002,
        "x_range": (0, 0.06),
        "x_ticks": np.array([0, 0.02, 0.04, 0.06]),
        "x_tick_labels": np.array([0, 0.02, 0.04, 0.06])
    },
    "Fe": {
        "populations": ["Fibroblast", "Cancer"],
        "threshold": 0.1,
        "x_range": (0, 0.3),
        "x_ticks": np.array([0, 0.1, 0.2, 0.3]),
        "x_tick_labels": np.array([0, 0.1, 0.2, 0.3]),
    }
}
supplementary_plot_helpers.functional_marker_thresholding(
    cell_table, functional_marker_viz_dir, marker_info=marker_info,
    figsize=(20, 40)
)


# Feature extraction
extraction_pipeline_tuning_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "extraction_pipeline_tuning")
if not os.path.exists(extraction_pipeline_tuning_dir):
    os.makedirs(extraction_pipeline_tuning_dir)

## 1: vary the features for each marker threshold
cell_table_full = pd.read_csv(
    os.path.join(ANALYSIS_DIR, "combined_cell_table_normalized_cell_labels_updated.csv")
)

threshold_multipliers = [0.25, 0.5, 0.75, 1, 2, 4]
threshold_test_vals = {
    m: [marker_info[m]["threshold"] * i for i in [0.25, 0.5, 0.75, 1, 2, 4]] for m in marker_info
}

func_thresholds_test = pd.DataFrame(
    index=threshold_test_vals.keys(),
    columns=[f"Count at preset thresh * {tm}" for tm in threshold_multipliers]
)
for tm in threshold_multipliers:
    func_thresholds_test[f"Raw preset thresh * {tm} value"] = np.nan

for marker, threshold_list in threshold_test_vals.items():
    for i, threshold in enumerate(threshold_list):
        marker_threshold_true_hits = np.sum(cell_table_full[marker].values >= threshold)
        func_thresholds_test.loc[
            marker, f"Count at preset thresh * {threshold_multipliers[i]}"
        ] = marker_threshold_true_hits
        func_thresholds_test.loc[
            marker, f"Raw preset thresh * {threshold_multipliers[i]} value"
        ] = threshold

func_thresholds_test["functional_marker"] = func_thresholds_test.index.values
func_thresholds_test.to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "functional_marker_threshold_tuning.csv"),
    index=False
)

## 2: vary the min cell param for the functional, morphology, diversity, and distance DataFrames
min_cell_multipliers = list(np.arange(0, 35, 5))

all_trials = ["functional", "morphology", "diversity", "distance"]
features = ["num_fovs", "num_fovs_deduped", "num_tissue_ids", "num_tissue_ids_deduped"]
results_df = pd.DataFrame(
    index=all_trials,
    columns=[f"{f}_min_cell_{mcm}" for f in features for mcm in min_cell_multipliers]
)

### 2.1: functional tests
print("Starting functional tests")
total_df_func = pd.read_csv(os.path.join(OUTPUT_DIR, "functional_df_per_core.csv"))
total_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cluster_df_per_core.csv'))
harmonized_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'harmonized_metadata.csv'))
for min_cells in min_cell_multipliers:
    filtered_dfs = []
    metrics = [['cluster_broad_count', 'cluster_broad_freq'],
               ['cluster_count', 'cluster_freq'],
               ['meta_cluster_count', 'meta_cluster_freq']]
    for metric in metrics:
        # subset count df to include cells at the relevant clustering resolution
        for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border',
                            'tls', 'tagg', 'all']:
            count_df = total_df[total_df.metric == metric[0]]
            count_df = count_df[count_df.subset == compartment]

            # subset functional df to only include functional markers at this resolution
            func_df = total_df_func[total_df_func.metric.isin(metric)]
            func_df = func_df[func_df.subset == compartment]

            # for each cell type, determine which FOVs have high enough counts to be included
            for cell_type in count_df.cell_type.unique():
                keep_df = count_df[count_df.cell_type == cell_type]
                keep_df = keep_df[keep_df.value >= min_cells]
                keep_fovs = keep_df.fov.unique()

                # subset functional df to only include FOVs with high enough counts
                keep_markers = func_df[func_df.cell_type == cell_type]
                keep_markers = keep_markers[keep_markers.fov.isin(keep_fovs)]

                # append to list of filtered dfs
                filtered_dfs.append(keep_markers)

    filtered_func_df = pd.concat(filtered_dfs)

    # # take subset for plotting average functional marker expression
    # filtered_func_df_plot = filtered_func_df.loc[filtered_func_df.subset == 'all', :]
    # filtered_func_df_plot = filtered_func_df_plot.loc[filtered_func_df_plot.metric.isin(['cluster_broad_freq', 'cluster_freq', 'meta_cluster_freq']), :]
    # filtered_func_df_plot = filtered_func_df_plot.loc[filtered_func_df_plot.functional_marker.isin(sp_markers), :]

    # # save filtered df, CHECKPOINT
    # filtered_func_df_plot.to_csv(os.path.join(output_dir, 'functional_df_per_core_filtered.csv'), index=False)
    # results_df.loc["functional", f"num_fovs_min_cell_{min_cells}"] = len(filtered_func_df_plot["fov"].unique())

    # load matrices
    broad_df_include = pd.read_csv(
        os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_broad.csv'),
        index_col=0
    )
    med_df_include = pd.read_csv(
        os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_med.csv'),
        index_col=0
    )
    meta_df_include = pd.read_csv(
        os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_meta.csv'),
        index_col=0
    )

    # identify metrics and dfs that will be filtered
    filtering = [['cluster_broad_count', 'cluster_broad_freq', broad_df_include],
               ['cluster_count', 'cluster_freq', med_df_include],
               ['meta_cluster_count', 'meta_cluster_freq', meta_df_include]]

    combo_dfs = []

    for filters in filtering:
        # get variables
        metric_names = filters[:2]
        metric_df = filters[2]

        # subset functional df to only include functional markers at this resolution
        func_df = filtered_func_df[filtered_func_df.metric.isin(metric_names)]

        # loop over each cell type, and get the corresponding markers
        for cell_type in metric_df.index:
            markers = metric_df.columns[metric_df.loc[cell_type] == True]

            # subset functional df to only include this cell type
            func_df_cell = func_df[func_df.cell_type == cell_type]

            # subset functional df to only include markers for this cell type
            func_df_cell = func_df_cell[func_df_cell.functional_marker.isin(markers)]

            # append to list of dfs
            combo_dfs.append(func_df_cell)

    # load inclusion matrices
    broad_df_include_dp = pd.read_csv(
        os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_broad_dp.csv'),
        index_col=0
    )
    med_df_include_dp = pd.read_csv(
        os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_med_dp.csv'),
        index_col=0
    )
    meta_df_include_dp = pd.read_csv(
        os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_meta_dp.csv'),
        index_col=0
    )

    # identify metrics and dfs that will be filtered
    filtering = [['cluster_broad_count', 'cluster_broad_freq', broad_df_include_dp],
               ['cluster_count', 'cluster_freq', med_df_include_dp],
               ['meta_cluster_count', 'meta_cluster_freq', meta_df_include_dp]]

    for filters in filtering:
        # get variables
        metric_names = filters[:2]
        metric_df = filters[2]

        # subset functional df to only include functional markers at this resolution
        func_df = filtered_func_df[filtered_func_df.metric.isin(metric_names)]

        # loop over each cell type, and get the corresponding markers
        for cell_type in metric_df.index:
            markers = metric_df.columns[metric_df.loc[cell_type] == True]

            # subset functional df to only include this cell type
            func_df_cell = func_df[func_df.cell_type == cell_type]

            # subset functional df to only include markers for this cell type
            func_df_cell = func_df_cell[func_df_cell.functional_marker.isin(markers)]

            # append to list of dfs
            combo_dfs.append(func_df_cell)

    # append to list of dfs
    long_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'total_func_per_core.csv'))
    combo_dfs.append(long_df)

    # combine
    combo_df = pd.concat(combo_dfs)
    # combo_df.to_csv(os.path.join(output_dir, 'functional_df_per_core_filtered.csv'), index=False)
    results_df.loc["functional", f"num_fovs_min_cell_{min_cells}"] = len(combo_df["fov"].unique())

    # create version of filtered df aggregated by timepoint
    combo_df_grouped_func = combo_df.groupby(['Tissue_ID', 'cell_type', 'functional_marker', 'metric', 'subset'])
    combo_df_timepoint_func = combo_df_grouped_func['value'].agg([np.mean, np.std])
    combo_df_timepoint_func.reset_index(inplace=True)
    combo_df_timepoint_func = combo_df_timepoint_func.merge(harmonized_metadata.drop(['fov', 'MIBI_data_generated'], axis=1).drop_duplicates(), on='Tissue_ID')

    # # save timepoint df, CHECKPOINT
    # combo_df_timepoint_func.to_csv(os.path.join(OUTPUT_DIR, 'functional_df_per_timepoint_filtered.csv'), index=False)
    results_df.loc["functional", f"num_tissue_ids_min_cell_{min_cells}"] = len(combo_df_timepoint_func["Tissue_ID"].unique())

    # use previously generated exclude list
    exclude_df = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'exclude_double_positive_markers.csv'))

    dedup_dfs = []

    cluster_resolution = [['cluster_broad_freq', 'cluster_broad_count'],
               ['cluster_freq', 'cluster_count'],
               ['meta_cluster_freq', 'meta_cluster_count']]

    for cluster in cluster_resolution:
        # subset functional df to only include functional markers at this resolution
        func_df = combo_df[combo_df.metric.isin(cluster)]

        # add unique identifier for cell + marker combo
        func_df['feature_name'] = func_df['cell_type'] + '__' + func_df['functional_marker']

        exclude_names = exclude_df.loc[exclude_df.metric == cluster[0], 'feature_name'].values

        # remove double positive markers that are highly correlated with single positive scores
        func_df = func_df[~func_df.feature_name.isin(exclude_names)]
        dedup_dfs.append(func_df)


    dedup_dfs.append(long_df)
    deduped_df = pd.concat(dedup_dfs)
    deduped_df = deduped_df.drop('feature_name', axis=1)

    # # save deduped df, CHECKPOINT
    # deduped_df.to_csv(os.path.join(OUTPUT_DIR, 'functional_df_per_core_filtered_deduped.csv'), index=False)
    results_df.loc["functional", f"num_fovs_deduped_min_cell_{min_cells}"] = len(deduped_df["fov"].unique())

    # create version aggregated by timepoint
    deduped_df_grouped = deduped_df.groupby(['Tissue_ID', 'cell_type', 'functional_marker', 'metric', 'subset'])
    deduped_df_timepoint = deduped_df_grouped['value'].agg([np.mean, np.std])
    deduped_df_timepoint.reset_index(inplace=True)
    deduped_df_timepoint = deduped_df_timepoint.merge(harmonized_metadata.drop(['fov', 'MIBI_data_generated'], axis=1).drop_duplicates(), on='Tissue_ID')

    # # save timepoint df, CHECKPOINT
    # deduped_df_timepoint.to_csv(os.path.join(OUTPUT_DIR, 'functional_df_per_timepoint_filtered_deduped.csv'), index=False)
    results_df.loc["functional", f"num_tissue_ids_deduped_min_cell_{min_cells}"] = len(deduped_df_timepoint["Tissue_ID"].unique())

results_df.to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "dfs_creation_tuning.csv"), index=False
)

### 2.2: morphology tests
print("Starting morphology tests")
total_df_morph = pd.read_csv(os.path.join(OUTPUT_DIR, "morph_df_per_core.csv"))
for min_cells in min_cell_multipliers:
    filtered_dfs = []
    metrics = [['cluster_broad_count', 'cluster_broad_freq'],
               ['cluster_count', 'cluster_freq'],
               ['meta_cluster_count', 'meta_cluster_freq']]
    for metric in metrics:
        # subset count df to include cells at the relevant clustering resolution
        for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border',
                            'tls', 'tagg', 'all']:
            count_df = total_df[total_df.metric == metric[0]]
            count_df = count_df[count_df.subset == compartment]

            # subset morphology df to only include morphology metrics at this resolution
            morph_df = total_df_morph[total_df_morph.metric == metric[1]]
            morph_df = morph_df[morph_df.subset == compartment]

            # for each cell type, determine which FOVs have high enough counts to be included
            for cell_type in count_df.cell_type.unique():
                keep_df = count_df[count_df.cell_type == cell_type]
                keep_df = keep_df[keep_df.value >= min_cells]
                keep_fovs = keep_df.fov.unique()

                # subset morphology df to only include FOVs with high enough counts
                keep_features = morph_df[morph_df.cell_type == cell_type]
                keep_features = keep_features[keep_features.fov.isin(keep_fovs)]

                # append to list of filtered dfs
                filtered_dfs.append(keep_features)

    filtered_dfs.append(long_df)
    filtered_morph_df = pd.concat(filtered_dfs)

    # # save filtered df, CHECKPOINT
    # filtered_morph_df.to_csv(os.path.join(output_dir, 'morph_df_per_core_filtered.csv'), index=False)
    results_df.loc["morphology", f"num_fovs_min_cell_{min_cells}"] = len(filtered_morph_df["fov"].unique())

    # create version aggregated by timepoint
    filtered_morph_df_grouped = filtered_morph_df.groupby(['Tissue_ID', 'cell_type', 'morphology_feature', 'metric', 'subset'])
    filtered_morph_df_timepoint = filtered_morph_df_grouped['value'].agg([np.mean, np.std])
    filtered_morph_df_timepoint.reset_index(inplace=True)
    filtered_morph_df_timepoint = filtered_morph_df_timepoint.merge(harmonized_metadata.drop(['fov', 'MIBI_data_generated'], axis=1).drop_duplicates(), on='Tissue_ID')

    # # save timepoint df, CHECKPOINT
    # filtered_morph_df_timepoint.to_csv(os.path.join(output_dir, 'morph_df_per_timepoint_filtered.csv'), index=False)
    results_df.loc["morphology", f"num_tissue_ids_min_cell_{min_cells}"] = len(filtered_morph_df_timepoint["Tissue_ID"].unique())

    # remove redundant morphology features
    block1 = ['area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'convex_area', 'equivalent_diameter']

    block2 = ['area_nuclear', 'major_axis_length_nuclear', 'minor_axis_length_nuclear', 'perimeter_nuclear', 'convex_area_nuclear', 'equivalent_diameter_nuclear']

    block3 = ['eccentricity', 'major_axis_equiv_diam_ratio']

    block4 = ['eccentricity_nuclear', 'major_axis_equiv_diam_ratio_nuclear', 'perim_square_over_area_nuclear']

    deduped_morph_df = filtered_morph_df.loc[~filtered_morph_df.morphology_feature.isin(block1[1:] + block2[1:] + block3[1:] + block4[1:]), :]

    # only keep complex morphology features for cancer cells, remove everything except area and nc for others
    cancer_clusters = ['Cancer', 'Cancer_EMT', 'Cancer_Other', 'Cancer_CD56', 'Cancer_CK17',
                       'Cancer_Ecad', 'Cancer_Mono', 'Cancer_SMA', 'Cancer_Vim']
    basic_morph_features = ['area', 'area_nuclear', 'nc_ratio']

    deduped_morph_df = deduped_morph_df.loc[~(~(deduped_morph_df.cell_type.isin(cancer_clusters)) & ~(deduped_morph_df.morphology_feature.isin(basic_morph_features))), :]

    # # saved deduped, CHECKPOINT
    # deduped_morph_df.to_csv(os.path.join(output_dir, 'morph_df_per_core_filtered_deduped.csv'), index=False)
    results_df.loc["morphology", f"num_fovs_deduped_min_cell_{min_cells}"] = len(deduped_morph_df["fov"].unique())

    # same for timepoints
    deduped_morph_df_timepoint = filtered_morph_df_timepoint.loc[~filtered_morph_df_timepoint.morphology_feature.isin(block1[1:] + block2[1:] + block3[1:] + block4[1:]), :]
    deduped_morph_df_timepoint = deduped_morph_df_timepoint.loc[~(~(deduped_morph_df_timepoint.cell_type.isin(cancer_clusters)) & ~(deduped_morph_df_timepoint.morphology_feature.isin(basic_morph_features))), :]

    # save morph timepoint, CHECKPOINT
    # deduped_morph_df_timepoint.to_csv(os.path.join(output_dir, 'morph_df_per_timepoint_filtered_deduped.csv'), index=False)
    results_df.loc["morphology", f"num_tissue_ids_deduped_min_cell_{min_cells}"] = len(deduped_morph_df_timepoint["Tissue_ID"].unique())

results_df.to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "dfs_creation_tuning.csv"), index=False
)

### 2.3: diversity tests
print("Starting diversity tests")
total_df_diversity = pd.read_csv(os.path.join(OUTPUT_DIR, 'diversity_df_per_core.csv'))
for min_cells in min_cell_multipliers:
    filtered_dfs = []
    metrics = [['cluster_broad_count', 'cluster_broad_freq'],
               ['cluster_count', 'cluster_freq'],
               ['meta_cluster_count', 'meta_cluster_freq']]
    for metric in metrics:
        # subset count df to include cells at the relevant clustering resolution
        for compartment in ['cancer_core', 'cancer_border', 'stroma_core',
                            'stroma_border', 'tagg', 'tls', 'all']:
            count_df = total_df[total_df.metric == metric[0]]
            count_df = count_df[count_df.subset == compartment]

            # subset diversity df to only include diversity metrics at this resolution
            diversity_df = total_df_diversity[total_df_diversity.metric == metric[1]]
            diversity_df = diversity_df[diversity_df.subset == compartment]

            # for each cell type, determine which FOVs have high enough counts to be included
            for cell_type in count_df.cell_type.unique():
                keep_df = count_df[count_df.cell_type == cell_type]
                keep_df = keep_df[keep_df.value >= min_cells]
                keep_fovs = keep_df.fov.unique()

                # subset morphology df to only include FOVs with high enough counts
                keep_features = diversity_df[diversity_df.cell_type == cell_type]
                keep_features = keep_features[keep_features.fov.isin(keep_fovs)]

                # append to list of filtered dfs
                filtered_dfs.append(keep_features)

    filtered_diversity_df = pd.concat(filtered_dfs)

    # # save filtered df, CHECKPOINT
    # filtered_diversity_df.to_csv(os.path.join(output_dir, 'diversity_df_per_core_filtered.csv'), index=False)
    results_df.loc["diversity", f"num_fovs_min_cell_{min_cells}"] = len(filtered_diversity_df["fov"].unique())

    # create version aggregated by timepoint
    filtered_diversity_df_grouped = filtered_diversity_df.groupby(['Tissue_ID', 'cell_type', 'diversity_feature', 'metric', 'subset'])
    filtered_diversity_df_timepoint = filtered_diversity_df_grouped['value'].agg([np.mean, np.std])
    filtered_diversity_df_timepoint.reset_index(inplace=True)
    filtered_diversity_df_timepoint = filtered_diversity_df_timepoint.merge(harmonized_metadata.drop(['fov', 'MIBI_data_generated'], axis=1).drop_duplicates(), on='Tissue_ID')

    # # save timepoint df, CHECKPOINT
    # filtered_diversity_df_timepoint.to_csv(os.path.join(output_dir, 'diversity_df_per_timepoint_filtered.csv'), index=False)
    results_df.loc["diversity", f"num_tissue_ids_min_cell_{min_cells}"] = len(filtered_diversity_df_timepoint["Tissue_ID"].unique())

    # investigate correlation between diversity scores
    fov_data = filtered_diversity_df.copy()
    fov_data['feature_name_unique'] = fov_data['cell_type'] + '_' + fov_data['diversity_feature']
    fov_data = fov_data.loc[(fov_data.subset == 'all') & (fov_data.metric == 'cluster_freq')]
    fov_data = fov_data.loc[fov_data.diversity_feature != 'diversity_cell_meta_cluster']
    fov_data_wide = fov_data.pivot(index='fov', columns='feature_name_unique', values='value')

    corr_df = fov_data_wide.corr(method='spearman')

    # replace Nans
    corr_df = corr_df.fillna(0)
    clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))

    # save deduped df that excludes cell meta cluster, CHECKPOINT
    deduped_diversity_df = filtered_diversity_df.loc[filtered_diversity_df.diversity_feature != 'diversity_cell_meta_cluster']
    # deduped_diversity_df.to_csv(os.path.join(output_dir, 'diversity_df_per_core_filtered_deduped.csv'), index=False)
    results_df.loc["diversity", f"num_fovs_deduped_min_cell_{min_cells}"] = len(deduped_diversity_df["fov"].unique())

    # create version aggregated by timepoint
    deduped_diversity_df_grouped = deduped_diversity_df.groupby(['Tissue_ID', 'cell_type', 'diversity_feature', 'metric', 'subset'])
    deduped_diversity_df_timepoint = deduped_diversity_df_grouped['value'].agg([np.mean, np.std])
    deduped_diversity_df_timepoint.reset_index(inplace=True)
    deduped_diversity_df_timepoint = deduped_diversity_df_timepoint.merge(harmonized_metadata.drop(['fov', 'MIBI_data_generated'], axis=1).drop_duplicates(), on='Tissue_ID')

    # # save timepoint df, CHECKPOINT
    # deduped_diversity_df_timepoint.to_csv(os.path.join(output_dir, 'diversity_df_per_timepoint_filtered_deduped.csv'), index=False)
    results_df.loc["diversity", f"num_tissue_ids_deduped_min_cell_{min_cells}"] = len(deduped_diversity_df_timepoint["Tissue_ID"].unique())

results_df.to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "dfs_creation_tuning.csv"), index=False
)

### 2.4: distance tests
print("Starting distance tests")
total_df_distance = pd.read_csv(os.path.join(OUTPUT_DIR, "distance_df_per_core.csv"))
for min_cells in min_cell_multipliers:
    filtered_dfs = []
    metrics = [['cluster_broad_count', 'cluster_broad_freq']]

    for metric in metrics:
        # subset count df to include cells at the relevant clustering resolution
        for compartment in ['cancer_core', 'cancer_border', 'stroma_core',
                            'stroma_border', 'tagg', 'tls', 'all']:
            count_df = total_df[total_df.metric == metric[0]]
            count_df = count_df[count_df.subset == compartment]

            # subset distance df to only include distance metrics at this resolution
            distance_df = total_df_distance[total_df_distance.metric == metric[1]]
            distance_df = distance_df[distance_df.subset == compartment]

            # for each cell type, determine which FOVs have high enough counts to be included
            for cell_type in count_df.cell_type.unique():
                keep_df = count_df[count_df.cell_type == cell_type]
                keep_df = keep_df[keep_df.value >= min_cells]
                keep_fovs = keep_df.fov.unique()

                # subset morphology df to only include FOVs with high enough counts
                keep_features = distance_df[distance_df.cell_type == cell_type]
                keep_features = keep_features[keep_features.fov.isin(keep_fovs)]

                # append to list of filtered dfs
                filtered_dfs.append(keep_features)

    filtered_distance_df = pd.concat(filtered_dfs)

    # # save filtered df, CHECKPOINT
    # filtered_distance_df.to_csv(os.path.join(output_dir, 'distance_df_per_core_filtered.csv'), index=False)
    results_df.loc["distance", f"num_fovs_min_cell_{min_cells}"] = len(filtered_distance_df["fov"].unique())

    # filter distance df to only include features with low correlation with abundance
    keep_df = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'distance_df_keep_features.csv'))

    deduped_dfs = []
    for cell_type in keep_df.cell_type.unique():
        keep_features = keep_df.loc[keep_df.cell_type == cell_type, 'feature_name'].unique()
        if len(keep_features) > 0:
            keep_df_subset = filtered_distance_df.loc[filtered_distance_df.cell_type == cell_type]
            keep_df_subset = keep_df_subset.loc[keep_df_subset.linear_distance.isin(keep_features)]
            deduped_dfs.append(keep_df_subset)

    deduped_distance_df = pd.concat(deduped_dfs)

    # # save filtered df, CHECKPOINT
    # deduped_distance_df.to_csv(os.path.join(output_dir, 'distance_df_per_core_deduped.csv'), index=False)
    results_df.loc["distance", f"num_fovs_deduped_min_cell_{min_cells}"] = len(deduped_distance_df["Tissue_ID"].unique())

    # create version aggregated by timepoint
    deduped_distance_df_grouped = deduped_distance_df.groupby(['Tissue_ID', 'cell_type', 'linear_distance', 'metric', 'subset'])
    deduped_distance_df_timepoint = deduped_distance_df_grouped['value'].agg([np.mean, np.std])
    deduped_distance_df_timepoint.reset_index(inplace=True)
    deduped_distance_df_timepoint = deduped_distance_df_timepoint.merge(harmonized_metadata.drop(['fov', 'MIBI_data_generated'], axis=1).drop_duplicates(), on='Tissue_ID')

    # # save timepoint df, CHECKPOINT
    # deduped_distance_df_timepoint.to_csv(os.path.join(output_dir, 'distance_df_per_timepoint_deduped.csv'), index=False)
    results_df.loc["distance", f"num_tissue_ids_deduped_min_cell_{min_cells}"] = len(filtered_distance_df["fov"].unique())

results_df["df_type"] = results_df.index.values
results_df = results_df.reset_index(drop=True)
results_df.to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "dfs_creation_tuning.csv"), index=False
)

## 3: vary the minimum_density function for cell abundance type analysis
cluster_df_core = pd.read_csv(os.path.join(OUTPUT_DIR, 'cluster_df_per_core.csv'))
results_dict_broad = {}

fov_data_list = []
fov_data_filtered_list = []
timepoint_data_list = []
timepoint_data_filtered_list = []

### 3.1: test for broad cell type abundance ratios
print("Running tests for broad cell type abundance")
input_df = cluster_df_core[cluster_df_core['metric'].isin(['cluster_broad_density'])]
for minimum_density in [0.0001, 0.00025, 0.0005, 0.00075, 0.0010, 0.00125, 0.0015]:
    fov_data = []

    for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
    #for compartment in ['all']:
        compartment_df = input_df[input_df.subset == compartment].copy()
        cell_types = compartment_df.cell_type.unique()

        # put cancer and stromal cells last
        cell_types = [cell_type for cell_type in cell_types if cell_type not in ['Cancer', 'Stroma']]
        cell_types = cell_types + ['Cancer', 'Stroma']

        for cell_type1, cell_type2 in itertools.combinations(cell_types, 2):
            cell_type1_df = compartment_df[compartment_df.cell_type == cell_type1].copy()
            cell_type2_df = compartment_df[compartment_df.cell_type == cell_type2].copy()

            # only keep FOVS with at least one cell type over the minimum density
            cell_type1_mask = cell_type1_df.value > minimum_density
            cell_type2_mask = cell_type2_df.value > minimum_density
            cell_mask = cell_type1_mask.values | cell_type2_mask.values

            cell_type1_df = cell_type1_df[cell_mask]
            cell_type2_df = cell_type2_df[cell_mask]

            # add minimum density to avoid log2(0)
            cell_type1_df['ratio'] = np.log2((cell_type1_df.value.values + minimum_density) /
                                             (cell_type2_df.value.values + minimum_density))

            cell_type1_df['value'] = cell_type1_df.ratio.values

            cell_type1_df['feature_name'] = cell_type1 + '__' + cell_type2 + '__ratio'
            cell_type1_df['compartment'] = compartment

            if compartment == 'all':
                cell_type1_df['feature_name_unique'] = cell_type1 + '__' + cell_type2 + '__ratio'
            else:
                cell_type1_df['feature_name_unique'] = cell_type1 + '__' + cell_type2 + '__ratio__' + compartment

            cell_type1_df['cell_pop'] = 'multiple'
            cell_type1_df['cell_pop_level'] = 'broad'
            cell_type1_df['feature_type'] = 'density_ratio'
            cell_type1_df['feature_type_detail'] = cell_type1
            cell_type1_df['feature_type_detail_2'] = cell_type2
            cell_type1_df = cell_type1_df[['fov', 'value', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                                           'cell_pop_level', 'feature_type', 'feature_type_detail', 'feature_type_detail_2']]
        fov_data.append(cell_type1_df)

    fov_data_df = pd.concat(fov_data)
    fov_data_df = fov_data_df.rename(columns={'value': 'raw_value'})
    fov_data_wide = fov_data_df.pivot(index='fov', columns='feature_name_unique', values='raw_value')
    zscore_df = (fov_data_wide - fov_data_wide.mean()) / fov_data_wide.std()

    # add z-scores to fov_data_df
    zscore_df = zscore_df.reset_index()
    zscore_df_long = pd.melt(zscore_df, id_vars='fov', var_name='feature_name_unique', value_name='normalized_value')
    fov_data_df = pd.merge(fov_data_df, zscore_df_long, on=['fov', 'feature_name_unique'], how='left')

    # add metadata
    fov_data_df = pd.merge(fov_data_df, harmonized_metadata[['Tissue_ID', 'fov']], on='fov', how='left')

    # rearrange columns
    fov_data_df = fov_data_df[['Tissue_ID', 'fov', 'raw_value', 'normalized_value', 'feature_name', 'feature_name_unique',
                                'compartment', 'cell_pop', 'cell_pop_level', 'feature_type', 'feature_type_detail', 'feature_type_detail_2']]
    # fov_data_df.to_csv(os.path.join(analysis_dir, 'fov_features.csv'), index=False)

    fov_data_df_grouped = fov_data_df.groupby(["compartment", "feature_name_unique"]).aggregate(
        {"raw_value": ["mean", "median"],
         "normalized_value": ["mean", "median"],
         "fov": "size"}
    ).reset_index()
    fov_data_df_grouped.columns = ["_".join(col).rstrip("_") for col in fov_data_df_grouped.columns.values]
    fov_data_df_grouped = fov_data_df_grouped.rename(columns={
        "fov_size": "num_fovs"
    })
    fov_data_df_grouped["minimum_density"] = minimum_density
    fov_data_list.append(fov_data_df_grouped)

    # create timepoint-level stats file
    grouped = fov_data_df.groupby(['Tissue_ID', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                                   'cell_pop_level', 'feature_type']).agg({'raw_value': ['mean', 'std'],
                                                                           'normalized_value': ['mean', 'std']})
    grouped.columns = ['raw_mean', 'raw_std', 'normalized_mean', 'normalized_std']
    grouped = grouped.reset_index()
    # grouped.to_csv(os.path.join(analysis_dir, 'timepoint_features.csv'), index=False)

    timepoint_data_df_grouped = grouped.groupby(["compartment", "feature_name_unique"]).aggregate(
        {"raw_mean": ["mean", "median"],
         "raw_std": ["mean", "median"],
         "normalized_mean": ["mean", "median"],
         "normalized_std": ["mean", "median"],
         "Tissue_ID": "size"}
    ).reset_index()
    timepoint_data_df_grouped.columns = ["_".join(col).rstrip("_") for col in timepoint_data_df_grouped.columns.values]
    timepoint_data_df_grouped = timepoint_data_df_grouped.rename(columns={
        "Tissue_ID_size": "count"
    })
    timepoint_data_df_grouped["minimum_density"] = minimum_density
    timepoint_data_list.append(timepoint_data_df_grouped)

    # use pre-defined list of features to exclude
    exclude_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'exclude_features_compartment_correlation.csv'))
    fov_data_df_filtered = fov_data_df.loc[~fov_data_df.feature_name_unique.isin(exclude_df.feature_name_unique.values), :]
    # fov_data_df_filtered.to_csv(os.path.join(analysis_dir, 'fov_features_filtered.csv'), index=False)

    fov_data_df_filtered_grouped = fov_data_df_filtered.groupby(["compartment", "feature_name_unique"]).aggregate(
        {"raw_value": ["mean", "median"],
         "normalized_value": ["mean", "median"],
         "fov": "size"}
    ).reset_index()
    fov_data_df_filtered_grouped.columns = ["_".join(col).rstrip("_") for col in fov_data_df_filtered_grouped.columns.values]
    fov_data_df_filtered_grouped = fov_data_df_filtered_grouped.rename(columns={
        "fov_size": "count"
    })
    fov_data_df_filtered_grouped["minimum_density"] = minimum_density
    fov_data_filtered_list.append(fov_data_df_filtered_grouped)

    # group by timepoint
    grouped = fov_data_df_filtered.groupby(['Tissue_ID', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                                     'cell_pop_level', 'feature_type']).agg({'raw_value': ['mean', 'std'],
                                                                                'normalized_value': ['mean', 'std']})

    grouped.columns = ['raw_mean', 'raw_std', 'normalized_mean', 'normalized_std']
    grouped = grouped.reset_index()

    # grouped.to_csv(os.path.join(analysis_dir, 'timepoint_features_filtered.csv'), index=False)

    timepoint_data_df_filtered_grouped = grouped.groupby(["compartment", "feature_name_unique"]).aggregate(
        {"raw_mean": ["mean", "median"],
         "raw_std": ["mean", "median"],
         "normalized_mean": ["mean", "median"],
         "normalized_std": ["mean", "median"],
         "Tissue_ID": "size"}
    ).reset_index()
    timepoint_data_df_filtered_grouped.columns = ["_".join(col).rstrip("_") for col in timepoint_data_df_filtered_grouped.columns.values]
    timepoint_data_df_filtered_grouped = timepoint_data_df_filtered_grouped.rename(columns={
        "Tissue_ID_size": "count"
    })
    timepoint_data_df_filtered_grouped["minimum_density"] = minimum_density
    timepoint_data_filtered_list.append(timepoint_data_df_filtered_grouped)

pd.concat(fov_data_list).to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "cell_broad_abundance_fov_level_min_density_tests.csv"), index=False
)
pd.concat(timepoint_data_list).to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "cell_broad_abundance_timepoint_level_min_density_tests.csv"), index=False
)
pd.concat(fov_data_filtered_list).to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "cell_broad_abundance_filtered_fov_level_min_density_tests.csv"), index=False
)
pd.concat(timepoint_data_filtered_list).to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "cell_broad_abundance_filtered_timepoint_level_min_density_tests.csv"), index=False
)


### 3.2: test for specific cell type abundance ratios
print("Running tests for specific cell type abundance")
fov_data_list = []
fov_data_filtered_list = []
timepoint_data_list = []
timepoint_data_filtered_list = []

cell_ratios = [('CD8T', 'CD4T'), ('CD4T', 'Treg'), ('CD8T', 'Treg'), ('M1_Mac', 'M2_Mac')]
input_df = cluster_df_core[cluster_df_core.metric == 'cluster_density'].copy()
for minimum_density in [0.0001, 0.00025, 0.0005, 0.00075, 0.0010]:
    fov_data = []

    for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
    #for compartment in ['all']:
        compartment_df = input_df[input_df.subset == compartment].copy()
        for cell_type1, cell_type2 in cell_ratios:
            cell_type1_df = compartment_df[compartment_df.cell_type == cell_type1].copy()
            cell_type2_df = compartment_df[compartment_df.cell_type == cell_type2].copy()

            # only keep FOVS with at least one cell type over the minimum density
            cell_type1_mask = cell_type1_df.value > minimum_density
            cell_type2_mask = cell_type2_df.value > minimum_density
            cell_mask = cell_type1_mask.values | cell_type2_mask.values

            cell_type1_df = cell_type1_df[cell_mask]
            cell_type2_df = cell_type2_df[cell_mask]

            cell_type1_df['ratio'] = np.log2((cell_type1_df.value.values + minimum_density) /
                                             (cell_type2_df.value.values + minimum_density))

            cell_type1_df['value'] = cell_type1_df.ratio.values
            cell_type1_df['feature_name'] = cell_type1 + '__' + cell_type2 + '__ratio'

            if compartment == 'all':
                cell_type1_df['feature_name_unique'] = cell_type1 + '__' + cell_type2 + '__ratio'
            else:
                cell_type1_df['feature_name_unique'] = cell_type1 + '__' + cell_type2 + '__ratio__' + compartment

            cell_type1_df['compartment'] = compartment
            cell_type1_df['cell_pop'] = 'Immune'
            cell_type1_df['cell_pop_level'] = 'med'
            cell_type1_df['feature_type'] = 'density_ratio'
            cell_type1_df['feature_type_detail'] = cell_type1
            cell_type1_df['feature_type_detail_2'] = cell_type2
            cell_type1_df = cell_type1_df[['fov', 'value', 'feature_name', 'feature_name_unique','compartment', 'cell_pop',
                                           'cell_pop_level',  'feature_type', 'feature_type_detail', 'feature_type_detail_2']]
            fov_data.append(cell_type1_df)

    fov_data_df = pd.concat(fov_data)
    fov_data_df = fov_data_df.rename(columns={'value': 'raw_value'})
    fov_data_wide = fov_data_df.pivot(index='fov', columns='feature_name_unique', values='raw_value')
    zscore_df = (fov_data_wide - fov_data_wide.mean()) / fov_data_wide.std()

    # add z-scores to fov_data_df
    zscore_df = zscore_df.reset_index()
    zscore_df_long = pd.melt(zscore_df, id_vars='fov', var_name='feature_name_unique', value_name='normalized_value')
    fov_data_df = pd.merge(fov_data_df, zscore_df_long, on=['fov', 'feature_name_unique'], how='left')

    # add metadata
    fov_data_df = pd.merge(fov_data_df, harmonized_metadata[['Tissue_ID', 'fov']], on='fov', how='left')

    # rearrange columns
    fov_data_df = fov_data_df[['Tissue_ID', 'fov', 'raw_value', 'normalized_value', 'feature_name', 'feature_name_unique',
                                'compartment', 'cell_pop', 'cell_pop_level', 'feature_type', 'feature_type_detail', 'feature_type_detail_2']]
    # fov_data_df.to_csv(os.path.join(analysis_dir, 'fov_features.csv'), index=False)

    fov_data_df_grouped = fov_data_df.groupby(["compartment", "feature_name_unique"]).aggregate(
        {"raw_value": ["mean", "median"],
         "normalized_value": ["mean", "median"],
         "fov": "size"}
    ).reset_index()
    fov_data_df_grouped.columns = ["_".join(col).rstrip("_") for col in fov_data_df_grouped.columns.values]
    fov_data_df_grouped = fov_data_df_grouped.rename(columns={
        "fov_size": "count"
    })
    fov_data_df_grouped["minimum_density"] = minimum_density
    fov_data_list.append(fov_data_df_grouped)

    # create timepoint-level stats file
    grouped = fov_data_df.groupby(['Tissue_ID', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                                   'cell_pop_level', 'feature_type']).agg({'raw_value': ['mean', 'std'],
                                                                           'normalized_value': ['mean', 'std']})
    grouped.columns = ['raw_mean', 'raw_std', 'normalized_mean', 'normalized_std']
    grouped = grouped.reset_index()
    # grouped.to_csv(os.path.join(analysis_dir, 'timepoint_features.csv'), index=False)

    timepoint_data_df_grouped = grouped.groupby(["compartment", "feature_name_unique"]).aggregate(
        {"raw_mean": ["mean", "median"],
         "raw_std": ["mean", "median"],
         "normalized_mean": ["mean", "median"],
         "normalized_std": ["mean", "median"],
         "Tissue_ID": "size"}
    ).reset_index()
    timepoint_data_df_grouped.columns = ["_".join(col).rstrip("_") for col in timepoint_data_df_grouped.columns.values]
    timepoint_data_df_grouped = timepoint_data_df_grouped.rename(columns={
        "Tissue_ID_size": "count"
    })
    timepoint_data_df_grouped["minimum_density"] = minimum_density
    timepoint_data_list.append(timepoint_data_df_grouped)

    # use pre-defined list of features to exclude
    exclude_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'exclude_features_compartment_correlation.csv'))
    fov_data_df_filtered = fov_data_df.loc[~fov_data_df.feature_name_unique.isin(exclude_df.feature_name_unique.values), :]
    # fov_data_df_filtered.to_csv(os.path.join(analysis_dir, 'fov_features_filtered.csv'), index=False)

    fov_data_df_filtered_grouped = fov_data_df_filtered.groupby(["compartment", "feature_name_unique"]).aggregate(
        {"raw_value": ["mean", "median"],
         "normalized_value": ["mean", "median"],
         "fov": "size"}
    ).reset_index()
    fov_data_df_filtered_grouped.columns = ["_".join(col).rstrip("_") for col in fov_data_df_filtered_grouped.columns.values]
    fov_data_df_filtered_grouped = fov_data_df_filtered_grouped.rename(columns={
        "fov_size": "count"
    })
    fov_data_df_filtered_grouped["minimum_density"] = minimum_density
    fov_data_filtered_list.append(fov_data_df_filtered_grouped)

    # group by timepoint
    grouped = fov_data_df_filtered.groupby(['Tissue_ID', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                                     'cell_pop_level', 'feature_type']).agg({'raw_value': ['mean', 'std'],
                                                                                'normalized_value': ['mean', 'std']})

    grouped.columns = ['raw_mean', 'raw_std', 'normalized_mean', 'normalized_std']
    grouped = grouped.reset_index()

    # grouped.to_csv(os.path.join(analysis_dir, 'timepoint_features_filtered.csv'), index=False)

    timepoint_data_df_filtered_grouped = grouped.groupby(["compartment", "feature_name_unique"]).aggregate(
        {"raw_mean": ["mean", "median"],
         "raw_std": ["mean", "median"],
         "normalized_mean": ["mean", "median"],
         "normalized_std": ["mean", "median"],
         "Tissue_ID": "size"}
    ).reset_index()
    timepoint_data_df_filtered_grouped.columns = ["_".join(col).rstrip("_") for col in timepoint_data_df_filtered_grouped.columns.values]
    timepoint_data_df_filtered_grouped = timepoint_data_df_filtered_grouped.rename(columns={
        "Tissue_ID_size": "count"
    })
    timepoint_data_df_filtered_grouped["minimum_density"] = minimum_density
    timepoint_data_filtered_list.append(timepoint_data_df_filtered_grouped)

pd.concat(fov_data_list).to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "cell_specific_abundance_fov_level_min_density_tests.csv"), index=False
)
pd.concat(timepoint_data_list).to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "cell_specific_abundance_timepoint_level_min_density_tests.csv"), index=False
)
pd.concat(fov_data_filtered_list).to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "cell_specific_abundance_filtered_fov_level_min_density_tests.csv"), index=False
)
pd.concat(timepoint_data_filtered_list).to_csv(
    os.path.join(extraction_pipeline_tuning_dir, "cell_specific_abundance_filtered_timepoint_level_min_density_tests.csv"), index=False
)
