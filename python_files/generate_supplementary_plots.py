# File with code for generating supplementary plots
import os
import random

import numpy as np
import pandas as pd
import matplotlib
from ark.utils.plot_utils import cohort_cluster_plot
# from toffy import qc_comp, qc_metrics_plots
from alpineer import io_utils


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, Normalize

import supplementary_plot_helpers

ANALYSIS_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files"
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


