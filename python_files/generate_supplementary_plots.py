# File with code for generating supplementary plots
import os
import random
from pathlib import Path
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

import python_files.supplementary_plot_helpers as supplementary_plot_helpers

ANALYSIS_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files"
CHANNEL_DIR = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/'
INTERMEDIATE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files"
OUTPUT_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/output_files"
METADATA_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files/metadata"
SEG_DIR = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output'
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
samples_fov = "TONIC_TMA24_R8C1"
samples_channels = sorted(io_utils.remove_file_extensions(
    io_utils.list_files(os.path.join(samples_dir, samples_fov), substrs=".tiff")
))
exclude_chans = ["Au", "CD11c_nuc_exclude", "CK17_smoothed", "ECAD_smoothed", "FOXP3_nuc_include",
                 "LAG", "Noodle", "chan_39", "chan_45", "chan_48", "chan_115", "chan_141"]
for ec in exclude_chans:
    if ec in samples_channels:
        samples_channels.remove(ec)
supplementary_plot_helpers.validate_panel(
    samples_dir, samples_fov, panel_validation_viz_dir, channels=samples_channels, font_size=320
)


# ROI selection
metadata = pd.read_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files/harmonized_metadata.csv')
metadata = metadata.loc[metadata.MIBI_data_generated, :]
metadata = metadata.loc[metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]

fov_counts = metadata.groupby('Tissue_ID').size().values
fov_counts = pd.DataFrame(fov_counts, columns=['FOV Count'])
sns.histplot(data=fov_counts, x='FOV Count')
sns.despine()
plt.title("Number of FOVs per Timepoint")
plt.xlabel("Number of FOVs")
plt.tight_layout()

plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR,'hne_core_fov_plots', "fov_counts_per_timepoint.pdf"), dpi=300)
plt.close()

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
## show a run with images pre- and post-Rosetta
rosetta_tiling = os.path.join(SUPPLEMENTARY_FIG_DIR, "rosetta_tiling")
if not os.path.exists(rosetta_before_after_viz):
    os.makedirs(rosetta_before_after_viz)

run_name = "2022-01-14_TONIC_TMA2_run1"
pre_rosetta_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/extracted"
post_rosetta_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/rosetta"

# NOTE: images not scaled up programmatically, this happens manually in Photoshop
supplementary_plot_helpers.stitch_before_after_rosetta(
    pre_rosetta_dir, post_rosetta_dir, rosetta_tiling, run_name,
    [11, 20, 35], "CD4", post_rosetta_subdir="normalized", padding=0, step=1,
    save_separate=True
)
supplementary_plot_helpers.stitch_before_after_rosetta(
    pre_rosetta_dir, post_rosetta_dir, rosetta_tiling, run_name,
    [17, 18, 39], "CD56", post_rosetta_subdir="normalized", padding=0, step=1,
    save_separate=True
)
supplementary_plot_helpers.stitch_before_after_rosetta(
    pre_rosetta_dir, post_rosetta_dir, rosetta_tiling, run_name,
    [30, 45], "CD31", post_rosetta_subdir="normalized", padding=0, step=1,
    save_separate=True
)
supplementary_plot_helpers.stitch_before_after_rosetta(
    pre_rosetta_dir, post_rosetta_dir, rosetta_tiling, run_name,
    [11, 15, 17, 20, 30, 42, 43], "CD8", post_rosetta_subdir="normalized", padding=0, step=1,
    save_separate=True
)

## show a run with images stitched in acquisition order pre- and post-normalization
norm_tiling = os.path.join(SUPPLEMENTARY_FIG_DIR, "acquisition_order")
if not os.path.exists(acquisition_order_viz_dir_norm):
    os.makedirs(acquisition_order_viz_dir_norm)

run_name = "2022-01-14_TONIC_TMA2_run1"
pre_norm_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/rosetta"
post_norm_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/normalized"

# NOTE: images not scaled up programmatically, this happens manually in Photoshop
supplementary_plot_helpers.stitch_before_after_norm(
    pre_norm_dir, post_norm_dir, norm_tiling, run_name,
    [
        11, 12, 13, 14, 15, 17, 18, 20, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 39, 40, 41, 42, 43, 44, 45, 46, 47
    ], "H3K9ac", pre_norm_subdir="normalized", padding=0, step=1
)


# Cell identification and classification
cell_table = pd.read_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files/cell_table_clusters.csv')
cluster_order = {'Cancer': 0, 'Cancer_EMT': 1, 'Cancer_Other': 2, 'CD4T': 3, 'CD8T': 4, 'Treg': 5,
                 'T_Other': 6, 'B': 7, 'NK': 8, 'M1_Mac': 9, 'M2_Mac': 10, 'Mac_Other': 11,
                 'Monocyte': 12, 'APC': 13, 'Mast': 14, 'Neutrophil': 15, 'Fibroblast': 16,
                 'Stroma': 17, 'Endothelium': 18, 'Other': 19, 'Immune_Other': 20}
cell_table = cell_table.sort_values(by=['cell_cluster'], key=lambda x: x.map(cluster_order))

save_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs'

cluster_stats_dir = os.path.join(save_dir, "cluster_stats")
if not os.path.exists(cluster_stats_dir):
    os.makedirs(cluster_stats_dir)

## cell cluster counts
sns.histplot(data=cell_table, x="cell_cluster")
sns.despine()
plt.title("Cell Cluster Counts")
plt.xlabel("Cell Cluster")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig(os.path.join(cluster_stats_dir, "cells_per_cluster.pdf"), dpi=300)

## fov cell counts
cluster_counts = np.unique(cell_table.fov, return_counts=True)[1]
plt.figure(figsize=(8, 6))
g = sns.histplot(data=cluster_counts, kde=True)
sns.despine()
plt.title("Histogram of Cell Counts per Image")
plt.xlabel("Number of Cells in an Image")
plt.tight_layout()
plt.savefig(os.path.join(cluster_stats_dir, "cells_per_fov.pdf"), dpi=300)

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
    plot_name = "cell_props_by_tissue_loc.pdf" if metric == 'Localization' else "cell_props_by_timepoint.pdf"
    plt.savefig(os.path.join(cluster_stats_dir, plot_name), dpi=300)

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

## Segmentation Channels and Overlays

cohort_path = Path("/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples")
seg_dir = Path("/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/")

save_dir = Path(SUPPLEMENTARY_FIG_DIR) / "segmentation_chans_overlays"
save_dir.mkdir(exist_ok=True, parents=True)

membrane_channels = ["CD14", "CD38", "CD45", "ECAD", "CK17"]
overlay_channels = ["membrane_channel", "nuclear_channel"]

fovs_mem_markers = [
    "TONIC_TMA3_R5C2",
    "TONIC_TMA3_R11C2",
    "TONIC_TMA10_R3C3",
    "TONIC_TMA11_R1C6",
    "TONIC_TMA13_R1C5",
    "TONIC_TMA16_R12C1",
    "TONIC_TMA23_R10C2",
    "TONIC_TMA24_R10C2",
]
for fov in fovs_mem_markers:
    p = supplementary_plot_helpers.MembraneMarkersSegmentationPlot(
        fov = fov,
        image_data=cohort_path,
        segmentation_dir=seg_dir,
        membrane_channels=membrane_channels,
        overlay_channels=overlay_channels,
        q=(0.05, 0.95),
        clip=False,
        figsize=(8,4),
        layout="constrained",
        image_type="pdf"
    )
    p.make_plot(save_dir = save_dir)

fovs_seg = [
    "TONIC_TMA3_R2C5",
    "TONIC_TMA4_R10C4",
    "TONIC_TMA5_R3C4",
    "TONIC_TMA8_R1C2",
    "TONIC_TMA9_R4C4",
    "TONIC_TMA12_R4C1",
    "TONIC_TMA12_R7C6",
    "TONIC_TMA18_R4C5",
    "TONIC_TMA21_R2C1",
    "TONIC_TMA21_R9C1",
    "TONIC_TMA23_R1C3",
    "TONIC_TMA24_R2C6",
]

for fov in fovs_seg:
    p = supplementary_plot_helpers.SegmentationOverlayPlot(
        fov=fov,
        segmentation_dir=seg_dir,
        overlay_channels=overlay_channels,
        q=(0.05, 0.95),
        figsize=(8, 4),
        clip=False,
        layout="constrained",
        image_type="pdf",
    )
    p.make_plot(save_dir = save_dir)

# HnE Core, FOV and Segmentation Overlays
hne_fovs = [
    "TONIC_TMA2_R7C4",
    "TONIC_TMA4_R11C2",
    "TONIC_TMA4_R12C4",
    "TONIC_TMA24_R2C3",
]
hne_path = Path(SUPPLEMENTARY_FIG_DIR) / "hne_core_fov_plots" / " cores_fov_seg_maps"
seg_dir = Path("/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/")

save_dir = Path(SUPPLEMENTARY_FIG_DIR) / "hne_core_fov_plots" / "figures"
save_dir.mkdir(exist_ok=True, parents=True)
for fov in hne_fovs:
    supplementary_plot_helpers.CorePlot(
        fov=fov, hne_path=hne_path, seg_dir=seg_dir
    ).make_plot(save_dir=save_dir)

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


# Feature parameter tuning
extraction_pipeline_tuning_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "extraction_pipeline_tuning")
if not os.path.exists(extraction_pipeline_tuning_dir):
    os.makedirs(extraction_pipeline_tuning_dir)

## vary the features for each marker threshold
cell_table_full = pd.read_csv(
    os.path.join(ANALYSIS_DIR, "combined_cell_table_normalized_cell_labels_updated.csv")
)
supplementary_plot_helpers.run_functional_marker_positivity_tuning_tests(
    cell_table_full, extraction_pipeline_tuning_dir, marker_info,
    threshold_mults=[1/4, 1/2, 3/4, 7/8, 1, 8/7, 4/3, 2, 4]
)

## vary min cell param to see how many FOVs get kept or not
total_df = pd.read_csv(os.path.join(OUTPUT_DIR, "cluster_df_per_core.csv"))
cluster_broad_df = pd.read_csv(os.path.join(OUTPUT_DIR, "cluster_df_per_core.csv"))
supplementary_plot_helpers.run_min_cell_feature_gen_fovs_dropped_tests(
    cluster_broad_df, min_cell_params=[1, 3, 5, 10, 20], compartments=["all"],
    metrics=["cluster_broad_count"], save_dir=extraction_pipeline_tuning_dir
)

## vary params for cancer mask and boundary definition inclusion
cell_table_clusters = pd.read_csv(os.path.join(ANALYSIS_DIR, 'cell_table_clusters.csv'))
supplementary_plot_helpers.run_cancer_mask_inclusion_tests(
    cell_table_clusters, channel_dir=CHANNEL_DIR, seg_dir=SEG_DIR,
    threshold_mults=[1/4, 1/2, 3/4, 7/8, 1, 8/7, 4/3, 2, 4],
    save_dir=extraction_pipeline_tuning_dir, base_sigma=10, base_channel_thresh=0.0015,
    base_min_mask_size=7000, base_max_hole_size=1000, base_border_size=50
)


# False positive analysis
## Analyse the significance scores of top features after randomization compared to the TONIC data.
fp_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, 'false_positive_analysis')
if not os.path.exists(fp_dir):
    os.makedirs(fp_dir)

# compute random feature sets
'''
combined_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'timepoint_combined_features.csv'))
feature_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_ranking.csv'))
feature_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_metadata.csv'))

repeated_features, repeated_features_num, scores = [], [], []
overlapping_features, random_top_features = [], []

sample_num = 100
np.random.seed(13)

for i, seed in enumerate(random.sample(range(1, 2000), sample_num)):
    print(f'{i+1}/100')
    intersection_of_features, jaccard_score, top_random_features = random_feature_generation(combined_df, seed, feature_df[:100], feature_metadata)

    shared_df = pd.DataFrame({
        'random_seed': [seed] * len(intersection_of_features),
        'repeated_features' : list(intersection_of_features),
        'jaccard_score': [jaccard_score] * len(intersection_of_features)
    })
    overlapping_features.append(shared_df)

    top_random_features['seed'] = seed
    random_top_features.append(top_random_features)

results = pd.concat(overlapping_features)
top_features = pd.concat(random_top_features)
# add TONIC features to data with seed 0
top_features = pd.concat([top_features, feature_df[:100]])
top_features['seed'] = top_features['seed'].fillna(0)

results.to_csv(os.path.join(fp_dir, 'overlapping_features.csv'), index=False)
top_features.to_csv(os.path.join(fp_dir, 'top_features.csv'), index=False)
'''

top_features = pd.read_csv(os.path.join(fp_dir, 'top_features.csv'))
results = pd.read_csv(os.path.join(fp_dir, 'overlapping_features.csv'))

avg_scores = top_features[['seed', 'pval', 'log_pval', 'fdr_pval', 'med_diff']].groupby(by='seed').mean()
avg_scores['abs_med_diff'] = abs(avg_scores['med_diff'])
top_features['abs_med_diff'] = abs(top_features['med_diff'])

# log p-value & effect size plots
for name, metric in zip(['Log p-value', 'Effect Size'], ['log_pval', 'abs_med_diff']):
    # plot metric dist in top features for TONIC data and one random set
    TONIC = top_features[top_features.seed == 0]
    random = top_features[top_features.seed == 8]
    g = sns.distplot(TONIC[metric], kde=True, color='#1f77b4')
    g = sns.distplot(random[metric], kde=True, color='#ff7f0e')
    g.set(xlim=(0, None))
    plt.xlabel(name)
    plt.title(f"{name} Distribution in TONIC vs a Random")
    g.legend(labels=["TONIC", "Randomized"])
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.9, 1))
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(fp_dir, f"{metric}_dists.pdf"), dpi=300)
    plt.show()

    # plot average metric across top features for each set
    g = sns.distplot(avg_scores[metric][1:], kde=True,  color='#ff7f0e')
    g.axvline(x=avg_scores[metric][0], color='#1f77b4')
    g.set(xlim=(0, avg_scores[metric][0]*1.2))
    plt.xlabel(f'Average {name} of Top 100 Features')
    plt.title(f"Average {name} in TONIC vs Random Sets")
    g.legend(labels=["Randomized", "TONIC"])
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.9, 1))
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(fp_dir, f"{metric}_avg_per_set.pdf"), dpi=300)
    plt.show()

# general feature overlap plots
high_features = results.groupby(by='repeated_features').count().sort_values(by='random_seed', ascending=False).reset_index()
high_features = high_features[high_features.random_seed>3].sort_values(by='random_seed')
plt.barh(high_features.repeated_features, high_features.random_seed)
plt.xlabel('How Many Random Sets Contain the Feature')
plt.title('Repeated Top Features')
sns.despine()
plt.savefig(os.path.join(fp_dir, "Repeated_Top_Features.pdf"), dpi=300, bbox_inches='tight')
plt.show()

repeated_features_num = results.groupby(by='random_seed').count().sort_values(by='repeated_features', ascending=False)
plt.hist(repeated_features_num.repeated_features)
plt.xlabel('Number of TONIC Top Features in each Random Set')
plt.title('Histogram of Overlapping Features')
sns.despine()
plt.savefig(os.path.join(fp_dir, f"Histogram_of_Overlapping_Features.pdf"), dpi=300)
plt.show()

plt.hist(results.jaccard_score, bins=10)
plt.xlim((0, 0.10))
plt.title('Histogram of Jaccard Scores')
sns.despine()
plt.xlabel('Jaccard Score')
plt.savefig(os.path.join(fp_dir, "Histogram_of_Jaccard_Scores.pdf"), dpi=300)
plt.show()
