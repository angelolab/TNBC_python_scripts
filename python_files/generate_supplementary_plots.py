# File with code for generating supplementary plots
import itertools
import os
import pathlib
import random

import numpy as np
import pandas as pd
import matplotlib
import skimage.io as io
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
INTERMEDIATE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files"
OUTPUT_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/output_files"
SUPPLEMENTARY_FIG_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"


# # Panel validation
# panel_validation_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "panel_validation")
# if not os.path.exists(panel_validation_viz_dir):
#     os.makedirs(panel_validation_viz_dir)

# controls_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/controls"
# controls_fov = "TONIC_TMA1_colon_bottom"
# supplementary_plot_helpers.validate_panel(
#     controls_dir, controls_fov, panel_validation_viz_dir, font_size=180
# )

# samples_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples"
# samples_fov = "TONIC_TMA2_R5C4"
# samples_channels = sorted(io_utils.remove_file_extensions(
#     io_utils.list_files(os.path.join(samples_dir, samples_fov), substrs=".tiff")
# ))
# exclude_chans = ["CD11c_nuc_exclude", "CK17_smoothed", "ECAD_smoothed", "FOXP3_nuc_include",
#                  "chan_39", "chan_45", "chan_48", "chan_115", "chan_141"]
# for ec in exclude_chans:
#     if ec in samples_channels:
#         samples_channels.remove(ec)
# supplementary_plot_helpers.validate_panel(
#     samples_dir, samples_fov, panel_validation_viz_dir, channels=samples_channels, font_size=320
# )


# # ROI selection


# # QC
# qc_metrics = ["Non-zero mean intensity"]
# channel_exclude = ["chan_39", "chan_45", "CD11c_nuc_exclude", "CD11c_nuc_exclude_update",
#                    "FOXP3_nuc_include", "FOXP3_nuc_include_update", "CK17_smoothed",
#                    "FOXP3_nuc_exclude_update", "chan_48", "chan_141", "chan_115", "LAG3"]

# ## FOV spatial location
# cohort_path = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples"
# qc_tma_metrics_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/qc_metrics/qc_tma_metrics"
# if not os.path.exists(qc_tma_metrics_dir):
#     os.makedirs(qc_tma_metrics_dir)

# fovs = io_utils.list_folders(cohort_path)
# tmas = list(set([fov.split('_R')[0] for fov in fovs]))

# qc_tmas = qc_comp.QCTMA(
#     qc_metrics=qc_metrics,
#     cohort_path=cohort_path,
#     metrics_dir=qc_tma_metrics_dir,
# )

# qc_tmas.compute_qc_tma_metrics(tmas=tmas)
# qc_tmas.qc_tma_metrics_zscore(tmas=tmas, channel_exclude=channel_exclude)
# qc_metrics_plots.qc_tmas_metrics_plot(qc_tmas=qc_tmas, tmas=tmas, save_figure=True, dpi=300)

# ## longitudinal controls
# control_path = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/controls"
# qc_control_metrics_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/qc_metrics/qc_longitudinal_control"
# if not os.path.exists(qc_control_metrics_dir):
#     os.makedirs(qc_control_metrics_dir)

# folders = io_utils.list_folders(control_path, "TMA3_")
# control_substrs = [name.split("_")[2] + '_' + name.split("_")[3] if len(name.split("_")) == 4
#                    else name.split("_")[2] + '_' + name.split("_")[3]+'_' + name.split("_")[4]
#                    for name in folders]

# all_folders = io_utils.list_folders(control_path)
# for i, control in enumerate(control_substrs):
#     control_sample_name = control
#     print(control)
#     if control == 'tonsil_bottom':
#         fovs = [folder for folder in all_folders if control in folder and len(folder) <= 25]
#     else:
#         fovs = [folder for folder in all_folders if control in folder]

#     qc_control = qc_comp.QCControlMetrics(
#         qc_metrics=qc_metrics,
#         cohort_path=control_path,
#         metrics_dir=qc_control_metrics_dir,
#     )

#     qc_control.compute_control_qc_metrics(
#         control_sample_name=control_sample_name,
#         fovs=fovs,
#         channel_exclude=channel_exclude,
#         channel_include=None,
#     )

#     qc_metrics_plots.longitudinal_control_heatmap(
#         qc_control=qc_control, control_sample_name=control_sample_name, save_figure=True, dpi=300
#     )

# dfs = []
# for control in control_substrs:
#     df = pd.read_csv(os.path.join(qc_control_metrics_dir, f"{control}_combined_nonzero_mean_stats.csv"))
#     df['fov'] = [i.replace('_' + control, '') for i in list(df['fov'])]
#     log2_norm_df: pd.DataFrame = df.pivot(
#         index="channel", columns="fov", values="Non-zero mean intensity"
#     ).transform(func=lambda row: np.log2(row / row.mean()), axis=1)
#     if control != 'tonsil_bottom_duplicate1':
#         dup_col = [col for col in log2_norm_df.columns if 'duplicate1' in col]
#         log2_norm_df = log2_norm_df.drop(columns=dup_col) if dup_col else log2_norm_df

#     mean_t_df: pd.DataFrame = (
#         log2_norm_df.mean(axis=0)
#         .to_frame(name="mean")
#         .transpose()
#         .sort_values(by="mean", axis=1)
#     )
#     transformed_df: pd.DataFrame = pd.concat(
#         objs=[log2_norm_df, mean_t_df]
#     ).sort_values(by="mean", axis=1, inplace=False)
#     transformed_df.rename_axis("channel", axis=0, inplace=True)
#     transformed_df.rename_axis("fov", axis=1, inplace=True)

#     dfs.append(transformed_df)
# all_data = pd.concat(dfs).replace([np.inf, -np.inf], 0, inplace=True)
# all_data = all_data.groupby(['channel']).mean()
# all_data = all_data.sort_values(by="mean", axis=1, inplace=False).round(2)


# fig = plt.figure(figsize=(12,12), dpi=300)
# fig.set_layout_engine(layout="constrained")
# gs = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, height_ratios=[len(all_data.index) - 1, 1])
# _norm = Normalize(vmin=-1, vmax=1)
# _cmap = sns.color_palette("vlag", as_cmap=True)
# fig.suptitle(f"Average per TMA - QC: Non-zero Mean Intensity ")

# annotation_kws = {
#     "horizontalalignment": "center",
#     "verticalalignment": "center",
#     "fontsize": 8,
# }

# ax_heatmap = fig.add_subplot(gs[0, 0])
# sns.heatmap(
#     data=all_data[~all_data.index.isin(["mean"])],
#     ax=ax_heatmap,
#     linewidths=1,
#     linecolor="black",
#     cbar_kws={"shrink": 0.5},
#     annot=True,
#     annot_kws=annotation_kws,
#     xticklabels=False,
#     norm=_norm,
#     cmap=_cmap,
# )

# ax_heatmap.collections[0].colorbar.ax.set_title(r"$\log_2(QC)$")
# ax_heatmap.set_yticks(
#     ticks=ax_heatmap.get_yticks(),
#     labels=ax_heatmap.get_yticklabels(),
#     rotation=0,
# )
# ax_heatmap.set_xlabel(None)

# ax_avg = fig.add_subplot(gs[1, 0])
# sns.heatmap(
#     data=all_data[all_data.index.isin(["mean"])],
#     ax=ax_avg,
#     linewidths=1,
#     linecolor="black",
#     annot=True,
#     annot_kws=annotation_kws,
#     fmt=".2f",
#     cmap=ListedColormap(["white"]),
#     cbar=False,
# )
# ax_avg.set_yticks(
#     ticks=ax_avg.get_yticks(),
#     labels=["Mean"],
#     rotation=0,
# )
# ax_avg.set_xticks(
#     ticks=ax_avg.get_xticks(),
#     labels=ax_avg.get_xticklabels(),
#     rotation=45,
#     ha="right",
#     rotation_mode="anchor",
# )
# ax_heatmap.set_ylabel("Channel")
# ax_avg.set_xlabel("FOV")

# fig.savefig(fname=os.path.join(qc_control_metrics_dir, "figures/log2_avgs.png"), dpi=300,
#             bbox_inches="tight")


# # Image processing
# ## show a run with images stitched in acquisition order pre- and post-normalization
# acquisition_order_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "acquisition_order")
# if not os.path.exists(acquisition_order_viz_dir):
#     os.makedirs(acquisition_order_viz_dir)

# run_name = "2022-01-14_TONIC_TMA2_run1"
# pre_norm_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/rosetta"
# post_norm_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/normalized"
# save_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"

# # NOTE: image not scaled up, this happens in Photoshop
# supplementary_plot_helpers.stitch_before_after_norm(
#     pre_norm_dir, post_norm_dir, acquisition_order_viz_dir, run_name,
#     "H3K9ac", pre_norm_subdir="normalized", padding=0, step=1
# )


# # Cell identification and classification
# cell_table = pd.read_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files/cell_table_clusters.csv')
# cluster_order = {'Cancer': 0, 'Cancer_EMT': 1, 'Cancer_Other': 2, 'CD4T': 3, 'CD8T': 4, 'Treg': 5,
#                  'T_Other': 6, 'B': 7, 'NK': 8, 'M1_Mac': 9, 'M2_Mac': 10, 'Mac_Other': 11,
#                  'Monocyte': 12, 'APC': 13, 'Mast': 14, 'Neutrophil': 15, 'Fibroblast': 16,
#                  'Stroma': 17, 'Endothelium': 18, 'Other': 19, 'Immune_Other': 20}
# cell_table = cell_table.sort_values(by=['cell_cluster'], key=lambda x: x.map(cluster_order))

# save_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs'

# ## cell cluster counts
# sns.histplot(data=cell_table, x="cell_cluster")
# sns.despine()
# plt.title("Cell Cluster Counts")
# plt.xlabel("Cell Cluster")
# plt.xticks(rotation=75)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "cells_per_cluster.png"), dpi=300)

# ## fov cell counts
# cluster_counts = np.unique(cell_table.fov, return_counts=True)[1]
# plt.figure(figsize=(8, 6))
# g = sns.histplot(data=cluster_counts, kde=True)
# sns.despine()
# plt.title("Histogram of Cell Counts per Image")
# plt.xlabel("Number of Cells in an Image")
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "cells_per_fov.png"), dpi=300)

# ## cell type composition by tissue location of met and timepoint
# meta_data = pd.read_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files/harmonized_metadata.csv')
# meta_data = meta_data[['fov', 'Patient_ID', 'Timepoint', 'Localization']]

# all_data = cell_table.merge(meta_data, on=['fov'], how='left')

# for metric in ['Localization', 'Timepoint']:
#     data = all_data[all_data.Timepoint == 'baseline'] if metric == 'Localization' else all_data

#     groups = np.unique(data.Localization) if metric == 'Localization' else \
#         ['primary', 'baseline', 'post_induction', 'on_nivo']
#     dfs = []
#     for group in groups:
#         sub_data = data[data[metric] == group]

#         df = sub_data.groupby("cell_cluster_broad").count().reset_index()
#         df = df.set_index('cell_cluster_broad').transpose()
#         sub_df = df.iloc[:1].reset_index(drop=True)
#         sub_df.insert(0, metric, [group])
#         sub_df[metric] = sub_df[metric].map(str)
#         sub_df = sub_df.set_index(metric)

#         dfs.append(sub_df)
#     prop_data = pd.concat(dfs).transform(func=lambda row: row / row.sum(), axis=1)

#     color_map = {'Cancer': 'dimgrey', 'Stroma': 'darksalmon', 'T': 'navajowhite',
#                  'Mono_Mac': 'red', 'B': 'darkviolet', 'Other': 'yellowgreen',
#                  'Granulocyte': 'aqua', 'NK': 'dodgerblue'}

#     means = prop_data.mean(axis=0).reset_index()
#     means = means.sort_values(by=[0], ascending=False)
#     prop_data = prop_data[means.cell_cluster_broad]

#     colors = [color_map[cluster] for cluster in means.cell_cluster_broad]
#     prop_data.plot(kind='bar', stacked=True, color=colors)
#     sns.despine()
#     plt.ticklabel_format(style='plain', useOffset=False, axis='y')
#     plt.gca().set_ylabel("Cell Proportions")
#     xlabel = "Tissue Location" if metric == 'Localization' else "Timepoint"
#     plt.gca().set_xlabel(xlabel)
#     plt.xticks(rotation=30)
#     plt.title(f"Cell Type Composition by {xlabel}")
#     handles, labels = plt.gca().get_legend_handles_labels()
#     plt.legend(handles[::-1], labels[::-1],
#                bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
#     plt.tight_layout()
#     plot_name = "cell_props_by_tissue_loc.png" if metric == 'Localization' else "cell_props_by_timepoint.png"
#     plt.savefig(os.path.join(save_dir, plot_name), dpi=300)

# ## colored cell cluster masks from random subset of 20 FOVs
# random.seed(13)
# seg_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output'

# all_fovs = list(cell_table['fov'].unique())
# fovs = random.sample(all_fovs, 20)
# cell_table_subset = cell_table[cell_table.fov.isin(fovs)]

# cohort_cluster_plot(
#     fovs=fovs,
#     seg_dir=seg_dir,
#     save_dir=save_dir,
#     cell_data=cell_table_subset,
#     erode=True,
#     fov_col='fov',
#     label_col='label',
#     cluster_col='cell_cluster_broad',
#     seg_suffix="_whole_cell.tiff",
#     cmap=color_map,
#     display_fig=False,
# )

# # Functional marker thresholding
# cell_table = pd.read_csv(
#     os.path.join(ANALYSIS_DIR, "combined_cell_table_normalized_cell_labels_updated.csv")
# )
# functional_marker_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "functional_marker_dist_thresholds_test")
# if not os.path.exists(functional_marker_viz_dir):
#     os.makedirs(functional_marker_viz_dir)

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
# supplementary_plot_helpers.functional_marker_thresholding(
#     cell_table, functional_marker_viz_dir, marker_info=marker_info,
#     figsize=(20, 40)
# )


# Feature extraction
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from alpineer.io_utils import list_folders
extraction_pipeline_tuning_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "extraction_pipeline_tuning")
if not os.path.exists(extraction_pipeline_tuning_dir):
    os.makedirs(extraction_pipeline_tuning_dir)

# ## 1: vary the features for each marker threshold
# cell_table_full = pd.read_csv(
#     os.path.join(ANALYSIS_DIR, "combined_cell_table_normalized_cell_labels_updated.csv")
# )

# threshold_mults = [1/4, 1/2, 3/4, 7/8, 1, 8/7, 4/3, 2, 4]
# marker_threshold_data = {}

# for marker in marker_info:
#     marker_threshold_data[marker] = {}

#     for threshold in threshold_mults:
#         multiplied_threshold = marker_info[marker]["threshold"] * threshold
#         marker_threshold_data[marker][threshold] = {
#             "multiplied_threshold": multiplied_threshold,
#             "num_positive_cells": np.sum(cell_table_full[marker].values >= multiplied_threshold),
#             "num_positive_cells_norm": np.sum(
#                 cell_table_full[marker].values >= multiplied_threshold
#             ) / np.sum(
#                 cell_table_full[marker].values >= marker_info[marker]["threshold"]
#             )
#         }

# # fig = plt.figure()
# # ax = fig.add_subplot(1, 1, 1)
# # threshold_mult_strs = [str(np.round(np.log2(tm), 3)) for tm in threshold_mults]

# # for i, marker in enumerate(marker_threshold_data):
# #     mult_data = [mtd["num_positive_cells_norm"] for mtd in marker_threshold_data[marker].values()]
# #     _ = ax.plot(threshold_mult_strs, mult_data, color="gray", label=marker)

# # _ = ax.set_title(
# #     f"Positive cells per threshold, normalized by 1x",
# #     fontsize=11
# # )
# # _ = ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
# # _ = ax.yaxis.get_major_formatter().set_scientific(False)
# # _ = ax.set_xlabel("log2(threshold multiplier)", fontsize=7)
# # _ = ax.set_ylabel("Positive cell counts, normalized by 1x", fontsize=7)
# # _ = ax.tick_params(axis="both", which="major", labelsize=7)

# # # save the figure to save_dir
# # _ = fig.savefig(
# #     pathlib.Path(extraction_pipeline_tuning_dir) / f"functional_marker_threshold_experiments_norm.png",
# #     dpi=300
# # )

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# threshold_mult_strs = [str(np.round(np.log2(tm), 3)) for tm in threshold_mults]

# for i, marker in enumerate(marker_threshold_data):
#     mult_data = [mtd["num_positive_cells"] for mtd in marker_threshold_data[marker].values()]
#     _ = ax.plot(threshold_mult_strs, mult_data, color="gray", label=marker)

# _ = ax.set_title(
#     f"Positive cells per threshold",
#     fontsize=11
# )
# _ = ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
# _ = ax.yaxis.get_major_formatter().set_scientific(False)
# _ = ax.set_xlabel("log2(threshold multiplier)", fontsize=7)
# _ = ax.set_ylabel("Positive cell counts", fontsize=7)
# _ = ax.tick_params(axis="both", which="major", labelsize=7)

# # save the figure to save_dir
# _ = fig.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"functional_marker_threshold_experiments.png",
#     dpi=300
# )


# ## 2. vary min cell param for functional, morphology, diversity, and distance DataFrames
min_cell_tests = [1, 3, 5, 10, 20]
total_fovs = len(list_folders("/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples"))

# # ### 2.1: functional tests
# print("Starting functional tests")
# total_df_func = pd.read_csv(os.path.join(OUTPUT_DIR, "functional_df_per_core.csv"))
total_df = pd.read_csv(os.path.join(OUTPUT_DIR, "cluster_df_per_core.csv"))
harmonized_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, "harmonized_metadata.csv"))

total_fovs_dropped = {}
# metrics = [['cluster_broad_count', 'cluster_broad_freq'],
#            ['cluster_count', 'cluster_freq'],
#            ['meta_cluster_count', 'meta_cluster_freq']]
metrics = [['cluster_broad_count', 'cluster_broad_freq'],
           ['cluster_count', 'cluster_freq'],
           ['meta_cluster_count', 'meta_cluster_freq']]

for metric in metrics:
    total_fovs_dropped[metric[0]] = {}
    total_fovs_dropped[metric[1]] = {}

for compartment in ['all']:
    for min_cells in min_cell_tests:
        for metric in metrics:
            total_fovs_dropped[metric[0]][min_cells] = {}
            count_df = total_df[total_df.metric == metric[0]]
            count_df = count_df[count_df.subset == compartment]
            all_fovs = count_df.fov.unique()

            for cell_type in count_df.cell_type.unique():
                keep_df = count_df[count_df.cell_type == cell_type]
                keep_df = keep_df[keep_df.value >= min_cells]
                keep_fovs = keep_df.fov.unique()
                # total_fovs_dropped[min_cells].append(len(all_fovs) - len(keep_fovs))
                # total_fovs_dropped[metric[0]][min_cells].append(len(all_fovs) - len(keep_fovs))
                total_fovs_dropped[metric[0]][min_cells][cell_type] = len(all_fovs) - len(keep_fovs)

            total_fovs_dropped[metric[1]][min_cells] = {}
            count_df = total_df[total_df.metric == metric[1]]
            count_df = count_df[count_df.subset == compartment]
            all_fovs = count_df.fov.unique()

            for cell_type in count_df.cell_type.unique():
                keep_df = count_df[count_df.cell_type == cell_type]
                keep_df = keep_df[keep_df.value >= min_cells]
                keep_fovs = keep_df.fov.unique()
                # total_fovs_dropped[min_cells].append(len(all_fovs) - len(keep_fovs))
                # total_fovs_dropped[metric[0]][min_cells].append(len(all_fovs) - len(keep_fovs))
                total_fovs_dropped[metric[1]][min_cells][cell_type] = len(all_fovs) - len(keep_fovs)

    # Flatten the data
    def flatten_data(data, feature_name):
        return pd.DataFrame([
            {'min_cells': min_cells, 'Number of FOVs dropped': value, 'Feature': feature_name, 'Cell Type': cell_type}
            for feature, min_cells_dict in data.items()
            for min_cells, cell_types in min_cells_dict.items()
            for cell_type, value in cell_types.items()
        ])

    df = pd.concat([
        flatten_data({'cluster_broad_count': total_fovs_dropped['cluster_broad_count']}, 'cluster_broad_count')
        # flatten_data({'cluster_broad_freq': total_fovs_dropped['cluster_broad_freq']}, 'cluster_broad_freq'),
        # flatten_data({'cluster_count': total_fovs_dropped['cluster_count']}, 'cluster_count')
    ])
    print(df)
    # print(df[(df["Feature"] == "cluster_count") & (df["min_cells"] == 1)])

    # Create the strip plot
    # plot = sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Feature', data=df, kind='strip', palette={'cluster_broad_count': 'blue', 'cluster_count': 'red'})
    # Create the strip plot, coloring by cell type
    plot = sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Cell Type', data=df, kind='strip', palette='Set2', dodge=False)
    plot.fig.subplots_adjust(top=0.9)  # Adjust the top spacing to fit the title

    # Annotate each point with the cell type
    # for index, row in df.iterrows():
    #     plot.ax.text(row['min_cells'], row['Number of FOVs dropped'], row['Cell Type'], horizontalalignment='left', size='medium', color='black', weight='semibold')

    plot.fig.suptitle('Distribution of FOVs dropped across min_cells trials')
    # plt.tight_layout()

    # # Flatten the data into a format suitable for seaborn
    # def flatten_data(data, feature_name):
    #     flattened = pd.DataFrame([
    #         {'min_cells': min_cells, 'Number of FOVs dropped': value, 'Feature': feature_name}
    #         for feature, values_dict in data.items()
    #         for min_cells, values in values_dict.items()
    #         for value in values
    #     ])
    #     return flattened

    # # Apply the function to each feature and concatenate the results
    # df = pd.concat([
    #     flatten_data({'cluster_broad_count': total_fovs_dropped['cluster_broad_count']}, 'cluster_broad_count'),
    #     # flatten_data({'cluster_count': total_fovs_dropped['cluster_count']}, 'cluster_count')
    # ])

    # # Create the strip plot with faceting
    # sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Feature', data=df, kind='strip', palette={'cluster_broad_count': 'blue', 'cluster_count': 'red'})

    # plt.title('Distribution of FOVs dropped across min_cells trials')
    # save the figure to save_dir
    plt.savefig(
        pathlib.Path(extraction_pipeline_tuning_dir) / f"{compartment}_min_cells_fovs_dropped_stripplot.png",
        dpi=300
    )

    df = pd.concat([
        flatten_data({'cluster_count': total_fovs_dropped['cluster_count']}, 'cluster_count'),
        # flatten_data({'cluster_broad_freq': total_fovs_dropped['cluster_broad_freq']}, 'cluster_broad_freq')
    ])
    print(df)
    # print(df[(df["Feature"] == "cluster_count") & (df["min_cells"] == 1)])

    # Create the strip plot
    # plot = sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Feature', data=df, kind='strip', palette={'cluster_broad_count': 'blue', 'cluster_count': 'red'})
    # Create the strip plot, coloring by cell type
    palette = sns.color_palette("husl", 21)
    plot = sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Cell Type', data=df, kind='strip', palette=palette, dodge=False)
    plot.fig.subplots_adjust(top=0.9)  # Adjust the top spacing to fit the title

    # Annotate each point with the cell type
    # for index, row in df.iterrows():
    #     plot.ax.text(row['min_cells'], row['Number of FOVs dropped'], row['Cell Type'], horizontalalignment='left', size='medium', color='black', weight='semibold')

    plot.fig.suptitle('Distribution of FOVs dropped across min_cells trials')
    # plt.tight_layout()

    # # Flatten the data into a format suitable for seaborn
    # def flatten_data(data, feature_name):
    #     flattened = pd.DataFrame([
    #         {'min_cells': min_cells, 'Number of FOVs dropped': value, 'Feature': feature_name}
    #         for feature, values_dict in data.items()
    #         for min_cells, values in values_dict.items()
    #         for value in values
    #     ])
    #     return flattened

    # # Apply the function to each feature and concatenate the results
    # df = pd.concat([
    #     flatten_data({'cluster_broad_count': total_fovs_dropped['cluster_broad_count']}, 'cluster_broad_count'),
    #     # flatten_data({'cluster_count': total_fovs_dropped['cluster_count']}, 'cluster_count')
    # ])

    # # Create the strip plot with faceting
    # sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Feature', data=df, kind='strip', palette={'cluster_broad_count': 'blue', 'cluster_count': 'red'})

    # plt.title('Distribution of FOVs dropped across min_cells trials')
    # save the figure to save_dir
    plt.savefig(
        pathlib.Path(extraction_pipeline_tuning_dir) / f"{compartment}_min_cells_raw_fovs_dropped_stripplot.png",
        dpi=300
    )

    df = pd.concat([
        flatten_data({'meta_cluster_count': total_fovs_dropped['meta_cluster_count']}, 'meta_cluster_count'),
        # flatten_data({'cluster_broad_freq': total_fovs_dropped['cluster_broad_freq']}, 'cluster_broad_freq')
    ])
    print(df)
    # print(df[(df["Feature"] == "cluster_count") & (df["min_cells"] == 1)])

    # Create the strip plot
    # plot = sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Feature', data=df, kind='strip', palette={'cluster_broad_count': 'blue', 'cluster_count': 'red'})
    # Create the strip plot, coloring by cell type
    palette = sns.color_palette("husl", 21)
    plot = sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Cell Type', data=df, kind='strip', palette=palette, dodge=False)
    plot.fig.subplots_adjust(top=0.9)  # Adjust the top spacing to fit the title

    # Annotate each point with the cell type
    # for index, row in df.iterrows():
    #     plot.ax.text(row['min_cells'], row['Number of FOVs dropped'], row['Cell Type'], horizontalalignment='left', size='medium', color='black', weight='semibold')

    plot.fig.suptitle('Distribution of FOVs dropped across min_cells trials')
    # plt.tight_layout()

    # # Flatten the data into a format suitable for seaborn
    # def flatten_data(data, feature_name):
    #     flattened = pd.DataFrame([
    #         {'min_cells': min_cells, 'Number of FOVs dropped': value, 'Feature': feature_name}
    #         for feature, values_dict in data.items()
    #         for min_cells, values in values_dict.items()
    #         for value in values
    #     ])
    #     return flattened

    # # Apply the function to each feature and concatenate the results
    # df = pd.concat([
    #     flatten_data({'cluster_broad_count': total_fovs_dropped['cluster_broad_count']}, 'cluster_broad_count'),
    #     # flatten_data({'cluster_count': total_fovs_dropped['cluster_count']}, 'cluster_count')
    # ])

    # # Create the strip plot with faceting
    # sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Feature', data=df, kind='strip', palette={'cluster_broad_count': 'blue', 'cluster_count': 'red'})

    # plt.title('Distribution of FOVs dropped across min_cells trials')
    # save the figure to save_dir
    plt.savefig(
        pathlib.Path(extraction_pipeline_tuning_dir) / f"{compartment}_min_cells_meta_fovs_dropped_stripplot.png",
        dpi=300
    )

    df = pd.concat([
        flatten_data({'cluster_broad_freq': total_fovs_dropped['cluster_broad_freq']}, 'cluster_broad_freq'),
        # flatten_data({'cluster_broad_freq': total_fovs_dropped['cluster_broad_freq']}, 'cluster_broad_freq')
    ])
    print(df)
    # print(df[(df["Feature"] == "cluster_count") & (df["min_cells"] == 1)])

    # Create the strip plot
    # plot = sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Feature', data=df, kind='strip', palette={'cluster_broad_count': 'blue', 'cluster_count': 'red'})
    # Create the strip plot, coloring by cell type
    palette = sns.color_palette("husl", 21)
    plot = sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Cell Type', data=df, kind='strip', palette=palette, dodge=False)
    plot.fig.subplots_adjust(top=0.9)  # Adjust the top spacing to fit the title

    # Annotate each point with the cell type
    # for index, row in df.iterrows():
    #     plot.ax.text(row['min_cells'], row['Number of FOVs dropped'], row['Cell Type'], horizontalalignment='left', size='medium', color='black', weight='semibold')

    plot.fig.suptitle('Distribution of FOVs dropped across min_cells trials')
    # plt.tight_layout()

    # # Flatten the data into a format suitable for seaborn
    # def flatten_data(data, feature_name):
    #     flattened = pd.DataFrame([
    #         {'min_cells': min_cells, 'Number of FOVs dropped': value, 'Feature': feature_name}
    #         for feature, values_dict in data.items()
    #         for min_cells, values in values_dict.items()
    #         for value in values
    #     ])
    #     return flattened

    # # Apply the function to each feature and concatenate the results
    # df = pd.concat([
    #     flatten_data({'cluster_broad_count': total_fovs_dropped['cluster_broad_count']}, 'cluster_broad_count'),
    #     # flatten_data({'cluster_count': total_fovs_dropped['cluster_count']}, 'cluster_count')
    # ])

    # # Create the strip plot with faceting
    # sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Feature', data=df, kind='strip', palette={'cluster_broad_count': 'blue', 'cluster_count': 'red'})

    # plt.title('Distribution of FOVs dropped across min_cells trials')
    # save the figure to save_dir
    plt.savefig(
        pathlib.Path(extraction_pipeline_tuning_dir) / f"{compartment}_min_cells_freq_fovs_dropped_stripplot.png",
        dpi=300
    )

total_df_func = pd.read_csv(os.path.join(OUTPUT_DIR, "functional_df_per_core.csv"))
total_fovs_dropped_func = {}
total_fovs_dropped_func["cluster_broad_freq"] = {}
for compartment in ["all"]:
    all_fovs = count_df[count_df.subset == compartment].fov.unique()
    for min_cells in min_cell_tests:
        total_fovs_dropped_func["cluster_broad_freq"][min_cells] = {}
        filtered_dfs = []
        metrics = [['cluster_broad_count', 'cluster_broad_freq'],
                   ['cluster_count', 'cluster_freq'],
                   ['meta_cluster_count', 'meta_cluster_freq']]
        for metric in metrics:
            total_fovs_dropped[metric[0]][min_cells] = {}
            count_df = total_df[total_df.metric == metric[0]]
            count_df = count_df[count_df.subset == compartment]

            # subset functional df to only include functional markers at this resolution
            func_df = total_df_func[total_df_func.metric.isin(metric)]
            func_df = func_df[func_df.subset == compartment]

            for cell_type in count_df.cell_type.unique():
                keep_df = count_df[count_df.cell_type == cell_type]
                keep_df = keep_df[keep_df.value >= min_cells]
                keep_fovs = keep_df.fov.unique()

                # 1. functional marker tests
                keep_markers = func_df[func_df.cell_type == cell_type]
                keep_markers = keep_markers[keep_markers.fov.isin(keep_fovs)]

                filtered_dfs.append(keep_markers)

        filtered_func_df = pd.concat(filtered_dfs)

        # load matrices
        broad_df_include = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_broad.csv'), index_col=0)
        med_df_include = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_med.csv'), index_col=0)
        meta_df_include = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_meta.csv'), index_col=0)

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
        broad_df_include_dp = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_broad_dp.csv'), index_col=0)
        med_df_include_dp = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_med_dp.csv'), index_col=0)
        meta_df_include_dp = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_meta_dp.csv'), index_col=0)

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

        # combine
        combo_df = pd.concat(combo_dfs)

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

        deduped_df_broad = deduped_df[deduped_df["metric"] == "cluster_broad_freq"].copy()

        for cell_type in deduped_df_broad["cell_type"].unique():
            deduped_df_fovs_subset = deduped_df_broad[deduped_df_broad["cell_type"] == cell_type].fov.unique()
            total_fovs_dropped_func["cluster_broad_freq"][min_cells][cell_type] = len(all_fovs) - len(deduped_df_fovs_subset)

    # Flatten the data
    def flatten_data(data, feature_name):
        return pd.DataFrame([
            {'min_cells': min_cells, 'Number of FOVs dropped': value, 'Feature': feature_name, 'Cell Type': cell_type}
            for feature, min_cells_dict in data.items()
            for min_cells, cell_types in min_cells_dict.items()
            for cell_type, value in cell_types.items()
        ])

    df = pd.concat([
        flatten_data({'cluster_broad_freq': total_fovs_dropped['cluster_broad_freq']}, 'cluster_broad_freq'),
        # flatten_data({'cluster_count': total_fovs_dropped['cluster_count']}, 'cluster_count')
    ])
    print(df)

    plot = sns.catplot(x='min_cells', y='Number of FOVs dropped', hue='Cell Type', data=df, kind='strip', palette='Set2', dodge=False)
    plot.fig.subplots_adjust(top=0.9)  # Adjust the top spacing to fit the title

    plot.fig.suptitle('Distribution of FOVs dropped across min_cells trials (functional deduped)')

    plt.savefig(
        pathlib.Path(extraction_pipeline_tuning_dir) / f"{compartment}_min_cells_meta_fovs_dropped_stripplot_functional.png",
        dpi=300
    )



# functional_feature_fov_counts = {}
# for min_cells in min_cell_tests:
#     filtered_dfs = []
#     metrics = [['cluster_broad_count', 'cluster_broad_freq'],
#                ['cluster_count', 'cluster_freq'],
#                ['meta_cluster_count', 'meta_cluster_freq']]
#     for metric in metrics:
#         # subset count df to include cells at the relevant clustering resolution
#         for compartment in ["all"]:
#             count_df = total_df[total_df.metric == metric[0]]
#             count_df = count_df[count_df.subset == compartment]

#             # subset functional df to only include functional markers at this resolution
#             func_df = total_df_func[total_df_func.metric.isin(metric)]
#             func_df = func_df[func_df.subset == compartment]

#             # for each cell type, determine which FOVs have high enough counts to be included
#             for cell_type in count_df.cell_type.unique():
#                 keep_df = count_df[count_df.cell_type == cell_type]
#                 keep_df = keep_df[keep_df.value >= min_cells]
#                 keep_fovs = keep_df.fov.unique()

#                 # subset functional df to only include FOVs with high enough counts
#                 keep_markers = func_df[func_df.cell_type == cell_type]
#                 keep_markers = keep_markers[keep_markers.fov.isin(keep_fovs)]

#                 # append to list of filtered dfs
#                 filtered_dfs.append(keep_markers)

#     filtered_func_df = pd.concat(filtered_dfs)

#     # load matrices
#     broad_df_include = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_broad.csv'), index_col=0)
#     med_df_include = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_med.csv'), index_col=0)
#     meta_df_include = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_meta.csv'), index_col=0)

#     # identify metrics and dfs that will be filtered
#     filtering = [['cluster_broad_count', 'cluster_broad_freq', broad_df_include],
#                ['cluster_count', 'cluster_freq', med_df_include],
#                ['meta_cluster_count', 'meta_cluster_freq', meta_df_include]]

#     combo_dfs = []

#     for filters in filtering:
#         # get variables
#         metric_names = filters[:2]
#         metric_df = filters[2]

#         # subset functional df to only include functional markers at this resolution
#         func_df = filtered_func_df[filtered_func_df.metric.isin(metric_names)]

#         # loop over each cell type, and get the corresponding markers
#         for cell_type in metric_df.index:
#             markers = metric_df.columns[metric_df.loc[cell_type] == True]

#             # subset functional df to only include this cell type
#             func_df_cell = func_df[func_df.cell_type == cell_type]

#             # subset functional df to only include markers for this cell type
#             func_df_cell = func_df_cell[func_df_cell.functional_marker.isin(markers)]

#             # append to list of dfs
#             combo_dfs.append(func_df_cell)

#     # load inclusion matrices
#     broad_df_include_dp = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_broad_dp.csv'), index_col=0)
#     med_df_include_dp = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_med_dp.csv'), index_col=0)
#     meta_df_include_dp = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'inclusion_matrix_meta_dp.csv'), index_col=0)

#     # identify metrics and dfs that will be filtered
#     filtering = [['cluster_broad_count', 'cluster_broad_freq', broad_df_include_dp],
#                ['cluster_count', 'cluster_freq', med_df_include_dp],
#                ['meta_cluster_count', 'meta_cluster_freq', meta_df_include_dp]]

#     for filters in filtering:
#         # get variables
#         metric_names = filters[:2]
#         metric_df = filters[2]

#         # subset functional df to only include functional markers at this resolution
#         func_df = filtered_func_df[filtered_func_df.metric.isin(metric_names)]

#         # loop over each cell type, and get the corresponding markers
#         for cell_type in metric_df.index:
#             markers = metric_df.columns[metric_df.loc[cell_type] == True]

#             # subset functional df to only include this cell type
#             func_df_cell = func_df[func_df.cell_type == cell_type]

#             # subset functional df to only include markers for this cell type
#             func_df_cell = func_df_cell[func_df_cell.functional_marker.isin(markers)]

#             # append to list of dfs
#             combo_dfs.append(func_df_cell)

#     # append to list of dfs
#     long_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'total_func_per_core.csv'))
#     combo_dfs.append(long_df)

#     # combine
#     combo_df = pd.concat(combo_dfs)

#     fovs_per_metric = combo_df.groupby("metric")["fov"].nunique().reset_index(name="unique_fov_count")
#     fovs_per_metric["unique_fov_count"] = total_fovs - fovs_per_metric["unique_fov_count"]
#     functional_feature_fov_counts[min_cells] = fovs_per_metric.set_index("metric")["unique_fov_count"].to_dict()

# # Flattening the dictionary and creating a DataFrame
# flat_data_functional = []
# for min_cells, metrics in functional_feature_fov_counts.items():
#     for metric, unique_fov_count in metrics.items():
#         flat_data_functional.append((min_cells, unique_fov_count))

# df_functional_viz = pd.DataFrame(flat_data_functional, columns=['min_cells', 'unique_fov_count'])

# # Ensuring 'min_cells' is treated as a categorical variable
# df_functional_viz['min_cells'] = pd.Categorical(df_functional_viz['min_cells'])

# # Creating a box plot
# plt.figure(figsize=(10, 6))
# ax = sns.boxplot(x='min_cells', y='unique_fov_count', data=df_functional_viz)
# plt.title('FOVs Excluded for Functional Features per min_cells')
# ax.set_xlabel("min_cells threshold")
# ax.set_ylabel("FOVs dropped")
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"functional_features_min_cells_boxplot.png",
#     dpi=300
# )

# # Creating a violin plot
# plt.figure(figsize=(10, 6))
# ax = sns.violinplot(x='min_cells', y='unique_fov_count', data=df_functional_viz)
# plt.title('FOVs Excluded for Functional Features per min_cells')
# ax.set_xlabel("min_cells threshold")
# ax.set_ylabel("FOVs dropped")
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"functional_features_min_cells_violinplot.png",
#     dpi=300
# )


# ### 2.2: morphology tests
# print("Starting morphology tests")
# total_df_morph = pd.read_csv(os.path.join(OUTPUT_DIR, "morph_df_per_core.csv"))
# cell_table_morph = pd.read_csv(os.path.join(ANALYSIS_DIR, "cell_table_morph.csv"))
# annotations_by_mask = pd.read_csv(os.path.join(INTERMEDIATE_DIR, "mask_dir/individual_masks-no_tagg_tls", "cell_annotation_mask.csv"))
# harmonized_annotations = annotations_by_mask
# harmonized_annotations = harmonized_annotations.rename(columns={"mask_name": "tumor_region"})
# cell_table_morph = cell_table_morph.merge(harmonized_annotations, on=["fov", "label"], how="left")

# # create manual df with total morphology marker average across all cells in an image
# morph_table_small = cell_table_morph.loc[:, ~cell_table_morph.columns.isin(['cell_cluster', 'cell_cluster_broad', 'cell_meta_cluster', 'label', 'tumor_region'])]

# # group by specified columns
# grouped_table = morph_table_small.groupby("fov")
# transformed = grouped_table.agg(np.mean)
# transformed.reset_index(inplace=True)

# # reshape to long df
# long_df = pd.melt(transformed, id_vars=["fov"], var_name="morphology_feature")
# long_df["metric"] = "total_freq"
# long_df["cell_type"] = "all"
# long_df["subset"] = "all"

# long_df = long_df.merge(harmonized_metadata, on='fov', how='inner')

# print(total_df_morph.metric.unique())
# morphology_feature_fov_counts = {}
# for min_cells in min_cell_tests:
#     filtered_dfs = []
#     metrics = [['cluster_broad_count', 'cluster_broad_freq'],
#                ['cluster_count', 'cluster_freq'],
#                ['meta_cluster_count', 'meta_cluster_freq']]
#     for metric in metrics:
#         # subset count df to include cells at the relevant clustering resolution
#         for compartment in ['all']:
#             count_df = total_df[total_df.metric == metric[0]]
#             count_df = count_df[count_df.subset == compartment]

#             # subset morphology df to only include morphology metrics at this resolution
#             morph_df = total_df_morph[total_df_morph.metric == metric[1]]
#             morph_df = morph_df[morph_df.subset == compartment]

#             # for each cell type, determine which FOVs have high enough counts to be included
#             for cell_type in count_df.cell_type.unique():
#                 keep_df = count_df[count_df.cell_type == cell_type]
#                 keep_df = keep_df[keep_df.value >= min_cells]
#                 keep_fovs = keep_df.fov.unique()

#                 # subset morphology df to only include FOVs with high enough counts
#                 keep_features = morph_df[morph_df.cell_type == cell_type]
#                 keep_features = keep_features[keep_features.fov.isin(keep_fovs)]

#                 # append to list of filtered dfs
#                 filtered_dfs.append(keep_features)

#     filtered_dfs.append(long_df)
#     filtered_morph_df = pd.concat(filtered_dfs)

#     # # # save filtered df, CHECKPOINT
#     # # filtered_morph_df.to_csv(os.path.join(output_dir, 'morph_df_per_core_filtered.csv'), index=False)
#     # results_df.loc["morphology", f"num_fovs_min_cell_{min_cells}"] = len(filtered_morph_df["fov"].unique())

#     # # create version aggregated by timepoint
#     # filtered_morph_df_grouped = filtered_morph_df.groupby(['Tissue_ID', 'cell_type', 'morphology_feature', 'metric', 'subset'])
#     # filtered_morph_df_timepoint = filtered_morph_df_grouped['value'].agg([np.mean, np.std])
#     # filtered_morph_df_timepoint.reset_index(inplace=True)
#     # filtered_morph_df_timepoint = filtered_morph_df_timepoint.merge(harmonized_metadata.drop(['fov', 'MIBI_data_generated'], axis=1).drop_duplicates(), on='Tissue_ID')

#     # # # save timepoint df, CHECKPOINT
#     # # filtered_morph_df_timepoint.to_csv(os.path.join(output_dir, 'morph_df_per_timepoint_filtered.csv'), index=False)
#     # results_df.loc["morphology", f"num_tissue_ids_min_cell_{min_cells}"] = len(filtered_morph_df_timepoint["Tissue_ID"].unique())

#     # remove redundant morphology features
#     block1 = ['area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'convex_area', 'equivalent_diameter']

#     block2 = ['area_nuclear', 'major_axis_length_nuclear', 'minor_axis_length_nuclear', 'perimeter_nuclear', 'convex_area_nuclear', 'equivalent_diameter_nuclear']

#     block3 = ['eccentricity', 'major_axis_equiv_diam_ratio']

#     block4 = ['eccentricity_nuclear', 'major_axis_equiv_diam_ratio_nuclear', 'perim_square_over_area_nuclear']

#     deduped_morph_df = filtered_morph_df.loc[~filtered_morph_df.morphology_feature.isin(block1[1:] + block2[1:] + block3[1:] + block4[1:]), :]

#     # only keep complex morphology features for cancer cells, remove everything except area and nc for others
#     cancer_clusters = ['Cancer', 'Cancer_EMT', 'Cancer_Other', 'Cancer_CD56', 'Cancer_CK17',
#                        'Cancer_Ecad', 'Cancer_Mono', 'Cancer_SMA', 'Cancer_Vim']
#     basic_morph_features = ['area', 'area_nuclear', 'nc_ratio']

#     deduped_morph_df = deduped_morph_df.loc[~(~(deduped_morph_df.cell_type.isin(cancer_clusters)) & ~(deduped_morph_df.morphology_feature.isin(basic_morph_features))), :]

#     fovs_per_metric = deduped_morph_df.groupby("metric")["fov"].nunique().reset_index(name="unique_fov_count")
#     fovs_per_metric["unique_fov_count"] = total_fovs - fovs_per_metric["unique_fov_count"]
#     morphology_feature_fov_counts[min_cells] = fovs_per_metric.set_index("metric")["unique_fov_count"].to_dict()

#     # # # saved deduped, CHECKPOINT
#     # # deduped_morph_df.to_csv(os.path.join(output_dir, 'morph_df_per_core_filtered_deduped.csv'), index=False)
#     # results_df.loc["morphology", f"num_fovs_deduped_min_cell_{min_cells}"] = len(deduped_morph_df["fov"].unique())

#     # # same for timepoints
#     # deduped_morph_df_timepoint = filtered_morph_df_timepoint.loc[~filtered_morph_df_timepoint.morphology_feature.isin(block1[1:] + block2[1:] + block3[1:] + block4[1:]), :]
#     # deduped_morph_df_timepoint = deduped_morph_df_timepoint.loc[~(~(deduped_morph_df_timepoint.cell_type.isin(cancer_clusters)) & ~(deduped_morph_df_timepoint.morphology_feature.isin(basic_morph_features))), :]

#     # # save morph timepoint, CHECKPOINT
#     # # deduped_morph_df_timepoint.to_csv(os.path.join(output_dir, 'morph_df_per_timepoint_filtered_deduped.csv'), index=False)
#     # results_df.loc["morphology", f"num_tissue_ids_deduped_min_cell_{min_cells}"] = len(deduped_morph_df_timepoint["Tissue_ID"].unique())

# print(morphology_feature_fov_counts)

# # Flattening the dictionary and creating a DataFrame
# flat_data_morphology = []
# for min_cells, metrics in morphology_feature_fov_counts.items():
#     for metric, unique_fov_count in metrics.items():
#         flat_data_morphology.append((min_cells, unique_fov_count))

# df_morphology_viz = pd.DataFrame(flat_data_morphology, columns=['min_cells', 'unique_fov_count'])

# # Ensuring 'min_cells' is treated as a categorical variable
# df_morphology_viz['min_cells'] = pd.Categorical(df_morphology_viz['min_cells'])

# # Creating a box plot
# plt.figure(figsize=(10, 6))
# ax = sns.boxplot(x='min_cells', y='unique_fov_count', data=df_morphology_viz)
# plt.title('FOVs Excluded for Morphology Features per min_cells')
# ax.set_xlabel("min_cells threshold")
# ax.set_ylabel("FOVs dropped")
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"morph_features_min_cells_boxplot.png",
#     dpi=300
# )

# # Creating a violin plot
# plt.figure(figsize=(10, 6))
# ax = sns.violinplot(x='min_cells', y='unique_fov_count', data=df_morphology_viz)
# plt.title('FOVs Excluded for Morphology Features per min_cells')
# ax.set_xlabel("min_cells threshold")
# ax.set_ylabel("FOVs dropped")
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"morph_features_min_cells_violinplot.png",
#     dpi=300
# )

# ### 2.3: diversity tests
# print("Starting diversity tests")
# total_df_diversity = pd.read_csv(os.path.join(OUTPUT_DIR, 'diversity_df_per_core.csv'))
# print(total_df_diversity.metric.unique())
# diversity_feature_fov_counts = {}
# for min_cells in min_cell_tests:
#     filtered_dfs = []
#     metrics = [['cluster_broad_count', 'cluster_broad_freq'],
#                ['cluster_count', 'cluster_freq'],
#                ['meta_cluster_count', 'meta_cluster_freq']]
#     for metric in metrics:
#         # subset count df to include cells at the relevant clustering resolution
#         for compartment in ['all']:
#             count_df = total_df[total_df.metric == metric[0]]
#             count_df = count_df[count_df.subset == compartment]

#             # subset diversity df to only include diversity metrics at this resolution
#             diversity_df = total_df_diversity[total_df_diversity.metric == metric[1]]
#             diversity_df = diversity_df[diversity_df.subset == compartment]

#             # for each cell type, determine which FOVs have high enough counts to be included
#             for cell_type in count_df.cell_type.unique():
#                 keep_df = count_df[count_df.cell_type == cell_type]
#                 keep_df = keep_df[keep_df.value >= min_cells]
#                 keep_fovs = keep_df.fov.unique()

#                 # subset morphology df to only include FOVs with high enough counts
#                 keep_features = diversity_df[diversity_df.cell_type == cell_type]
#                 keep_features = keep_features[keep_features.fov.isin(keep_fovs)]

#                 # append to list of filtered dfs
#                 filtered_dfs.append(keep_features)

#     filtered_diversity_df = pd.concat(filtered_dfs)

#     # # save filtered df, CHECKPOINT
#     # filtered_diversity_df.to_csv(os.path.join(output_dir, 'diversity_df_per_core_filtered.csv'), index=False)

#     # # create version aggregated by timepoint
#     # filtered_diversity_df_grouped = filtered_diversity_df.groupby(['Tissue_ID', 'cell_type', 'diversity_feature', 'metric', 'subset'])
#     # filtered_diversity_df_timepoint = filtered_diversity_df_grouped['value'].agg([np.mean, np.std])
#     # filtered_diversity_df_timepoint.reset_index(inplace=True)
#     # filtered_diversity_df_timepoint = filtered_diversity_df_timepoint.merge(harmonized_metadata.drop(['fov', 'MIBI_data_generated'], axis=1).drop_duplicates(), on='Tissue_ID')

#     # # save timepoint df, CHECKPOINT
#     # filtered_diversity_df_timepoint.to_csv(os.path.join(output_dir, 'diversity_df_per_timepoint_filtered.csv'), index=False)
#     # results_df.loc["diversity", f"num_tissue_ids_min_cell_{min_cells}"] = len(filtered_diversity_df_timepoint["Tissue_ID"].unique())

#     # investigate correlation between diversity scores
#     fov_data = filtered_diversity_df.copy()
#     fov_data['feature_name_unique'] = fov_data['cell_type'] + '_' + fov_data['diversity_feature']
#     fov_data = fov_data.loc[(fov_data.subset == 'all') & (fov_data.metric == 'cluster_freq')]
#     fov_data = fov_data.loc[fov_data.diversity_feature != 'diversity_cell_meta_cluster']
#     fov_data_wide = fov_data.pivot(index='fov', columns='feature_name_unique', values='value')

#     corr_df = fov_data_wide.corr(method='spearman')

#     # replace Nans
#     corr_df = corr_df.fillna(0)
#     clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))

#     # save deduped df that excludes cell meta cluster, CHECKPOINT
#     deduped_diversity_df = filtered_diversity_df.loc[filtered_diversity_df.diversity_feature != 'diversity_cell_meta_cluster']

#     fovs_per_metric = deduped_diversity_df.groupby("metric")["fov"].nunique().reset_index(name="unique_fov_count")
#     fovs_per_metric["unique_fov_count"] = total_fovs - fovs_per_metric["unique_fov_count"]
#     diversity_feature_fov_counts[min_cells] = fovs_per_metric.set_index("metric")["unique_fov_count"].to_dict()
#     # deduped_diversity_df.to_csv(os.path.join(output_dir, 'diversity_df_per_core_filtered_deduped.csv'), index=False)
#     # results_df.loc["diversity", f"num_fovs_deduped_min_cell_{min_cells}"] = len(deduped_diversity_df["fov"].unique())

#     # # create version aggregated by timepoint
#     # deduped_diversity_df_grouped = deduped_diversity_df.groupby(['Tissue_ID', 'cell_type', 'diversity_feature', 'metric', 'subset'])
#     # deduped_diversity_df_timepoint = deduped_diversity_df_grouped['value'].agg([np.mean, np.std])
#     # deduped_diversity_df_timepoint.reset_index(inplace=True)
#     # deduped_diversity_df_timepoint = deduped_diversity_df_timepoint.merge(harmonized_metadata.drop(['fov', 'MIBI_data_generated'], axis=1).drop_duplicates(), on='Tissue_ID')

#     # # # save timepoint df, CHECKPOINT
#     # # deduped_diversity_df_timepoint.to_csv(os.path.join(output_dir, 'diversity_df_per_timepoint_filtered_deduped.csv'), index=False)
#     # results_df.loc["diversity", f"num_tissue_ids_deduped_min_cell_{min_cells}"] = len(deduped_diversity_df_timepoint["Tissue_ID"].unique())

# print(diversity_feature_fov_counts)

# # Flattening the dictionary and creating a DataFrame
# flat_data_diversity = []
# for min_cells, metrics in diversity_feature_fov_counts.items():
#     for metric, unique_fov_count in metrics.items():
#         flat_data_diversity.append((min_cells, unique_fov_count))

# df_diversity_viz = pd.DataFrame(flat_data_diversity, columns=['min_cells', 'unique_fov_count'])

# # Ensuring 'min_cells' is treated as a categorical variable
# df_diversity_viz['min_cells'] = pd.Categorical(df_diversity_viz['min_cells'])

# # Creating a box plot
# plt.figure(figsize=(10, 6))
# ax = sns.boxplot(x='min_cells', y='unique_fov_count', data=df_diversity_viz)
# plt.title('FOVs Excluded for Diversity Features per min_cells')
# ax.set_xlabel("min_cells threshold")
# ax.set_ylabel("FOVs dropped")
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"diversity_features_min_cells_boxplot.png",
#     dpi=300
# )

# # Creating a violin plot
# plt.figure(figsize=(10, 6))
# ax = sns.violinplot(x='min_cells', y='unique_fov_count', data=df_diversity_viz)
# plt.title('FOVs Excluded for Diversity Features per min_cells')
# ax.set_xlabel("min_cells threshold")
# ax.set_ylabel("FOVs dropped")
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"diversity_features_min_cells_violinplot.png",
#     dpi=300
# )

# ### 2.4: distance tests
# print("Starting distance tests")
# total_df_distance = pd.read_csv(os.path.join(OUTPUT_DIR, "distance_df_per_core.csv"))
# print(total_df_distance.metric.unique())
# distance_feature_fov_counts = {}
# for min_cells in min_cell_tests:
#     filtered_dfs = []
#     metrics = [['cluster_broad_count', 'cluster_broad_freq']]

#     for metric in metrics:
#         # subset count df to include cells at the relevant clustering resolution
#         for compartment in ['all']:
#             count_df = total_df[total_df.metric == metric[0]]
#             count_df = count_df[count_df.subset == compartment]

#             # subset distance df to only include distance metrics at this resolution
#             distance_df = total_df_distance[total_df_distance.metric == metric[1]]
#             distance_df = distance_df[distance_df.subset == compartment]

#             # for each cell type, determine which FOVs have high enough counts to be included
#             for cell_type in count_df.cell_type.unique():
#                 keep_df = count_df[count_df.cell_type == cell_type]
#                 keep_df = keep_df[keep_df.value >= min_cells]
#                 keep_fovs = keep_df.fov.unique()

#                 # subset morphology df to only include FOVs with high enough counts
#                 keep_features = distance_df[distance_df.cell_type == cell_type]
#                 keep_features = keep_features[keep_features.fov.isin(keep_fovs)]

#                 # append to list of filtered dfs
#                 filtered_dfs.append(keep_features)

#     filtered_distance_df = pd.concat(filtered_dfs)

#     # # # save filtered df, CHECKPOINT
#     # # filtered_distance_df.to_csv(os.path.join(output_dir, 'distance_df_per_core_filtered.csv'), index=False)
#     # results_df.loc["distance", f"num_fovs_min_cell_{min_cells}"] = len(filtered_distance_df["fov"].unique())

#     # filter distance df to only include features with low correlation with abundance
#     keep_df = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'post_processing', 'distance_df_keep_features.csv'))

#     deduped_dfs = []
#     for cell_type in keep_df.cell_type.unique():
#         keep_features = keep_df.loc[keep_df.cell_type == cell_type, 'feature_name'].unique()
#         if len(keep_features) > 0:
#             keep_df_subset = filtered_distance_df.loc[filtered_distance_df.cell_type == cell_type]
#             keep_df_subset = keep_df_subset.loc[keep_df_subset.linear_distance.isin(keep_features)]
#             deduped_dfs.append(keep_df_subset)

#     deduped_distance_df = pd.concat(deduped_dfs)

#     fovs_per_metric = deduped_distance_df.groupby("metric")["fov"].nunique().reset_index(name="unique_fov_count")
#     fovs_per_metric["unique_fov_count"] = total_fovs - fovs_per_metric["unique_fov_count"]
#     distance_feature_fov_counts[min_cells] = fovs_per_metric.set_index("metric")["unique_fov_count"].to_dict()

#     # # # save filtered df, CHECKPOINT
#     # # deduped_distance_df.to_csv(os.path.join(output_dir, 'distance_df_per_core_deduped.csv'), index=False)
#     # results_df.loc["distance", f"num_fovs_deduped_min_cell_{min_cells}"] = len(deduped_distance_df["Tissue_ID"].unique())

#     # # create version aggregated by timepoint
#     # deduped_distance_df_grouped = deduped_distance_df.groupby(['Tissue_ID', 'cell_type', 'linear_distance', 'metric', 'subset'])
#     # deduped_distance_df_timepoint = deduped_distance_df_grouped['value'].agg([np.mean, np.std])
#     # deduped_distance_df_timepoint.reset_index(inplace=True)
#     # deduped_distance_df_timepoint = deduped_distance_df_timepoint.merge(harmonized_metadata.drop(['fov', 'MIBI_data_generated'], axis=1).drop_duplicates(), on='Tissue_ID')

#     # # # save timepoint df, CHECKPOINT
#     # # deduped_distance_df_timepoint.to_csv(os.path.join(output_dir, 'distance_df_per_timepoint_deduped.csv'), index=False)
#     # results_df.loc["distance", f"num_tissue_ids_deduped_min_cell_{min_cells}"] = len(filtered_distance_df["fov"].unique())

# print(distance_feature_fov_counts)

# # Flattening the dictionary and creating a DataFrame
# flat_data_distance = []
# for min_cells, metrics in distance_feature_fov_counts.items():
#     for metric, unique_fov_count in metrics.items():
#         flat_data_distance.append((min_cells, unique_fov_count))

# df_distance_viz = pd.DataFrame(flat_data_distance, columns=['min_cells', 'unique_fov_count'])

# # Ensuring 'min_cells' is treated as a categorical variable
# df_distance_viz['min_cells'] = pd.Categorical(df_distance_viz['min_cells'])

# # Creating a box plot
# plt.figure(figsize=(10, 6))
# ax = sns.boxplot(x='min_cells', y='unique_fov_count', data=df_distance_viz)
# plt.title('FOVs Excluded for Distance Features per min_cells')
# ax.set_xlabel("min_cells threshold")
# ax.set_ylabel("FOVs dropped")
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"distance_features_min_cells_boxplot.png",
#     dpi=300
# )

# # Creating a violin plot
# plt.figure(figsize=(10, 6))
# ax = sns.violinplot(x='min_cells', y='unique_fov_count', data=df_distance_viz)
# plt.title('FOVs Excluded for Distance Features per min_cells')
# ax.set_xlabel("min_cells threshold")
# ax.set_ylabel("FOVs dropped")
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"distance_features_min_cells_violinplot.png",
#     dpi=300
# )

# 3: cancer mask tests
import utils
channel_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/'
seg_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output'
mask_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files/mask_dir/'
analysis_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files'
cell_table_clusters = pd.read_csv(os.path.join(analysis_dir, 'cell_table_clusters.csv'))

folders = list_folders(channel_dir)

cell_mask_sigmas = [0, 5, 10, 15, 20]
cell_mask_min_mask_sizes = [0, 5, 10, 15, 20]
cell_mask_max_hole_sizes = [10000, 50000, 100000, 150000, 200000]
cell_mask_smooth_threshes = [0.1, 0.2, 0.3, 0.4, 0.5]

cell_border_border_sizes = [30, 40, 50, 60, 70]
cell_border_min_mask_sizes = [2500, 3000, 3500, 4000, 4500]
cell_border_max_hole_sizes = [100, 1000, 5000, 10000, 15000]
cell_border_channel_threshes = [0.0005, 0.001, 0.0015, 0.002, 0.0025]

cell_mask_sigma_data = {s: [] for s in cell_mask_sigmas}
cell_mask_min_mask_size_data = {mms: [] for mms in cell_mask_min_mask_sizes}
cell_mask_max_hole_size_data = {mhs: [] for mhs in cell_mask_max_hole_sizes}
cell_mask_smooth_thresh_data = {st: [] for st in cell_mask_smooth_threshes}

cell_border_border_size_data = {bs: {"full": [], "external": [], "interior": []} for bs in cell_border_border_sizes}
cell_border_min_mask_size_data = {mms: {"full": [], "external": [], "interior": []} for mms in cell_border_min_mask_sizes}
cell_border_max_hole_size_data = {mhs: {"full": [], "external": [], "interior": []} for mhs in cell_border_max_hole_sizes}
cell_border_channel_thresh_data = {ct: {"full": [], "external": [], "interior": []} for ct in cell_border_channel_threshes}

i = 0
# for folder in folders:
#     ecad = io.imread(os.path.join(channel_dir, folder, 'ECAD.tiff'))

    # intermediate_folder = os.path.join(intermediate_dir, folder)
    # if not os.path.exists(intermediate_folder):
    #     os.mkdir(intermediate_folder)

    # generate cancer/stroma mask by combining segmentation mask with ECAD channel
    # seg_label = io.imread(os.path.join(seg_dir, folder + '_whole_cell.tiff'))[0]

    # # test different create_cell_mask parameters
    # for s in cell_mask_sigmas:
    #     seg_mask = utils.create_cell_mask(seg_label, cell_table_clusters, folder, ['Cancer'], sigma=s)
    #     percent_hit = np.sum(seg_mask) / seg_mask.size
    #     cell_mask_sigma_data[s].append(percent_hit)

    # for mms in cell_mask_min_mask_sizes:
    #     seg_mask = utils.create_cell_mask(seg_label, cell_table_clusters, folder, ['Cancer'], min_mask_size=mms)
    #     percent_hit = np.sum(seg_mask) / seg_mask.size
    #     cell_mask_min_mask_size_data[mms].append(percent_hit)

    # for mhs in cell_mask_max_hole_sizes:
    #     seg_mask = utils.create_cell_mask(seg_label, cell_table_clusters, folder, ['Cancer'], max_hole_size=mhs)
    #     percent_hit = np.sum(seg_mask) / seg_mask.size
    #     cell_mask_max_hole_size_data[mhs].append(percent_hit)

    # for st in cell_mask_smooth_threshes:
    #     seg_mask = utils.create_cell_mask(seg_label, cell_table_clusters, folder, ['Cancer'], smooth_thresh=st)
    #     percent_hit = np.sum(seg_mask) / seg_mask.size
    #     cell_mask_smooth_thresh_data[st].append(percent_hit)

    # # given the base create_cell_mask parameters, analyze the cancer boundary params
    # seg_mask = utils.create_cell_mask(seg_label, cell_table_clusters, folder, ['Cancer'])

    # # test different create_cancer_boundary parameters
    # for bs in cell_border_border_sizes:
    #     cancer_mask = utils.create_cancer_boundary(ecad, seg_mask, border_size=bs, min_mask_size=7000)
    #     # percent_full = np.sum(seg_mask == 4) / seg_mask.size
    #     # cell_border_border_size_data[bs]["full"].append(percent_full)

    #     percent_external = np.sum(cancer_mask == 2) / cancer_mask.size
    #     cell_border_border_size_data[bs]["external"].append(percent_external)

    #     percent_interior = np.sum(cancer_mask == 3) / cancer_mask.size
    #     cell_border_border_size_data[bs]["interior"].append(percent_interior)

    # # test different create_cancer_boundary parameters
    # for mms in cell_border_min_mask_sizes:
    #     cancer_mask = utils.create_cancer_boundary(ecad, seg_mask, min_mask_size=mms)
    #     # percent_full = np.sum(seg_mask == 4) / seg_mask.size
    #     # cell_border_min_mask_size_data[mms]["full"].append(percent_full)

    #     percent_external = np.sum(cancer_mask == 2) / cancer_mask.size
    #     cell_border_min_mask_size_data[mms]["external"].append(percent_external)

    #     percent_interior = np.sum(cancer_mask == 3) / cancer_mask.size
    #     cell_border_min_mask_size_data[mms]["interior"].append(percent_interior)

    # # test different create_cancer_boundary parameters
    # for mhs in cell_border_max_hole_sizes:
    #     cancer_mask = utils.create_cancer_boundary(ecad, seg_mask, min_mask_size=7000, max_hole_size=mhs)
    #     # percent_full = np.sum(seg_mask == 4) / seg_mask.size
    #     # cell_border_max_hole_size_data[mhs]["full"].append(percent_full)

    #     percent_external = np.sum(cancer_mask == 2) / cancer_mask.size
    #     cell_border_max_hole_size_data[mhs]["external"].append(percent_external)

    #     percent_interior = np.sum(cancer_mask == 3) / cancer_mask.size
    #     cell_border_max_hole_size_data[mhs]["interior"].append(percent_interior)

    # # test different create_cancer_boundary parameters
    # for ct in cell_border_channel_threshes:
    #     cancer_mask = utils.create_cancer_boundary(ecad, seg_mask, min_mask_size=7000, channel_thresh=ct)
    #     # percent_full = np.sum(seg_mask == 4) / seg_mask.size
    #     # cell_border_channel_thresh_data[ct]["full"].append(percent_full)

    #     percent_external = np.sum(cancer_mask == 2) / cancer_mask.size
    #     cell_border_channel_thresh_data[ct]["external"].append(percent_external)

    #     percent_interior = np.sum(cancer_mask == 3) / cancer_mask.size
    #     cell_border_channel_thresh_data[ct]["interior"].append(percent_interior)

    # i += 1
    # if i % 10 == 0:
    #     print(f"Processed {i} folders")
    # cancer_mask = utils.create_cancer_boundary(ecad, seg_mask, min_mask_size=7000)
    # cancer_mask = cancer_mask.astype(np.uint8)

# # Preparing the data for plotting
# data = []
# labels = []
# for key, values in cell_mask_sigma_data.items():
#     data.extend(values)
#     labels.extend([key] * len(values))

# # Creating the boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=labels, y=data)
# plt.title('Distribution of % mask included in cancer across sigma')
# plt.xlabel('sigma')
# plt.ylabel('% of mask included in cancer')

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"sigma_cancer_mask_inclusion_box.png",
#     dpi=300
# )

# # Creating the violin plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x=labels, y=data)
# plt.title('Distribution of % mask included in cancer across sigma')
# plt.xlabel('sigma')
# plt.ylabel('% of mask included in cancer')

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"sigma_cancer_mask_inclusion_violin.png",
#     dpi=300
# )

# # Preparing the data for plotting
# data = []
# labels = []
# for key, values in cell_mask_min_mask_size_data.items():
#     data.extend(values)
#     labels.extend([key] * len(values))

# # Creating the boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=labels, y=data)
# plt.title('Distribution of % mask included in cancer across min mask sizes')
# plt.xlabel('min mask size')
# plt.ylabel('% of mask included in cancer')

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"min_mask_size_cancer_mask_inclusion_box.png",
#     dpi=300
# )

# # Creating the violin plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x=labels, y=data)
# plt.title('Distribution of % mask included in cancer across min mask sizes')
# plt.xlabel('min mask size')
# plt.ylabel('% of mask included in cancer')

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"min_mask_size_cancer_mask_inclusion_violin.png",
#     dpi=300
# )

# # Preparing the data for plotting
# data = []
# labels = []
# for key, values in cell_mask_max_hole_size_data.items():
#     data.extend(values)
#     labels.extend([key] * len(values))

# # Creating the boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=labels, y=data)
# plt.title('Distribution of % mask included in cancer across max hole sizes')
# plt.xlabel('max hole size')
# plt.ylabel('% of mask included in cancer')

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"max_hole_size_cancer_mask_inclusion_box.png",
#     dpi=300
# )

# # Creating the violin plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x=labels, y=data)
# plt.title('Distribution of % mask included in cancer across max hole sizes')
# plt.xlabel('max hole size')
# plt.ylabel('% of mask included in cancer')

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"max_hole_size_cancer_mask_inclusion_violin.png",
#     dpi=300
# )

# # Preparing the data for plotting
# data = []
# labels = []
# for key, values in cell_mask_smooth_thresh_data.items():
#     data.extend(values)
#     labels.extend([key] * len(values))

# # Creating the boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=labels, y=data)
# plt.title('Distribution of % mask included in cancer across smoothing thresholds')
# plt.xlabel('smooth thresh')
# plt.ylabel('% of mask included in cancer')

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"smooth_thresh_cancer_mask_inclusion_box.png",
#     dpi=300
# )

# # Creating the violin plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x=labels, y=data)
# plt.title('Distribution of % mask included in cancer across smoothing thresholds')
# plt.xlabel('smooth thresh')
# plt.ylabel('% of mask included in cancer')

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"smooth_thresh_cancer_mask_inclusion_violin.png",
#     dpi=300
# )

# # Transforming the data into a DataFrame
# data_to_plot = {'border_size': [], 'percentage': [], 'type': []}
# for bs, types in cell_border_border_size_data.items():
#     for type_label, values in types.items():
#         data_to_plot['border_size'].extend([bs] * len(values))
#         data_to_plot['percentage'].extend(values)
#         data_to_plot['type'].extend([type_label] * len(values))

# df = pd.DataFrame(data_to_plot)

# # Creating the boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='border_size', y='percentage', hue='type', data=df)
# plt.title('Distribution of % cancer borders across border sizes')
# plt.xlabel('border size')
# plt.ylabel('% of mask assigned to boundary type')
# plt.legend(title='Mask Type', labels=['% total external cancer boundary', '% total interior cancer boundary'])

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"border_size_cancer_border_inclusion_box.png",
#     dpi=300
# )

# # Creating the violin plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='border_size', y='percentage', hue='type', data=df, split=True)
# plt.title('Distribution of % cancer borders across border sizes')
# plt.xlabel('border size')
# plt.ylabel('% of mask assigned to boundary type')
# plt.legend(title='Mask Type', labels=['% total external cancer boundary', '% total interior cancer boundary'])

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"border_size_cancer_border_inclusion_violin.png",
#     dpi=300
# )

# # Transforming the data into a DataFrame
# data_to_plot = {'min_mask_size': [], 'percentage': [], 'type': []}
# for mms, types in cell_border_min_mask_size_data.items():
#     for type_label, values in types.items():
#         data_to_plot['min_mask_size'].extend([mms] * len(values))
#         data_to_plot['percentage'].extend(values)
#         data_to_plot['type'].extend([type_label] * len(values))

# df = pd.DataFrame(data_to_plot)

# # Creating the boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='min_mask_size', y='percentage', hue='type', data=df)
# plt.title('Distribution of % cancer borders across min mask sizes')
# plt.xlabel('min mask size')
# plt.ylabel('% of mask assigned to boundary type')
# plt.legend(title='Mask Type', labels=['% total external cancer boundary', '% total interior cancer boundary'])

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"min_mask_size_cancer_border_inclusion_box.png",
#     dpi=300
# )

# # Creating the violin plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='min_mask_size', y='percentage', hue='type', data=df, split=True)
# plt.title('Distribution of % cancer borders across min mask sizes')
# plt.xlabel('min mask size')
# plt.ylabel('% of mask assigned to boundary type')
# plt.legend(title='Mask Type', labels=['% total external cancer boundary', '% total interior cancer boundary'])

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"min_mask_size_cancer_border_inclusion_violin.png",
#     dpi=300
# )

# # Transforming the data into a DataFrame
# data_to_plot = {'max_hole_size': [], 'percentage': [], 'type': []}
# for mhs, types in cell_border_max_hole_size_data.items():
#     for type_label, values in types.items():
#         data_to_plot['max_hole_size'].extend([mhs] * len(values))
#         data_to_plot['percentage'].extend(values)
#         data_to_plot['type'].extend([type_label] * len(values))

# df = pd.DataFrame(data_to_plot)

# # Creating the boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='max_hole_size', y='percentage', hue='type', data=df)
# plt.title('Distribution of % cancer borders across max hole sizes')
# plt.xlabel('max hole size')
# plt.ylabel('% of mask assigned to boundary type')
# plt.legend(title='Mask Type', labels=['% total external cancer boundary', '% total interior cancer boundary'])

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"max_hole_size_cancer_border_inclusion_box.png",
#     dpi=300
# )

# # Creating the violin plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='max_hole_size', y='percentage', hue='type', data=df, split=True)
# plt.title('Distribution of % cancer borders across max hole sizes')
# plt.xlabel('max hole size')
# plt.ylabel('% of mask assigned to boundary type')
# plt.legend(title='Mask Type', labels=['% total external cancer boundary', '% total interior cancer boundary'])

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"max_hole_size_cancer_border_inclusion_violin.png",
#     dpi=300
# )

# # Transforming the data into a DataFrame
# data_to_plot = {'max_hole_size': [], 'percentage': [], 'type': []}
# for ct, types in cell_border_channel_thresh_data.items():
#     for type_label, values in types.items():
#         data_to_plot['channel_thresh'].extend([ct] * len(values))
#         data_to_plot['percentage'].extend(values)
#         data_to_plot['type'].extend([type_label] * len(values))

# df = pd.DataFrame(data_to_plot)

# # Creating the boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='channel_thresh', y='percentage', hue='type', data=df)
# plt.title('Distribution of % cancer borders across channel thresholds')
# plt.xlabel('channel thresh')
# plt.ylabel('% of mask assigned to boundary type')
# plt.legend(title='Mask Type', labels=['% total external cancer boundary', '% total interior cancer boundary'])

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"channel_thresh_cancer_border_inclusion_box.png",
#     dpi=300
# )

# # Creating the violin plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='channel_thresh', y='percentage', hue='type', data=df, split=True)
# plt.title('Distribution of % cancer borders across channel thresholds')
# plt.xlabel('channel thresh')
# plt.ylabel('% of mask assigned to boundary type')
# plt.legend(title='Mask Type', labels=['% total external cancer boundary', '% total interior cancer boundary'])

# # save the figure to save_dir
# plt.savefig(
#     pathlib.Path(extraction_pipeline_tuning_dir) / f"channel_thresh_cancer_border_inclusion_violin.png",
#     dpi=300
# )
