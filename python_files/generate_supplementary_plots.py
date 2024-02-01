# File with code for generating supplementary plots
import math
import os
import pathlib
import shutil
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from alpineer.io_utils import validate_paths
from alpineer.misc_utils import make_iterable, verify_in_list

ANALYSIS_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files"
SUPPLEMENTARY_FIG_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"

class MarkerDict(TypedDict):
    populations: List[str]
    threshold: float
    x_range: Optional[Tuple[float, float]]
    x_ticks: Optional[np.ndarray]
    x_tick_labels: Optional[np.ndarray]


# Panel validation


# ROI selection


# QC


# Image processing


# Cell identification and classification



# Functional marker thresholding
def functional_marker_thresholding(
    cell_table: pd.DataFrame, save_dir: Union[str, pathlib.Path],
    marker_info: Dict[str, MarkerDict], pop_col: str = "cell_cluster",
    figsize: Optional[Tuple[float, float]] = None
):
    """For a set of markers, visualize their distribution across the entire cohort, plus just 
    against the specified populations.

    Args:
        cell_table (pd.DataFrame):
            Cell table with clustered cell populations
        save_dir (Union[str, pathlib.Path]):
            The directory to save the marker distribution histograms
        marker_info (str):
            For each marker, define the populations, threshold, x-range, and x-tick locations
            NOTE: assumes that each marker is being visualized against the same number of
            populations
        pop_col (str):
            Column containing the names of the cell populations
        fig_size (Optional[Tuple[float, float]]):
            The figure size to use for the image.
            If None use default sizing (18.6, 6.6 * len(populations))
    """
    # verify save_dir is valid
    validate_paths([save_dir])

    # verify figsize is valid if set
    if figsize and (len(figsize) != 2 or figsize[0] <= 0 or figsize[1] <= 0):
        raise ValueError(
            "Invalid figsize: it must be in the form (size_x, size_y), size_x > 0, size_y > 0"
        )

    # define the subplots
    markers = list(marker_info.keys())
    figsize = figsize if figsize else (18.6, 6.6 * len(populations))
    fig, axs = plt.subplots(
        len(marker_info),
        len(marker_info[markers[0]]["populations"]) + 1,
        figsize=figsize
    )

    # retrieve all the markers and populations in the cell table (done for validation)
    all_markers: np.ndarray = cell_table.columns.values
    all_populations: np.ndarray = cell_table[pop_col].unique()

    # set axs_row and axs_col as counters to position the titles correctly
    axs_row: int = 0
    axs_col: int = 0

    # iterate over each marker
    for marker in markers:
        # retrieve all populations associated with the marker
        populations: List[str] = marker_info[marker]["populations"]

        # Verify that the marker and all populations specified are valid
        verify_in_list(
            specified_marker=marker,
            cell_table_columns=all_markers
        )

        verify_in_list(
            specified_populations=populations,
            cell_table_populations=all_populations
        )

        # limit x_range to 99.9% of the marker in question if x_range not specified
        x_range = marker_info[marker].get(
            "x_range", (0, np.quantile(cell_table[marker].values, 0.999))
        )

        # retrieve the x ticks and x tick labels
        x_ticks = marker_info[marker].get("x_ticks", None)
        x_tick_labels = marker_info[marker].get("x_tick_labels", None)

        # the first subplot should always be the distribution of the marker against all populations
        threshold: float = marker_info[marker]["threshold"]
        axs[axs_row][0].hist(
            cell_table[marker].values,
            50,
            density=True,
            facecolor='g',
            alpha=0.75,
            range=x_range
        )
        axs[axs_row][0].set_title(
            "{} in all populations".format(marker),
            fontsize=28
        )
        axs[axs_row][0].axvline(x=threshold)

        if isinstance(x_ticks, np.ndarray):
            axs[axs_row][0].set_xticks(x_ticks)

        if isinstance(x_tick_labels, np.ndarray):
            axs[axs_row][0].set_xticklabels(x_tick_labels, fontsize=24)

        axs[axs_row][0].tick_params(axis="y", labelsize=24)

        # add additional subplots to the figure based on the specified populations
        for i, pop in zip(np.arange(1, len(populations) + 1), populations):
            cell_table_marker_sub: pd.DataFrame = cell_table.loc[
                cell_table[pop_col] == pop, marker
            ].values
            axs[axs_row][i].hist(
                cell_table_marker_sub,
                50,
                density=True,
                facecolor='g',
                alpha=0.75,
                range=x_range
            )
            axs[axs_row][i].set_title(
                "{} in {}".format(marker, pop),
                fontsize=28
            )
            axs[axs_row][i].axvline(x=threshold)

            if isinstance(x_ticks, np.ndarray):
                axs[axs_row][i].set_xticks(x_ticks)

            if isinstance(x_tick_labels, np.ndarray):
                axs[axs_row][i].set_xticklabels(x_tick_labels, fontsize=24)

            axs[axs_row][i].tick_params(axis="y", labelsize=24)

        # update axs_row to the next column
        axs_row += 1

    plt.tight_layout()

    # save the figure to save_dir
    fig.savefig(
        pathlib.Path(save_dir) / f"functional_marker_thresholds.png",
        dpi=300
    )


cell_table = pd.read_csv(
    os.path.join(ANALYSIS_DIR, "combined_cell_table_normalized_cell_labels_updated.csv")
)
functional_marker_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "functional_marker_dist_thresholds")
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
functional_marker_thresholding(
    cell_table, functional_marker_viz_dir, marker_info=marker_info,
    figsize=(20, 40)
)


# Feature extraction


