# File with code for generating supplementary plots
import os
import pathlib
import shutil
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
def functional_marker_thresholding_grid(
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

    # # verify x_range is valid if set
    # if x_range and (len(x_range) != 2 or x_range[0] >= x_range[1]):
    #     raise ValueError(
    #         "Invalid x_range: it must be in the form (low, high), low < high"
    #     )

    # verify figsize is valid if set
    if figsize and (len(figsize) != 2 or figsize[0] <= 0 or figsize[1] <= 0):
        raise ValueError(
            "Invalid figsize: it must be in the form (size_x, size_y), size_x > 0, size_y > 0"
        )

    # define the subplots
    markers = list(marker_info.keys())
    figsize = figsize if figsize else (18.6, 6.6 * len(populations))
    fig, axs = plt.subplots(
        len(marker_info[markers[0]]["populations"]) + 1,
        len(marker_info),
        figsize=figsize
    )

    # retrieve all the markers and populations in the cell table (done for validation)
    all_markers: np.ndarray = cell_table.columns.values
    all_populations: np.ndarray = cell_table[pop_col].unique()

    # set axs_col as a counter to position the titles correctly
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
        x_range = marker_info[marker].get("x_range", np.quantile(cell_table[marker].values, 0.999))

        # retrieve the x ticks and x tick labels
        x_ticks = marker_info[marker].get("x_ticks", None)
        x_tick_labels = marker_info[marker].get("x_tick_labels", None)

        # the first subplot should always be the distribution of the marker against all populations
        threshold: float = marker_info[marker]["threshold"]
        axs[0][axs_col].hist(
            cell_table[marker].values,
            50,
            density=True,
            facecolor='g',
            alpha=0.75,
            range=x_range
        )
        axs[0][axs_col].set_title("Distribution of {} in all populations".format(marker))
        axs[0][axs_col].axvline(x=threshold)

        if isinstance(x_ticks, np.ndarray):
            axs[0][axs_col].set_xticks(x_ticks)

        if isinstance(x_tick_labels, np.ndarray):
            axs[0][axs_col].set_xticklabels(x_tick_labels)

        # add additional subplots to the figure based on the specified populations
        for i, pop in zip(np.arange(1, len(populations) + 1), populations):
            cell_table_marker_sub: pd.DataFrame = cell_table.loc[
                cell_table[pop_col] == pop, marker
            ].values
            axs[i][axs_col].hist(
                cell_table_marker_sub,
                50,
                density=True,
                facecolor='g',
                alpha=0.75,
                range=x_range
            )
            axs[i][axs_col].set_title("Distribution of {} in {}".format(marker, pop))
            axs[i][axs_col].axvline(x=threshold)

            if isinstance(x_ticks, np.ndarray):
                axs[i][axs_col].set_xticks(x_ticks)

            if isinstance(x_tick_labels, np.ndarray):
                axs[i][axs_col].set_xticklabels(x_tick_labels)

        # update axs_col to the next column
        axs_col += 1

    plt.tight_layout()

    # save the figure to save_dir
    fig.savefig(
        pathlib.Path(save_dir) / f"functional_marker_thresholds.png",
        dpi=300
    )


def functional_marker_thresholding(
    cell_table: pd.DataFrame, save_dir: Union[str, pathlib.Path],
    marker: str, populations: List[str], threshold: float,
    pop_col: str = "cell_meta_cluster", percentile: float = 0.999,
    x_range: Optional[Tuple[float, float]] = None, figsize: Optional[Tuple[float, float]] = None):
    """For a particular marker, visualize its distribution across the entire cohort, plus just 
    against the specified populations.

    Args:
        cell_table (pd.DataFrame):
            Cell table with clustered cell populations
        save_dir (Union[str, pathlib.Path]):
            The directory to save the marker distribution histograms
        marker (str):
            The marker to visualize the distributions for
        populations (List[str]):
            Additional populations to subset on for more distribution plots
        threshold (float):
            Value to plot a horizontal line for visualization, determined from Mantis
        pop_col (str):
            Column containing the names of the cell populations
        percentile (float):
            Cap used to control x axis limits of the plot
        x_range (Optional[Tuple[float, float]]):
            The range of x-values to visualize
        fig_size (Optional[Tuple[float, float]]):
            The figure size to use for the image.
            If None use default sizing (6.2, 2.2 * len(populations))
    """
    # verify save_dir is valid
    validate_paths([save_dir])

    # verify x_range is valid if set
    if x_range and (len(x_range) != 2 or x_range[0] >= x_range[1]):
        raise ValueError(
            "Invalid x_range: it must be in the form (low, high), low < high"
        )

    # verify figsize is valid if set
    if figsize and (len(figsize) != 2 or figsize[0] <= 0 or figsize[1] <= 0):
        raise ValueError(
            "Invalid figsize: it must be in the form (size_x, size_y), size_x > 0, size_y > 0"
        )

    # Make populations a list if it's str
    populations: List[str] = make_iterable(populations, ignore_str=True)

    # Verify that the marker and all populations specified are valid
    verify_in_list(
        specified_marker=marker,
        cell_table_columns=cell_table.columns.values
    )

    all_populations: np.ndarray = cell_table[pop_col].unique()
    verify_in_list(
        specified_populations=populations,
        cell_table_populations=all_populations
    )

    # define the subplot grid
    figsize = figsize if figsize else (6.2, 2.2 * len(populations))
    fig, axs = plt.subplots(1 + len(populations), 1, figsize=figsize, squeeze=False)

    # determine max value to show on histograms based on the specified percentile
    x_range = x_range if x_range else (0, np.quantile(cell_table[marker].values, percentile))

    # the first subplot should always be the distribution of the marker against all populations
    axs[0][axs_col].hist(
        cell_table[marker].values,
        50,
        density=True,
        facecolor='g',
        alpha=0.75,
        range=x_range
    )
    axs[0][axs_col].set_title("Distribution of {} in all populations".format(marker))
    axs[0][axs_col].axvline(x=threshold)

    # add additional subplots to the figure based on the specified populations
    for i, pop in zip(np.arange(1, len(populations) + 1), populations):
        cell_table_marker_sub: pd.DataFrame = cell_table.loc[
            cell_table[pop_col] == pop, marker
        ].values
        axs[i][axs_col].hist(
            cell_table_marker_sub,
            50,
            density=True,
            facecolor='g',
            alpha=0.75,
            range=x_range
        )
        axs[i][axs_col].set_title("Distribution of {} in {}".format(marker, pop))
        axs[i][axs_col].axvline(x=threshold)


cell_table = pd.read_csv(
    os.path.join(ANALYSIS_DIR, "combined_cell_table_normalized_cell_labels_updated.csv")
)
functional_marker_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "functional_marker_dist_thresholds")
if not os.path.exists(functional_marker_viz_dir):
    os.makedirs(functional_marker_viz_dir)

marker_info = {
    "CD45RO": {
        "populations": ["CD4T", "Fibroblast"],
        "threshold": 0.002,
        "x_range": (0, 0.02),
        "x_ticks": np.array([0, 0.005, 0.01, 0.015, 0.02]),
        "x_tick_labels": np.array([0, 0.005, 0.01, 0.015, 0.02]),
    },
    "CD38": {
        "populations": ["Endothelium", "Cancer_EMT"],
        "threshold": 0.004,
        "x_range": (0, 0.01),
        "x_ticks": np.array([0, 0.005, 0.01, 0.015, 0.02]),
        "x_tick_labels": np.array([0, 0.005, 0.01, 0.015, 0.02]),
    },
    "PDL1": {
        "populations": ["Cancer", "Stroma"],
        "threshold": 0.001,
        "x_range": (0, 0.003),
        "x_ticks": np.array([0, 0.001, 0.002, 0.003]),
        "x_tick_labels": np.array([0, 0.001, 0.002, 0.003]),
    }
}
functional_marker_thresholding_grid(
    cell_table, functional_marker_viz_dir, marker_info=marker_info,
    figsize=(25, 10)
)


# Feature extraction


