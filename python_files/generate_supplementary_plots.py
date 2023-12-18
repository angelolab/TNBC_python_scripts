# File with code for generating supplementary plots
import os
import pathlib
import shutil
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns

from alpineer.io_utils import validate_paths
from alpineer.misc_utils import make_iterable, verify_in_list

# Panel validation


# ROI selection


# QC


# Image processing


# Cell identification and classification



# Functional marker thresholding
def functional_marker_thresholding(
    cell_table: pd.DataFrame, save_dir: Union[str, pathlib.Path], marker: str,
    populations: List[str], threshold: float, pop_col: str = "cell_meta_cluster",
    percentile: float = 0.999, x_range: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None):
    """For a particular marker, visualize its distribution across the entire cohort, plus just 
    against the specified populations.

    Args:
        cell_table (pd.DataFrame):
            Cell table with clustered cell populations
        marker (str):
            The marker to visualize the distributions for
        save_dir (Union[str, pathlib.Path]):
            The directory to save the marker distribution histograms
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
    print(f"The x_range to use is: {x_range}")

    # the first subplot should always be the distribution of the marker against all populations
    axs[0][0].hist(
        cell_table[marker].values,
        50,
        density=True,
        facecolor='g',
        alpha=0.75,
        range=x_range
    )
    axs[0][0].set_title("Distribution of {} in all populations".format(marker))
    axs[0][0].axvline(x=threshold)

    # add additional subplots to the figure based on the specified populations
    for ax, pop in zip(axs[1:].flat, populations):
        cell_table_marker_sub: pd.DataFrame = cell_table.loc[
            cell_table[pop_col] == pop, marker
        ].values
        ax.hist(
            cell_table_marker_sub,
            50,
            density=True,
            facecolor='g',
            alpha=0.75,
            range=x_range
        )
        ax.set_title("Distribution of {} in {}".format(marker, pop))
        ax.axvline(x=threshold)

    plt.tight_layout()

    # save the figure to save_dir
    fig.savefig(
        pathlib.Path(save_dir) / f"{marker}_thresholds_{'_'.join(populations)}.png",
        dpi=300
    )

# Feature extraction


