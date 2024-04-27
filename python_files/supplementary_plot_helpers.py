import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import natsort as ns
import numpy as np
import os
import pandas as pd
import pathlib
from scipy.ndimage import gaussian_filter
import seaborn as sns
import skimage.io as io
from skimage.measure import label
from skimage import morphology
import xarray as xr
import utils as cancer_mask_utils

from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from alpineer.io_utils import list_folders, list_files, remove_file_extensions, validate_paths
from alpineer.load_utils import load_imgs_from_tree
from alpineer.misc_utils import verify_in_list


ACQUISITION_ORDER_INDICES = [
    11, 12, 13, 14, 15, 17, 18, 20, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 39, 40, 41, 42, 43, 44, 45, 46, 47
]


# generate stitching/annotation function, used by panel validation and acquisition order tiling
def stitch_and_annotate_padded_img(image_data: xr.DataArray, padding: int = 25,
                                   font_size: int = 100, annotate: bool = False,
                                   step: int = 1):
    """Stitch an image with (c, x, y) dimensions. If specified, annotate each image with labels
    contained in the cth dimension.

    Args:
        image_data (xr.DataArray):
            The image data to tile, should be 3D
        padding (int):
            Amount of padding to add around each channel in the stitched image
        font_size (int):
            The font size to use for annotations
        annotate (bool):
            Whether to annotate the images with labels in dimension c
        step (int):
            The step size to use before adding an image to the tile

    Returns:
        Image:
            The PIL image instance which contains the stitched (with padding) and annotated image
    """
    # param validation
    if padding < 0:
        raise ValueError("padding must be a non-negative integer")
    if font_size <= 0:
        raise ValueError("font_size must be a positive integer")

    images_to_select = np.arange(0, image_data.shape[0], step=step)
    image_data = image_data[images_to_select, ...]

    # define the number of rows and columns
    num_cols: int = math.isqrt(image_data.shape[0])
    num_rows: int = math.ceil(image_data.shape[0] / num_cols)
    row_len: int = image_data.shape[1]
    col_len: int = image_data.shape[2]

    # create the blank image, start with a fully white slate
    stitched_image: np.ndarray = np.zeros(
        (
            num_rows * row_len + (num_rows - 1) * padding,
            num_cols * col_len + (num_cols - 1) * padding
        )
    )
    stitched_image.fill(255)

    # retrieve the annotation labels
    annotation_labels: List[Any] = list(image_data.coords[image_data.dims[0]].values)

    # stitch the channels
    img_idx: int = 0
    for row in range(num_rows):
        for col in range(num_cols):
            stitched_image[
                (row * row_len + padding * row) : ((row + 1) * row_len + padding * row),
                (col * col_len + padding * col) : ((col + 1) * col_len + padding * col)
            ] = image_data[img_idx, ...]
            img_idx += 1
            if img_idx == len(annotation_labels):
                break

    # define a draw instance for annotating the channel name
    stitched_image_im: Image = Image.fromarray(stitched_image)

    # annotate with labels in c-axis if arg set
    if annotate:
        imdraw: ImageDraw = ImageDraw.Draw(stitched_image_im)
        imfont: ImageFont = ImageFont.truetype("Arial Unicode.ttf", font_size)

        img_idx = 0
        fill_value: int = np.max(stitched_image)
        for row in range(num_rows):
            for col in range(num_cols):
                imdraw.text(
                    (col * col_len + padding * col, row * row_len + padding * row),
                    annotation_labels[img_idx],
                    font=imfont,
                    fill=fill_value
                )
                img_idx += 1
                if img_idx == len(annotation_labels):
                    break

    return stitched_image_im


# panel validation helpers
def validate_panel(
    data_dir: Union[str, pathlib.Path], fov: str, save_dir: Union[str, pathlib.Path], 
    channels: Optional[List[str]] = None, img_sub_folder: str = "", padding: int = 10,
    font_size: int = 200
):
    """Given a FOV in an image folder, stitch and annotate each channel

    Args:
        data_dir (Union[str, pathlib.Path]):
            The directory containing the image data for each FOV
        fov (str):
            The name of the FOV to stitch
        save_dir (Union[str, pathlib.Path]):
            The directory to save the stitched image
        channels (Optional[List[str]]):
            The list of channels to tile. If None, uses all.
        img_sub_folder (str):
            The sub folder name inside each FOV directory, set to "" if None
        padding (int):
            Amount of padding to add around each channel in the stitched image
        font_size (int):
            The font size to use for annotations
    """
    # verify the FOV is valid
    all_fovs: List[str] = list_folders(data_dir)
    verify_in_list(
        specified_fov=fov,
        valid_fovs=all_fovs
    )

    # verify save_dir is valid before defining the save path
    validate_paths([save_dir])
    stitched_img_path: pathlib.Path = pathlib.Path(save_dir) / f"{fov}_channels_stitched.tiff"

    # validate the channels provided, or set to all if None
    all_channels = remove_file_extensions(list_files(os.path.join(data_dir, fov), substrs=".tiff"))
    if not channels:
        channels = all_channels
    verify_in_list(
        specified_channels=channels,
        valid_channels=all_channels
    )

    # sort the channels to ensure they get tiled in alphabetical order, regardless of case
    channels = sorted(channels, key=str.lower)

    # load the data and get the channel names and image dimensions
    image_data: xr.DataArray = load_imgs_from_tree(
        data_dir=data_dir, fovs=[fov], channels=channels, img_sub_folder=img_sub_folder
    )[0, ...]

    # normalize each channel by their 99.9% value, for clearer visualization
    image_data = image_data / image_data.quantile(0.999, dim=["rows", "cols"])

    # ensure the channels dimension is the 0th for annotation purposes
    image_data = image_data.transpose("channels", "rows", "cols")

    # generate the stitched image and save
    print(f"Font size is: {font_size}")
    panel_tiled: Image = stitch_and_annotate_padded_img(
        image_data, padding=padding, font_size=font_size, annotate=True
    )

    panel_tiled.save(stitched_img_path)


# functional marker thresholding helpers
class MarkerDict(TypedDict):
    populations: List[str]
    threshold: float
    x_range: Optional[Tuple[float, float]]
    x_ticks: Optional[np.ndarray]
    x_tick_labels: Optional[np.ndarray]


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


# acquisition order helpers
def stitch_before_after_norm(
    pre_norm_dir: Union[str, pathlib.Path], post_norm_dir: Union[str, pathlib.Path],
    save_dir: Union[str, pathlib.Path],
    run_name: str, channel: str, pre_norm_subdir: str = "", post_norm_subdir: str = "",
    padding: int = 25, font_size: int = 100, step: int = 1
):
    """Generates two stitched images: before and after normalization

    pre_norm_dir (Union[str, pathlib.Path]):
        The directory containing the run data before normalization
    post_norm_dir (Union[str, pathlib.Path]):
        The directory containing the run data after normalization
    save_dir (Union[str, pathlib.Path]):
        The directory to save both the pre- and post-norm tiled images
    run_name (str):
        The name of the run to tile, should be present in both pre_norm_dir and post_norm_dir
    channel (str):
        The name of the channel to tile inside run_name
    pre_norm_subdir (str):
        If applicable, the name of the subdirectory inside each FOV folder of the pre-norm data
    post_norm_subdir (str):
        If applicable, the name of the subdirectory inside each FOV folder of the post-norm data
    padding (int):
        Amount of padding to add around each channel in the stitched image
    font_size (int):
        The font size to use for annotations
    step (int):
        The step size to use before adding an image to the tile
    """
    # verify that the run_name specified appears in both pre and post norm folders
    all_pre_norm_runs: List[str] = list_folders(pre_norm_dir)
    all_post_norm_runs: List[str] = list_folders(post_norm_dir)
    verify_in_list(
        specified_run=run_name,
        all_pre_norm_runs=all_pre_norm_runs
    )
    verify_in_list(
        specified_run=run_name,
        all_post_norm_runs=all_post_norm_runs
    )

    # verify save_dir is valid before defining the save paths
    validate_paths([save_dir])
    pre_norm_stitched_path: pathlib.Path = \
        pathlib.Path(save_dir) / f"{run_name}_{channel}_pre_norm_stitched.tiff"
    post_norm_stitched_path: pathlib.Path = \
        pathlib.Path(save_dir) / f"{run_name}_{channel}_post_norm_stitched.tiff"

    pre_norm_run_path: pathlib.Path = pathlib.Path(pre_norm_dir) / run_name
    post_norm_run_path: pathlib.Path = pathlib.Path(post_norm_dir) / run_name

    # get all the FOVs in natsorted order
    # NOTE: assumed that the FOVs are the same pre and post, since the run names are the same
    all_fovs: List[str] = ns.natsorted(list_folders(pre_norm_run_path))

    # load pre- and post-norm data in acquisition order, drop channel axis as it's 1-D
    pre_norm_data: xr.DataArray = load_imgs_from_tree(
        data_dir=pre_norm_run_path, fovs=all_fovs, channels=[channel],
        img_sub_folder=pre_norm_subdir, max_image_size=2048
    )[..., 0]
    post_norm_data: xr.DataArray = load_imgs_from_tree(
        data_dir=post_norm_run_path, fovs=all_fovs, channels=[channel],
        img_sub_folder=post_norm_subdir, max_image_size=2048
    )[..., 0]

    pre_norm_data = pre_norm_data[ACQUISITION_ORDER_INDICES, ...]
    post_norm_data = post_norm_data[ACQUISITION_ORDER_INDICES, ...]

    # reassign coordinate with FOV names that don't contain "-scan-1" or additional dashes
    fovs_condensed: np.ndarray = np.array([f"FOV{af.split('-')[1]}" for af in all_fovs])
    fovs_condensed = fovs_condensed[ACQUISITION_ORDER_INDICES]
    pre_norm_data = pre_norm_data.assign_coords({"fovs": fovs_condensed})
    post_norm_data = post_norm_data.assign_coords({"fovs": fovs_condensed})

    # generate and save the pre- and post-norm tiled images
    pre_norm_tiled: Image = stitch_and_annotate_padded_img(
        pre_norm_data, padding, font_size, step=step
    )
    post_norm_tiled: Image = stitch_and_annotate_padded_img(
        post_norm_data, padding, font_size, step=step
    )

    pre_norm_tiled.save(pre_norm_stitched_path)
    post_norm_tiled.save(post_norm_stitched_path)


def run_functional_marker_positivity_tuning_tests(
    cell_table: pd.DataFrame, save_dir: Union[str, pathlib.Path],
    marker_info: Dict[str, MarkerDict], threshold_mults: List[float]
):
    """At different multipliers of each functional marker threshold, visualize across all how they change
    using a line plot

    Args:
        cell_table (pd.DataFrame):
            Cell table with clustered cell populations
        save_dir (Union[str, pathlib.Path]):
            The directory to save the line plots showing changes in number of positive cells
            per functional marker
        marker_info (str):
            For each marker, define the populations, threshold, x-range, and x-tick locations
            NOTE: used as convenience with `functional_marker_thresholding`, but only threshold value needed
        threshold_mults (List[str]):
            The multipliers to use and visualize across all markers
    """
    marker_threshold_data = {}

    for marker in marker_info:
        marker_threshold_data[marker] = {}

        for threshold in threshold_mults:
            multiplied_threshold = marker_info[marker]["threshold"] * threshold
            marker_threshold_data[marker][threshold] = {
                "multiplied_threshold": multiplied_threshold,
                "num_positive_cells": np.sum(cell_table[marker].values >= multiplied_threshold),
                "num_positive_cells_norm": np.sum(
                    cell_table[marker].values >= multiplied_threshold
                ) / np.sum(
                    cell_table[marker].values >= marker_info[marker]["threshold"]
                )
            }

    # # plot the num positive cells normalized by 1x per marker
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # threshold_mult_strs = [str(np.round(np.log2(tm), 3)) for tm in threshold_mults]

    # for i, marker in enumerate(marker_threshold_data):
    #     mult_data = [mtd["num_positive_cells_norm"] for mtd in marker_threshold_data[marker].values()]
    #     _ = ax.plot(threshold_mult_strs, mult_data, color="gray", label=marker)

    # _ = ax.set_title(
    #     f"Positive cells per threshold, normalized by 1x",
    #     fontsize=11
    # )
    # _ = ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    # _ = ax.yaxis.get_major_formatter().set_scientific(False)
    # _ = ax.set_xlabel("log2(threshold multiplier)", fontsize=7)
    # _ = ax.set_ylabel("Positive cell counts, normalized by 1x", fontsize=7)
    # _ = ax.tick_params(axis="both", which="major", labelsize=7)

    # # save the figure to save_dir
    # _ = fig.savefig(
    #     pathlib.Path(extraction_pipeline_tuning_dir) / f"functional_marker_threshold_experiments_norm.png",
    #     dpi=300
    # )

    # plot the raw num positive cells per marker
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    threshold_mult_strs = [str(np.round(np.log2(tm), 3)) for tm in threshold_mults]

    for i, marker in enumerate(marker_threshold_data):
        mult_data = [mtd["num_positive_cells"] for mtd in marker_threshold_data[marker].values()]
        _ = ax.plot(threshold_mult_strs, mult_data, color="gray", label=marker)

    _ = ax.set_title(
        f"Positive cells per threshold",
        fontsize=11
    )
    _ = ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    _ = ax.yaxis.get_major_formatter().set_scientific(False)
    _ = ax.set_xlabel("log2(threshold multiplier)", fontsize=7)
    _ = ax.set_ylabel("Positive cell counts", fontsize=7)
    _ = ax.tick_params(axis="both", which="major", labelsize=7)

    # save the figure to save_dir
    _ = fig.savefig(
        pathlib.Path(save_dir) / f"functional_marker_threshold_experiments.png",
        dpi=300
    )


def run_min_cell_feature_gen_fovs_dropped_tests(
    cluster_broad_df: pd.DataFrame, min_cell_params: List[int],
    compartments: List[str], metrics: List[str], save_dir: Union[str, pathlib.Path]
):
    """For the feature generation, visualize a strip plot of how varying min_cell thresholds
    determine which FOVs get dropped

    Args:
        cluster_broad_df (pd.DataFrame):
            The data in `"cluster_core_per_df.csv"`. While the process for functional, morph,
            diversity, and distance varies, the way the FOVs are deselected is the same across all
            since this dataset is always used.
        min_cell_params (List[int]):
            The min_cell thresholds to use
        compartments (List[str]):
            The compartments to visualize
        metrics (List[str]):
            The specific metric to facet the strip plot on (ex. `"cluster_count_broad"`)
        save_dir (Union[str, pathlib.Path]):
            The directory to save the strip plot(s)
    """
    total_fovs_dropped = {}
    for metric in metrics:
        total_fovs_dropped[metric] = {}

    for compartment in compartments:
        for min_cells in min_cell_params:
            for metric in metrics:
                total_fovs_dropped[metric][min_cells] = {}
                count_df = cluster_broad_df[cluster_broad_df.metric == metric]
                count_df = count_df[count_df.subset == compartment]
                all_fovs = count_df.fov.unique()

                for cell_type in count_df.cell_type.unique():
                    keep_df = count_df[count_df.cell_type == cell_type]
                    keep_df = keep_df[keep_df.value >= min_cells]
                    keep_fovs = keep_df.fov.unique()
                    total_fovs_dropped[metric][min_cells][cell_type] = \
                        len(all_fovs) - len(keep_fovs)

        for metric in metrics:
            # visualize a strip plot of FOVs dropped per min_cell test per cluster broad assignment
            fovs_dropped_dict = {metric: total_fovs_dropped[metric]}
            df = pd.DataFrame(
                [
                    {
                        "min_cells": min_cells,
                        "num_fovs_dropped": value,
                        "feature": metric,
                        "cell_type": cell_type
                    }
                    for feature, min_cells_dict in fovs_dropped_dict.items()
                    for min_cells, cell_types in min_cells_dict.items()
                    for cell_type, value in cell_types.items()
                ]
            )
            plot = sns.catplot(
                x="min_cells",
                y="num_fovs_dropped",
                hue="cell_type",
                data=df,
                kind="strip",
                palette="Set2",
                dodge=False
            )
            plot.fig.subplots_adjust(top=0.9)
            plot.fig.suptitle("Distribution of FOVs dropped across min_cells trials")
            plt.savefig(
                pathlib.Path(save_dir) /
                f"{compartment}_min_cells_{metric}_fovs_dropped_stripplot.png",
                dpi=300
            )


def run_cancer_mask_inclusion_tests(
    cell_table_clusters: pd.DataFrame, channel_dir: pathlib.Path, seg_dir: pathlib.Path,
    threshold_mults: List[float], save_dir: Union[str, pathlib.Path], base_sigma: int = 10,
    base_channel_thresh: float = 0.0015, base_min_mask_size: int = 7000,
    base_max_hole_size: int = 1000, base_border_size: int = 50
):
    """Create box plots showing how much of the intermediate cancer mask in `create_cancer_boundary`
    after Gaussian blurring, channel thresholding, small object removal, and hole filling remain
    at individual tuning of these params.

    Finally, create box plot showing how much of the cancer mask gets defined as cancer or stroma 
    boundary at different border_size params. Note that the default sigma,  channel threhsolding, 
    small object removal, and hole filling params are used

    Args:
        cell_table_clusters (pd.DataFrame):
            The data contained in "cell_table_clusters.csv"
        channel_dir (pathlib.Path):
            The path to the "samples" folder containing all the data
        seg_dir (pathlib.Path):
            The path to the segmentation data
        threshold_mults (List[float]):
            What value to multiply the base params to test in this function
        save_dir (Union[str, pathlib.Path]):
            The directory to save the box plots
        base_sigma (int):
            The sigma value currently used for Gaussian blurring
        base_channel_thresh (float):
            The base threshold value to use after Gaussian blurring
        base_min_mask_size (int):
            The base minimum value for what objects get kept in the image
        base_max_hole_size (int):
            The base maximum size of a hole that doesn't get filled in the image
        base_border_size (int):
            The base border size to use for erosion and dilation that defines the boundaries
    """
    folders = list_folders(channel_dir)
    threshold_mult_strs = [str(np.round(np.log2(tm), 3)) for tm in threshold_mults]

    cell_boundary_sigmas = [int(tm * base_sigma) for tm in threshold_mults]
    cell_boundary_channel_threshes = [tm * base_channel_thresh for tm in threshold_mults]
    cell_boundary_min_mask_sizes = [int(tm * base_min_mask_size) for tm in threshold_mults]
    # cell_boundary_max_hole_sizes = [int(tm * max_hole_size) for tm in threshold_mults]
    cell_boundary_border_sizes = [int(tm * base_border_size) for tm in threshold_mults]

    cell_boundary_sigma_data = {s: [] for s in cell_boundary_sigmas}
    cell_boundary_channel_thresh_data = {ct: [] for ct in cell_boundary_channel_threshes}
    cell_boundary_min_mask_size_data = {mms: [] for mms in cell_boundary_min_mask_sizes}
    # cell_boundary_max_hole_size_data = {mhs: [] for mhs in cell_boundary_max_hole_sizes}
    cell_boundary_border_size_data = {bs: [] for bs in cell_boundary_border_sizes}

    i = 0
    for folder in folders:
        ecad = io.imread(os.path.join(channel_dir, folder, "ECAD.tiff"))

        # generate cancer/stroma mask by combining segmentation mask with ECAD channel
        seg_label = io.imread(os.path.join(seg_dir, folder + "_whole_cell.tiff"))[0]
        seg_mask = cancer_mask_utils.create_cell_mask(
            seg_label, cell_table_clusters, folder, ["Cancer"]
        )

        for s in cell_boundary_sigmas:
            img_smoothed = gaussian_filter(ecad, sigma=s)
            img_mask = img_smoothed > base_channel_thresh

            # clean up mask prior to analysis
            img_mask = np.logical_or(img_mask, seg_mask)
            label_mask = label(img_mask)
            label_mask = morphology.remove_small_objects(
                label_mask, min_size=base_min_mask_size
            )
            label_mask = morphology.remove_small_holes(
                label_mask, area_threshold=base_max_hole_size
            )

            percent_hit = np.sum(label_mask) / label_mask.size
            cell_boundary_sigma_data[s].append(percent_hit)

        img_smoothed = gaussian_filter(ecad, sigma=base_sigma)
        for ct in cell_boundary_channel_threshes:
            img_mask = img_smoothed > ct

            # clean up mask prior to analysis
            img_mask = np.logical_or(img_mask, seg_mask)
            label_mask = label(img_mask)
            label_mask = morphology.remove_small_objects(
                label_mask, min_size=base_min_mask_size)
            label_mask = morphology.remove_small_holes(
                label_mask, area_threshold=base_max_hole_size
            )

            percent_hit = np.sum(label_mask) / label_mask.size
            cell_boundary_channel_thresh_data[ct].append(percent_hit)

        img_smoothed = gaussian_filter(ecad, sigma=base_sigma)
        img_mask = img_smoothed > base_channel_thresh
        img_mask = np.logical_or(img_mask, seg_mask)
        label_mask_base = label(img_mask)
        for mms in cell_boundary_min_mask_sizes:
            label_mask = morphology.remove_small_objects(
                label_mask_base, min_size=mms
            )
            label_mask = morphology.remove_small_holes(
                label_mask, area_threshold=base_max_hole_size
            )

            percent_hit = np.sum(label_mask) / label_mask.size
            cell_boundary_min_mask_size_data[mms].append(percent_hit)

        # for mhs in cell_boundary_max_hole_sizes:
        #     label_mask = morphology.remove_small_objects(
        #         label_mask_base, min_size=base_max_hole_size
        #     )
        #     label_mask = morphology.remove_small_holes(
        #         label_mask, area_threshold=mhs
        #     )

        #     percent_hit = np.sum(label_mask) / label_mask.size
        #     cell_boundary_max_hole_size_data[mhs].append(percent_hit)

        label_mask = morphology.remove_small_objects(
            label_mask_base, min_size=base_min_mask_size
        )
        label_mask = morphology.remove_small_holes(
            label_mask, area_threshold=base_max_hole_size
        )
        for bs in cell_boundary_border_sizes:
            # define external borders
            external_boundary = morphology.binary_dilation(label_mask)

            for _ in range(bs):
                external_boundary = morphology.binary_dilation(external_boundary)

            external_boundary = external_boundary.astype(int) - label_mask.astype(int)
            # create interior borders
            interior_boundary = morphology.binary_erosion(label_mask)

            for _ in range(bs):
                interior_boundary = morphology.binary_erosion(interior_boundary)

            interior_boundary = label_mask.astype(int) - interior_boundary.astype(int)

            combined_mask = np.ones_like(ecad)
            combined_mask[label_mask] = 4
            combined_mask[external_boundary > 0] = 2
            combined_mask[interior_boundary > 0] = 3

            percent_border = np.sum(
                (combined_mask == 2) | (combined_mask == 3)
            ) / combined_mask.size
            cell_boundary_border_size_data[bs].append(percent_border)

        i += 1
        if i % 10 == 0:
            print(f"Processed {i} folders")

    # plot the sigma experiments
    data_sigma = []
    labels_sigma = []
    for i, (_, values_s) in enumerate(cell_boundary_sigma_data.items()):
        data_sigma.extend(values_s)
        labels_sigma.extend([threshold_mult_strs[i]] * len(values_s))

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=labels_sigma, y=data_sigma)
    plt.title("Distribution of % mask included in cancer across sigma (1x = 10)")
    plt.xlabel("log2(sigma multiplier)")
    plt.ylabel("% of mask included in cancer")
    plt.savefig(
        pathlib.Path(save_dir) /
        f"sigma_cancer_mask_inclusion_box.png",
        dpi=300
    )

    # plot the channel thresh experiments
    data_channel_thresh = []
    labels_channel_thresh = []
    for i, (_, values_ct) in enumerate(cell_boundary_channel_thresh_data.items()):
        data_channel_thresh.extend(values)
        labels_channel_thresh.extend([threshold_mult_strs[i]] * len(values_ct))

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=labels_channel_thresh, y=data_channel_thresh)
    plt.title("Distribution of % mask included in cancer across smoothing thresholds (1x = 0.0015)")
    plt.xlabel("log2(smooth thresh multiplier)")
    plt.ylabel("% of mask included in cancer")
    plt.savefig(
        pathlib.Path(save_dir) /
        f"smooth_thresh_cancer_mask_inclusion_box.png",
        dpi=300
    )

    # plot the min mask size experiments
    data_min_mask_size = []
    labels_min_mask_size = []
    for i, (_, values_mms) in enumerate(cell_boundary_min_mask_size_data.items()):
        data_min_mask_size.extend(values_mms)
        labels_min_mask_size.extend([threshold_mult_strs[i]] * len(values_mms))

    # Creating the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=labels_min_mask_size, y=data_min_mask_size)
    plt.title("Distribution of % mask included in cancer across min mask sizes (1x = 7000)")
    plt.xlabel("log2(min mask size multiplier)")
    plt.ylabel("% of mask included in cancer")
    plt.savefig(
        pathlib.Path(save_dir) /
        f"min_mask_size_cancer_mask_inclusion_box.png",
        dpi=300
    )

    # # plot the max hole size experiments
    # data_max_hole_size = []
    # labels_max_hole_size = []
    # for i, (_, values_mhs) in enumerate(cell_boundary_max_hole_size_data.items()):
    #     data_max_hole_size.extend(values_mhs)
    #     labels_max_hole_size.extend([threshold_mult_strs[i]] * len(values_mhs))

    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x=labels_max_hole_size, y=data_max_hole_size)
    # plt.title("Distribution of % mask included in cancer across max hole sizes (1x = 1000)")
    # plt.xlabel("log2(max hole size multiplier)")
    # plt.ylabel("% of mask included in cancer")
    # plt.savefig(
    #     pathlib.Path(save_dir) /
    #     f"max_hole_size_cancer_mask_inclusion_box.png",
    #     dpi=300
    # )

    # plot the border size experiments
    data_border_size = []
    labels_border_size = []
    for i, (_, values_bs) in enumerate(cell_boundary_border_size_data.items()):
        data_border_size.extend(values)
        labels_border_size.extend([threshold_mult_strs[i]] * len(values_bs))

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=labels_border_size, y=data_border_size)
    plt.title('Distribution of % cancer boundary across border sizes (1x = 50)')
    plt.xlabel('log2(border size multiplier)')
    plt.ylabel('% of mask identified as cancer boundary')
    plt.savefig(
        pathlib.Path(save_dir) / f"border_size_cancer_region_percentages_box.png",
        dpi=300
    )
