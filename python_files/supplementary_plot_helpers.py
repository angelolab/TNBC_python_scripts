import itertools
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import natsort as ns
import numpy as np
import pandas as pd
import pathlib
import xarray as xr

from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from alpineer.io_utils import list_folders, list_files, remove_file_extensions, validate_paths
from alpineer.load_utils import load_imgs_from_tree
from alpineer.misc_utils import verify_in_list


ACQUISITION_ORDER_INDICES = [
    11, 12, 13, 14, 15, 17, 18, 20, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 39, 40, 41, 42, 43, 44, 45, 46, 47
]


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
        imfont: ImageFont = ImageFont.truetype("Arial Unicode.ttf", 100)

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


# occupancy statistic helpers
def visualize_occupancy_statistics(
    cell_table: pd.DataFrame, save_dir: Union[str, pathlib.Path], max_image_size: int = 2048,
    tiles_per_row_col: int = 8, positive_threshold: int = 5
):
    """Define occupancy statistics over a cohort, and visualize distribution of positive
    tile counts across FOV.

    Args:
        cell_table (pd.DataFrame):
            The cell table associated with the cohort, need to contain all FOVs
        save_dir (Union[str, pathlib.Path]):
            Directory to save the visualizations in
        max_image_size (int):
            The max image size to load in
        tiles_per_row_col (int):
            The row/col dims of tiles to define over each image
        positive_threshold (int):
            The cell count in a tile required for positivity
    """
    # get all the FOV names
    fov_names: np.ndarray = cell_table["fov"].unique()

    # define the tile size in pixels
    tile_size: int = max_image_size // tiles_per_row_col

    # define the occupancy_stats array
    occupancy_stats: xr.DataArray = xr.DataArray(
        np.zeros((len(fov_names), tile_size, tile_size), dtype=np.int16),
        coords=[fov_names, np.zeros(tile_size), np.zeros(tile_size)],
        dims=["fov", "x", "y"]
    )

    # for each FOV, update the corresponding tile count
    for fov in fov_names:
        cell_table_fov: pd.DataFrame = cell_table[cell_table["fov"] == fov].copy()
        cell_table_fov["tile_row"] = cell_table_fov["centroid-1"] // tile_size
        cell_table_row["tile_col"] = cell_table_fov["centroid-0"] // tile_size
        fov_occupancy_table: pd.DataFrame = cell_table_fov.groupby(
            ["tile_row", "tile_col"]
        ).size().reset_index(name="occupancy")
        occupancy_stats.loc[
            fov, fov_occupancy_table["tile_row"].values, fov_occupancy_table["tile_col"].values
        ] += fov_occupancy_table["occupancy"].values

    # define the tiles that are positive based on threshold
    occupancy_stats_positivity: xr.DataArray = occupancy_stats > positive_threshold

    # count number of positive tiles per FOV
    fov_positive_tiles: Dict[str, int] = occupancy_stats_positivity.sum(
        dim=["tile_row", "tile_col"]
    ).to_dict()["data"]

    # 1. visualize histogram of positive tile counts across cohort
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(10, 10)
    )
    axs[0].hist(list(fov_positive_tiles.values()))

    # 2. visualize heatmap of average positivity of tiles across FOVs
    cmap_positivity = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"])
    c_positivity = axs[1].imshow(
        occupancy_stats_positivity.mean(dim="fov").values,
        cmap=cmap_positivity,
        aspect="auto",
        origin="lower"
    )
    for x, y in itertools.pairwise(occupancy_stats_positivity.tile_row.values):
        axs[1].text(y, x, f"R{x}C{y}", ha="center", va="center", color="black")
    fig.colorbar(c_positivity, ax=axs[1])

    # 3. visualize heatmap of average number of counts across FOVs
    cmap_counts = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"])
    c_counts = axs[2].imshow(
        occupancy_stats.mean(dim="fov").values,
        cmap=cmap_counts,
        aspect="auto"
    )
    for x, y in itertools.pairwise(occupancy_stats.tile_row.values):
        axs[2].text(y, x, f"R{x}C{y}", ha="center", va="center", color="black")
    fig.colorbar(c_positivity, ax=axs[2])

    # save the figure to save_dir
    fig.savefig(
        pathlib.Path(save_dir) / f"occupancy_stats_{tiles_per_row_col}_{positive_threshold}.png",
        dpi=300
    )
