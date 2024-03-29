import math
import matplotlib.pyplot as plt
import natsort as ns
import numpy as np
import os
import pandas as pd
import pathlib
import xarray as xr
from os import PathLike
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, TypedDict, Union, Literal
from ark.utils import plot_utils, data_utils
from alpineer.io_utils import list_folders, list_files, remove_file_extensions, validate_paths
from alpineer.load_utils import load_imgs_from_tree, load_imgs_from_dir
from alpineer.misc_utils import verify_in_list
from toffy.image_stitching import rescale_images
from .utils import remove_ticks, QuantileNormalization, mask_erosion_ufunc
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
def stitch_before_after_rosetta(
    pre_rosetta_dir: Union[str, pathlib.Path], post_rosetta_dir: Union[str, pathlib.Path],
    save_dir: Union[str, pathlib.Path],
    run_name: str, fov_indices: Optional[List[int]], target_channel: str, source_channel: str = "Noodle",
    pre_rosetta_subdir: str = "", post_rosetta_subdir: str = "",
    img_size_scale: float = 0.5, percent_norm: Optional[float] = 99.999,
    padding: int = 25, font_size: int = 175, step: int = 1,
    save_separate: bool = False
):
    """Generates two stitched images: before and after Rosetta

    pre_rosetta_dir (Union[str, pathlib.Path]):
        The directory containing the run data before Rosetta
    post_rosetta_dir (Union[str, pathlib.Path]):
        The directory containing the run data after Rosetta
    save_dir (Union[str, pathlib.Path]):
        The directory to save both the pre- and post-Rosetta tiled images
    run_name (str):
        The name of the run to tile, should be present in both pre_rosetta_dir and post_rosetta_dir
    fov_indices (Optional[List[int]]):
        The list of indices to select. If None, use all.
    target_channel (str):
        The name of the channel to tile inside run_name
    source_channel (str):
        The name of the source channel that was subtracted from target_channel
    pre_rosetta_subdir (str):
        If applicable, the name of the subdirectory inside each FOV folder of the pre-Rosetta data
    post_rosetta_subdir (str):
        If applicable, the name of the subdirectory inside each FOV folder of the post-Rosetta data
    percent_norm (int):
        Percentile normalization param to enable easy visualization, if None then skips this step
    img_size_scale (float):
        Amount to scale down image. Set to None for no scaling
    padding (int):
        Amount of padding to add around each channel in the stitched image
    font_size (int):
        The font size to use for annotations
    step (int):
        The step size to use before adding an image to the tile
    save_separate (bool):
        If set, then save each FOV separately, otherwise save full tile
    """
    # verify that the run_name specified appears in both pre and post norm folders
    all_pre_rosetta_runs: List[str] = list_folders(pre_rosetta_dir)
    all_post_rosetta_runs: List[str] = list_folders(post_rosetta_dir)
    verify_in_list(
        specified_run=run_name,
        all_pre_norm_runs=all_pre_rosetta_runs
    )
    verify_in_list(
        specified_run=run_name,
        all_post_norm_runs=all_post_rosetta_runs
    )

    # verify save_dir is valid before defining the save paths
    validate_paths([save_dir])
    rosetta_stitched_path: pathlib.Path = \
        pathlib.Path(save_dir) / f"{run_name}_{target_channel}_pre_post_Rosetta.tiff"

    # define full paths to pre- and post-Rosetta data
    pre_rosetta_run_path: pathlib.Path = pathlib.Path(pre_rosetta_dir) / run_name
    post_rosetta_run_path: pathlib.Path = pathlib.Path(post_rosetta_dir) / run_name

    # get all the FOVs in natsorted order
    # NOTE: assumed that the FOVs are the same pre and post, since the run names are the same
    all_fovs: List[str] = ns.natsorted(list_folders(pre_rosetta_run_path))

    # load Noodle, pre-, and post-Rosetta data in acquisition order, drop channel axis as it's 1-D
    # ensure the pre-Rosetta Noodle is loaded
    noodle_data: xr.DataArray = load_imgs_from_tree(
        data_dir=pre_rosetta_run_path, fovs=all_fovs, channels=[source_channel],
        img_sub_folder=pre_rosetta_subdir, max_image_size=2048
    )[..., 0]
    pre_rosetta_data: xr.DataArray = load_imgs_from_tree(
        data_dir=pre_rosetta_run_path, fovs=all_fovs, channels=[target_channel],
        img_sub_folder=pre_rosetta_subdir, max_image_size=2048
    )[..., 0]
    post_rosetta_data: xr.DataArray = load_imgs_from_tree(
        data_dir=post_rosetta_run_path, fovs=all_fovs, channels=[target_channel],
        img_sub_folder=post_rosetta_subdir, max_image_size=2048
    )[..., 0]

    # divide pre-Rosetta by 200 to ensure same scale
    pre_rosetta_data = pre_rosetta_data / 200

    # reassign coordinate with FOV names that don't contain "-scan-1" or additional dashes
    fovs_condensed: np.ndarray = np.array([f"FOV{af.split('-')[1]}" for af in all_fovs])
    noodle_data = noodle_data.assign_coords({"fovs": fovs_condensed})
    pre_rosetta_data = pre_rosetta_data.assign_coords({"fovs": fovs_condensed})
    post_rosetta_data = post_rosetta_data.assign_coords({"fovs": fovs_condensed})

    # the top should be original, middle Noodle, bottom Rosetta-ed
    # NOTE: leave out Noodle row from dimensions for now for proper rescaling and percentile norm
    stitched_pre_post_rosetta: np.ndarray = np.zeros(
        (2048 * 2, 2048 * len(fovs_condensed))
    )
    for fov_i, fov_name in enumerate(fovs_condensed):
        # add the rescaled pre- and post-Rosetta images first
        stitched_pre_post_rosetta[
            0:2048, (2048 * fov_i):(2048 * (fov_i + 1))
        ] = pre_rosetta_data[fov_i, ...].values
        stitched_pre_post_rosetta[
            2048:4096, (2048 * fov_i):(2048 * (fov_i + 1))
        ] = post_rosetta_data[fov_i, ...].values

    # define the Noodle row
    stitched_noodle: np.ndarray = np.zeros((2048, 2048 * len(fovs_condensed)))
    for fov_i, fov_name in enumerate(fovs_condensed):
        stitched_noodle[
            :, (2048 * fov_i):(2048 * (fov_i + 1))
        ] = noodle_data[fov_i, ...].values

    # run percent normalization on Noodle data if specified
    if percent_norm:
        source_percentile: float = np.percentile(stitched_noodle, percent_norm)
        non_source_percentile: float = np.percentile(stitched_pre_post_rosetta, percent_norm)
        perc_ratio: float = source_percentile / non_source_percentile
        stitched_noodle = stitched_noodle / perc_ratio

    # combine the Noodle data with the stitched data, swap so that Noodle is in the middle
    stitched_pre_post_rosetta = np.vstack(
        (stitched_pre_post_rosetta[:2048, :], stitched_noodle, stitched_pre_post_rosetta[2048:, :])
    )

    # subset on just the FOV indices selected
    # NOTE: because of how percent norm works, better to run on all FOVs first to ensure brightness
    # as opposed to subsetting first, which often leads to dimmer images
    if fov_indices:
        indices_select = []
        for fi in fov_indices:
            indices_select.extend(list(np.arange(2048 * fi, 2048 * (fi + 1))))
        stitched_pre_post_rosetta = stitched_pre_post_rosetta[:, indices_select]

    if save_separate:
        # save each individual image separately
        for fov_i, fov_num in enumerate(fov_indices):
            stitched_rosetta_pil: Image = Image.fromarray(
                stitched_pre_post_rosetta[:, (2048 * fov_i):(2048 * (fov_i + 1))]
            )
            rosetta_stitched_path: pathlib.Path = \
                pathlib.Path(save_dir) / f"{target_channel}_image_{fov_num}.tiff"
            stitched_rosetta_pil.save(rosetta_stitched_path)

    else:
        # save the full stitched image
        stitched_rosetta_pil: Image = Image.fromarray(np.round(stitched_pre_post_rosetta, 3))
        stitched_rosetta_pil.save(rosetta_stitched_path)


def stitch_before_after_norm(
    pre_norm_dir: Union[str, pathlib.Path], post_norm_dir: Union[str, pathlib.Path],
    save_dir: Union[str, pathlib.Path], run_name: str,
    fov_indices: Optional[List[int]], channel: str, pre_norm_subdir: str = "",
    post_norm_subdir: str = "", padding: int = 25, font_size: int = 100, step: int = 1
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
    fov_indices (Optional[List[int]]):
        The list of indices to select. If None, use all.
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

    if fov_indices:
        pre_norm_data = pre_norm_data[fov_indices, ...]
        post_norm_data = post_norm_data[fov_indices, ...]

    # reassign coordinate with FOV names that don't contain "-scan-1" or additional dashes
    fovs_condensed: np.ndarray = np.array([f"FOV{af.split('-')[1]}" for af in all_fovs])
    if fov_indices:
        fovs_condensed = fovs_condensed[fov_indices]
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

class MembraneMarkersSegmentationPlot:
    def __init__(
        self,
        fov: str,
        image_data: PathLike,
        segmentation_dir: PathLike,
        membrane_channels: List[str],
        overlay_channels: str | List[str],
        q: tuple[float, float] = (0.05, 0.95),
        clip: bool = False,
        figsize: Tuple[int, int] = (12, 4),
        layout: Literal["constrained", "tight"] = None,
        image_type: Literal["png", "pdf", "svg"] = "pdf",
    ):
        """Creates a figure with two subplots, one for each membrane marker used for segmentation,
        and one for the overlay of the membrane and nuclear markers.

        Args
        ----------
        fov : str
            The name of the FOV to be plotted
        image_data : PathLike
            The directory containing the image data.
        segmentation_dir : PathLike
            The directory containing the segmentation data.
        membrane_channels : List[str]
            The names of the membrane markers to be plotted.
        overlay_channels : str | List[str]
            The overlay channels to be plotted, can be either "nuclear_channel",
            "membrane_channel", or both.
        overlay_cmap: str, optional
            The colormap to use for the overlay, by default "viridis_r"
        q : tuple[float, float], optional
            A tuple of quatiles where the smallest element is the minimum quantile
            and the largest element is the maximum percentile, by default (0.05, 0.95). Must
            be between 0 and 1 inclusive.
        clip : bool, optional
            If True, the normalized values are clipped to the range [0, 1], by default False
        figsize : Tuple[int, int], optional
            The size of the figure, by default (8, 4)
        layout : Literal["constrained", "tight"], optional
            The layout engine, defaults to None, by default None
        image_type : Literal["png", "pdf", "svg"], optional
            The file type to save the plot as, by default "pdf"
        """
        self.fov_name = fov
        self.membrane_channels = membrane_channels
        self.overlay_channels = overlay_channels
        self.figsize = figsize
        self.layout = layout
        self.n_chans = len(set(membrane_channels))
        self.q = q
        self.image_data = image_data
        self.seg_dir = segmentation_dir
        self.image_type = image_type
        self.clip = clip

        self.fig = plt.figure(figsize=figsize, layout=layout)
        self.subfigs = self.fig.subfigures(
            nrows=1, ncols=2, wspace=0.05, width_ratios=[1, 1]
        )

    def make_plot(self, save_dir: PathLike):
        """Plots the membrane markers and overlay and saves the figure to the specified directory.

        Args
        ----------
        save_dir : PathLike
            The directory to save the figure to.
        """
        self.fov_xr = load_imgs_from_tree(
            data_dir=self.image_data,
            fovs=[self.fov_name],
            channels=self.membrane_channels,
        )

        self.fov_overlay = plot_utils.create_overlay(
            fov=self.fov_name,
            segmentation_dir=self.seg_dir / "deepcell_output",
            data_dir=self.seg_dir / "deepcell_input",
            img_overlay_chans=self.overlay_channels,
            seg_overlay_comp="whole_cell",
        )

        self.fov_cell_segmentation = load_imgs_from_dir(
            data_dir=self.seg_dir / "deepcell_output",
            files=[f"{self.fov_name}_whole_cell.tiff"],
        )

        self.fig.suptitle(
            t=f"{self.fov_name} Membrane Markers and Segmentation", fontsize=8
        )

        self._plot_mem_markers()
        self._plot_overlay_segmentation()
        self.fig.savefig(
            fname=save_dir / f"{self.fov_name}_membrane_markers_overlay.{self.image_type}",
        )
        plt.close(self.fig)

    def _plot_mem_markers(self):
        self.subfigs[0].suptitle("Membrane Markers", fontsize=6)

        markers_subplots = self.subfigs[0].subplots(
            nrows=2,
            ncols=int(np.ceil((self.n_chans + 1) / 2)),
            sharex=True,
            sharey=True,
            gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        )

        channel_axes = markers_subplots.flat[: self.n_chans]

        self.subfigs[0].add_subplot

        for ax, channel in zip(channel_axes, ns.natsorted(self.membrane_channels)):
            chan_data = self.fov_xr.sel({"channels": channel}).squeeze()

            ax.imshow(
                X=chan_data,
                cmap="gray",
                norm=QuantileNormalization(q=self.q, clip=self.clip),
                interpolation="none",
                aspect="equal",
            )
            ax.set_title(channel, fontsize=6)
            remove_ticks(ax, "xy")

        ax_sum = markers_subplots.flat[self.n_chans]

        ax_sum.imshow(
            X=self.fov_xr.sum("channels").squeeze(),
            cmap="gray",
            norm=QuantileNormalization(q=self.q, clip=self.clip),
            interpolation="none",
            aspect="equal",
        )

        ax_sum.set_title("Sum", fontsize=6)
        remove_ticks(ax_sum, "xy")

        # Clean up and remove the empty subplots
        for ax in markers_subplots.flat[self.n_chans + 1 :]:
            ax.remove()

    def _plot_overlay_segmentation(self)-> None:
        cell_seg_ax, overlay_ax = self.subfigs[1].subplots(
            nrows=1, ncols=2, sharex=True, sharey=True
        )
        overlay_ax.set_title("Nuclear and Membrane Overlay", fontsize=6)
        overlay_ax.imshow(
            X=self.fov_overlay,
            norm=QuantileNormalization(q=self.q, clip=self.clip),
            interpolation="none",
            aspect="equal",
        )

        cell_seg_ax.set_title("Cell Segmentation", fontsize=6)
        cell_seg_ax.imshow(
            X=xr.apply_ufunc(
                mask_erosion_ufunc,
                self.fov_cell_segmentation.squeeze(),
                input_core_dims=[["rows", "cols"]],
                output_core_dims=[["rows", "cols"]],
                kwargs={"connectivity": 1, "mode": "thick"},
            ).pipe(lambda x: x.where(cond=x < 1, other=0.5)),
            cmap="grey",
            interpolation="none",
            aspect="equal",
            vmin=0,
            vmax=1,
        )

        remove_ticks(overlay_ax, "xy")
        remove_ticks(cell_seg_ax, "xy")


class SegmentationOverlayPlot:
    def __init__(
        self,
        fov: str,
        segmentation_dir: PathLike,
        overlay_channels: str | List[str],
        q: tuple[float, float] = (0.05, 0.95),
        clip: bool = False,
        figsize: tuple[float, float] =(8, 4),
        layout: Literal["constrained", "tight"] = "constrained",
        image_type: Literal["png", "pdf", "svg"] = "svg",
    ) -> None:
        """Creates a figure with two subplots, one for the cell segmentation and one for the overlay

        Parameters
        ----------
        fov : str
            The name of the FOV to be plotted
        segmentation_dir : PathLike
            The directory containing the segmentation data.
        overlay_channels : str | List[str]
            The overlay channels to be plotted, can be either "nuclear_channel",
        q : tuple[float, float], optional
            A tuple of quatiles where the smallest element is the minimum quantile
            and the largest element is the maximum percentile, by default (0.05, 0.95). Must
            be between 0 and 1 inclusive.
        clip : bool, optional
            If True, the normalized values are clipped to the range [0, 1], by default False
        figsize : Tuple[int, int], optional
            The size of the figure, by default (8, 4)
        layout : Literal["constrained", "tight"], optional
            The layout engine, defaults to None, by default None
        image_type : Literal["png", "pdf", "svg"], optional
            The file type to save the plot as, by default "pdf"
        """
        self.fov_name = fov
        self.seg_dir = segmentation_dir
        self.overlay_channels = overlay_channels
        self.q = q
        self.clip = clip
        self.figsize = figsize
        self.layout = layout
        self.image_type = image_type

        self.fig = plt.figure(figsize=self.figsize, layout=layout)

    def make_plot(self, save_dir: PathLike) -> None:
        self.fov_overlay = plot_utils.create_overlay(
            fov=self.fov_name,
            segmentation_dir=self.seg_dir / "deepcell_output",
            data_dir=self.seg_dir / "deepcell_input",
            img_overlay_chans=self.overlay_channels,
            seg_overlay_comp="whole_cell",
        )
        self.fov_cell_segmentation = load_imgs_from_dir(
            data_dir=self.seg_dir / "deepcell_output",
            files=[f"{self.fov_name}_whole_cell.tiff"],
        )
        self.fig.suptitle(
            t=f"{self.fov_name} Cell Segmentation and Overlay", fontsize=8
        )
        self._plot_overlay_segmentation()
        self.fig.savefig(
            save_dir / f"{self.fov_name}_segmentation_overlay.{self.image_type}"
        )
        
        plt.close(self.fig)

    def _plot_overlay_segmentation(self):
        cell_seg_ax, overlay_ax = self.fig.subplots(
            nrows=1, ncols=2, sharex=True, sharey=True
        )
        overlay_ax.set_title("Nuclear and Membrane Overlay", fontsize=6)
        overlay_ax.imshow(
            X=self.fov_overlay,
            norm=QuantileNormalization(q=self.q, clip=self.clip),
            interpolation="none",
            aspect="equal",
        )

        cell_seg_ax.set_title("Cell Segmentation", fontsize=6)
        cell_seg_ax.imshow(
            X=xr.apply_ufunc(
                mask_erosion_ufunc,
                self.fov_cell_segmentation.squeeze(),
                input_core_dims=[["rows", "cols"]],
                output_core_dims=[["rows", "cols"]],
                kwargs={"connectivity": 2, "mode": "thick"},
            ).pipe(lambda x: x.where(cond=x < 1, other=0.5)),
            cmap="grey",
            interpolation="none",
            aspect="equal",
            vmin=0,
            vmax=1,
        )

        remove_ticks(overlay_ax, "xy")
        remove_ticks(cell_seg_ax, "xy")
