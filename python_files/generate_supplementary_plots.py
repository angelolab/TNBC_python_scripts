# File with code for generating supplementary plots
import math
import os
import pathlib
import shutil
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
import natsort as ns
import xarray as xr
from PIL import Image, ImageDraw, ImageFont

from alpineer.io_utils import list_folders, list_files, remove_file_extensions, validate_paths
from alpineer.load_utils import load_imgs_from_tree
from alpineer.misc_utils import verify_in_list

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns

SUPPLEMENTARY_FIG_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"
ACQUISITION_INDICES = [
    11, 12, 13, 14, 15, 17, 18, 20, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 39, 40, 41, 42, 43, 44, 45, 46, 47
]


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


# Panel validation


# ROI selection


# QC


# Image processing


# Cell identification and classification
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
        pathlib.Path(save_dir) / f"{run_name}_{channel}_pre_norm_stitched_{step}.tiff"
    post_norm_stitched_path: pathlib.Path = \
        pathlib.Path(save_dir) / f"{run_name}_{channel}_post_norm_stitched_{step}.tiff"

    pre_norm_run_path: pathlib.Path = pathlib.Path(pre_norm_dir) / run_name
    post_norm_run_path: pathlib.Path = pathlib.Path(post_norm_dir) / run_name

    # get all the FOVs in natsorted order
    # NOTE: assumed that the FOVs are the same pre and post, since the run names are the same
    all_fovs: List[str] = ns.natsorted(list_folders(pre_norm_run_path))

    # load pre- and post-norm data in acquisition order, drop channel axis as it's 1-D
    pre_norm_data = load_imgs_from_tree(
        data_dir=pre_norm_run_path, fovs=all_fovs, channels=[channel],
        img_sub_folder=pre_norm_subdir, max_image_size=2048
    )[..., 0]
    post_norm_data = load_imgs_from_tree(
        data_dir=post_norm_run_path, fovs=all_fovs, channels=[channel],
        img_sub_folder=post_norm_subdir, max_image_size=2048
    )[..., 0]

    pre_norm_data = pre_norm_data[ACQUISITION_INDICES, ...]
    post_norm_data = post_norm_data[ACQUISITION_INDICES, ...]

    # reassign coordinate with FOV names that don't contain "-scan-1" or additional dashes
    fovs_condensed: List[str] = [f"FOV{af.split('-')[1]}" for af in all_fovs]
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

acquisition_order_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "acquisition_order_test_step")
if not os.path.exists(acquisition_order_viz_dir):
    os.makedirs(acquisition_order_viz_dir)

run_name = "2022-01-14_TONIC_TMA2_run1"
pre_norm_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/rosetta"
post_norm_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/normalized"
save_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"

for step in np.arange(1, 5):
    stitch_before_after_norm(
        pre_norm_dir, post_norm_dir, acquisition_order_viz_dir, run_name,
        "H3K9ac", pre_norm_subdir="normalized", padding=0, step=step
    )
    stitch_before_after_norm(
        pre_norm_dir, post_norm_dir, acquisition_order_viz_dir, run_name,
        "H3K27me3", pre_norm_subdir="normalized", padding=0, step=step
    )


# Functional marker thresholding



# Feature extraction


