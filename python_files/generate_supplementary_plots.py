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


def stitch_and_annotate_padded_img(image_data: xr.DataArray, padding: int = 25,
                                   font_size: int = 100):
    """Stitch an image with (c, x, y) dimensions, and annotate each image with labels contained
    in the xth dimension.

    Args:
        image_data (xr.DataArray):
            The image data to tile, should be 3D
        padding (int):
            Amount of padding to add around each channel in the stitched image
        font_size (int):
            The font size to use for annotations

    Returns:
        Image:
            The PIL image instance which contains the stitched (with padding) and annotated image
    """
    # param validation
    if padding <= 0:
        raise ValueError("padding must be a positive integer")
    if font_size <= 0:
        raise ValueError("font_size must be a positive integer")

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
    imdraw: ImageDraw = ImageDraw.Draw(stitched_image_im)
    imfont: ImageFont = ImageFont.truetype("Arial Unicode.ttf", 100)

    # annotate each channel
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
            if img_idx == len(channels):
                break

    return stitched_image_im


# Panel validation


# ROI selection


# QC


# Image processing


# Cell identification and classification
def stitch_before_after_norm(
    pre_norm_dir: Union[str, pathlib.Path], norm_dir: Union[str, pathlib.Path],
    save_dir: Union[str, pathlib.Path],
    run_name: str, channel: str, pre_norm_subdir: str = "", norm_subdir: str = "",
    padding: int = 25, font_size: int = 100):
    """Generates two stitched images: before and after normalization

    pre_norm_dir (Union[str, pathlib.Path]):
        The directory containing the run data before normalization
    norm_dir (Union[str, pathlib.Path]):
        The directory containing the run data after normalization
    save_dir (Union[str, pathlib.Path]):
        The directory to save both the pre- and post-norm tiled images
    run_name (str):
        The name of the run to tile, should be present in both pre_norm_dir and norm_dir
    channel (str):
        The name of the channel to tile inside run_name
    pre_norm_subdir (str):
        If applicable, the name of the subdirectory inside each FOV folder of the pre-norm data
    norm_subdir (str):
        If applicable, the name of the subdirectory inside each FOV folder of the norm data
    padding (int):
        Amount of padding to add around each channel in the stitched image
    font_size (int):
        The font size to use for annotations
    """
    # verify that the run_name specified appears in both pre and post norm folders
    all_pre_norm_runs: List[str] = list_folders(pre_norm_dir)
    all_norm_runs: List[str] = list_folders(norm_dir)
    verify_in_list(
        specified_run=run_name,
        all_pre_norm_runs=all_pre_runs
    )
    verify_in_list(
        specified_run=run_name,
        all_post_norm_runs=all_norm_runs
    )

    # verify save_dir is valid before defining the save paths
    validate_paths([save_dir])
    pre_norm_stitched_path: pathlib.Path = \
        pathlib.Path(save_dir) / f"{channel}_pre_norm_stitched.tiff"
    post_norm_stitched_path: pathlib.Path = \
        pathlib.Path(save_dir) / f"{channel}_post_norm_stitched.tiff"

    pre_norm_run_path: pathlib.Path = pathlib.Path(pre_norm_dir) / run_name
    norm_run_path: pathlib.Path = pathlib.Path(norm_dir) / run_name

    # get all the FOVs in natsorted order
    # NOTE: currently assumed that the FOVs are the same, since the runs are the same
    all_fovs: List[str] = ns.natsorted(list_folders(pre_norm_run_path))

    # load pre- and post-norm data in acquisition order, drop channel axis as it's 1-D
    pre_norm_data = load_imgs_from_tree(
        data_dir=pre_norm_dir, fovs=all_fovs, channels=[channel], img_sub_folder=pre_norm_subdir
    )[..., 0]
    post_norm_data = load_imgs_from_tree(
        data_dir=pre_norm_dir, fovs=all_fovs, channels=[channel], img_sub_folder=post_norm_subdir
    )[..., 0]

    # reassign coordinate with FOV names that don't contain "-scan-1" or additional dashes
    fovs_condensed: List[str] = [f"FOV{af.split('-')[1]}" for af in all_fovs]
    pre_norm_data = pre_norm_data.assign_coords({"fov": fovs_condensed})
    post_norm_data = post_norm_data.assign_coords({"fov": fovs_condensed})

    # generate and save the pre- and post-norm tiled images
    pre_norm_tiled: Image = stitch_and_annotate_padded_img(pre_norm_data, padding, font_size)
    post_norm_tiled: Image = stitch_and_annotate_padded_img(post_norm_data, padding, font_size)

    pre_norm_tiled.save(pre_norm_stitched_path)
    post_norm_tiled.save(post_norm_stitched_path)

run_name = "2022-01-14_TONIC_TMA2_run1"
pre_norm_dir = f"/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/rosetta"
post_norm_dir = f"/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/normalized/{run_name}"
save_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"

stitch_before_after_norm(pre_norm_dir, post_norm_dir, save_dir, run_name)


# Functional marker thresholding



# Feature extraction


