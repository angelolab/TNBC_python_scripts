# File with code for generating supplementary plots
import math
import os
import pathlib
import shutil
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
import xarray as xr
from PIL import Image, ImageDraw, ImageFont

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
from alpineer.io_utils import list_folders, list_files, validate_paths
from alpineer.load_utils import load_imgs_from_tree
from alpineer.misc_utils import verify_in_list


# Panel validation, must include: all channels tiled in a single image, images scaled down
def validate_panel(
    data_dir: Union[str, pathlib.Path], fov: str, save_dir: Union[str, pathlib.Path],
    img_sub_folder: str = "", padding: int = 10
):
    """Given a FOV in an image folder, stitch and annotate each channel

    Args:
        data_dir (Union[str, pathlib.Path]):
            The directory containing the image data for each FOV
        fov (str):
            The name of the FOV to stitch
        save_dir (Union[str, pathlib.Path]):
            The directory to save the stitched image
        img_sub_folder (str):
            The sub folder name inside each FOV directory, set to "" if None
        padding (int):
            Amount of padding to add around each channel in the stitched image
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

    # load the data and get the channel names and image dimensions
    image_data: xr.DataArray = load_imgs_from_tree(
        data_dir=data_dir, fovs=[fov], img_sub_folder=img_sub_folder
    )[0, ...]
    channels: np.ndarray = image_data.channels.values

    # normalize each channel by their 99.9% value, for clearer visualization
    image_data = image_data / image_data.quantile(0.999, dim=["rows", "cols"])

    # define the number of rows and columns
    num_cols: int = math.isqrt(len(channels))
    num_rows: int = math.ceil(len(channels) / num_cols)
    row_len: int = test_img_data.shape[0]
    col_len: int = test_img_data.shape[1]

    # create the blank image, start with a fully white slate
    stitched_image: np.ndarray = np.zeros(
        (
            num_rows * row_len + (num_rows - 1) * padding,
            num_cols * col_len + (num_rows - 1) * padding
        )
    )
    stitched_image = stitched_image.fill(255)

    # stitch the channels
    img_idx: int = 0
    for row in range(num_rows):
        for col in range(num_cols):
            stitched_image[
                (row * row_len + padding * row) : ((row + 1) * row_len + padding * row),
                (col * col_len + padding * col) : ((col + 1) * col_len + padding * col)
            ] = data_xr[..., img_idx]
            img_idx += 1
            if img_idx == len(channels):
                break

    # define a draw instance for annotating the channel name
    stitched_image_im: Image = Image.fromarray(stitched_image)
    imdraw: ImageDraw = ImageDraw.Draw(stitched_image_im)
    imfont: ImageFont = ImageFont.truetype("Arial Unicode.ttf", 40)

    # annotate each channel
    img_idx = 0
    fill_value: int = np.max(stitched_image)
    for row in range(num_rows):
        for col in range(num_cols):
            imdraw.text(
                (row * img_dim[0] + padding * row, col * img_dim[1] + padding * col),
                channels[img_idx],
                font=imfont,
                fill=fill_value
            )
            img_idx += 1
            if img_idx == len(channels):
                break

    # save the stitched image
    stitched_image_im.save(stitched_img_path)


# ROI selection


# QC


# Image processing


# Cell identification and classification



# Functional marker thresholding



# Feature extraction


