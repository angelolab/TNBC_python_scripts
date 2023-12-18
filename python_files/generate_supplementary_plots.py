# File with code for generating supplementary plots
import math
import os
import pathlib
import shutil
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
import xarray as xr
from PIL import Image, ImageDraw, ImageFont

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
from alpineer.io_utils import list_folders, list_files, remove_file_extensions, validate_paths
from alpineer.load_utils import load_imgs_from_tree
from alpineer.misc_utils import verify_in_list


# Panel validation, must include: all channels tiled in a single image, images scaled down
def validate_panel(
    data_dir: Union[str, pathlib.Path], save_dir: Union[str, pathlib.Path], fov: str,
    channels: Optional[List[str]] = None, img_sub_folder: str = "", padding: int = 25,
    font_size: int = 100
):
    """Given a FOV in an image folder, stitch and annotate each channel

    Args:
        data_dir (Union[str, pathlib.Path]):
            The directory containing the image data for each FOV
        save_dir (Union[str, pathlib.Path]):
            The directory to save the stitched image
        fov (str):
            The name of the FOV to stitch
        channels (List[str]):
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
    print(f"All the channels are: {image_data.channels.values}")
    print(f"The total number of channels is: {len(image_data.channels.values)}")

    # normalize each channel by their 99.9% value, for clearer visualization
    image_data = image_data / image_data.quantile(0.999, dim=["rows", "cols"])

    # define the number of rows and columns
    num_cols: int = math.isqrt(len(channels))
    num_rows: int = math.ceil(len(channels) / num_cols)
    row_len: int = image_data.shape[0]
    col_len: int = image_data.shape[1]

    # create the blank image, start with a fully white slate
    stitched_image: np.ndarray = np.zeros(
        (
            num_rows * row_len + (num_rows - 1) * padding,
            num_cols * col_len + (num_rows - 1) * padding
        )
    )
    stitched_image.fill(255)

    # stitch the channels
    img_idx: int = 0
    for row in range(num_rows):
        for col in range(num_cols):
            stitched_image[
                (row * row_len + padding * row) : ((row + 1) * row_len + padding * row),
                (col * col_len + padding * col) : ((col + 1) * col_len + padding * col)
            ] = image_data[..., img_idx]
            img_idx += 1
            if img_idx == len(channels):
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


