# File with code for generating supplementary plots
import math
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import matplotlib
import xarray as xr

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns

import supplementary_plot_helpers

SUPPLEMENTARY_FIG_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"


# Panel validation


# ROI selection


# QC


# Image processing


# Cell identification and classification
acquisition_order_viz_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "acquisition_order")
if not os.path.exists(acquisition_order_viz_dir):
    os.makedirs(acquisition_order_viz_dir)

run_name = "2022-01-14_TONIC_TMA2_run1"
pre_norm_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/rosetta"
post_norm_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Acquisition/normalized"
save_dir = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"

# NOTE: image not scaled up, this happens in Photoshop
supplementary_plot_helpers.stitch_before_after_norm(
    pre_norm_dir, post_norm_dir, acquisition_order_viz_dir, run_name,
    "H3K9ac", pre_norm_subdir="normalized", padding=0, step=1
)


# Functional marker thresholding



# Feature extraction


