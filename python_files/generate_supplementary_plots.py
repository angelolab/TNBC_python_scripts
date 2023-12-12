# File with code for generating supplementary plots
import os
import shutil
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns


# Panel validation


# ROI selection


# QC


# Image processing


# Cell identification and classification



# Functional marker thresholding
def functional_marker_thresholding(
    cell_table: pd.DataFrame, populations, List[str], marker: str,
    pop_col: str = "cell_meta_cluster", threshold: Optional[float] = None,
    percentile: float = 0.999):
    pass


# Feature extraction


