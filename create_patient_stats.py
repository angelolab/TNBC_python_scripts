import os

import ark.utils.misc_utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ark.utils.io_utils import list_folders
from ark.utils.misc_utils import verify_same_elements

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'

total_df = pd.read_csv(os.path.join(data_dir, 'summary_df_timepoint.csv'))

x1 = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C']
y1 = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c']
x2 = ['A', 'B', 'C', 'D']
y2 = ['a', 'b', 'c', 'd']

df1 = pd.DataFrame({'letter': x1, 'metadata': y1})

df2 = pd.DataFrame({'letter': x2, 'metadata2': y2})

df1.merge(df2, on='letter')

