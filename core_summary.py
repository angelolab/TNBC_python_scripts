import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dataset
core_df = pd.read_csv(os.path.join(data_dir, 'summary_df_core.csv'))
core_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))

core_metadata = core_metadata.loc[:, ['Tissue_ID', 'fov']]
core_df = core_df.merge(core_metadata, on='Tissue_ID')

plot_df = core_df.loc[core_df.metric == 'cluster_broad_freq', :]
plot_df.loc[plot_df.fov == 'TONIC_TMA7_R7C3', :]

timepoint_metadata.loc[timepoint_metadata.Tissue_ID == 'T18-60536', ['Timepoint', 'Localization']]

test_df = pd.DataFrame({'fov': [1, 1, 1, 1, 2, 2, 2, 2],
                        'celltype': ['bcell', 'tcell', 'mac', 'endo',
                                     'bcell', 'tcell', 'mac', 'endo'],
                        'mean': [5, 10, 15, 20, 6, 11, 16, 21]})
grouped = test_df.groupby('fov')

for name, group in grouped:
    print(name)
    print(group)
    print(group['mean'])


np.linalg.norm(x1 - x2)
