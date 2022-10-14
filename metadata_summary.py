import os
import json

import pandas as pd
import numpy as np
import seaborn as sns

base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'

# summary plots for metadata counting

# plot tissue sites of mets
plot_df = timepoint_table.loc[np.isin(timepoint_table['TONIC_ID'], primary_baseline), :]
plot_df = plot_df.loc[plot_df['Timepoint'] == 'baseline', :]

#plot_df = timepoint_table.loc[np.isin(timepoint_table['Timepoint'], ['baseline', 'post_induction', 'on_nivo', 'progression']), :]
p = sns.countplot(x='Localization', data=plot_df, order=['Lymphnode', 'Skin', 'Liver', 'Breast',
                                                         'Lung', 'Peritoneum', 'Stomach',
                                                         'Subcutaneous', 'Thoracal',
                                                         'Muscle'])
p.tick_params(labelsize=15)
p.set_ylabel('Number of patients', fontsize=15)
p.set_xlabel('Metastasis Location', fontsize=15)

