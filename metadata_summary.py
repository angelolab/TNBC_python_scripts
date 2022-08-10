import os

import pandas as pd
import numpy as np
import seaborn as sns

base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'

timepoint_table = pd.read_csv(os.path.join(base_dir, 'TONIC_data_per_timepoint.csv'))
timepoint_table = timepoint_table.loc[timepoint_table['On_TMA'] == 'Yes', :]

# get list of IDs with matching primary and baseline
primary_ids = timepoint_table.loc[np.isin(timepoint_table['Timepoint'], ['primary', 'biopsy']), 'TONIC_ID'].unique()

baseline_ids = timepoint_table.loc[timepoint_table['Timepoint'] == 'baseline', 'TONIC_ID'].unique()
induction_ids = timepoint_table.loc[timepoint_table['Timepoint'] == 'post_induction', 'TONIC_ID'].unique()
nivo_ids = timepoint_table.loc[timepoint_table['Timepoint'] == 'on_nivo', 'TONIC_ID'].unique()
ln_pos_ids = timepoint_table.loc[timepoint_table['Timepoint'] == 'lymphnode_pos', 'TONIC_ID'].unique()
ln_neg_ids = timepoint_table.loc[timepoint_table['Timepoint'] == 'lymphnode_neg', 'TONIC_ID'].unique()

primary_baseline = list(set(primary_ids).intersection(set(baseline_ids)))
baseline_induction = list(set(baseline_ids).intersection(set(induction_ids)))
baseline_nivo = list(set(baseline_ids).intersection(set(nivo_ids)))
baseline_induction_nivo = list(set(baseline_induction).intersection(set(nivo_ids)))
ln_both_ids = list(set(ln_neg_ids).intersection(set(ln_pos_ids)))

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

