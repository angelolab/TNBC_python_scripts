import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr, ttest_ind, ttest_rel

from python_files.utils import find_conserved_features

plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
timepoint_features = pd.read_csv(os.path.join(data_dir, 'timepoint_features_no_compartment.csv'))
evolution_cats = ['primary__baseline', 'baseline__post_induction', 'baseline__on_nivo', 'post_induction__on_nivo']
timepoint_features = timepoint_features.merge(harmonized_metadata[['Tissue_ID', 'Timepoint', 'Localization', 'Patient_ID'] + evolution_cats].drop_duplicates(), on='Tissue_ID', how='left')

evolution_dfs = []
# generate evolution df based on difference in timepoints
for evolution_col in evolution_cats:
    timepoint_1, timepoint_2 = evolution_col.split('__')
    evolution_df = timepoint_features[timepoint_features[evolution_col]].copy()
    evolution_df = evolution_df.loc[evolution_df.Timepoint.isin([timepoint_1, timepoint_2])]

    # get the paired features
    evolution_df_wide = evolution_df.pivot(index=['feature_name', 'Patient_ID'], columns='Timepoint', values=['normalized_mean', 'raw_mean'])
    evolution_df_wide.columns = ['_'.join(col).strip() for col in evolution_df_wide.columns.values]

    # calculate difference between normalised and raw values across timepoints
    evolution_df_wide = evolution_df_wide.reset_index()
    evolution_df_wide = evolution_df_wide.dropna(axis=0)
    evolution_df_wide['comparison'] = evolution_col
    evolution_df_wide['normalized_value'] = evolution_df_wide['normalized_mean_' + timepoint_2] - evolution_df_wide['normalized_mean_' + timepoint_1]
    evolution_df_wide['raw_value'] = evolution_df_wide['raw_mean_' + timepoint_2] - evolution_df_wide['raw_mean_' + timepoint_1]
    evolution_df_wide = evolution_df_wide[['feature_name', 'Patient_ID', 'comparison', 'normalized_value', 'raw_value']]

    evolution_dfs.append(evolution_df_wide)

evolution_df = pd.concat(evolution_dfs)
evolution_df.to_csv(os.path.join(data_dir, 'evolution/evolution_df.csv'), index=False)