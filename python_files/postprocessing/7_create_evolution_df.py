import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr, ttest_ind, ttest_rel

from python_files.utils import find_conserved_features

plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
intermediate_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files'
output_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/output_files'
analysis_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files'

harmonized_metadata = pd.read_csv(os.path.join(analysis_dir, 'harmonized_metadata.csv'))
timepoint_features = pd.read_csv(os.path.join(analysis_dir, 'timepoint_features_filtered.csv'))
evolution_cats = ['primary__baseline', 'baseline__post_induction', 'baseline__on_nivo', 'post_induction__on_nivo']
timepoint_features = timepoint_features.merge(harmonized_metadata[['Tissue_ID', 'Timepoint', 'Localization', 'Patient_ID'] + evolution_cats].drop_duplicates(), on='Tissue_ID', how='left')
patient_metadata = pd.read_csv(os.path.join(intermediate_dir, 'metadata/TONIC_data_per_patient.csv'))

evolution_dfs = []
# generate evolution df based on difference in timepoints
for evolution_col in evolution_cats:
    timepoint_1, timepoint_2 = evolution_col.split('__')
    if timepoint_1 == 'primary':
        timepoint_1 = 'primary_untreated'
    evolution_df = timepoint_features[timepoint_features[evolution_col]].copy()
    evolution_df = evolution_df.loc[evolution_df.Timepoint.isin([timepoint_1, timepoint_2])]

    # get the paired features
    evolution_df_wide = evolution_df.pivot(index=['feature_name_unique', 'Patient_ID'], columns='Timepoint', values=['normalized_mean', 'raw_mean'])
    evolution_df_wide.columns = ['_'.join(col).strip() for col in evolution_df_wide.columns.values]

    # calculate difference between normalised and raw values across timepoints
    evolution_df_wide = evolution_df_wide.reset_index()
    evolution_df_wide = evolution_df_wide.dropna(axis=0)
    evolution_df_wide['comparison'] = evolution_col
    evolution_df_wide['normalized_value'] = evolution_df_wide['normalized_mean_' + timepoint_2] - evolution_df_wide['normalized_mean_' + timepoint_1]
    evolution_df_wide['raw_value'] = evolution_df_wide['raw_mean_' + timepoint_2] - evolution_df_wide['raw_mean_' + timepoint_1]
    evolution_df_wide = evolution_df_wide[['feature_name_unique', 'Patient_ID', 'comparison', 'normalized_value', 'raw_value']]

    evolution_dfs.append(evolution_df_wide)

evolution_df = pd.concat(evolution_dfs)
evolution_df.to_csv(os.path.join(output_dir, 'nivo_outcomes_evolution_df.csv'), index=False)


# create combined df
timepoint_features = pd.read_csv(os.path.join(analysis_dir, 'timepoint_features_filtered.csv'))
timepoint_features = timepoint_features.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint', 'primary__baseline',
                                                                   'baseline__on_nivo', 'baseline__post_induction', 'post_induction__on_nivo']].drop_duplicates(), on='Tissue_ID')
timepoint_features = timepoint_features.merge(patient_metadata[['Patient_ID', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']].drop_duplicates(), on='Patient_ID', how='left')

# Hacky, remove once metadata is updated
timepoint_features = timepoint_features.loc[timepoint_features.Clinical_benefit.isin(['Yes', 'No']), :]
timepoint_features = timepoint_features.loc[timepoint_features.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
timepoint_features = timepoint_features[['Tissue_ID', 'feature_name', 'feature_name_unique', 'raw_mean', 'raw_std', 'normalized_mean', 'normalized_std', 'Patient_ID', 'Timepoint', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]

# look at evolution
evolution_df = pd.read_csv(os.path.join(output_dir, 'nivo_outcomes_evolution_df.csv'))
evolution_df = evolution_df.merge(patient_metadata[['Patient_ID', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']].drop_duplicates(), on='Patient_ID', how='left')
evolution_df = evolution_df.rename(columns={'raw_value': 'raw_mean', 'normalized_value': 'normalized_mean', 'comparison': 'Timepoint'})
evolution_df = evolution_df[['feature_name_unique', 'raw_mean', 'normalized_mean', 'Patient_ID', 'Timepoint', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]

# combine together into single df
combined_df = timepoint_features.copy()
combined_df = combined_df[['feature_name_unique', 'raw_mean', 'normalized_mean', 'Patient_ID', 'Timepoint', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]
combined_df = pd.concat([combined_df, evolution_df[['feature_name_unique', 'raw_mean', 'normalized_mean',
                                                    'Patient_ID', 'Timepoint', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]])
combined_df['combined_name'] = combined_df.feature_name_unique + '__' + combined_df.Timepoint

combined_df.to_csv(os.path.join(analysis_dir, 'nivo_outcomes_combined_df.csv'), index=False)
