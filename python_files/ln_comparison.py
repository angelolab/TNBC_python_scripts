import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr, ttest_ind, ttest_rel

from python_files.utils import find_conserved_features, compare_timepoints, compare_populations
from python_files.utils import summarize_population_enrichment, summarize_timepoint_enrichment


local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
patient_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/TONIC_data_per_patient.csv'))

timepoint_features = pd.read_csv(os.path.join(data_dir, 'timepoint_features_filtered.csv'))
timepoint_features = timepoint_features.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint']].drop_duplicates(), on='Tissue_ID')
#timepoint_features = timepoint_features.merge(patient_metadata[['Patient_ID', 'LN_pos']].drop_duplicates(), on='Patient_ID', how='left')


# compare different lymph nodes
compare_df = compare_timepoints(feature_df=timepoint_features, timepoint_1_name='lymphnode_pos', timepoint_1_list=['lymphnode_pos'],
                                timepoint_2_name='lymphnode_neg', timepoint_2_list=['lymphnode_neg'])
# plot results
output_dir = plot_dir + '/ln_comparison'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

summarize_timepoint_enrichment(input_df=compare_df, feature_df=timepoint_features, timepoints=['lymphnode_pos', 'lymphnode_neg'],
                             pval_thresh=2, diff_thresh=0.3, output_dir=output_dir)


# compare progression with on-nivo
compare_df = compare_timepoints(feature_df=timepoint_features, timepoint_1_name='on_nivo', timepoint_1_list=['on_nivo'],
                                timepoint_2_name='progression', timepoint_2_list=['progression'])
# plot results
output_dir = plot_dir + '/progression'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

summarize_timepoint_enrichment(input_df=compare_df, feature_df=timepoint_features, timepoints=['on_nivo', 'progression'],
                             pval_thresh=2, diff_thresh=0.3, output_dir=output_dir)



