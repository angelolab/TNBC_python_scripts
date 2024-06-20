import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib_venn import venn2


BASE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/"

# supplementary tables
save_dir = os.path.join(BASE_DIR, 'supplementary_figs/supplementary_tables')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# sample summary
harmonized_metadata = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/harmonized_metadata.csv'))
wes_metadata = pd.read_csv(os.path.join(BASE_DIR, 'sequencing_data/preprocessing/TONIC_WES_meta_table.tsv'), sep='\t')
rna_metadata = pd.read_csv(os.path.join(BASE_DIR, 'sequencing_data/preprocessing/TONIC_tissue_rna_id.tsv'), sep='\t')
rna_metadata = rna_metadata.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint']].drop_duplicates(), on='Tissue_ID', how='left')

harmonized_metadata = harmonized_metadata.loc[harmonized_metadata.MIBI_data_generated, :]

modality = ['MIBI'] * 4 + ['RNA'] * 3 + ['DNA'] * 1
timepoint = ['primary_untreated', 'baseline', 'post_induction', 'on_nivo'] + ['baseline', 'post_induction', 'on_nivo'] + ['baseline']

sample_summary_df = pd.DataFrame({'modality': modality, 'timepoint': timepoint, 'sample_num': [0] * 8, 'patient_num': [0] * 8})

# populate dataframe
for idx, row in sample_summary_df.iterrows():
    if row.modality == 'MIBI':
        sample_summary_df.loc[idx, 'sample_num'] = len(harmonized_metadata.loc[harmonized_metadata.Timepoint == row.timepoint, :])
        sample_summary_df.loc[idx, 'patient_num'] = len(harmonized_metadata.loc[harmonized_metadata.Timepoint == row.timepoint, 'Patient_ID'].unique())
    elif row.modality == 'RNA':
        sample_summary_df.loc[idx, 'sample_num'] = len(rna_metadata.loc[rna_metadata.Timepoint == row.timepoint, :])
        sample_summary_df.loc[idx, 'patient_num'] = len(rna_metadata.loc[rna_metadata.Timepoint == row.timepoint, 'Patient_ID'].unique())
    elif row.modality == 'DNA':
        sample_summary_df.loc[idx, 'sample_num'] = len(wes_metadata.loc[wes_metadata.timepoint == row.timepoint, :])
        sample_summary_df.loc[idx, 'patient_num'] = len(wes_metadata.loc[wes_metadata.timepoint == row.timepoint, 'Individual.ID'].unique())

sample_summary_df.to_csv(os.path.join(save_dir, 'Supplementary_Table_3.csv'), index=False)

# feature metadata
feature_metadata = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/feature_metadata.csv'))

feature_metadata.columns = ['Feature name', 'Feature name including compartment', 'Compartment the feature is calculated in',
                            'Cell types used to calculate feature', 'Level of clustering granularity for cell types',
                            'Type of feature', 'Additional information about the feature', 'Additional information about the feature']

feature_metadata.to_csv(os.path.join(save_dir, 'Supplementary_Table_4.csv'), index=False)
