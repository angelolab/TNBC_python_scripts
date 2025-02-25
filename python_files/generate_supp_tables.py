import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os
import pandas as pd
import numpy as np


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
clinical_data = pd.read_csv(os.path.join(BASE_DIR, 'intermediate_files/metadata/patient_clinical_data.csv'))

harmonized_metadata = harmonized_metadata.loc[harmonized_metadata.MIBI_data_generated, :]
harmonized_metadata = harmonized_metadata[['Patient_ID', 'Timepoint']].merge(clinical_data, on='Patient_ID').drop_duplicates()
harmonized_metadata = harmonized_metadata[harmonized_metadata.Clinical_benefit.isin(['Yes', 'No'])]
wes_metadata = wes_metadata.rename(columns={'Individual.ID': 'Patient_ID', 'timepoint': 'Timepoint'})
wes_metadata = wes_metadata[['Patient_ID', 'Timepoint']].merge(clinical_data, on='Patient_ID').drop_duplicates()
wes_metadata = wes_metadata[wes_metadata.Clinical_benefit.isin(['Yes', 'No'])]
rna_metadata = rna_metadata[['Patient_ID', 'Timepoint']].merge(clinical_data, on='Patient_ID').drop_duplicates()
rna_metadata = rna_metadata[rna_metadata.Clinical_benefit.isin(['Yes', 'No'])]

modality = ['MIBI'] * 4 + ['RNA'] * 3 + ['DNA'] * 1
timepoint = ['primary', 'baseline', 'pre_nivo', 'on_nivo'] + ['baseline', 'pre_nivo', 'on_nivo'] + ['baseline']

sample_summary_df = pd.DataFrame({'modality': modality, 'timepoint': timepoint, 'sample_num': [0] * 8, 'patient_num': [0] * 8, 'responder_num': [0] * 8, 'nonresponder_num': [0] * 8})

# populate dataframe
for idx, row in sample_summary_df.iterrows():
    if row.modality == 'MIBI':
        sample_summary_df.loc[idx, 'sample_num'] = len(harmonized_metadata.loc[harmonized_metadata.Timepoint == row.timepoint, :])
        sample_summary_df.loc[idx, 'patient_num'] = len(harmonized_metadata.loc[harmonized_metadata.Timepoint == row.timepoint, 'Patient_ID'].unique())
        sample_summary_df.loc[idx, 'responder_num'] = len(harmonized_metadata.loc[np.logical_and(harmonized_metadata.Timepoint == row.timepoint, harmonized_metadata.Clinical_benefit == 'Yes')])
        sample_summary_df.loc[idx, 'nonresponder_num'] = len(harmonized_metadata.loc[np.logical_and(harmonized_metadata.Timepoint == row.timepoint, harmonized_metadata.Clinical_benefit == 'No')])
    elif row.modality == 'RNA':
        sample_summary_df.loc[idx, 'sample_num'] = len(rna_metadata.loc[rna_metadata.Timepoint == row.timepoint, :])
        sample_summary_df.loc[idx, 'patient_num'] = len(rna_metadata.loc[rna_metadata.Timepoint == row.timepoint, 'Patient_ID'].unique())
        sample_summary_df.loc[idx, 'responder_num'] = len(rna_metadata.loc[np.logical_and(rna_metadata.Timepoint == row.timepoint, rna_metadata.Clinical_benefit == 'Yes')])
        sample_summary_df.loc[idx, 'nonresponder_num'] = len(rna_metadata.loc[np.logical_and(rna_metadata.Timepoint == row.timepoint, rna_metadata.Clinical_benefit == 'No')])
    elif row.modality == 'DNA':
        sample_summary_df.loc[idx, 'sample_num'] = len(wes_metadata.loc[wes_metadata.Timepoint == row.timepoint, :])
        sample_summary_df.loc[idx, 'patient_num'] = len(wes_metadata.loc[wes_metadata.Timepoint == row.timepoint, 'Patient_ID'].unique())
        sample_summary_df.loc[idx, 'responder_num'] = len(wes_metadata.loc[np.logical_and(wes_metadata.Timepoint == row.timepoint, wes_metadata.Clinical_benefit == 'Yes')])
        sample_summary_df.loc[idx, 'nonresponder_num'] = len(wes_metadata.loc[np.logical_and(wes_metadata.Timepoint == row.timepoint, wes_metadata.Clinical_benefit == 'No')])

sample_summary_df.to_csv(os.path.join(save_dir, 'Supplementary_Table_3.csv'), index=False)

# feature metadata
feature_metadata = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/feature_metadata.csv'))

feature_metadata.columns = ['Feature name', 'Feature name including compartment', 'Compartment the feature is calculated in',
                            'Cell types used to calculate feature', 'Level of clustering granularity for cell types',
                            'Type of feature', 'Additional information about the feature', 'Additional information about the feature']

feature_metadata.to_csv(os.path.join(save_dir, 'Supplementary_Table_4.csv'), index=False)

# sequencing features
sequencing_features = pd.read_csv(os.path.join(BASE_DIR, 'sequencing_data/processed_genomics_features.csv'))
sequencing_features = sequencing_features[['feature_name', 'data_type', 'feature_type']].drop_duplicates()
sequencing_features = sequencing_features.loc[sequencing_features.feature_type != 'gene_rna', :]

sequencing_features.to_csv(os.path.join(save_dir, 'Supplementary_Table_5.csv'), index=False)

