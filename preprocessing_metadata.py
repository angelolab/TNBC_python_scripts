import os
import pandas as pd
import numpy as np

from ark.utils.io_utils import list_folders

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'

# create consolidated cell table with only cell populations
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated.csv'))

cell_table = cell_table.loc[:, ['fov', 'cell_meta_cluster', 'label', 'cell_cluster',
                             'cell_cluster_broad']]
cell_table.to_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_clusters_only.csv'),
                  index=False)

# create consolidated cell table with only functional marker freqs
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated.csv'))

func_cols = [col for col in cell_table.columns if '_threshold' in col]
cell_table_func = cell_table.loc[:, ['fov', 'cell_cluster_broad', 'cell_cluster', 'cell_meta_cluster'] + func_cols]
cell_table_func.columns = [col.split('_threshold')[0] for col in cell_table_func.columns]
cell_table_func.to_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'),
                       index=False)


# determine which cores have valid image data
all_fovs = list_folders('/Volumes/Big_Boy/TONIC_Cohort/image_data/samples')
core_df['MIBI_data'] = core_df['fov'].isin(all_fovs)
core_df.to_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))

# identify which primaries are treated vs untreated
patient_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_patient.csv'))
timepoint_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint.csv'))

untreated_pts = patient_metadata.loc[patient_metadata.NAC_received_for_primary_tumor == 'No', 'Study_ID']
treated_pts = patient_metadata.loc[patient_metadata.NAC_received_for_primary_tumor == 'Yes', 'Study_ID']

# designate biopsy or primary as untreated
timepoint_metadata.loc[(timepoint_metadata.TONIC_ID.isin(untreated_pts)) & (timepoint_metadata.Timepoint == 'primary'), 'Timepoint'] = 'primary_untreated'
timepoint_metadata.loc[(timepoint_metadata.TONIC_ID.isin(treated_pts)) & (timepoint_metadata.Timepoint == 'biopsy'), 'Timepoint'] = 'primary_untreated'


# annotate IDs from specific combinations of timepoints
subset_metadata = timepoint_metadata.loc[timepoint_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
metadata_wide = pd.pivot(subset_metadata, index='TONIC_ID', columns='Timepoint', values='Tissue_ID')

# patients with both a primary and baseline sample
primary_baseline = metadata_wide.index[np.logical_and(~metadata_wide.primary_untreated.isnull(),
                                                      ~metadata_wide.baseline.isnull())]
patient_metadata['primary_baseline'] = patient_metadata.Study_ID.isin(primary_baseline)
# timepoint_metadata['primary_baseline'] = np.logical_and(timepoint_metadata.Timepoint.isin(['primary_untreated', 'baseline']),
#                                                         timepoint_metadata.TONIC_ID.isin(primary_baseline))

# patients with both a baseline and induction sample
baseline_induction = metadata_wide.index[np.logical_and(~metadata_wide.baseline.isnull(),
                                                        ~metadata_wide.post_induction.isnull())]
patient_metadata['baseline_induction'] = patient_metadata.Study_ID.isin(baseline_induction)

# patients with both a baseline and nivo sample
baseline_nivo = metadata_wide.index[np.logical_and(~metadata_wide.baseline.isnull(),
                                                        ~metadata_wide.on_nivo.isnull())]
patient_metadata['baseline_nivo'] = patient_metadata.Study_ID.isin(baseline_nivo)

# patients with a baseline, induction, and nivo sample
baseline_induction_nivo = metadata_wide.index[np.logical_and(~metadata_wide.baseline.isnull(),
                                                             np.logical_and(~metadata_wide.post_induction.isnull(),
                                                                            ~metadata_wide.on_nivo.isnull()))]
patient_metadata['baseline_induction_nivo'] = patient_metadata.Study_ID.isin(baseline_induction_nivo)

