import os
import pandas as pd
import numpy as np

from ark.utils.io_utils import list_folders

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'


#
# Determine which cores have valid image data, and update all metadata tables
#

# get list of acquired FOVs
all_fovs = list_folders('/Volumes/Big_Boy/TONIC_Cohort/image_data/samples')
fov_df = pd.DataFrame({'imaged_fovs': all_fovs})
fov_df.to_csv(os.path.join(data_dir, 'imaged_fovs.csv'), index=False)

# annotate acquired FOVs
core_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
fov_df = pd.read_csv(os.path.join(data_dir, 'imaged_fovs.csv'))
core_metadata['MIBI_data_generated'] = core_metadata['fov'].isin(fov_df.imaged_fovs)

# TODO: get list of pathologist excluded FOVs


# TODO: relabel tumor cells as epithelial

core_metadata.to_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'), index=False)

# annotate timepoints with at least 1 valid core
timepoint_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint.csv'))
timepoint_grouped = core_metadata.loc[:, ['Tissue_ID', 'MIBI_data_generated']].groupby('Tissue_ID').agg(np.sum)
include_ids = timepoint_grouped.index[(timepoint_grouped['MIBI_data_generated'] > 0)]
timepoint_metadata['MIBI_include'] = timepoint_metadata.Tissue_ID.isin(include_ids)

#
# Consolidate biopsy and primary labels into single 'untreated_primary' label
#

# identify patients with and without neo-adjuvant chemotherapy (NAC)
patient_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_patient.csv'))
untreated_pts = patient_metadata.loc[patient_metadata.NAC_received_for_primary_tumor == 'No', 'Study_ID']
treated_pts = patient_metadata.loc[patient_metadata.NAC_received_for_primary_tumor == 'Yes', 'Study_ID']

# designate biopsy as untreated for NAC patients, primary as untreated for patients without NAC
timepoint_metadata['Timepoint_broad'] = timepoint_metadata.Timepoint
timepoint_metadata.loc[(timepoint_metadata.TONIC_ID.isin(untreated_pts)) & (timepoint_metadata.Timepoint == 'primary'), 'Timepoint'] = 'primary_untreated'
timepoint_metadata.loc[(timepoint_metadata.TONIC_ID.isin(treated_pts)) & (timepoint_metadata.Timepoint == 'biopsy'), 'Timepoint'] = 'primary_untreated'

# some patients without NAC only have a biopsy sample available; we need to identify those patients
assigned_pts = timepoint_metadata.loc[timepoint_metadata.Timepoint == 'primary_untreated', 'TONIC_ID'].unique()
unassigned_untreated_pts = set(untreated_pts).difference(set(assigned_pts))
timepoint_metadata.loc[(timepoint_metadata.TONIC_ID.isin(unassigned_untreated_pts)) &
                       (timepoint_metadata.Timepoint == 'biopsy'), 'Timepoint'] = 'primary_untreated'

timepoint_metadata.to_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint.csv'), index=False)


#
# Identify patients with specific combinations of timepoints present
#

# reshape data to allow for easy boolean comparisons
subset_metadata = timepoint_metadata.loc[timepoint_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
subset_metadata = subset_metadata.loc[subset_metadata.MIBI_include, :]
metadata_wide = pd.pivot(subset_metadata, index='TONIC_ID', columns='Timepoint', values='Tissue_ID')

# patients with both a primary and baseline sample
primary_baseline = metadata_wide.index[np.logical_and(~metadata_wide.primary_untreated.isnull(),
                                                      ~metadata_wide.baseline.isnull())]
patient_metadata['primary_baseline'] = patient_metadata.Study_ID.isin(primary_baseline)

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

patient_metadata.to_csv(os.path.join(data_dir, 'TONIC_data_per_patient.csv'), index=False)
