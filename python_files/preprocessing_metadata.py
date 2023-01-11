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
core_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core_unprocessed.csv'))
fov_df = pd.read_csv(os.path.join(data_dir, 'imaged_fovs.csv'))
core_metadata['MIBI_data_generated'] = core_metadata['fov'].isin(fov_df.imaged_fovs)

# TODO: get list of pathologist excluded FOVs


# TODO: relabel tumor cells as epithelial

# annotate timepoints with at least 1 valid core
timepoint_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint_unprocessed.csv'))
timepoint_grouped = core_metadata.loc[:, ['Tissue_ID', 'MIBI_data_generated']].groupby('Tissue_ID').agg(np.sum)
include_ids = timepoint_grouped.index[(timepoint_grouped['MIBI_data_generated'] > 0)]
timepoint_metadata['MIBI_data_generated'] = timepoint_metadata.Tissue_ID.isin(include_ids)

#
# Consolidate biopsy and primary labels into single 'untreated_primary' label
#

# identify patients with and without neo-adjuvant chemotherapy (NAC)
patient_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_patient_unprocessed.csv'))
untreated_pts = patient_metadata.loc[patient_metadata.NAC_received_for_primary_tumor == 'No', 'TONIC_ID']
treated_pts = patient_metadata.loc[patient_metadata.NAC_received_for_primary_tumor == 'Yes', 'TONIC_ID']

# designate biopsy as untreated for NAC patients, primary as untreated for patients without NAC
timepoint_metadata['Timepoint_broad'] = timepoint_metadata.Timepoint
timepoint_metadata.loc[(timepoint_metadata.TONIC_ID.isin(untreated_pts)) & (timepoint_metadata.Timepoint == 'primary') & timepoint_metadata.MIBI_data_generated, 'Timepoint'] = 'primary_untreated'
timepoint_metadata.loc[(timepoint_metadata.TONIC_ID.isin(treated_pts)) & (timepoint_metadata.Timepoint == 'biopsy') & timepoint_metadata.MIBI_data_generated, 'Timepoint'] = 'primary_untreated'

# some patients without NAC only have a biopsy sample available; we need to identify those patients
assigned_pts = timepoint_metadata.loc[timepoint_metadata.Timepoint == 'primary_untreated', 'TONIC_ID'].unique()
unassigned_untreated_pts = set(untreated_pts).difference(set(assigned_pts))
timepoint_metadata.loc[(timepoint_metadata.TONIC_ID.isin(unassigned_untreated_pts)) &
                       (timepoint_metadata.Timepoint == 'biopsy') &
                        timepoint_metadata.MIBI_data_generated, 'Timepoint'] = 'primary_untreated'

#
# Consolidate rare metastatic locations into single 'other' label
#

# get rare metastatic sites per patient
metastatic_sites = timepoint_metadata.loc[timepoint_metadata.Timepoint.isin(['baseline', 'post_induction', 'on_nivo']), 'Localization'].value_counts()
rare_sites = metastatic_sites[4:].index

# consolidate rare sites into 'other' label
timepoint_metadata['Localization_broad'] = timepoint_metadata.Localization
timepoint_metadata.loc[timepoint_metadata.Localization.isin(rare_sites), 'Localization'] = 'Other'

#
# Identify patients with specific combinations of timepoints present
#

# find the patients with baseline and on-nivo timepoints from the same location
timepoint_sums = timepoint_metadata.loc[timepoint_metadata.Timepoint.isin(['baseline', 'on_nivo']), :]

# get the number of entries in Location_studytissue that are equal to 'A' grouped by TONIC_ID
timepoint_sums = timepoint_sums.loc[:, ['TONIC_ID', 'Location_studytissue']].groupby('TONIC_ID').agg(lambda x: np.sum(x == 'A'))
timepoint_sums.reset_index(inplace=True)
timepoint_sums = timepoint_sums.loc[timepoint_sums.Location_studytissue == 2, :]

# include only patients with 2 counts of studytissue 'A'
subset_metadata = timepoint_metadata.loc[timepoint_metadata.TONIC_ID.isin(timepoint_sums.TONIC_ID) &
                                        timepoint_metadata.MIBI_data_generated, :]

patient_metadata['baseline_on_nivo'] = patient_metadata.TONIC_ID.isin(subset_metadata.TONIC_ID.unique())

# reshape data to allow for easy boolean comparisons
subset_metadata = timepoint_metadata.loc[timepoint_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
subset_metadata = subset_metadata.loc[subset_metadata.MIBI_data_generated, :]
metadata_wide = pd.pivot(subset_metadata, index='TONIC_ID', columns='Timepoint', values='Tissue_ID')

# patients with both a primary and baseline sample
primary_baseline = metadata_wide.index[np.logical_and(~metadata_wide.primary_untreated.isnull(),
                                                      ~metadata_wide.baseline.isnull())]
patient_metadata['primary_baseline'] = patient_metadata.TONIC_ID.isin(primary_baseline)

# This doesn't take into account tissue location: TBD what we do with this
# # patients with both a baseline and induction sample
# baseline_induction = metadata_wide.index[np.logical_and(~metadata_wide.baseline.isnull(),
#                                                         ~metadata_wide.post_induction.isnull())]
# patient_metadata['baseline_induction'] = patient_metadata.Study_ID.isin(baseline_induction)
#
# # patients with both a baseline and nivo sample
# baseline_nivo = metadata_wide.index[np.logical_and(~metadata_wide.baseline.isnull(),
#                                                         ~metadata_wide.on_nivo.isnull())]
# patient_metadata['baseline_nivo'] = patient_metadata.Study_ID.isin(baseline_nivo)
#
# # patients with a baseline, induction, and nivo sample
# baseline_induction_nivo = metadata_wide.index[np.logical_and(~metadata_wide.baseline.isnull(),
#                                                              np.logical_and(~metadata_wide.post_induction.isnull(),
#                                                                             ~metadata_wide.on_nivo.isnull()))]
# patient_metadata['baseline_induction_nivo'] = patient_metadata.Study_ID.isin(baseline_induction_nivo)

#
# create harmonized metadata sheet that has all relevant information for analyses
#

# handle NAs in Tissue_ID
core_missing = core_metadata.Tissue_ID.isnull()
imaged_cores = core_metadata.MIBI_data_generated
np.where(np.logical_and(core_missing, imaged_cores))

# all of the missing cores were not imaged, can be dropped
core_metadata = core_metadata.loc[~core_missing, :]

# check for Tissue_IDs present in core metadata but not timepoint metadata
timepoint_missing = ~core_metadata.Tissue_ID.isin(timepoint_metadata.Tissue_ID)
timepoint_missing = core_metadata.Tissue_ID[timepoint_missing].unique()
print(timepoint_missing)

# get metadata on missing cores
core_metadata.loc[core_metadata.Tissue_ID.isin(timepoint_missing[3:]), :]

# check for Tissue_IDs present in timepoint metadata but not core metadata
core_missing = ~timepoint_metadata.Tissue_ID.isin(core_metadata.Tissue_ID) & (timepoint_metadata.On_TMA == 'Yes')
core_missing = timepoint_metadata.Tissue_ID[core_missing].unique()
print(core_missing)

# remove missing cores
core_metadata = core_metadata.loc[~core_metadata.Tissue_ID.isin(timepoint_missing), :]


# select relevant columns from cores
harmonized_metadata = core_metadata[['fov', 'Tissue_ID']]

# select and merge relevant columns from timepoints
harmonized_metadata = pd.merge(harmonized_metadata, timepoint_metadata.loc[:, ['Tissue_ID', 'TONIC_ID', 'Timepoint', 'Localization']], on='Tissue_ID', how='left')
assert np.sum(harmonized_metadata.Tissue_ID.isnull()) == 0

# select and merge relevant columns from patients
harmonized_metadata = pd.merge(harmonized_metadata, patient_metadata.loc[:, ['TONIC_ID', 'baseline_on_nivo', 'primary_baseline']], on='TONIC_ID', how='left')
assert np.sum(harmonized_metadata.Tissue_ID.isnull()) == 0

harmonized_metadata.to_csv(os.path.join(data_dir, 'harmonized_metadata.csv'), index=False)

# add in the harmonized metadata
core_metadata = pd.merge(core_metadata, harmonized_metadata, on=['fov', 'Tissue_ID'], how='left')

# add in the harmonized metadata, dropping the fov column
harmonized_metadata = harmonized_metadata.drop(['fov', 'TONIC_ID', 'Timepoint', 'Localization'], axis=1)
harmonized_metadata = harmonized_metadata.drop_duplicates()
timepoint_metadata = pd.merge(timepoint_metadata, harmonized_metadata, on='Tissue_ID', how='left')

# save all modified metadata sheets
core_metadata.to_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'), index=False)
timepoint_metadata.to_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint.csv'), index=False)
patient_metadata.to_csv(os.path.join(data_dir, 'TONIC_data_per_patient.csv'), index=False)
