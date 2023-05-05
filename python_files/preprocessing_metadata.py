import os
import pandas as pd
import numpy as np

from alpineer.io_utils import list_folders

local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data'
metadata_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/metadata'


#
# Determine which cores have valid image data, and update all metadata tables
#

# get list of acquired FOVs
all_fovs = list_folders('/Volumes/Big_Boy/TONIC_Cohort/image_data/samples')
fov_df = pd.DataFrame({'imaged_fovs': all_fovs})
fov_df.to_csv(os.path.join(data_dir, 'imaged_fovs.csv'), index=False)

# annotate acquired FOVs
core_metadata = pd.read_csv(os.path.join(metadata_dir, 'TONIC_data_per_core_unprocessed.csv'))
fov_df = pd.read_csv(os.path.join(metadata_dir, 'imaged_fovs.csv'))
core_metadata['MIBI_data_generated'] = core_metadata['fov'].isin(fov_df.imaged_fovs)

# TODO: get list of pathologist excluded FOVs


# TODO: relabel tumor cells as epithelial

# annotate timepoints with at least 1 valid core
timepoint_metadata = pd.read_csv(os.path.join(metadata_dir, 'TONIC_data_per_timepoint_unprocessed.csv'))
timepoint_grouped = core_metadata.loc[:, ['Tissue_ID', 'MIBI_data_generated']].groupby('Tissue_ID').agg(np.sum)
include_ids = timepoint_grouped.index[(timepoint_grouped['MIBI_data_generated'] > 0)]
timepoint_metadata['MIBI_data_generated'] = timepoint_metadata.Tissue_ID.isin(include_ids)

#
# Consolidate biopsy and primary labels into single 'untreated_primary' label
#

# identify patients with and without neo-adjuvant chemotherapy (NAC)
patient_metadata = pd.read_csv(os.path.join(metadata_dir, 'TONIC_data_per_patient_unprocessed.csv'))
untreated_pts = patient_metadata.loc[patient_metadata.NAC_received_for_primary_tumor == 'No', 'Patient_ID']
treated_pts = patient_metadata.loc[patient_metadata.NAC_received_for_primary_tumor == 'Yes', 'Patient_ID']

# designate biopsy as untreated for NAC patients, primary as untreated for patients without NAC
timepoint_metadata['Timepoint_broad'] = timepoint_metadata.Timepoint
timepoint_metadata.loc[(timepoint_metadata.Patient_ID.isin(untreated_pts)) & (timepoint_metadata.Timepoint == 'primary') & timepoint_metadata.MIBI_data_generated, 'Timepoint'] = 'primary_untreated'
timepoint_metadata.loc[(timepoint_metadata.Patient_ID.isin(treated_pts)) & (timepoint_metadata.Timepoint == 'biopsy') & timepoint_metadata.MIBI_data_generated, 'Timepoint'] = 'primary_untreated'

# some patients without NAC only have a biopsy sample available; we need to identify those patients
assigned_pts = timepoint_metadata.loc[timepoint_metadata.Timepoint == 'primary_untreated', 'Patient_ID'].unique()
unassigned_untreated_pts = set(untreated_pts).difference(set(assigned_pts))
timepoint_metadata.loc[(timepoint_metadata.Patient_ID.isin(unassigned_untreated_pts)) &
                       (timepoint_metadata.Timepoint == 'biopsy') &
                        timepoint_metadata.MIBI_data_generated, 'Timepoint'] = 'primary_untreated'

#
# Consolidate rare metastatic locations into single 'other' label
#

# get rare metastatic sites per patient
site_counts = timepoint_metadata.Localization.value_counts()
rare_sites = site_counts[7:].index
rare_sites = rare_sites.append(pd.Index(['Unknown']))

# consolidate rare sites into 'other' label
timepoint_metadata['Localization_detailed'] = timepoint_metadata.Localization
timepoint_metadata.loc[timepoint_metadata.Localization.isin(rare_sites), 'Localization'] = 'Other'

#
# Identify patients with specific combinations of timepoints present
#

# reshape data to allow for easy boolean comparisons
subset_metadata = timepoint_metadata.loc[timepoint_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
subset_metadata = subset_metadata.loc[subset_metadata.MIBI_data_generated, :]
metadata_wide = pd.pivot(subset_metadata, index='Patient_ID', columns='Timepoint', values='Tissue_ID')

# add studytissue to primary tumors to identify patients with missing data
subset_metadata.loc[subset_metadata.Timepoint == 'primary_untreated', 'Location_studytissue'] = 'A'
metadata_wide = pd.pivot(subset_metadata, index='Patient_ID', columns='Timepoint', values='Location_studytissue')

# primary baseline comparison requires any tissue present in either
current_wide = metadata_wide.loc[:, ['primary_untreated', 'baseline']]
current_wide = current_wide.dropna(axis=0)

patient_metadata['primary__baseline'] = patient_metadata.Patient_ID.isin(current_wide.index)

# other comparisons require tissue from the same location
comparison_pairs = [['baseline', 'post_induction'], ['baseline', 'on_nivo'], ['post_induction', 'on_nivo']]

# loop through pairs, find patients with matching tissue, add to patient_metadata
for pair in comparison_pairs:
    current_wide = metadata_wide.loc[:, pair]
    current_wide = current_wide.dropna(axis=0)
    current_wide = current_wide.loc[current_wide[pair[0]] == current_wide[pair[1]], :]

    patient_metadata['__'.join(pair)] = patient_metadata.Patient_ID.isin(current_wide.index)


# handle NAs in Tissue_ID
core_missing = core_metadata.Tissue_ID.isnull()
imaged_cores = core_metadata.MIBI_data_generated
np.where(np.logical_and(core_missing, imaged_cores))

# all of the missing cores were not imaged, can be dropped
core_metadata = core_metadata.loc[~core_missing, :]

# This is the wrong tissue and should be excldued
bad_cores = ['T20-04891']
core_metadata = core_metadata.loc[~core_metadata.Tissue_ID.isin(bad_cores), :]

# check for Tissue_IDs present in core metadata but not timepoint metadata
timepoint_missing = ~core_metadata.Tissue_ID.isin(timepoint_metadata.Tissue_ID)
timepoint_missing = core_metadata.Tissue_ID[timepoint_missing].unique()
print(timepoint_missing)

# get metadata on missing cores
core_metadata.loc[core_metadata.Tissue_ID.isin(timepoint_missing[3:]), :]

# These can all be excluded
core_metadata = core_metadata.loc[~core_metadata.Tissue_ID.isin(timepoint_missing), :]

# check for Tissue_IDs present in timepoint metadata but not core metadata
core_missing = ~timepoint_metadata.Tissue_ID.isin(core_metadata.Tissue_ID) & (timepoint_metadata.On_TMA == 'Yes')
core_missing = timepoint_metadata.Tissue_ID[core_missing].unique()
print(core_missing)

# select relevant columns from cores
harmonized_metadata = core_metadata[['fov', 'Tissue_ID', 'MIBI_data_generated']]

# select and merge relevant columns from timepoints
harmonized_metadata = pd.merge(harmonized_metadata, timepoint_metadata.loc[:, ['Tissue_ID', 'Patient_ID', 'Timepoint', 'Localization']], on='Tissue_ID', how='left')
assert np.sum(harmonized_metadata.Tissue_ID.isnull()) == 0

# select and merge relevant columns from patients
harmonized_metadata = pd.merge(harmonized_metadata, patient_metadata.loc[:, ['Patient_ID', 'primary__baseline', 'baseline__post_induction', 'baseline__on_nivo', 'post_induction__on_nivo']], on='Patient_ID', how='inner')
assert np.sum(harmonized_metadata.Tissue_ID.isnull()) == 0

# save harmonized metadata
harmonized_metadata.to_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'), index=False)

# add in the harmonized metadata
core_metadata = pd.merge(core_metadata, harmonized_metadata, on=['fov', 'Tissue_ID'], how='left')

# add in the harmonized metadata, dropping the fov column
# TODO: determine if we want to keep the timepoints without any image data
harmonized_metadata = harmonized_metadata.drop(['fov', 'Patient_ID', 'Timepoint', 'Localization', 'MIBI_data_generated'], axis=1)
harmonized_metadata = harmonized_metadata.drop_duplicates()
timepoint_metadata = pd.merge(timepoint_metadata, harmonized_metadata, on='Tissue_ID', how='left')

# save all modified metadata sheets
core_metadata.to_csv(os.path.join(metadata_dir, 'TONIC_data_per_core.csv'), index=False)
timepoint_metadata.to_csv(os.path.join(metadata_dir, 'TONIC_data_per_timepoint.csv'), index=False)
patient_metadata.to_csv(os.path.join(metadata_dir, 'TONIC_data_per_patient.csv'), index=False)
