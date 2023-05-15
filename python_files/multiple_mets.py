import os
import pandas as pd
import numpy as np

from alpineer.io_utils import list_folders

import seaborn as sns
import matplotlib.pyplot as plt

local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data'
metadata_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/metadata'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

timepoint_features = pd.read_csv(os.path.join(data_dir, 'timepoint_features_no_compartment.csv'))
harmonized_metadata = pd.read_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'))
timepoint_features = timepoint_features.merge(harmonized_metadata[['Tissue_ID', 'Timepoint', 'Patient_ID']].drop_duplicates(), on='Tissue_ID', how='left')

# calculate distances between primary/baseline and primary/met1
input_df = timepoint_features.loc[timepoint_features.Timepoint.isin(['primary_untreated', 'baseline']), :]

baseline_distances = generate_grouped_distances(sample='Tissue_ID', group_by='Patient_ID',
                                                 data_df=input_df, harmonized_metadata=harmonized_metadata)
baseline_distances = baseline_distances.rename(columns={'euc_distance': 'primary_baseline_distance'})


input_df = timepoint_features.loc[timepoint_features.Timepoint.isin(['primary_untreated', 'metastasis_1']), :]

met1_distances = generate_grouped_distances(sample='Tissue_ID', group_by='Patient_ID',
                                                    data_df=input_df, harmonized_metadata=harmonized_metadata)
met1_distances = met1_distances.rename(columns={'euc_distance': 'primary_met1_distance'})

input_df = timepoint_features.loc[timepoint_features.Timepoint.isin(['baseline', 'metastasis_1']), :]
met1_baseline_distances = generate_grouped_distances(sample='Tissue_ID', group_by='Patient_ID',
                                                    data_df=input_df, harmonized_metadata=harmonized_metadata)
met1_baseline_distances = met1_baseline_distances.rename(columns={'euc_distance': 'met1_baseline_distance'})

combined_distances = baseline_distances[['primary_baseline_distance', 'metadata']].merge(met1_distances[['primary_met1_distance', 'metadata']], on='metadata')
combined_distances = combined_distances.merge(met1_baseline_distances[['met1_baseline_distance', 'metadata']], on='metadata')
combined_distances = combined_distances.dropna()

sns.scatterplot(data=combined_distances, x='primary_baseline_distance', y='primary_met1_distance')
plt.savefig(os.path.join(plot_dir, 'metastatic_distance_baseline_met.png'))
plt.close()

sns.scatterplot(data=combined_distances, x='primary_baseline_distance', y='met1_baseline_distance')
plt.savefig(os.path.join(plot_dir, 'metastatic_distance_baseline_met_baseline.png'))
plt.close()

# for each feature, calculate if the change from primary to met is shared across mets



# for patients with many mets, perform heirarchical clustering to see if mets cluster together
