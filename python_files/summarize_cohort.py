import matplotlib.pyplot as plt
from upsetplot import from_contents, plot
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

metadata_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/metadata'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

timepoint_metadata = pd.read_csv(os.path.join(metadata_dir, 'TONIC_data_per_timepoint.csv'))

# upset plot for cohort summary

# set index for which combinations of timepoints to look at
arrays = [[True,  False, False, False, False, True,  True,  True,  False, False, False, True,],  # primary
          [False, True,  False, False, False, True,  False, True,  False, False, False, False,],   # met
          [False, False, True,  False, False, False, True,  True,  True,  True,  True,  True,],   # baseline
          [False, False, False, True,  False, False, False, False, True,  False, True,  True,],    # induction
          [False, False, False, False, True,  False, False, False, False, True,  True,  True,]]   # run in

counts = []

# format metadata to wide with a the same tissue type for metastatic samples
subset_metadata = timepoint_metadata.loc[timepoint_metadata.Timepoint.isin(['primary_untreated', 'metastasis_1', 'baseline', 'post_induction', 'on_nivo']), :]
subset_metadata = subset_metadata.loc[subset_metadata.MIBI_data_generated, :]
subset_metadata.loc[subset_metadata.Timepoint.isin(['primary_untreated', 'metastasis_1']), 'Location_studytissue'] = 'A'
metadata_wide = pd.pivot(subset_metadata, index='Patient_ID', columns='Timepoint', values='Location_studytissue')

# reorder columns
metadata_wide = metadata_wide[['primary_untreated', 'metastasis_1', 'baseline', 'post_induction', 'on_nivo']]

# loop through pairs, find patients with matching tissue, add to patient_metadata
for keep_idx in zip(*arrays):
    current_wide = metadata_wide.loc[:, np.array(keep_idx)]
    current_wide = current_wide.dropna(axis=0)
    equal_vals = current_wide.apply(lambda x: np.all(x == 'A'), axis=1)
    counts.append(np.sum(equal_vals))

plot_input = pd.DataFrame({'value': counts}, index=arrays)
plot_input.index.names = ['primary_untreated', 'metastasis_1', 'baseline', 'post_induction', 'on_nivo']
plot(plot_input, sum_over='value', subset_size='sum', sort_by='input', sort_categories_by='-input')
plt.savefig(os.path.join(plot_dir, 'upset_plot.pdf'), dpi=300)
plt.close()


# create default upset plot
contents_dict = {'primary': timepoint_metadata.loc[timepoint_metadata.Timepoint == 'primary_untreated', 'Patient_ID'].values,
                 'metastasis': timepoint_metadata.loc[timepoint_metadata.Timepoint == 'metastasis_1', 'Patient_ID'].values,
                 'baseline': timepoint_metadata.loc[timepoint_metadata.Timepoint == 'baseline', 'Patient_ID'].values,
                    'induction': timepoint_metadata.loc[timepoint_metadata.Timepoint == 'post_induction', 'Patient_ID'].values,
                    'nivo': timepoint_metadata.loc[timepoint_metadata.Timepoint == 'on_nivo', 'Patient_ID'].values}

upset_data = from_contents(contents_dict)

plot(upset_data, sort_categories_by='-input')
plt.savefig(os.path.join(plot_dir, 'upset_plot_default.pdf'), dpi=300)
plt.close()

