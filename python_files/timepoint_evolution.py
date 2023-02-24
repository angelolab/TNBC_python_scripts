import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'


# combine overlays together into a single image for easier viewing of what changes are happening over time
cluster_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay'
plot_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/baseline_nivo_overlay'
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'harmonized_metadata.csv'))

#patients = harmonized_metadata.loc[harmonized_metadata.primary_baseline == True, 'TONIC_ID'].unique()
patients = harmonized_metadata.loc[harmonized_metadata.baseline_on_nivo == True, 'TONIC_ID'].unique()
fov_df = harmonized_metadata.loc[harmonized_metadata.TONIC_ID.isin(patients), ['TONIC_ID', 'fov', 'Timepoint']]
fov_df = fov_df.loc[fov_df.fov.isin(cell_table_clusters.fov.unique())]
for patient in patients:

    # get all primary samples
    timepoint_1 = fov_df.loc[(fov_df.TONIC_ID == patient) & (fov_df.Timepoint == 'baseline'), 'fov'].unique()
    timepoint_2 = fov_df.loc[(fov_df.TONIC_ID == patient) & (fov_df.Timepoint == 'on_nivo'), 'fov'].unique()

    max_len = max(len(timepoint_1), len(timepoint_2))

    fig, axes = plt.subplots(2, max_len, figsize=(max_len*5, 10))
    for i in range(len(timepoint_1)):
        try:
            axes[0, i].imshow(plt.imread(os.path.join(cluster_dir, timepoint_1[i] + '.png')))
            axes[0, i].axis('off')
            axes[0, i].set_title('Baseline')
        except:
            print('No primary image for {}'.format(patient))

    for i in range(len(timepoint_2)):
        try:
            axes[1, i].imshow(plt.imread(os.path.join(cluster_dir, timepoint_2[i] + '.png')))
            axes[1, i].axis('off')
            axes[1, i].set_title('On Nivo')
        except:
            print('No baseline image for {}'.format(patient))

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'TONIC_{}.png'.format(patient)), dpi=300)
    plt.close()
