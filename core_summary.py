import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools


def compute_pairwise_distances(input_df):
    distances = []
    for col_1, col_2 in itertools.combinations(input_df.columns, 2):
        distances.append(np.linalg.norm(input_df[col_1].array - input_df[col_2].array))

    return np.mean(distances)


def compute_distances_between_groups(df_1, df_2):
    distances = []
    for col_1, col_2 in itertools.product(df_1.columns, df_2.columns):
        distances.append(np.linalg.norm(df_1[col_1].array - df_2[col_2].array))

    return np.mean(distances)


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dataset
core_df = pd.read_csv(os.path.join(data_dir, 'summary_df_core.csv'))
core_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))

# core_metadata = core_metadata.loc[:, ['Tissue_ID', 'fov']]
# core_df = core_df.merge(core_metadata, on='Tissue_ID')

plot_df = core_df.loc[core_df.metric == 'cluster_broad_freq', :]

#test_ids = plot_df.Tissue_ID.unique()[:3]
#plot_df = plot_df.loc[plot_df.Tissue_ID.isin(test_ids), :]

grouped = plot_df.groupby('Tissue_ID')

# calculate l2 distance between cores from the same timepoint for each patient
distances = []
IDs = []
for name, group in grouped:
    wide_df = pd.pivot(group, index='cell_type', columns='fov', values='value')
    if wide_df.shape[1] > 1:
        distances.append(compute_pairwise_distances(wide_df))
        IDs.append(name)

# plot distances between cores for each timepoint
distances_df = pd.DataFrame({'Tissue_ID': IDs, 'distances': distances})
timepoint_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint.csv'))
timepoint_metadata = timepoint_metadata.loc[:, ['Tissue_ID', 'TONIC_ID', 'Timepoint', 'Localization']]
distances_df = distances_df.merge(timepoint_metadata, on='Tissue_ID')

sns.set_style("white")
sns.set_context("talk")

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.stripplot(distances_df, x='Timepoint', y='distances')
plt.title("Variation in cell prevalence across cores from the same timepoint")
plt.xticks(rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Heterogeneity_across_cores_by_timepoint.png'))
plt.close()