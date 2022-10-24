import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

# this file contains code for summarizing the data on an individual core level


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

# load dataset
core_df = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
core_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))


#
# Evaluate heterogeneity of cell clusters prevalence across cores
#


# calculate l2 distance between cores from the same timepoint for each patient
plot_df = core_df.loc[core_df.metric == 'cluster_broad_freq', :]
grouped = plot_df.groupby('Tissue_ID')

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
timepoints = ['primary_untreated', 'baseline', 'post_induction', 'on_nivo']

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.stripplot(distances_df.loc[distances_df.Timepoint.isin(timepoints), :], x='Timepoint', y='distances')
plt.title("Variation in cell prevalence across cores from the same timepoint and same patient")
plt.xticks(rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Heterogeneity_across_cores_by_timepoint.png'))
plt.close()


# calculate l2 distance between cores from different patients and same timepoint
plot_df = plot_df.merge(timepoint_metadata, on='Tissue_ID')

grouped = plot_df.groupby('Timepoint')

distances = []
IDs = []
for name, group in grouped:
    fov_names = group.fov.unique()
    np.random.shuffle(fov_names)
    batch_size = 3
    for batch_start in range(0, len(group), batch_size):
        # subset df to just include selected FOVs
        batch_df = group.loc[group.fov.isin(fov_names[batch_start: batch_start + batch_size]), :]
        wide_df = pd.pivot(batch_df, index='cell_type', columns='fov', values='value')
        if wide_df.shape[1] > 1:
            distances.append(compute_pairwise_distances(wide_df))
            IDs.append(name)

# plot distances between cores for each timepoint
distances_df_random = pd.DataFrame({'distances': distances, 'Timepoint': IDs})

distances_df = distances_df.loc[:, distances_df.columns.isin(['Timepoint', 'distances'])]

distances_df_random['metric'] = 'distance_across_patients'
distances_df['metric'] = 'distance_within_patients'

distances_df = pd.concat([distances_df_random, distances_df])

sns.set_style("white")
sns.set_context("talk")
timepoints = ['primary_untreated', 'baseline', 'post_induction', 'on_nivo']

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.stripplot(distances_df.loc[distances_df.Timepoint.isin(timepoints), :], x='Timepoint', y='distances', hue='metric', dodge=True)
plt.title("Variation in cell prevalence across cores from the same timepoint and different patients")
plt.xticks(rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Heterogeneity_across_cores_by_timepoint.png'))
plt.close()



