import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

# this file contains code for summarizing the data on an individual core level


def compute_pairwise_distances(input_df, metadata_names, metadata_values):
    distances = []
    for col_1, col_2 in itertools.combinations(input_df.columns, 2):
        distances.append(np.linalg.norm(input_df[col_1].array - input_df[col_2].array))

    # create dataframe with distances and metadata
    return_df = pd.DataFrame({'distance': [np.mean(distances)]})

    for name, value in zip(metadata_names, metadata_values):
        return_df[name] = value

    return return_df


def compute_pairwise_distances_by_element(input_df):
    distances = []
    elements = []
    for element in input_df.index:
        current_df = input_df.loc[[element], :]
        for col_1, col_2 in itertools.combinations(current_df.columns, 2):
            current_distances = np.linalg.norm(current_df[col_1].array - current_df[col_2].array)
            distances.append(np.mean(current_distances))
            elements.append(element)

    return distances, elements


def compute_distances_between_groups(df_1, df_2):
    distances = []
    for col_1, col_2 in itertools.product(df_1.columns, df_2.columns):
        distances.append(np.linalg.norm(df_1[col_1].array - df_2[col_2].array))

    return np.mean(distances)


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# load dataset
core_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
timepoint_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'))
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core.csv'))


#
# Evaluate heterogeneity of cell clusters prevalence across cores
#

# TODO: investigate how tcell_Freq same timepoint dots can be greater than 1
def compute_difference_in_cell_prev(core_df, timepoint_df, metric):
    """Computes the difference in cell prevalances across timepoints"""

    # subset provided DFs
    timepoints = ['primary_untreated', 'baseline', 'post_induction', 'on_nivo']
    core_df_plot = core_df.loc[core_df.Timepoint.isin(timepoints), :]
    timepoint_df_plot = timepoint_df.loc[timepoint_df.Timepoint.isin(timepoints), :]

    core_df_plot = core_df_plot.loc[core_df_plot.metric == metric, :]
    timepoint_df_plot = timepoint_df_plot.loc[timepoint_df_plot.metric == metric, :]

    # compute distances between replicate cores from same patient and timepoint
    grouped = core_df_plot.groupby(['Tissue_ID', 'Timepoint'])
    distances_df_core = []

    for name, group in grouped:
        wide_df = pd.pivot(group, index='cell_type', columns='fov', values='value')
        if wide_df.shape[1] > 1:
            distances_df_core.append(compute_pairwise_distances(input_df=wide_df,
                                                           metadata_names=['Tissue_ID', 'Timepoint'],
                                                           metadata_values=name))

    # compute distances between cores for each timepoint from random patients
    # plot_df_timepoints = main_df.loc[main_df.metric == metric, :]
    # grouped_shuffle_timepoint = plot_df_timepoints.groupby(['Timepoint'])
    # distances_df_shuffle_timepoint = []
    # for name, group in grouped_shuffle_timepoint:
    #     # fov_names = group.fov.unique()
    #     fov_names = group.Tissue_ID.unique()
    #     np.random.shuffle(fov_names)
    #
    #     # select 3 FOVs at a time within each timepoint
    #     batch_size = 3
    #     for batch_start in range(0, len(group), batch_size):
    #         # subset df to just include selected FOVs
    #         # batch_df = group.loc[group.fov.isin(fov_names[batch_start: batch_start + batch_size]), :]
    #         batch_df = group.loc[group.Tissue_ID.isin(fov_names[batch_start: batch_start + batch_size]), :]
    #
    #         wide_df = pd.pivot(batch_df, index='cell_type', columns='Tissue_ID', values='mean')
    #         if wide_df.shape[1] > 1:
    #             distances_df_shuffle_timepoint.append(compute_pairwise_distances(input_df=wide_df,
    #                                                                              metadata_names=['Tissue_ID', 'Timepoint'],
    #                                                                              metadata_values=['NA'] + [name]))

    # compute distances between cores fully randomized
    distances_df_shuffle = []
    fov_names = timepoint_df_plot.Tissue_ID.unique()
    np.random.shuffle(fov_names)

    # select 3 FOVs at a time within each timepoint
    batch_size = 3
    for batch_start in range(0, len(timepoint_df_plot), batch_size):
        # subset df to just include selected FOVs
        batch_df = timepoint_df_plot.loc[timepoint_df_plot.Tissue_ID.isin(fov_names[batch_start: batch_start + batch_size]), :]
        wide_df = pd.pivot(batch_df, index='cell_type', columns='Tissue_ID', values='mean')
        if wide_df.shape[1] > 1:
            distances_df_shuffle.append(compute_pairwise_distances(input_df=wide_df,
                                                                   metadata_names=['Tissue_ID','Timepoint'],
                                                                   metadata_values=['NA', 'NA']))

    # compute distances between cores within patients across timepoints
    grouped_shuffle_patient = timepoint_df_plot.groupby(['TONIC_ID'])
    distances_df_shuffle_patient = []
    for name, group in grouped_shuffle_patient:
        wide_df = pd.pivot(group, index='cell_type', columns='Tissue_ID', values='mean')
        if wide_df.shape[1] > 1:
            distances_df_shuffle_patient.append(compute_pairwise_distances(input_df=wide_df,
                                                                 metadata_names=['Tissue_ID',
                                                                                 'Timepoint'],
                                                                 metadata_values=['NA', 'NA']))

    # plot distances between cores for each timepoint
    distances_df_shuffle_patient = pd.concat(distances_df_shuffle_patient)
    distances_df_shuffle_patient['metric'] = 'Same patients'

    distances_df_shuffle = pd.concat(distances_df_shuffle)
    distances_df_shuffle['metric'] = 'Different patients'

    distances_df_core = pd.concat(distances_df_core)
    distances_df_core['metric'] = 'Same timepoints'

    distances_df_combined = pd.concat([distances_df_shuffle, distances_df_core, distances_df_shuffle_patient])

    return distances_df_combined


distances_df_new = compute_difference_in_cell_prev(core_df=core_df_cluster, timepoint_df=timepoint_df_cluster, metric='tcell_freq')

fig, ax = plt.subplots()
ax = sns.boxplot(distances_df_new, x='metric', y='distance', order=['Same timepoints',  'Same patients', 'Different patients'])
plt.title("Variation in cell prevalence in {} across {}".format(metric, 'condition'))
#plt.xticks(rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Heterogeneity_of_{}_across_{}_by_metric.png'.format(metric, 'timepoint')))
plt.close()



