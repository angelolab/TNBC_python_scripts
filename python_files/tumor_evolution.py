import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

# this file contains code for summarizing the data on an individual core level


def compute_euclidian_distance(input_df):
    # remove rows with any nans
    output_df = input_df.dropna(axis=0)

    # unit normalize each column
    output_df = output_df.apply(lambda x: (x / np.linalg.norm(x)), axis=0)

    # compute euclidian distance
    dist = np.linalg.norm(output_df.values[:, :1] - output_df.values[:, 1:])

    return dist


def compute_pairwise_distances(input_df, metadata_name):
    distances = []
    for col_1, col_2 in itertools.combinations(input_df.columns, 2):
        subset_df = input_df[[col_1, col_2]]
        distances.append(compute_euclidian_distance(subset_df))

    # create dataframe with distances and metadata
    return_df = pd.DataFrame({'distance': [np.mean(distances)],
                              'metadata': [metadata_name]})

    return return_df





local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

# load dataset
fov_df = pd.read_csv(os.path.join(data_dir, 'fov_features_no_compartment.csv'))
timepoint_df = pd.read_csv(os.path.join(data_dir, 'timepoint_features_no_compartment.csv'))
ranked_features = pd.read_csv(os.path.join(data_dir, 'conserved_features/ranked_features_no_compartment.csv'))
fov_pairs = pd.read_csv(os.path.join(data_dir, 'conserved_features/fov_pairs.csv'))
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
harmonized_metadata = harmonized_metadata.loc[harmonized_metadata.MIBI_data_generated, :]
#
# Evaluate heterogeneity of cell clusters prevalence across cores
#

# TODO: investigate how tcell_Freq same timepoint dots can be greater than 1
def compute_difference_in_cell_prev(core_df, timepoint_df, metric):
    """Computes the difference in cell prevalances across timepoints"""

    # subset provided DFs
    timepoints = ['primary_untreated', 'baseline', 'post_induction', 'on_nivo']
    timepoints = ['primary_untreated', 'baseline']
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
        print(name)
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


harmonized_metadata['Tissue_ID_random'] = harmonized_metadata['Tissue_ID'].sample(frac=1).values
harmonized_metadata['TONIC_ID_random'] = harmonized_metadata['TONIC_ID'].sample(frac=1).values


def generate_grouped_distances(sample='fov', group_by='Tissue_ID',
                               harmonized_metadata=harmonized_metadata, data_df=fov_df):
    grouped_distances = []
    grouped = harmonized_metadata[[group_by, sample]].groupby(group_by)
    value = 'normalized_value' if sample == 'fov' else 'normalized_mean'
    for group_name, group_members in grouped:
        if len(group_members) > 1:
            group_df = data_df.loc[data_df[sample].isin(group_members[sample].values), :]
            wide_df = pd.pivot(group_df, index='feature_name_unique', columns=sample, values=value)
            grouped_distances.append(compute_pairwise_distances(input_df=wide_df,
                                                                metadata_name=group_name))
    grouped_distances = pd.concat(grouped_distances)

    return grouped_distances


fov_distances = generate_grouped_distances(sample='fov', group_by='Tissue_ID')
fov_distances_random = generate_grouped_distances(sample='fov', group_by='Tissue_ID_random')

# evaluate difference in similarity between paired and unpaired fovs
include_features = ranked_features.feature_name_unique[ranked_features.combined_rank < 450].values
fov_df_subset = fov_df_subset.loc[fov_df_subset.feature_name_unique.isin(include_features), :]

paired_distances = []
for row in range(len(fov_pairs)):
    fov_1, fov_2 = fov_pairs.iloc[row, :2].values

    fov_df_subset = fov_df.loc[fov_df.fov.isin([fov_1, fov_2]), :]
    wide_df = pd.pivot(fov_df_subset, index='feature_name_unique', columns='fov', values='normalized_value')

    paired_distances.append(compute_euclidian_distance(wide_df))

randomized_distances = []
fov_pairs['fov_random'] = fov_pairs.fov_2
fov_pairs.fov_random = fov_pairs.fov_random.sample(frac=1).values

for row in range(len(fov_pairs)):
    fov_1, fov_3 = fov_pairs.iloc[row, [0, 2]].values

    fov_df_subset = fov_df.loc[fov_df.fov.isin([fov_1, fov_3]), :]
    fov_df_subset = fov_df_subset.loc[fov_df_subset.feature_name_unique.isin(include_features), :]
    wide_df = pd.pivot(fov_df_subset, index='feature_name_unique', columns='fov', values='normalized_value')

    randomized_distances.append(compute_euclidian_distance(wide_df))


