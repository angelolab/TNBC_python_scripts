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
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core.csv'))


#
# Evaluate heterogeneity of cell clusters prevalence across cores
#


# calculate l2 distance between cores from the same timepoint for each patient
timepoints = ['primary_untreated', 'baseline', 'post_induction', 'on_nivo']

# TODO: Make sure distances are calculated from timepoint Df in addition to core DF for each comparison to ensure that isn't driving the difference
#for metric in ['cluster_broad_freq', 'cluster_freq', 'immune_freq']:
for metric in ['immune_freq']:
    plot_df = core_df_cluster.loc[core_df_cluster.metric == metric, :]
    plot_df = plot_df.loc[plot_df.Timepoint.isin(timepoints)]

    # compute distances between cores for each timepoint from each patient
    grouped = plot_df.groupby(['Tissue_ID', 'Timepoint'])

    distances_df_cores = []

    for name, group in grouped:
        wide_df = pd.pivot(group, index='cell_type', columns='fov', values='value')
        if wide_df.shape[1] > 1:
            distances_df_cores.append(compute_pairwise_distances(input_df=wide_df,
                                                           metadata_names=['Tissue_ID', 'Timepoint'],
                                                           metadata_values=name))
            #
            # dist_long, met_long = compute_pairwise_distances_by_element(wide_df)
            # metadata_long.extend([name] * len(met_long))
            # distances_long.extend(dist_long)
            # cell_type_long.extend(met_long)

    # distances_df_long = pd.DataFrame(metadata_long, columns=['Tissue_ID', 'Timepoint'])
    # distances_df_long['distance'] = distances_long
    # distances_df_long['cell_type'] = cell_type_long


    # compute distances between cores for each timepoint from random patients
    grouped_shuffle_timepoint = plot_df.groupby(['Timepoint'])
    distances_df_shuffle_timepoint = []
    for name, group in grouped_shuffle_timepoint:
        fov_names = group.fov.unique()
        np.random.shuffle(fov_names)

        # select 3 FOVs at a time within each timepoint
        batch_size = 3
        for batch_start in range(0, len(group), batch_size):
            # subset df to just include selected FOVs
            batch_df = group.loc[group.fov.isin(fov_names[batch_start: batch_start + batch_size]), :]
            wide_df = pd.pivot(batch_df, index='cell_type', columns='fov', values='value')
            if wide_df.shape[1] > 1:
                distances_df_shuffle_timepoint.append(compute_pairwise_distances(input_df=wide_df,
                                                                                 metadata_names=['Tissue_ID', 'Timepoint'],
                                                                                 metadata_values=['NA'] + [name]))

    # compute distances between cores fully randomized
    grouped_shuffle = plot_df.loc[plot_df.Timepoint.isin(timepoints), :]
    distances_df_shuffle = []
    fov_names = grouped_shuffle.fov.unique()
    np.random.shuffle(fov_names)

    # select 3 FOVs at a time within each timepoint
    batch_size = 3
    for batch_start in range(0, len(grouped_shuffle), batch_size):
        # subset df to just include selected FOVs
        batch_df = grouped_shuffle.loc[grouped_shuffle.fov.isin(fov_names[batch_start: batch_start + batch_size]), :]
        wide_df = pd.pivot(batch_df, index='cell_type', columns='fov', values='value')
        if wide_df.shape[1] > 1:
            distances_df_shuffle.append(compute_pairwise_distances(input_df=wide_df,
                                                                   metadata_names=['Tissue_ID','Timepoint'],
                                                                   metadata_values=['NA', 'NA']))

    # compute distances between cores within patients across timepoints
    grouped_shuffle_patient = plot_df.groupby(['TONIC_ID'])
    distances_df_shuffle_patient = []

    for name, group in grouped_shuffle_patient:
        wide_df = pd.pivot(group, index='cell_type', columns='fov', values='value')
        if wide_df.shape[1] > 1:
            distances_df_shuffle_patient.append(compute_pairwise_distances(input_df=wide_df,
                                                                 metadata_names=['Tissue_ID',
                                                                                 'Timepoint'],
                                                                 metadata_values=['NA', 'NA']))

    # plot distances between cores for each timepoint
    distances_df_shuffle_patient = pd.concat(distances_df_shuffle_patient)
    distances_df_shuffle_patient['metric'] = 'across time'

    distances_df_shuffle = pd.concat(distances_df_shuffle)
    distances_df_shuffle['metric'] = 'all cores'

    distances_df_shuffle_timepoint = pd.concat(distances_df_shuffle_timepoint)
    distances_df_shuffle_timepoint['metric'] = 'within timepoint'

    distances_df_cores = pd.concat(distances_df_cores)
    distances_df_cores['metric'] = 'within replicate cores'

    distances_df_combined = pd.concat([distances_df_shuffle, distances_df_shuffle_timepoint,
                                       distances_df_cores, distances_df_shuffle_patient])


    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.stripplot(distances_df_combined, x='metric', y='distance', hue='metric')
    #ax = sns.stripplot(distances_df.loc[distances_df.Timepoint.isin(timepoints), :], x='metric', y='distance', dodge=True)
    plt.title("Variation in cell prevalence in {} across cores".format(metric))
    #plt.xticks(rotation=90)

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Heterogeneity_of_{}_across_cores_by_metric.png'.format(metric)))
    plt.close()



