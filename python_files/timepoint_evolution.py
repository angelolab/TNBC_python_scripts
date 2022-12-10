import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dictionary with mapping of cell populations to functional markers
lymphocyte = ['B', 'CD4T', 'CD8T', 'Immune_Other', 'NK', 'T_Other', 'Treg']
cancer = ['Cancer', 'Cancer_EMT', 'Cancer_Other']
monocyte = ['APC', 'M1_Mac', 'M2_Mac', 'Monocyte', 'Mac_Other']
stroma = ['Fibroblast', 'Stroma', 'Endothelium']
granulocyte = ['Mast', 'Neutrophil']

keep_dict = {'CD38': ['B', 'Immune_Other', 'NK', 'Endothelium'], 'CD45RB': lymphocyte, 'CD45RO': lymphocyte,
             'CD57': lymphocyte + cancer, 'CD69': lymphocyte,
             'GLUT1': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'HLA1': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'HLADR': lymphocyte + monocyte, 'IDO': ['APC', 'B'], 'Ki67': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'LAG3': ['B'], 'PD1': lymphocyte, 'PDL1_combined': lymphocyte + monocyte + granulocyte + cancer,
             'TBET': lymphocyte, 'TCF1': lymphocyte, 'TIM3': lymphocyte + monocyte + granulocyte}



# create dataset
timepoint_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'))
timepoint_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_timepoint.csv'))

cluster_evolution = timepoint_df_cluster.loc[timepoint_df_cluster.primary_baseline == True, :]
cluster_evolution = cluster_evolution.loc[cluster_evolution.Timepoint.isin(['primary_untreated', 'baseline']), :]

# compute ratio across relevant metrics
metric = 'cluster_freq'
cluster_evolution_plot = cluster_evolution.loc[cluster_evolution.metric == metric, :]

cells = cluster_evolution_plot.cell_type.unique()

evolution_dfs = []
for cell_type in cells:
    cluster_evolution_wide = cluster_evolution_plot.loc[cluster_evolution_plot.cell_type == cell_type, :]
    cluster_evolution_wide = cluster_evolution_wide.pivot(index=['TONIC_ID'], columns=['Timepoint'], values='mean')
    cluster_evolution_wide.reset_index(inplace=True)

    # drop rows with NAs
    cluster_evolution_wide.dropna(inplace=True)

    # drop rows with NAs
    cluster_evolution_wide['ratio'] = np.log2(cluster_evolution_wide['baseline'] / cluster_evolution_wide['primary_untreated'])
    cluster_evolution_wide['ratio_adjusted'] = np.log2((cluster_evolution_wide['baseline'] + 0.001) / (cluster_evolution_wide['primary_untreated'] + 0.001))
    cluster_evolution_wide['metric'] = metric
    cluster_evolution_wide['cell_type'] = cell_type
    cluster_evolution_wide['comparison'] = 'paired'
    evolution_dfs.append(cluster_evolution_wide)

    # create a copy with ranodmized values in ratio
    cluster_evolution_wide_random = cluster_evolution_wide.copy()
    cluster_evolution_wide_random.baseline = np.random.permutation(cluster_evolution_wide.baseline)
    cluster_evolution_wide_random['ratio_adjusted'] = np.log2((cluster_evolution_wide_random['baseline'] + 0.001) / (cluster_evolution_wide_random['primary_untreated'] + 0.001))
    cluster_evolution_wide_random['metric'] = metric
    cluster_evolution_wide_random['cell_type'] = cell_type
    cluster_evolution_wide_random['comparison'] = 'random'
    evolution_dfs.append(cluster_evolution_wide_random)


evolution_df = pd.concat(evolution_dfs, axis=0)

# plot
g = sns.catplot(x='cell_type', y='ratio_adjusted', data=evolution_df, kind='strip', aspect=2, hue='comparison', dodge=True)
g.set_xticklabels(rotation=90)
# add a horizonal line at y=0
plt.axhline(y=0, color='black', linestyle='--')
g.savefig(os.path.join(plot_dir, '{}_evolution.png'.format(metric)), dpi=300)
plt.close()

# create evolution plot for functional markers
functional_evolution = timepoint_df_func.loc[timepoint_df_func.primary_baseline == True, :]
functional_evolution = functional_evolution.loc[functional_evolution.Timepoint.isin(['primary_untreated', 'baseline']), :]

metric = 'cluster_freq'
functional_evolution_plot = functional_evolution.loc[functional_evolution.metric == metric, :]
markers = functional_evolution_plot.functional_marker.unique()
markers = [x for x in markers if x not in ['CD45RO_CD45RB_ratio', 'H3K9ac_H3K27me3_ratio', 'PDL1_cancer_dim', 'PDL1']]

evolution_dfs = []
for marker in markers:
    functional_evolution_wide_all = functional_evolution_plot.loc[functional_evolution_plot.functional_marker == marker, :]
    keep_cells = keep_dict[marker]
    for cell in keep_cells:
        functional_evolution_wide = functional_evolution_wide_all.loc[functional_evolution_wide_all.cell_type == cell, :]
        functional_evolution_wide = functional_evolution_wide.pivot(index=['TONIC_ID'], columns=['Timepoint'], values='mean')
        functional_evolution_wide.reset_index(inplace=True)

        # replace nan with 0
        functional_evolution_wide.fillna(0, inplace=True)

        # drop rows with NAs
        functional_evolution_wide['ratio'] = np.log2(functional_evolution_wide['baseline'] / functional_evolution_wide['primary_untreated'])
        functional_evolution_wide['ratio_adjusted'] = np.log2((functional_evolution_wide['baseline'] + 0.001) / (functional_evolution_wide['primary_untreated'] + 0.001))
        functional_evolution_wide['metric'] = metric
        functional_evolution_wide['functional_marker'] = marker
        functional_evolution_wide['cell_type'] = cell
        functional_evolution_wide['comparison'] = 'paired'
        evolution_dfs.append(functional_evolution_wide)

        # create a copy with ranodmized values in ratio
        functional_evolution_wide_random = functional_evolution_wide.copy()
        functional_evolution_wide_random.baseline = np.random.permutation(functional_evolution_wide.baseline)
        functional_evolution_wide_random['ratio_adjusted'] = np.log2((functional_evolution_wide_random['baseline'] + 0.001) / (functional_evolution_wide_random['primary_untreated'] + 0.001))
        functional_evolution_wide_random['metric'] = metric
        functional_evolution_wide_random['functional_marker'] = marker
        functional_evolution_wide_random['cell_type'] = cell
        functional_evolution_wide_random['comparison'] = 'random'
        evolution_dfs.append(functional_evolution_wide_random)

functional_evolution_df = pd.concat(evolution_dfs, axis=0)

# plot
for marker in markers:
    functional_evolution_df_marker = functional_evolution_df.loc[functional_evolution_df.functional_marker == marker, :]
    g = sns.catplot(x='cell_type', y='ratio_adjusted', data=functional_evolution_df_marker, kind='strip', aspect=2, hue='comparison', dodge=True)
    g.set_xticklabels(rotation=90)

    # add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--')
    g.savefig(os.path.join(plot_dir, 'functional_{}_{}_evolution.png'.format(metric, marker)), dpi=300)
    plt.close()


# create overlays for specified patients
core_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
plot_patients = cluster_evolution.TONIC_ID.unique()[0:5]

cell_table_short = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_clusters_only.csv'))
for patient in plot_patients:
    patient_metadata = core_metadata.loc[core_metadata.TONIC_ID == str(patient), :]
    patient_metadata = patient_metadata[patient_metadata.Timepoint.isin(['primary_untreated', 'baseline'])]
    patient_metadata = patient_metadata[['fov', 'Timepoint']]

    # convert pandas df to a list for each column
    fovs, timepoints = [patient_metadata[x].tolist() for x in ['fov', 'Timepoint']]

    save_names = ['overlay_{}_{}_{}.png'.format(patient, x, y) for x, y in zip(fovs, timepoints)]

    create_cell_overlay(cell_table_short, seg_folder='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output',
                        fovs=fovs, cluster_col='cell_cluster_broad', plot_dir=plot_dir,
                        save_names=save_names)

