import os
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# These scripts generate plots comparing patients across timepoints

# load dfs
timepoint_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'))
timepoint_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_timepoint.csv'))
patient_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_patient.csv'))

# compute ratio between timepoints
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint.isin(['primary_untreated', 'baseline']), :]
plot_df = plot_df.loc[plot_df.TONIC_ID.isin(patient_metadata.loc[patient_metadata.primary_baseline, 'Study_ID'])]
plot_df = plot_df.loc[(plot_df.metric == 'cluster_broad_freq'), :]

grouped = pd.pivot(plot_df, index=['TONIC_ID', 'cell_type'], columns='Timepoint', values='mean')
grouped['ratio'] = np.log2(grouped['baseline'] / grouped['primary_untreated'])
grouped.reset_index(inplace=True)

sns.catplot(grouped, x='cell_type', y='ratio', aspect=2)

plt.title('Ratio of baseline mets to primary tumors for broad clusters')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Evolution_primary_baseline_ratio_cell_cluster_broad.png'))
plt.close()


# compute ratio of specific metrics between specified timepoints
def compute_functional_marker_ratio(long_df, include_timepoints, metric):
    """Computes the ratio between the specified timepoints for the provided metric"""

    plot_df = long_df.loc[long_df.Timepoint.isin(include_timepoints), :]
    plot_df = plot_df.loc[plot_df.TONIC_ID.isin(patient_metadata.loc[patient_metadata.primary_baseline, 'Study_ID'])]
    plot_df = plot_df.loc[(plot_df.metric == metric), :]

    grouped = pd.pivot(plot_df, index=['TONIC_ID', 'cell_type', 'functional_marker'],
                       columns='Timepoint', values='mean')
    grouped['ratio'] = np.log2(grouped[include_timepoints[1]] / grouped[include_timepoints[0]])
    grouped.reset_index(inplace=True)

    return grouped

# compute ratio of functional markers
func_ratio_df = compute_functional_marker_ratio(timepoint_df_func, ['primary_untreated', 'baseline'], 'cluster_freq')
func_ratio_df = func_ratio_df.loc[~func_ratio_df.functional_marker.isin(['PDL1_tumor_dim', 'H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio']), :]

sns.catplot(func_ratio_df, x='functional_marker', y='ratio', aspect=2, kind='box', col='cell_type',
            col_wrap=5)

# set xtick labels to be vertical
for ax in plt.gcf().axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
plt.savefig(os.path.join(plot_dir, 'Evolution_primary_baseline_ratio_functional_markers.png'))
plt.close()

# primary/baseline dotplot
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint.isin(['primary', 'baseline']), :]
plot_df = plot_df.loc[plot_df.TONIC_ID.isin(primary_baseline)]
plot_df = plot_df.loc[(plot_df.metric == 'cluster_broad_freq') & (plot_df.cell_type == 'tumor'), :]

sns.lineplot(data=plot_df, x='Timepoint', y='mean', hue='TONIC_ID', palette=['Black'],
             legend=False)
plt.title("Change in tumor cell proportion across timepoints", fontsize=15)
plt.savefig(os.path.join(plot_dir, 'Evolution_primary_baseline_tumor_proportion_lineplot.png'))
plt.close()

# baseline/induction/nivo dotplot
plot_df = timepoint_df.loc[timepoint_df.Timepoint.isin(['baseline', 'post_induction', 'on_nivo']), :]
plot_df = plot_df.loc[plot_df.TONIC_ID.isin(baseline_induction_nivo)]
plot_df = plot_df.loc[(plot_df.metric == 'cluster_broad_freq') & (plot_df.cell_type == 'tumor'), :]
sns.set_style("white")
sns.set_context("talk")
sns.lineplot(data=plot_df, x='Timepoint', y='mean', hue='TONIC_ID', palette=['Black'],
             legend=False)
plt.title("Change in tumor cell proportion across timepoints")

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Evolution_baseline_induction_nivo_tumor_proportion_lineplot.png'))
plt.close()

# facetted across all cell types
plot_df = timepoint_df.loc[timepoint_df.Timepoint.isin(['primary', 'baseline']), :]
plot_df = plot_df.loc[(plot_df.TONIC_ID.isin(primary_baseline)) & (plot_df.metric == 'cluster_broad_freq'), :]
g = sns.FacetGrid(plot_df, col='cell_type', col_wrap=4, hue='TONIC_ID', palette=['Black'], sharey=False)
g.map(sns.lineplot, 'Timepoint', 'mean')


plt.savefig(os.path.join(plot_dir, 'Evolution_primary_baseline_facet_proportion_lineplot.png'))
plt.close()


# catplot of cell prevalance per patient, stratified by disease stage
