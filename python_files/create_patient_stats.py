import os
import json

import ark.utils.misc_utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ark.utils.io_utils import list_folders
from ark.utils.misc_utils import verify_same_elements

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# These scripts generate plots comparing patients across timepoints

# load dfs
timepoint_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'))
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
