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

# load json file containing patient IDs for each grouping
with open(os.path.join(data_dir, 'cohort_ids.json'), mode='r') as jp:
    id_dict = json.load(jp)

primary_ids = id_dict['primary']
baseline_ids = id_dict['baseline']
induction_ids = id_dict['induction']
nivo_ids = id_dict['nivo']
ln_pos_ids = id_dict['ln_pos']
ln_neg_ids = id_dict['ln_neg']

primary_baseline = id_dict['primary_baseline']
baseline_induction = id_dict['baseline_induction']
baseline_nivo = id_dict['baseline_nivo']
baseline_induction_nivo = id_dict['baseline_ind_nivo']

# annotate timepoint-level dataset with necessary colum
timepoint_df = pd.read_csv(os.path.join(data_dir, 'summary_df_timepoint.csv'))
timepoint_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint.csv'))
timepoint_metadata = timepoint_metadata.loc[:, ['Tissue_ID', 'TONIC_ID', 'Timepoint']]
timepoint_df = timepoint_df.merge(timepoint_metadata, on='Tissue_ID')


# compute ratio between timepoints
plot_df = timepoint_df.loc[timepoint_df.Timepoint.isin(['primary', 'baseline']), :]
plot_df = plot_df.loc[plot_df.TONIC_ID.isin(primary_baseline)]
plot_df = plot_df.loc[(plot_df.metric == 'cluster_broad_freq'), :]
grouped = plot_df.groupby(['Timepoint', 'TONIC_ID'])

grouped = pd.pivot(plot_df, index=['TONIC_ID', 'cell_type'], columns='Timepoint', values='mean')
grouped['ratio'] = np.log2(grouped['baseline'] / grouped['primary'])
grouped.reset_index(inplace=True)

sns.catplot(grouped, x='cell_type', y='ratio', aspect=1.7)
plt.title('Ratio of between baseline mets and primary tumors for major cell proportions')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Evolution_primary_baseline_ratio_cell_cluster_broad.png'))
plt.close()



# primary/baseline dotplot
plot_df = timepoint_df.loc[timepoint_df.Timepoint.isin(['primary', 'baseline']), :]
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


# density plot of two variables colored by covariate
https://seaborn.pydata.org/examples/multiple_bivariate_kde.html

# dotplot single value estimate across many variables
https://seaborn.pydata.org/examples/pairgrid_dotplot.html


# create a palette in seaborn
my_colors = {'b_cell': 'red', 'granulocyte': 'green', 'mono_macs': 'blue', 'nk': 'orange', 'other': 'yellow', 'stroma': 'purple',
       't_cell': 'magenta', 'tumor': 'black'}
palette = my_colors


# create colormap for matplotlib based on colormap
# Get Unique continents
color_labels = df['continent'].unique()

# List of colors in the color palettes
rgb_values = sns.color_palette("Set2", 4)

# Map continents to the colors
color_map = dict(zip(color_labels, rgb_values))

# Finally use the mapped values
plt.scatter(df['population'], df['Area'], c=df['continent'].map(color_map))