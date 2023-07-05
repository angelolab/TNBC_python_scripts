import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os

from matplotlib_venn import venn3
import matplotlib.pyplot as plt


import os

import natsort
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
from scipy.stats import spearmanr

data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data'
metadata_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/metadata'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/'

cell_ordering = ['Cancer', 'Cancer_EMT', 'Cancer_Other', 'CD4T', 'CD8T', 'Treg', 'T_Other', 'B',
                 'NK', 'M1_Mac', 'M2_Mac', 'Mac_Other', 'Monocyte', 'APC','Mast', 'Neutrophil',
                 'Immune_Other',  'Fibroblast', 'Stroma','Endothelium', 'Other']
# create dataset
core_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core_filtered_deduped.csv'))
cell_table_func = pd.read_csv(os.path.join(data_dir, 'post_processing', 'combined_cell_table_normalized_cell_labels_updated_func_only.csv'))
timepoint_metadata = pd.read_csv(os.path.join(metadata_dir, 'TONIC_data_per_timepoint.csv'))


#
# Figure 1
#

# create venn diagrams
timepoint_metadata = timepoint_metadata.loc[timepoint_metadata.MIBI_data_generated, :]
baseline_ids = timepoint_metadata.loc[timepoint_metadata.Timepoint == 'baseline', 'Patient_ID'].values
induction_ids = timepoint_metadata.loc[timepoint_metadata.Timepoint == 'post_induction', 'Patient_ID'].values
nivo_ids = timepoint_metadata.loc[timepoint_metadata.Timepoint == 'on_nivo', 'Patient_ID'].values

venn3([set(baseline_ids), set(induction_ids), set(nivo_ids)], set_labels=('Baseline', 'Induction', 'Nivo'))
plt.savefig(os.path.join(plot_dir, 'figure1/venn_diagram.pdf'), dpi=300)
plt.close()


#
# Figure 2
#

# cell cluster heatmap

# Markers to include in the heatmap
markers = ["ECAD", "CK17", "CD45", "CD3", "CD4", "CD8", "FOXP3", "CD20", "CD56", "CD14", "CD68",
           "CD163", "CD11c", "HLADR", "ChyTr", "Calprotectin", "FAP", "SMA", "Vim", "Fibronectin",
           "Collagen1", "CD31"]

# Get average across each cell phenotype

# dat = pd.read_csv(os.path.join(data_dir, "post_processing/cell_table_counts.csv"))
# phenotype_col_name = "cell_cluster"
# mean_dat = dat.groupby(phenotype_col_name)[markers].mean()
# mean_dat['phenotype'] = mean_dat.cell_cluster
#
# mean_dat.to_csv(os.path.join(plot_dir, "figure2/cell_cluster_marker_means.csv"))

# read previously generated
mean_dat = pd.read_csv(os.path.join(plot_dir, "figure2/cell_cluster_marker_means.csv"))
mean_dat.index = mean_dat.cell_cluster.values
mean_dat = mean_dat.drop(['cell_cluster'], axis=1)
mean_dat = mean_dat.reindex(cell_ordering)

# set column order
mean_dat = mean_dat[markers]

# Make heatmap
f = sns.clustermap(data=mean_dat,
                   z_score=1,
                   cmap="vlag",
                   center=0,
                   vmin=-3,
                   vmax=3,
                   xticklabels=True,
                   yticklabels=True,
                    row_cluster=False,
            col_cluster=False,)
                   #row_colors=mean_dat.color.values)
f.fig.subplots_adjust(wspace=0.01)
f.ax_cbar.set_position((0.1, 0.82, 0.03, 0.15))
f.ax_heatmap.set_xlabel("Marker")
f.ax_heatmap.set_ylabel("Cell cluster")

f.savefig(os.path.join(plot_dir, "figure1/cell_cluster_marker_manual.pdf"))
plt.close()



# heatmap of functional marker expression per cell type
plot_df = core_df_func.loc[core_df_func.Timepoint.isin(['baseline', 'post_induction', 'on_nivo']), :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_freq', :]
plot_df = plot_df.loc[plot_df.subset == 'all', :]

sp_markers = [x for x in core_df_func.functional_marker.unique() if '__' not in x]
plot_df = plot_df.loc[plot_df.functional_marker.isin(sp_markers), :]

# # compute z-score within each functional marker
# plot_df['zscore'] = plot_df.groupby('functional_marker')['mean'].transform(lambda x: (x - x.mean()) / x.std())

# average the z-score across cell types
plot_df = plot_df.groupby(['cell_type', 'functional_marker']).mean().reset_index()
plot_df = pd.pivot(plot_df, index='cell_type', columns='functional_marker', values='value')
#plot_df = plot_df.apply(lambda x: (x - x.min()), axis=0)

# subtract min from each column, unless that column only has a single value
for col in plot_df.columns:
    if plot_df[col].max() == plot_df[col].min():
        continue
    else:
        plot_df[col] = plot_df[col] - plot_df[col].min()
plot_df = plot_df.apply(lambda x: (x / x.max()), axis=0)
plot_df = plot_df + 0.1

# set index based on cell_ordering
plot_df = plot_df.reindex(cell_ordering)

# set column order
cols = ['PDL1','Ki67','GLUT1','CD45RO', 'CD45RO_CD45RB_ratio','CD69', 'PD1','CD57','TBET', 'TCF1',
        'CD45RB', 'TIM3', 'Fe','HLADR','IDO','CD38','H3K9ac_H3K27me3_ratio', 'HLA1', 'Vim']

plot_df = plot_df[cols]

# plot heatmap
plt.figure(figsize=(12, 10))
#sns.clustermap(plot_df, cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=0, vmax=1, row_cluster=False)
sns.heatmap(plot_df, cmap=sns.color_palette("Greys", as_cmap=True), vmin=0, vmax=1.1)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Functional_marker_heatmap_min_max_normalized.pdf'))
plt.close()
