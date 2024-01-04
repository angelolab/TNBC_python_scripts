# File with code for generating supplementary plots
import os
import random

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from ark.utils.plot_utils import cohort_cluster_plot

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# Panel validation


# ROI selection


# QC

# Image processing


# Cell identification and classification
cell_table = pd.read_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files/cell_table_clusters.csv')
cluster_order = {'Cancer': 0, 'Cancer_EMT': 1, 'Cancer_Other': 2, 'CD4T': 3, 'CD8T': 4, 'Treg': 5,
                'T_Other': 6, 'B': 7, 'NK': 8, 'M1_Mac': 9, 'M2_Mac': 10, 'Mac_Other': 11,
                'Monocyte': 12, 'APC': 13, 'Mast': 14, 'Neutrophil': 15, 'Fibroblast': 16,
                'Stroma': 17, 'Endothelium': 18, 'Other': 19, 'Immune_Other': 20
                }
cell_table = cell_table.sort_values(by=['cell_cluster'], key=lambda x: x.map(cluster_order))

save_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs'

## cell cluster counts
sns.histplot(data=cell_table, x="cell_cluster")
matplotlib.pyplot.title("Cell Cluster Counts")
matplotlib.pyplot.xlabel("Cell Cluster")
matplotlib.pyplot.xticks(rotation=75)
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(os.path.join(save_dir, "cells_per_cluster.png"), dpi=300)

## fov cell counts
cluster_counts = np.unique(cell_table.fov, return_counts=True)[1]
matplotlib.pyplot.figure(figsize=(8, 6))
g = sns.histplot(data=cluster_counts, kde=True)
matplotlib.pyplot.title("Histogram of Cell Counts per Image")
matplotlib.pyplot.xlabel("Number of Cells in an Image")
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(os.path.join(save_dir, "cells_per_fov.png"), dpi=300)

## cell type composition by tissue location of met
meta_data = pd.read_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files/harmonized_metadata.csv')
meta_data = meta_data[['fov', 'Patient_ID', 'Timepoint', 'Localization']]

all_data = cell_table.merge(meta_data, on=['fov'], how='left')
base_data = all_data[all_data.Timepoint == 'baseline']

all_locals = np.unique(base_data.Localization)
dfs = []
for region in all_locals:
    localization_data = base_data[base_data.Localization == region]

    df = localization_data.groupby("cell_cluster_broad").count().reset_index()
    df = df.set_index('cell_cluster_broad').transpose()
    sub_df = df.iloc[:1].reset_index(drop=True)
    sub_df.insert(0, "Localization", [region])
    sub_df['Localization'] = sub_df['Localization'].map(str)
    sub_df = sub_df.set_index('Localization')

    dfs.append(sub_df)
prop_data = pd.concat(dfs).transform(func=lambda row: row / row.sum(), axis=1)

color_map = {'cell_cluster_broad': ['Cancer', 'Stroma', 'Mono_Mac', 'T','Other', 'Granulocyte', 'NK', 'B'],
             'color': ['dimgrey', 'darksalmon', 'red', 'navajowhite',  'yellowgreen', 'aqua', 'dodgerblue', 'darkviolet']}
prop_data = prop_data[color_map['cell_cluster_broad']]

sns.set(rc={'figure.figsize':(14,10)})
colors = color_map['color']
prop_data.plot(kind='bar', stacked=True, color=colors)
matplotlib.pyplot.ticklabel_format(style='plain', useOffset=False, axis='y')
matplotlib.pyplot.gca().set_ylabel("Cell Proportions")
matplotlib.pyplot.gca().set_xlabel("Tissue Location")
matplotlib.pyplot.xticks(rotation=30)
matplotlib.pyplot.title("Cell Type Composition by Tissue Location")
matplotlib.pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(os.path.join(save_dir, "cell_props_by_tissue_loc.png"), dpi=300)

## colored cell cluster masks from random subset of 20 FOVs
random.seed(13)
seg_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output'

all_fovs = list(cell_table['fov'].unique())
fovs = random.sample(all_fovs, 20)
cell_table_subset = cell_table[cell_table.fov.isin(fovs)]

cohort_cluster_plot(
    fovs=fovs,
    seg_dir=seg_dir,
    save_dir=save_dir,
    cell_data=cell_table_subset,
    erode=True,
    fov_col='fov',
    label_col='label',
    cluster_col='cell_cluster_broad',
    seg_suffix="_whole_cell.tiff",
    cmap=color_map,
    display_fig=False,
)

# Functional marker thresholding



# Feature extraction


