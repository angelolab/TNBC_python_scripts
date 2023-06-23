import os

import natsort
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
from scipy.stats import spearmanr

local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data'
metadata_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/metadata'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

cell_ordering = ['Cancer', 'Cancer_EMT', 'Cancer_Other', 'CD4T', 'CD8T', 'Treg', 'T_Other', 'B',
                 'NK', 'M1_Mac', 'M2_Mac', 'Mac_Other', 'Monocyte', 'APC','Mast', 'Neutrophil',
                 'Immune_Other',  'Fibroblast', 'Stroma','Endothelium', 'Other']
# create dataset
core_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core_filtered_deduped.csv'))
cell_table_func = pd.read_csv(os.path.join(data_dir, 'post_processing', 'combined_cell_table_normalized_cell_labels_updated_func_only.csv'))

# create a list of all possible combinations of markers
cell_type_broad = cell_table_func.cell_cluster_broad.unique()
cell_types = cell_table_func.cell_cluster.unique()

ratios = []
for cell_type in cell_types:
    functional_markers = core_df_func.loc[core_df_func.cell_type == cell_type, 'functional_marker'].unique()
    functional_markers = [x for x in functional_markers if x not in ['H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio']]
    observed_ratio = np.zeros((len(functional_markers), len(functional_markers)))
    observed_ratio = pd.DataFrame(observed_ratio, index=functional_markers, columns=functional_markers)

    expected_ratio = observed_ratio.copy()
    cell_table_subset = cell_table_func[cell_table_func['cell_cluster'] == cell_type]

    for marker1, marker2 in combinations(functional_markers, 2):
        # calculate the observed ratio of double positive cells
        marker1_pos = cell_table_subset[marker1].values
        marker2_pos = cell_table_subset[marker2].values
        double_pos_observed = np.logical_and(marker1_pos, marker2_pos)
        observed_ratio.loc[marker1, marker2] = np.sum(double_pos_observed) / len(cell_table_subset)

        # calculated the expected ratio of double positive cells
        double_pos_expected = (np.sum(marker1_pos) / len(cell_table_subset)) * (np.sum(marker2_pos) / len(cell_table_subset))
        expected_ratio.loc[marker1, marker2] = double_pos_expected

    obs_exp = np.log2(observed_ratio / expected_ratio)
    ratios.append(obs_exp)
    # create heatmap of observed and observed vs expected ratios

    g = sns.heatmap(obs_exp, cmap='vlag', vmin=-2, vmax=2)
    plt.title(cell_type)
    plt.savefig(os.path.join(plot_dir, cell_type + '_functional_marker_heatmap.png'), dpi=300)
    plt.close()


# collapse ratios into single array
ratios = np.stack(ratios, axis=0)
avg = np.nanmean(ratios, axis=0)
avg = pd.DataFrame(avg, index=functional_markers, columns=functional_markers)
sns.heatmap(avg, cmap='vlag', vmin=-3, vmax=3)
plt.tight_layout()
plt.savefig('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Plots/Functional_Markers/avg_functional_marker_heatmap.png', dpi=300)
plt.close()





# heatmap of functional marker expression per cell type
plot_df = core_df_func.loc[core_df_func.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_freq', :]
plot_df = plot_df.loc[plot_df.subset == 'all', :]

sp_markers = [x for x in core_df_func.functional_marker.unique() if '__' not in x]
# plot_df = plot_df.loc[plot_df.functional_marker.isin(sp_markers), :]

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

# plot heatmap
plt.figure(figsize=(30, 10))
#sns.heatmap(plot_df, cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=0, vmax=1)
sns.heatmap(plot_df, cmap=sns.color_palette("Greys", as_cmap=True), vmin=0, vmax=1.1)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Functional_marker_heatmap_min_max_normalized.png'))
plt.close()


# check correlation between single positive and double positive cells
working_df = deduped_df.loc[deduped_df.subset == 'all', :]
working_df = working_df.loc[working_df.metric == 'cluster_freq', :]
all_markers = working_df.functional_marker.unique()
dp_markers = [x for x in all_markers if '__' in x]
dp_markers = natsort.natsorted(dp_markers)
cell_types = working_df.cell_type.unique()

correlation_df = pd.DataFrame(index=cell_types, columns=dp_markers)
for marker in dp_markers:
    for cell_type in cell_types:
        current_df = working_df.loc[working_df.cell_type == cell_type, :]

        if marker not in current_df.functional_marker.unique():
            continue

        marker_1, marker_2 = marker.split('__')
        current_df = current_df.loc[current_df.functional_marker.isin([marker, marker_1, marker_2]), :]

        current_df_wide = current_df.pivot(index='fov', columns='functional_marker', values='value')
        if len(current_df_wide.columns) != 3:
            print("cell type: " + cell_type + " marker: " + marker + " has less than 3 columns")
            continue
        current_df_wide['expected'] = current_df_wide[marker_1] * current_df_wide[marker_2]

        cor, p_val = spearmanr(current_df_wide['expected'].values, current_df_wide[marker].values)
        correlation_df.loc[cell_type, marker] = cor

correlation_df = correlation_df.astype(float)

# set figure size
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(correlation_df, cmap='Reds', vmin=0, vmax=1)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Functional_marker_expected_actual_correlation_heatmap.png'), dpi=300)
plt.close()


# plot correlation between individual pairs of features
marker = 'PD1__TIM3'
cell_type = 'CD4T'

current_df = working_df.loc[working_df.cell_type == cell_type, :]
marker_1, marker_2 = marker.split('__')
current_df = current_df.loc[current_df.functional_marker.isin([marker, marker_1, marker_2]), :]

current_df_wide = current_df.pivot(index='fov', columns='functional_marker', values='value')
current_df_wide['expected'] = current_df_wide[marker_1] * current_df_wide[marker_2]
sns.scatterplot(x='expected', y=marker, data=current_df_wide)
plt.savefig(os.path.join(plot_dir, 'CD4T_TIM3_PD1_DP_expected_correlation.png'), dpi=300)
plt.close()


# generate correlation matrix between all features
working_df['feature_name'] = working_df['cell_type'] + '__' + working_df['functional_marker']
working_df_subset = working_df.loc[~working_df.feature_name.isin(exclude_features), :]
working_df_wide = working_df_subset.pivot(index='fov', columns='feature_name', values='value')

#working_df_wide = working_df_wide.loc[:, working_df_wide.columns.str.contains('HLA1')]
corr_df = working_df_wide.corr(method='spearman')

# replace Nans
corr_df = corr_df.fillna(0)


clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(plot_dir, 'spearman_correlation_dp_functional_markers_clustermap.png'), dpi=300)
plt.close()

# get names of features from clustergrid
feature_names = clustergrid.data2d.columns

clustergrid_small = sns.clustermap(corr_df.loc[feature_names[440:480], feature_names[440:480]], cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid_small.savefig(os.path.join(plot_dir, 'spearman_correlation_dp_functional_markers_clustermap_small_3.png'), dpi=300)
plt.close()


corr_df.loc[['B__TBET', 'B__TBET__TIM3'],['B__TBET', 'B__TBET__TIM3']]
# remove correlated features
exclude_features = []
cell_types = working_df.cell_type.unique()
for cell_type in cell_types:
    for marker in dp_markers:
        marker_1, marker_2 = marker.split('__')
        current_df = working_df.loc[working_df.cell_type == cell_type, :]
        current_df = current_df.loc[current_df.functional_marker.isin([marker, marker_1, marker_2]), :]
        if len(current_df) == 0:
            continue
        if marker not in current_df.functional_marker.unique():
            continue

        current_df_wide = current_df.pivot(index='fov', columns='functional_marker', values='value')

        if len(current_df_wide.columns) != 3:
            exclude_features.append(cell_type + '__' + marker)
            continue

        corr_1, _ = spearmanr(current_df_wide[marker_1].values, current_df_wide[marker].values)
        corr_2, _ = spearmanr(current_df_wide[marker_2].values, current_df_wide[marker].values)

        if (corr_1 > 0.8) | (corr_2 > 0.8):
            exclude_features.append(cell_type + '__' + marker)
