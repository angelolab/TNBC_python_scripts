import os

import natsort
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
from scipy.stats import spearmanr
import shutil

plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
output_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/output_files'
analysis_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files'

filtered_morph_df = pd.read_csv(os.path.join(output_dir, 'morph_df_per_core_filtered.csv'))

# heatmap of functional marker expression per cell type
plot_df = filtered_morph_df.loc[filtered_morph_df.Timepoint.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo']), :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_broad_freq', :]

# # compute z-score within each functional marker
# plot_df['zscore'] = plot_df.groupby('functional_marker')['mean'].transform(lambda x: (x - x.mean()) / x.std())

# average the z-score across cell types
plot_df = plot_df.groupby(['cell_type', 'morphology_feature']).mean().reset_index()
plot_df = pd.pivot(plot_df, index='cell_type', columns='morphology_feature', values='value')
plot_df = plot_df.apply(lambda x: (x - x.mean()), axis=0)
plot_df = plot_df.apply(lambda x: (x / x.std()), axis=0)

plot_df = plot_df.apply(lambda x: (x - x.min()), axis=0)
plot_df = plot_df.apply(lambda x: (x / x.max()), axis=0)

# plot heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(plot_df, cmap=sns.color_palette("Greys", as_cmap=True), vmin=-2, vmax=2)
plt.savefig(os.path.join(plot_dir, 'Functional_marker_heatmap_min_max_normalized_lag3.png'))
plt.close()



# generate correlation matrix between all features
working_df = filtered_morph_df.copy()
working_df_subset = working_df.loc[working_df.subset == 'all', :]
working_df_subset = working_df_subset.loc[~working_df_subset.morphology_feature.isin(block1[1:] + block2[1:] + block3[1:] + block4[1:]), :]
working_df_subset = working_df_subset.loc[working_df.metric == 'cluster_broad_freq', :]

for cell_type in working_df_subset.cell_type.unique():
    plot_subset = working_df_subset.loc[working_df_subset.cell_type == cell_type, :]
    df_wide = plot_subset.pivot(index='fov', columns='morphology_feature', values='value')

    corr_df = df_wide.corr(method='spearman')

    clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
    clustergrid.savefig(os.path.join(plot_dir, 'morphology/morph_correlation_subset_{}.png'.format(cell_type)), dpi=300)
    plt.close()

    # # get names of features from clustergrid
    # feature_names = clustergrid.data2d.columns
    #
    # clustergrid_small = sns.clustermap(corr_df.loc[feature_names[440:480], feature_names[440:480]], cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
    # clustergrid_small.savefig(os.path.join(plot_dir, 'spearman_correlation_dp_functional_markers_clustermap_small_3.png'), dpi=300)
    # plt.close()

block1 = ['area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'convex_area', 'equivalent_diameter']

block2 = ['area_nuclear', 'major_axis_length_nuclear', 'minor_axis_length_nuclear', 'perimeter_nuclear', 'convex_area_nuclear', 'equivalent_diameter_nuclear']

block3 = ['eccentricity', 'major_axis_equiv_diam_ratio']

block4 = ['eccentricity_nuclear', 'major_axis_equiv_diam_ratio_nuclear', 'perim_square_over_area_nuclear']


# look at correlation between cell types
working_df_subset = working_df_subset.loc[working_df_subset.metric == 'cluster_freq', :]
plot_subset = working_df_subset.loc[working_df_subset.cell_type.isin(['Cancer_1', 'Cancer_2', 'Cancer_3']), :]
plot_subset['feature_name'] = plot_subset.cell_type + '_' + plot_subset.morphology_feature
df_wide = plot_subset.pivot(index='fov', columns='feature_name', values='value')

corr_df = df_wide.corr(method='spearman')

clustergrid = sns.clustermap(corr_df, cmap='vlag', vmin=-1, vmax=1, figsize=(20, 20))
clustergrid.savefig(os.path.join(plot_dir, 'morphology/morph_correlation_cell_subset_cancer.png'), dpi=300)
plt.close()


# look at images that are high for each feature to assess quality
feature_df = pd.read_csv(os.path.join(analysis_dir, 'feature_ranking.csv'))
feature_name = 'area__Structural'
data_subset = feature_df.loc[feature_df.feature_name_unique == feature_name, :]
data_subset.sort_values(by='raw_value', ascending=False, inplace=True)

out_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/morphology_extremes'
plot_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay'

for i in range(1, 10):
    fov_low = data_subset.iloc[-i, :].fov
    fov_high = data_subset.iloc[i, :].fov

    low_path = os.path.join(plot_dir, fov_low + '.png')
    low_save_path = os.path.join(out_dir, 'low_{}.png'.format(i))
    shutil.copyfile(low_path, low_save_path)

    high_path = os.path.join(plot_dir, fov_high + '.png')
    high_save_path = os.path.join(out_dir, 'high_{}.png'.format(i))
    shutil.copyfile(high_path, high_save_path)

