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
                 'Fibroblast', 'Stroma','Endothelium']
# create dataset
core_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core_filtered_all_combos.csv'))
#cell_table_func = pd.read_csv(os.path.join(data_dir, 'post_processing', 'combined_cell_table_normalized_cell_labels_updated_func_only.csv'))
timepoint_metadata = pd.read_csv(os.path.join(metadata_dir, 'TONIC_data_per_timepoint.csv'))
harmonized_metadata = pd.read_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'))

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

# cell_counts = pd.read_csv(os.path.join(data_dir, "post_processing/cell_table_counts.csv"))
# phenotype_col_name = "cell_cluster"
# keep_fovs = harmonized_metadata.loc[harmonized_metadata.Timepoint.isin(['baseline', 'post_induction', 'on_nivo']), 'fov'].values
# cell_counts = cell_counts.loc[cell_counts.fov.isin(keep_fovs), :]
# mean_counts = cell_counts.groupby(phenotype_col_name)[markers].mean()
# mean_counts.to_csv(os.path.join(plot_dir, "figure2/cell_cluster_marker_means.csv"))

# read previously generated
mean_counts = pd.read_csv(os.path.join(plot_dir, "figure2/cell_cluster_marker_means.csv"))
mean_counts = mean_counts.reindex(cell_ordering)

# set column order
mean_counts = mean_counts[markers]

# Make heatmap
f = sns.clustermap(data=mean_counts,
                   z_score=1,
                   cmap="vlag",
                   center=0,
                   vmin=-3,
                   vmax=3,
                   xticklabels=True,
                   yticklabels=True,
                    row_cluster=False,
               col_cluster=False)
plt.tight_layout()
# f.fig.subplots_adjust(wspace=0.01)
# f.ax_cbar.set_position((0.1, 0.82, 0.03, 0.15))
# f.ax_heatmap.set_xlabel("Marker")
# f.ax_heatmap.set_ylabel("Cell cluster")

f.savefig(os.path.join(plot_dir, "figure2_cell_cluster_marker_manual.pdf"))
plt.close()



# heatmap of functional marker expression per cell type
plot_df = core_df_func.loc[core_df_func.Timepoint.isin(['baseline', 'post_induction', 'on_nivo']), :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_freq', :]
plot_df = plot_df.loc[plot_df.subset == 'all', :]
plot_df = plot_df.loc[~plot_df.functional_marker.isin(['Vim', 'CD45RO_CD45RB_ratio', 'H3K9ac_H3K27me3_ratio', 'HLA1'])]

#sp_markers = [x for x in core_df_func.functional_marker.unique() if '__' not in x]
#plot_df = plot_df.loc[plot_df.functional_marker.isin(sp_markers), :]

# # compute z-score within each functional marker
# plot_df['zscore'] = plot_df.groupby('functional_marker')['mean'].transform(lambda x: (x - x.mean()) / x.std())

# average the z-score across cell types
plot_df = plot_df.groupby(['cell_type', 'functional_marker']).mean().reset_index()
plot_df = pd.pivot(plot_df, index='cell_type', columns='functional_marker', values='value')

# subtract min from each column, unless that column only has a single value
# for col in plot_df.columns:
#     if plot_df[col].max() == plot_df[col].min():
#         continue
#     else:
#         plot_df[col] = plot_df[col] - plot_df[col].min()
# plot_df = plot_df.apply(lambda x: (x / x.max()), axis=0)
# plot_df = plot_df + 0.1

# set index based on cell_ordering
plot_df = plot_df.reindex(cell_ordering)

# set column order
# cols = ['PDL1','Ki67','GLUT1','CD45RO', 'CD45RO_CD45RB_ratio','CD69', 'PD1','CD57','TBET', 'TCF1',
#         'CD45RB', 'TIM3', 'Fe','HLADR','IDO','CD38','H3K9ac_H3K27me3_ratio', 'HLA1', 'Vim']

cols = ['PDL1','Ki67','GLUT1','CD45RO','CD69', 'PD1','CD57','TBET', 'TCF1',
        'CD45RB', 'TIM3', 'Fe','IDO','CD38']

plot_df = plot_df[cols]

# plot heatmap
#sns.clustermap(plot_df, cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=0, vmax=1, row_cluster=False)
f = sns.clustermap(data=plot_df, z_score=1,cmap="vlag", center=0, vmin=-3, vmax=3, xticklabels=True, yticklabels=True,
                    row_cluster=False,col_cluster=False)

#sns.heatmap(plot_df, cmap=sns.color_palette("Greys", as_cmap=True), vmin=0, vmax=1.1)
plt.tight_layout()
f.savefig(os.path.join(plot_dir, 'Figure2_Functional_marker_heatmap.pdf'))
plt.close()

# plot combined heatmap
# combine together
combined_df = pd.concat([mean_counts, plot_df], axis=1)

# plot heatmap
sns.clustermap(combined_df, z_score=1, cmap="vlag", center=0, vmin=-3, vmax=3, xticklabels=True, yticklabels=True,
                    row_cluster=False,col_cluster=False)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure2_combined_heatmap.pdf'))
plt.close()


# create barplot with total number of cells per cluster
plot_df = core_df_cluster.loc[core_df_cluster.Timepoint.isin(['baseline', 'post_induction', 'on_nivo']), :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_count', :]
plot_df = plot_df.loc[plot_df.subset == 'all', :]
plot_df = plot_df.loc[plot_df.cell_type.isin(cell_ordering), :]

plot_df_sum = plot_df[['cell_type', 'value']].groupby(['cell_type']).sum().reset_index()
plot_df_sum.index = plot_df_sum.cell_type
plot_df_sum = plot_df_sum.reindex(cell_ordering)
plot_df_sum['logcounts'] = np.log10(plot_df_sum['value'])

# plot barplot
fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=plot_df_sum, y='cell_type', x='value', color='grey', ax=ax)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure2_cell_count_barplot.pdf'))


#
# Figure 3
#

total_dfs = pd.read_csv(os.path.join(data_dir, 'nivo_outcomes/outcomes_df.csv'))


# plot total volcano
fig, ax = plt.subplots(figsize=(6,4))
sns.scatterplot(data=total_dfs, x='med_diff', y='log_pval', alpha=0.7, hue='importance_score', palette=sns.color_palette("Greys", as_cmap=True), ax=ax)
ax.set_xlim(-3, 3)
sns.despine()

# add gradient legend
norm = plt.Normalize(total_dfs.importance_score.min(), total_dfs.importance_score.max())
sm = plt.cm.ScalarMappable(cmap="Greys", norm=norm)
ax.get_legend().remove()
ax.figure.colorbar(sm, ax=ax)

plt.savefig(os.path.join(plot_dir, 'Figure3_volcano.pdf'))
plt.close()

# plot specific highlighted volcanos

# plot diversity volcano
total_dfs['diversity'] = total_dfs.feature_type.isin(['region_diversity', 'cell_diversity'])

fig, ax = plt.subplots(figsize=(2,2))
sns.scatterplot(data=total_dfs, x='med_diff', y='log_pval', hue='diversity', alpha=0.7, palette=['lightgrey', 'black'], s=10)
ax.set_xlim(-3, 3)
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Figure3_volcano_diversity.pdf'))
plt.close()

# plot phenotype volcano
total_dfs['phenotype'] = total_dfs.feature_type_broad == 'phenotype'

fig, ax = plt.subplots(figsize=(2,2))
sns.scatterplot(data=total_dfs, x='med_diff', y='log_pval', hue='phenotype', alpha=0.7, palette=['lightgrey', 'black'], s=10)
ax.set_xlim(-3, 3)
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Figure3_volcano_phenotype.pdf'))
plt.close()


# plot density volcano
total_dfs['density'] = total_dfs.feature_type.isin(['density'])

fig, ax = plt.subplots(figsize=(2,2))
sns.scatterplot(data=total_dfs, x='med_diff', y='log_pval', hue='density', alpha=0.7, palette=['lightgrey', 'black'], s=10)
ax.set_xlim(-3, 3)
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Figure3_volcano_density.pdf'))
plt.close()

# plot proportion volcano
total_dfs['proportion'] = total_dfs.feature_type.isin(['density_ratio', 'compartment_area_ratio', 'density_proportion'])

fig, ax = plt.subplots(figsize=(2,2))
sns.scatterplot(data=total_dfs, x='med_diff', y='log_pval', hue='proportion', alpha=0.7, palette=['lightgrey', 'black'], s=10)
ax.set_xlim(-3, 3)
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Figure3_volcano_proportion.pdf'))
plt.close()

# plot top features
plot_features = total_dfs.iloc[:20, :]

sns.barplot(data=plot_features, y='feature_name_unique', x='importance_score', hue='feature_type')


# get importance score of top 5 examples for functional markers

def get_top_x_features(df, feature_type, x=5, plot_val='importance_score', ascending=False):
    features = df.loc[df.feature_type == feature_type, 'feature_type_detail'].unique()

    scores, names = [], []
    for feature in features:
        plot_df = total_dfs.loc[(total_dfs.feature_type_detail == feature) &
                                (total_dfs.feature_type == feature_type), :]
        plot_df = plot_df.sort_values(by=plot_val, ascending=ascending)
        temp_scores = plot_df.iloc[:x, :][plot_val].values
        scores.append(temp_scores)
        names.append([feature] * len(temp_scores))

    score_df = pd.DataFrame({'score': np.concatenate(scores), 'name': np.concatenate(names)})
    return score_df

func_score_df = get_top_x_features(total_dfs, 'functional_marker', x=5, plot_val='combined_rank', ascending=True)
func_score_df = func_score_df.loc[func_score_df.name.isin(cols), :]
meds = func_score_df.groupby('name').median().sort_values(by='score', ascending=True)
func_score_df = func_score_df.loc[func_score_df.name.isin(meds.loc[meds.values > 0.85, :].index), :]

fig, ax = plt.subplots(figsize=(4, 6))
sns.stripplot(data=func_score_df, x='name', y='score', ax=ax, order=meds.loc[meds.values > 0.85, :].index, color='black')
sns.boxplot(data=func_score_df, x='name', y='score', order=meds.loc[meds.values > 0.85, :].index, color='grey', ax=ax, showfliers=False, width=0.5)
ax.set_title('Functional Markers Ranking')
ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure3_functinal_marker_enrichment.pdf'))
plt.close()


# get importance score of top 5 examples for densities
density_score_df = get_top_x_features(total_dfs, 'density', x=5)

meds = density_score_df.groupby('name').median().sort_values(by='score', ascending=False)

fig, ax = plt.subplots(figsize=(4, 6))
sns.stripplot(data=density_score_df, x='name', y='score', ax=ax, order=meds.index, color='black')
sns.boxplot(data=density_score_df, x='name', y='score', order=meds.index, color='grey', ax=ax, showfliers=False, width=0.5)
ax.set_title('Densities Ranking')
ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure3_density_enrichment.pdf'))
plt.close()

# get importance score of top 5 examples for diversity
diversity_score_df = get_top_x_features(total_dfs, 'cell_diversity', x=5)
meds = diversity_score_df.groupby('name').median().sort_values(by='score', ascending=False)

fig, ax = plt.subplots(figsize=(4, 6))
sns.stripplot(data=diversity_score_df, x='name', y='score', ax=ax, order=meds.index, color='black')
sns.boxplot(data=diversity_score_df, x='name', y='score', order=meds.index, color='grey', ax=ax, showfliers=False, width=0.5)
ax.set_title('Diversity Ranking')
ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure3_diversity_enrichment.pdf'))
plt.close()

#
# Figure 4
#

# plot specific top features
combined_df = pd.read_csv(os.path.join(data_dir, 'nivo_outcomes/combined_df.csv'))

# PDL1+__APC in induction
feature_name = 'PDL1+__APC'
timepoint = 'post_induction'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                                    (combined_df.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='grey', ax=ax, showfliers=False)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([0, 1])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure4_feature_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()

# change in CD8T density in cancer border
feature_name = 'CD8T__cluster_density__cancer_border'
timepoint = 'post_induction__on_nivo'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                            (combined_df.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='grey', ax=ax, showfliers=False)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([-.05, .2])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure4_feature_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()

# diversity of stroma in on nivo
feature_name = 'cluster_broad_diversity_cancer_border'
timepoint = 'post_induction'

plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                                    (combined_df.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='grey', ax=ax, showfliers=False)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([0, 2])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure4_feature_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()


# ratio of stroma to t cells
feature_name = 'Stroma__T__ratio__cancer_core'
timepoint = 'on_nivo'

plot_df = timepoint_features.loc[(timepoint_features.feature_name_unique == feature_name) &
                                    (timepoint_features.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(2, 4))
sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='grey', ax=ax)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([-10, 10])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'response_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()


# ratio of cancer to t cells
feature_name = 'Cancer__T__ratio'
timepoint = 'on_nivo'

plot_df = timepoint_features.loc[(timepoint_features.feature_name_unique == feature_name) &
                                    (timepoint_features.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='grey', ax=ax)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([-10, 15])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'response_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()



# # p value evaluation
# df = pd.read_csv('/Users/noahgreenwald/Downloads/daisy_data/feature_ranking_ttest.csv')
# plot_df = df.loc[df.comparison == 'post_induction__on_nivo', :]
# plot_df['rank'] = np.arange(1, plot_df.shape[0] + 1)
#
# sns.scatterplot(data=plot_df.iloc[:100], x='rank', y='pval')
# # add a line with slope of alpha / n
# slope = 0.1 / len(plot_df)
#
# plt.plot([0, 100], [0, 100 * slope], color='red')
#
# # calculate rank
# plot_df['pval_rank'] = plot_df.log_pval.rank(ascending=False)
# plot_df['cor_rank'] = plot_df.med_diff.abs().rank(ascending=False)
# plot_df['combined_rank'] = (plot_df.pval_rank.values + plot_df.cor_rank.values) / 2
#
#
# # p value evaluation
# df2 = pd.read_csv('/Users/noahgreenwald/Downloads/daisy_data/feature_ranking_manwhitney.csv')
# plot_df2 = df2.loc[df2.comparison == 'post_induction__on_nivo', :]
# plot_df2['rank'] = np.arange(1, plot_df2.shape[0] + 1)
#
# sns.scatterplot(data=plot_df2.iloc[:100], x='rank', y='pval')
# # add a line with slope of alpha / n
# slope = 0.1 / len(plot_df2)
#
# plt.plot([0, 100], [0, 100 * slope], color='red')
#
# # calculate rank
# plot_df2['pval_rank'] = plot_df2.log_pval.rank(ascending=False)
# plot_df2['cor_rank'] = plot_df2.med_diff.abs().rank(ascending=False)
# plot_df2['combined_rank_rank'] = (plot_df2.pval_rank.values + plot_df2.cor_rank.values) / 2
#
#
#
# combined_df = pd.merge(plot_df, plot_df2[['comparison', 'feature_name_unique', 'combined_rank_rank']], on=['comparison', 'feature_name_unique'], how='outer')
# combined_df = combined_df.sort_values(by='combined_rank_rank', ascending=True)


p_df = total_dfs.loc[total_dfs.pval < 0.05, :]