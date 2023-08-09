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

from python_files.utils import compute_feature_enrichment


data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data'
metadata_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/metadata'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/'
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))

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
total_dfs['importance_score_exp10'] = np.power(10, total_dfs.importance_score)
fig, ax = plt.subplots(figsize=(6,4))
sns.scatterplot(data=total_dfs, x='med_diff', y='log_pval', alpha=1, hue='importance_score', palette=sns.color_palette("Greys", as_cmap=True),
                s=2.5, edgecolor='none', ax=ax)
ax.set_xlim(-3.5, 3.5)
sns.despine()

# add gradient legend
norm = plt.Normalize(total_dfs.importance_score_exp10.min(), total_dfs.importance_score_exp10.max())
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
        plot_df = df.loc[(df.feature_type_detail == feature) &
                                (df.feature_type == feature_type), :]
        plot_df = plot_df.sort_values(by=plot_val, ascending=ascending)
        temp_scores = plot_df.iloc[:x, :][plot_val].values
        scores.append(temp_scores)
        names.append([feature] * len(temp_scores))

    score_df = pd.DataFrame({'score': np.concatenate(scores), 'name': np.concatenate(names)})
    return score_df

func_score_df = get_top_x_features(total_dfs, 'functional_marker', x=5, plot_val='combined_rank', ascending=True)
func_score_df = func_score_df.loc[func_score_df.name.isin(cols), :]
meds = func_score_df.groupby('name').median().sort_values(by='score', ascending=True)
#func_score_df = func_score_df.loc[func_score_df.name.isin(meds.loc[meds.values > 0.85, :].index), :]

fig, ax = plt.subplots(figsize=(4, 5))
sns.stripplot(data=func_score_df, x='name', y='score', ax=ax, order=meds.index, color='black')
sns.boxplot(data=func_score_df, x='name', y='score', order=meds.index, color='grey', ax=ax, showfliers=False, width=0.5)
ax.set_title('Functional Markers Ranking')
#ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure3_functional_marker_enrichment_rank.pdf'))
plt.close()


# get importance score of top 5 examples for densities
density_score_df = get_top_x_features(total_dfs, 'density', x=5, plot_val='combined_rank', ascending=True)

meds = density_score_df.groupby('name').median().sort_values(by='score', ascending=True)

fig, ax = plt.subplots(figsize=(4, 5))
sns.stripplot(data=density_score_df, x='name', y='score', ax=ax, order=meds.index, color='black')
sns.boxplot(data=density_score_df, x='name', y='score', order=meds.index, color='grey', ax=ax, showfliers=False, width=0.5)
ax.set_title('Densities Ranking')
#ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure3_density_enrichment_rank.pdf'))
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

# look at enrichment of specific features

# # spatial features
# total_dfs['spatial_feature'] = total_dfs.feature_type.isin(['density', 'region_diversity', 'cell_diversity',
#                                                             'pixie_ecm', 'fiber', 'compartment_area_ratio', 'ecm_cluster', 'compartment_area',
#                                                             'mixing_score', 'linear_distance', 'ecm_fraction'])
#
# enriched_features = compute_feature_enrichment(feature_df=total_dfs, inclusion_col='top_feature', analysis_col='spatial_feature')
#
# # # plot as a barplot
# # fig, ax = plt.subplots(figsize=(10,8))
# # sns.barplot(data=enriched_features, x='log2_ratio', y='spatial_feature', color='grey', ax=ax)
# # plt.xlabel('Log2 ratio of proportion of top features')
# # ax.set_xlim(-1.5, 1.5)
# # sns.despine()
# # plt.savefig(os.path.join(plot_dir, 'top_feature_enrichment.pdf'))
# # plt.close()
#
#
# # look at enriched cell types
# enriched_features = compute_feature_enrichment(feature_df=total_dfs, inclusion_col='top_feature', analysis_col='cell_pop')
#
# # plot as a barplot
# fig, ax = plt.subplots(figsize=(10,8))
# sns.barplot(data=enriched_features, x='log2_ratio', y='cell_pop', color='grey')
# plt.xlabel('Log2 ratio of proportion of top features')
# ax.set_xlim(-1.5, 1.5)
# sns.despine()
# plt.savefig(os.path.join(plot_dir, 'top_feature_celltype_enrichment.pdf'))
# plt.close()


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


#
# Figure 5
#

# plot top features
top_features = total_dfs.loc[total_dfs.top_feature, :]
top_features = top_features.sort_values('importance_score', ascending=False)

for idx, (feature_name, comparison) in enumerate(zip(top_features.feature_name_unique, top_features.comparison)):
    plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                              (combined_df.Timepoint == comparison), :]

    # plot
    sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'noinduction_responders', 'non-responders'],
                color='grey')
    plt.title(feature_name + ' in ' + comparison)
    plt.savefig(os.path.join(plot_dir, 'top_features_noinduction', f'{idx}_{feature_name}.png'))
    plt.close()


# summarize distribution of top features
top_features_by_comparison = top_features[['feature_name_unique', 'comparison']].groupby('comparison').count().reset_index()
top_features_by_comparison.columns = ['comparison', 'num_features']
top_features_by_comparison = top_features_by_comparison.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_comparison, x='comparison', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure5_num_features_per_comparison.pdf'))
plt.close()


# summarize overlap of top features
top_features_by_feature = top_features[['feature_name_unique', 'comparison']].groupby('feature_name_unique').count().reset_index()
feature_counts = top_features_by_feature.groupby('comparison').count().reset_index()
feature_counts.columns = ['num_comparisons', 'num_features']

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=feature_counts, x='num_comparisons', y='num_features', color='grey', ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure5_num_comparisons_per_feature.pdf'))
plt.close()


def get_top_x_features_by_list(df, detail_names, x=5, plot_val='importance_score', ascending=False):
    scores, names = [], []
    for feature in detail_names:
        keep_idx = np.logical_or(df.feature_type_detail == feature, df.feature_type_detail_2 == feature)
        plot_df = df.loc[keep_idx, :]
        plot_df = plot_df.sort_values(by=plot_val, ascending=ascending)
        temp_scores = plot_df.iloc[:x, :][plot_val].values
        scores.append(temp_scores)
        names.append([feature] * len(temp_scores))

    score_df = pd.DataFrame({'score': np.concatenate(scores), 'name': np.concatenate(names)})
    return score_df


# get importance score of top 5 examples for cell-based features
cell_type_list, cell_prop_list, comparison_list = [], [], []

for groupings in [[['nivo'], ['post_induction__on_nivo', 'on_nivo', 'baseline__on_nivo']],
                  [['baseline'], ['baseline']],
                  [['induction'], ['baseline__post_induction', 'post_induction']]]:

    # score of top 5 features
    name, comparisons = groupings
    # cell_type_features = get_top_x_features_by_list(df=total_dfs.loc[total_dfs.comparison.isin(comparisons)],
    #                                                 detail_names=cell_ordering + ['T', 'Mono_Mac'], x=5, plot_val='combined_rank',
    #                                                 ascending=True)
    #
    # meds = cell_type_features.groupby('name').median().sort_values(by='score', ascending=True)
    #
    # fig, ax = plt.subplots(figsize=(4, 6))
    # sns.stripplot(data=cell_type_features, x='name', y='score', ax=ax, order=meds.index, color='black')
    # sns.boxplot(data=cell_type_features, x='name', y='score', order=meds.index, color='grey', ax=ax, showfliers=False, width=0.5)
    # ax.set_title('Densities Ranking')
    # #ax.set_ylim([0, 1])
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # sns.despine()
    # plt.tight_layout()
    # plt.savefig(os.path.join(plot_dir, 'Figure5_cell_type_rank_{}.pdf'.format(name)))
    # plt.close()

    # proportion of features belonging to each cell type

    current_comparison_features = top_features.loc[top_features.comparison.isin(comparisons), :]
    for cell_type in cell_ordering + ['T', 'Mono_Mac']:
        cell_idx = np.logical_or(current_comparison_features.feature_type_detail == cell_type,
                                    current_comparison_features.feature_type_detail_2 == cell_type)
        cell_type_list.append(cell_type)
        cell_prop_list.append(np.sum(cell_idx) / len(current_comparison_features))
        comparison_list.append(name[0])

proportion_df = pd.DataFrame({'cell_type': cell_type_list, 'proportion': cell_prop_list, 'comparison': comparison_list})

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=proportion_df, x='cell_type', y='proportion', hue='comparison', hue_order=['nivo', 'baseline', 'induction'], ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure5_cell_type_proportion.pdf'))
plt.close()

# plot top featurse across all comparisons
all_top_features = total_dfs.loc[total_dfs.feature_name_unique.isin(top_features.feature_name_unique), :]
all_top_features = all_top_features.pivot(index='feature_name_unique', columns='comparison', values='signed_importance_score')
all_top_features = all_top_features.fillna(0)

sns.clustermap(data=all_top_features, cmap='RdBu_r', vmin=-1, vmax=1, figsize=(10, 10))
plt.savefig(os.path.join(plot_dir, 'top_features_clustermap_all.pdf'))
plt.close()

# identify features with opposite effects at different timepoints
opposite_features = []
for feature in all_top_features.index:
    feature_vals = all_top_features.loc[feature, :]

    # get sign of the feature with the max absolute value
    max_idx = np.argmax(np.abs(feature_vals))
    max_sign = np.sign(feature_vals[max_idx])

    # determine if any features of opposite sign have absolute value > 0.9
    opposite_idx = np.logical_and(np.abs(feature_vals) > 0.85, np.sign(feature_vals) != max_sign)
    if np.sum(opposite_idx) > 0:
        opposite_features.append(feature)

opposite_features = all_top_features.loc[opposite_features, :]

# create connected lineplots for features with opposite effects

# select patients with data at all timepoints
pats = harmonized_metadata.loc[harmonized_metadata.baseline__on_nivo, 'Patient_ID'].unique().tolist()
pats2 = harmonized_metadata.loc[harmonized_metadata.post_induction__on_nivo, 'Patient_ID'].unique().tolist()
#pats = set(pats).intersection(set(pats2))
#pats = set(pats).union(set(pats2))
pats = pats2

for feature in opposite_features.index:
    plot_df = combined_df.loc[(combined_df.feature_name_unique == feature) &
                                        (combined_df.Timepoint.isin(['baseline', 'post_induction', 'on_nivo'])) &
                                        (combined_df.Patient_ID.isin(pats)), :]

    plot_df_wide = plot_df.pivot(index=['Patient_ID', 'Clinical_benefit'], columns='Timepoint', values='raw_mean')
    #plot_df_wide.dropna(inplace=True)
    # divide each row by the baseline value
    #plot_df_wide = plot_df_wide.divide(plot_df_wide.loc[:, 'baseline'], axis=0)
    #plot_df_wide = plot_df_wide.subtract(plot_df_wide.loc[:, 'baseline'], axis=0)
    plot_df_wide = plot_df_wide.reset_index()

    plot_df_norm = pd.melt(plot_df_wide, id_vars=['Patient_ID', 'Clinical_benefit'], value_vars=['post_induction', 'on_nivo'])

    plot_df_1 = plot_df_norm.loc[plot_df_norm.Clinical_benefit == 'No', :]
    plot_df_2 = plot_df_norm.loc[plot_df_norm.Clinical_benefit == 'Yes', :]
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    sns.lineplot(data=plot_df_1, x='Timepoint', y='value', units='Patient_ID', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[0])
    sns.lineplot(data=plot_df_2, x='Timepoint', y='value', units='Patient_ID', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[1])
    sns.lineplot(data=plot_df_norm, x='Timepoint', y='value', units='Patient_ID',  hue='Clinical_benefit', estimator=None, alpha=0.5, marker='o', ax=ax[2])

    # set ylimits
    # ax[0].set_ylim([-0.6, 0.6])
    # ax[1].set_ylim([-0.6, 0.6])
    # ax[2].set_ylim([-0.6, 0.6])

    # add responder and non-responder titles
    ax[0].set_title('non-responders')
    ax[1].set_title('responders')
    ax[2].set_title('combined')

    # set figure title
    fig.suptitle(feature)
    plt.savefig(os.path.join(plot_dir, 'longitudinal_response_raw_{}.png'.format(feature)))
    plt.close()

