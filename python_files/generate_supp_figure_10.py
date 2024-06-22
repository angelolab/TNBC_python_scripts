import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib_venn import venn2



BASE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/"
SUPPLEMENTARY_FIG_DIR = os.path.join(BASE_DIR, "supplementary_figs")


ranked_features_all = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/feature_ranking.csv'))
ranked_features = ranked_features_all.loc[ranked_features_all.comparison.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
ranked_features = ranked_features.loc[ranked_features.feature_rank_global <= 100, :]

ranked_features = ranked_features.loc[ranked_features.feature_type.isin(['density', 'density_ratio', 'density_proportion']), :]
ranked_features['feature_type'] = ranked_features['feature_type'].replace('density_proportion', 'density_ratio')
ranked_features = ranked_features[['feature_name_unique', 'feature_type']]

ranked_feature_counts = ranked_features.groupby('feature_type').count().reset_index()

# plot
fig, ax = plt.subplots(1, 1, figsize=(2, 4))
sns.barplot(data=ranked_feature_counts, x='feature_type', y='feature_name_unique', color='grey', ax=ax)
sns.despine()
plt.xlabel('Feature Type')
plt.ylabel('Number of Features')
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_10a.pdf'), dpi=300)
plt.close()


# volcano plot for RNA features
ranked_features_df = pd.read_csv(os.path.join(BASE_DIR, 'sequencing_data/genomics_outcome_ranking.csv'))
ranked_features_df = ranked_features_df.loc[ranked_features_df.data_type == 'RNA', :]
ranked_features_df = ranked_features_df.sort_values(by='combined_rank', ascending=True)

ranked_features_df[['feature_name_unique']].to_csv(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_10c.csv'), index=False)

# plot  volcano
fig, ax = plt.subplots(figsize=(3,3))
sns.scatterplot(data=ranked_features_df, x='med_diff', y='log_pval', alpha=1, hue='importance_score',
                palette=sns.color_palette("icefire", as_cmap=True),
                s=2.5, edgecolor='none', ax=ax)
ax.set_xlim(-3, 3)
sns.despine()

# add gradient legend
norm = plt.Normalize(ranked_features_df.importance_score.min(), ranked_features_df.importance_score.max())
sm = plt.cm.ScalarMappable(cmap="icefire", norm=norm)
ax.get_legend().remove()
ax.figure.colorbar(sm, ax=ax)
plt.tight_layout()

plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_10b.pdf'))
plt.close()

# by comparison
top_features = ranked_features_df.iloc[:100, :]

top_features_by_comparison = top_features[['data_type', 'comparison']].groupby(['comparison']).size().reset_index()
top_features_by_comparison.columns = ['comparison', 'num_features']
top_features_by_comparison = top_features_by_comparison.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_comparison, x='comparison', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_10e.pdf'))
plt.close()

# by data type
ranked_features_df = pd.read_csv(os.path.join(BASE_DIR, 'sequencing_data/genomics_outcome_ranking.csv'))
top_features = ranked_features_df.iloc[:100, :]

top_features_by_data_type = top_features[['data_type', 'comparison']].groupby(['data_type']).size().reset_index()
top_features_by_data_type.columns = ['data_type', 'num_features']
top_features_by_data_type = top_features_by_data_type.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_data_type, x='data_type', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_10d.pdf'))
plt.close()


# summarize overlap of top features
ranked_features = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/feature_ranking.csv'))
top_features = ranked_features.loc[ranked_features.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo']), :]
top_features = top_features.iloc[:100, :]
top_features_by_feature = top_features[['feature_name_unique', 'comparison']].groupby('feature_name_unique').count().reset_index()
feature_counts = top_features_by_feature.groupby('comparison').count().reset_index()
feature_counts.columns = ['num_comparisons', 'num_features']

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=feature_counts, x='num_comparisons', y='num_features', color='grey', ax=ax)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_10f.pdf'))
plt.close()


# get overlap between static and evolution top features
overlap_type_dict = {'global': [['primary_', 'baseline', 'pre_nivo', 'on_nivo'],
                                ['primary__baseline', 'baseline__pre_nivo', 'baseline__on_nivo', 'pre_nivo__on_nivo']],
                     'primary': [['primary'], ['primary__baseline']],
                     'baseline': [['baseline'], ['primary__baseline', 'baseline__pre_nivo', 'baseline__on_nivo']],
                     'pre_nivo': [['pre_nivo'], ['baseline__pre_nivo', 'pre_nivo__on_nivo']],
                     'on_nivo': [['on_nivo'], ['baseline__on_nivo', 'pre_nivo__on_nivo']]}

overlap_results = {}
for overlap_type, comparisons in overlap_type_dict.items():
    static_comparisons, evolution_comparisons = comparisons

    overlap_top_features = ranked_features.copy()
    overlap_top_features = overlap_top_features.loc[overlap_top_features.comparison.isin(static_comparisons + evolution_comparisons)]
    overlap_top_features.loc[overlap_top_features.comparison.isin(static_comparisons), 'comparison'] = 'static'
    overlap_top_features.loc[overlap_top_features.comparison.isin(evolution_comparisons), 'comparison'] = 'evolution'
    overlap_top_features = overlap_top_features[['feature_name_unique', 'comparison']].drop_duplicates()
    overlap_top_features = overlap_top_features.iloc[:100, :]
    # keep_features = overlap_top_features.feature_name_unique.unique()[:100]
    # overlap_top_features = overlap_top_features.loc[overlap_top_features.feature_name_unique.isin(keep_features), :]
    # len(overlap_top_features.feature_name_unique.unique())
    static_ids = overlap_top_features.loc[
        overlap_top_features.comparison == 'static', 'feature_name_unique'].unique()
    evolution_ids = overlap_top_features.loc[
        overlap_top_features.comparison == 'evolution', 'feature_name_unique'].unique()

    overlap_results[overlap_type] = {'static_ids': static_ids, 'evolution_ids': evolution_ids}


# get counts of features in each category
static_ids = overlap_results['global']['static_ids']
evolution_ids = overlap_results['global']['evolution_ids']
venn2([set(static_ids), set(evolution_ids)], set_labels=('Static', 'Evolution'))
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_10k.pdf'))
plt.close()


# plot key features from primary tumors
combined_df = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/timepoint_combined_features.csv'))

for timepoint in ['primary', 'baseline', 'pre_nivo', 'on_nivo']:
    for feature, plot_name, lims in zip(['B__NK__ratio__cancer_core', 'HLADR+__APC', 'Cancer__Structural__ratio__cancer_border'],
                                  ['10X1', '10X2', '10X3'], [[-5, 7], [0, 1.2], [1, 12]]):

        plot_df = combined_df.loc[(combined_df.feature_name_unique == feature) &
                                  (combined_df.Timepoint == timepoint), :]

        fig, ax = plt.subplots(1, 1, figsize=(2, 4))
        sns.stripplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                        color='black', ax=ax)
        sns.boxplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                        color='grey', ax=ax, showfliers=False, width=0.3)
        ax.set_title(feature + ' ' + timepoint)
        ax.set_ylim(lims)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_{}_{}.pdf'.format(plot_name, timepoint)))
        plt.close()

# top features from baseline
for timepoint in ['primary', 'baseline', 'pre_nivo', 'on_nivo']:
    for feature, plot_name, lims in zip(['diversity_cell_cluster__Cancer_2__stroma_border', 'CD38+__Immune_Other', 'Other__Structural__ratio'],
                                  ['10X4', '10X5', '10X6'], [[0, 3], [-0.2, 1], [-10, 3]]):

        plot_df = combined_df.loc[(combined_df.feature_name_unique == feature) &
                                  (combined_df.Timepoint == timepoint), :]

        fig, ax = plt.subplots(1, 1, figsize=(2, 4))
        sns.stripplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                        color='black', ax=ax)
        sns.boxplot(data=plot_df, x='Clinical_benefit', y='raw_mean', order=['Yes', 'No'],
                        color='grey', ax=ax, showfliers=False, width=0.3)
        ax.set_title(feature + ' ' + timepoint)
        ax.set_ylim(lims)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_{}_{}.pdf'.format(plot_name, timepoint)))
        plt.close()


# shared/non-shared features over time
ranked_features = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/feature_ranking.csv'))
top_features = ranked_features.loc[ranked_features.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo']), :]
top_feature_names = top_features.feature_name_unique[:100].unique()
top_features = top_features.loc[top_features.feature_name_unique.isin(top_feature_names), :]
top_features_wide = pd.pivot(top_features, index='feature_name_unique', columns='comparison', values='feature_rank_global')
top_features_wide['top_100_count'] = top_features_wide.apply(lambda x: np.sum(x[:4] <= 100), axis=1)
top_features_wide['top_200_count'] = top_features_wide.apply(lambda x: np.sum(x[:4] <= 200), axis=1)
top_features_wide['top_350_count'] = top_features_wide.apply(lambda x: np.sum(x[:4] <= 350), axis=1)

# look at top 100 repeats
len([x for x in top_features_wide.loc[top_features_wide.top_100_count > 1, ].index if 'cancer_border' in x])
len([x for x in top_features_wide.index if 'cancer_border' in x])

np.sum(top_features_wide.top_350_count > 1)
top_features_wide.loc[top_features_wide.top_350_count > 3, ].index

shared_3 = top_features_wide.loc[top_features_wide.top_350_count > 2, ].index
len(shared_3)
len([x for x in shared_3 if 'diversity' in x])
len([x for x in shared_3 if 'ratio' in x])