import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/"
SUPPLEMENTARY_FIG_DIR = os.path.join(BASE_DIR, "supplementary_figs")


ranked_features_all = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/feature_ranking.csv'))
ranked_features = ranked_features_all.loc[ranked_features_all.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo'])]
ranked_features = ranked_features.loc[ranked_features.feature_rank_global <= 100, :]

# densities vs ratios in top 100
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
ax.set_ylim([0, 45])
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
ax.set_ylim(0, 7)
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
ax.set_ylim(0, 90)
sns.barplot(data=feature_counts, x='num_comparisons', y='num_features', color='grey', ax=ax)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_10f.pdf'))
plt.close()
