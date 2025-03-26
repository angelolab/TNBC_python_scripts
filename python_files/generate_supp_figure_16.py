import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/"
SUPPLEMENTARY_FIG_DIR = os.path.join(BASE_DIR, "supplementary_figs")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_files")

# plot top baseline features
NT_DIR = '/Volumes/Shared/Noah Greenwald/NTPublic'
combined_df = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat/analysis_files/timepoint_combined_features_immunotherapy+chemotherapy.csv'))
for timepoint in ['Baseline', 'On-treatment']:
    for feature, plot_name, lims in zip(['Ki67+__Treg__cancer_border', 'Ki67+__Treg__stroma_core', 'T_diversity__cancer_core'],
                                  ['11d', '11e', '11f'], [[0, 0.8], [0, 0.6], [0, 1.7]]):

        plot_df = combined_df.loc[(combined_df.feature_name_unique == feature) &
                                  (combined_df.Timepoint == timepoint), :]

        fig, ax = plt.subplots(1, 1, figsize=(2, 4))
        sns.stripplot(data=plot_df, x='pCR', y='raw_mean', order=['pCR', 'RD'], color='black', ax=ax)
        sns.boxplot(data=plot_df, x='pCR', y='raw_mean', order=['pCR', 'RD'], color='grey', ax=ax, showfliers=False, width=0.3)
        ax.set_title(timepoint)
        ax.set_ylim(lims)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'review_figures/NTPublic/baseline_top_features', '{}_{}.pdf'.format(feature, timepoint)))
        plt.close()

# NT feature enrichment on TONIC data
harmonized_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'harmonized_metadata.csv'))
ranked_features_all = pd.read_csv(os.path.join(BASE_DIR, 'TONIC_SpaceCat/NT_features_only/analysis_files/feature_ranking.csv'))
ranked_features = ranked_features_all.loc[ranked_features_all.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo'])]
top_features = ranked_features.loc[ranked_features.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo']), :]
top_features = top_features.iloc[:100, :]

# summarize distribution of top features
top_features_by_comparison = top_features[['feature_name_unique', 'comparison']].groupby('comparison').count().reset_index()
top_features_by_comparison.columns = ['comparison', 'num_features']
top_features_by_comparison = top_features_by_comparison.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_comparison, x='comparison', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_16j.pdf'))
plt.close()

