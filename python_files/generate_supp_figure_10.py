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

