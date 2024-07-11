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

# plot key features from primary tumors
combined_df = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/timepoint_combined_features.csv'))

for timepoint in ['primary', 'baseline', 'pre_nivo', 'on_nivo']:
    for feature, plot_name, lims in zip(['B__NK__ratio__cancer_core', 'HLADR+__APC', 'Cancer__Structural__ratio__cancer_border'],
                                  ['11a', '11b', '11c'], [[-5, 7], [0, 1.2], [1, 12]]):

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
                                  ['11d', '11e', '11f'], [[0, 3], [-0.2, 1], [-10, 3]]):

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
