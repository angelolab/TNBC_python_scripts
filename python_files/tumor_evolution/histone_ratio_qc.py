import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import skimage.io as io

from python_files.utils import find_conserved_features, compare_timepoints, compare_populations
from python_files.utils import summarize_population_enrichment, summarize_timepoint_enrichment
from python_files.utils import create_long_df_by_functional, create_long_df_by_cluster

from python_files import utils

plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

timepoint_features = pd.read_csv(os.path.join(data_dir, 'timepoint_features_filtered.csv'))
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
patient_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/TONIC_data_per_patient.csv'))

timepoint_features = timepoint_features.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint', 'primary__baseline']].drop_duplicates(), on='Tissue_ID')



# compare features prevalence in primary vs metastatic
primary_met_means = compare_timepoints(feature_df=timepoint_features, timepoint_1_name='primary', timepoint_1_list=['primary_untreated'],
                     timepoint_2_name='metastatic', timepoint_2_list=['baseline'], paired='primary__baseline',
                   feature_suff='mean')

timepoint_features_subset = timepoint_features.loc[timepoint_features.feature_name == 'H3K9ac_H3K27me3_ratio+__all', :]
#timepoint_features_subset = timepoint_features_subset.loc[timepoint_features_subset.primary__baseline, :]
timepoint_features_subset = timepoint_features_subset.loc[timepoint_features_subset.Timepoint.isin(['primary_untreated', 'metastasis_1']), :]

histone_wide = timepoint_features_subset.pivot(index='Patient_ID', columns='Timepoint', values='raw_mean')
histone_wide.reset_index(inplace=True)
histone_wide['histone_shift'] = histone_wide['metastasis_1'] - histone_wide['primary_untreated']

keep_cols = ['Interval_primary_TONIC_baseline_months', 'Disease_Free_Interval_(months)_primto1strelapse']
histone_wide = histone_wide.merge(patient_metadata[['Patient_ID'] + keep_cols], on='Patient_ID')

sns.lineplot(data=timepoint_features_subset, x='Timepoint', y='raw_mean', units='Patient_ID', estimator=None, color='grey', alpha=0.5, marker='o')

sns.scatterplot(data=histone_wide, x='Interval_primary_TONIC_baseline_months', y='histone_shift', palette='viridis')
plt.savefig(os.path.join(plot_dir, 'histone_shift_vs_interval_met1.png'), dpi=300)
plt.close()

sns.scatterplot(data=histone_wide, x='Disease_Free_Interval_(months)_primto1strelapse', y='histone_shift', palette='viridis')
plt.savefig(os.path.join(plot_dir, 'histone_shift_vs_dfi_met1.png'), dpi=300)
plt.close()


# look at correlation between shift in primary met1 and primary baseline
timepoint_features_subset = timepoint_features.loc[timepoint_features.feature_name == 'H3K9ac_H3K27me3_ratio+__all', :]
timepoint_features_subset = timepoint_features_subset.loc[timepoint_features_subset.Timepoint.isin(['primary_untreated', 'metastasis_1', 'baseline']), :]

histone_wide = timepoint_features_subset.pivot(index='Patient_ID', columns='Timepoint', values='raw_mean')
histone_wide.reset_index(inplace=True)
histone_wide['histone_shift_baseline'] = histone_wide['baseline'] - histone_wide['primary_untreated']
histone_wide['histone_shift_met1'] = histone_wide['metastasis_1'] - histone_wide['primary_untreated']

sns.scatterplot(data=histone_wide, x='histone_shift_baseline', y='histone_shift_met1', palette='viridis')
plt.savefig(os.path.join(plot_dir, 'histone_shift_baseline_vs_met1.png'), dpi=300)
plt.close()

# look at correlation between DOS and primary ratio
timepoint_features_subset = timepoint_features[timepoint_features.feature_name == 'H3K9ac_H3K27me3_ratio+__all']
timepoint_features_subset = timepoint_features_subset.loc[timepoint_features_subset.Timepoint == 'primary_untreated', :]
timepoint_features_subset = timepoint_features_subset.merge(patient_metadata[['Patient_ID', 'Interval_primary_TONIC_baseline_months']], on='Patient_ID')

sns.scatterplot(data=timepoint_features_subset, x='Interval_primary_TONIC_baseline_months', y='raw_mean', palette='viridis')
plt.savefig(os.path.join(plot_dir, 'dos_vs_primary_ratio.png'), dpi=300)
plt.close()

cell_table_func = pd.read_csv(os.path.join(data_dir, 'post_processing', 'cell_table_func_single_positive.csv'))
cell_table_counts = pd.read_csv(os.path.join(data_dir, 'post_processing','cell_table_counts.csv'))

cell_table_subset = cell_table_func.loc[cell_table_func.cell_cluster_broad == 'Cancer', :]
cell_counts_subset = cell_table_counts.loc[cell_table_counts.cell_cluster_broad == 'Cancer', :]

cell_table_subset['H3K9ac'] = cell_counts_subset['H3K9ac']
cell_table_subset['H3K27me3'] = cell_counts_subset['H3K27me3']
del cell_counts_subset

sns.kdeplot(data=cell_table_subset.loc[:100000, :], x='H3K9ac', y='H3K27me3', fill=True, clip=[0, 0.04])
plt.savefig(os.path.join(plot_dir, 'H3K9ac_H3K27me3_kde.png'), dpi=300)
plt.close()

sns.kdeplot(data=cell_table_subset.loc[:300000, :], x='H3K9ac_H3K27me3_ratio', y='H3K27me3', fill=True, clip=None)
plt.savefig(os.path.join(plot_dir, 'H3K9ac_H3K27me3_ratio_H3K27me3_kde.png'), dpi=300)
plt.close()

sns.kdeplot(data=cell_table_subset.loc[:300000, :], x='H3K9ac_H3K27me3_ratio', y='H3K9ac', fill=True, clip=None)
plt.savefig(os.path.join(plot_dir, 'H3K9ac_H3K27me3_ratio_H3K9ac_kde.png'), dpi=300)
plt.close()


# create per-FOV averages of total counts and ratio

func_df_params = [['cluster_broad_count', 'cell_cluster_broad', False],
                  ['cluster_broad_freq', 'cell_cluster_broad', True],
                  ['cluster_count', 'cell_cluster', False],
                  ['cluster_freq', 'cell_cluster', True],
                  ['meta_cluster_count', 'cell_meta_cluster', False],
                  ['meta_cluster_freq', 'cell_meta_cluster', True]]
                  #['kmeans_freq', 'kmeans_labels', True]]

# columns which are not functional markers need to be dropped from the df
drop_cols = ['cell_meta_cluster', 'cell_cluster', 'label',  'Ki67', 'CD38', 'CD45RB', 'CD45RO', 'CD57', 'CD69',
       'GLUT1', 'IDO', 'LAG3', 'PD1', 'PDL1', 'HLA1', 'HLADR', 'TBET', 'TCF1',
       'TIM3', 'Vim', 'Fe']

# create df
func_df = create_long_df_by_functional(func_table=cell_table_subset,
                                       result_name='cluster_broad_freq',
                                       cluster_col_name='cell_cluster_broad',
                                       drop_cols=drop_cols,
                                       normalize=True)


func_df_wide = func_df.pivot(index='fov', columns='functional_marker', values='value')

# look at relationship between markers
sns.kdeplot(data=func_df_wide, x='H3K9ac', y='H3K27me3', fill=True)
plt.savefig(os.path.join(plot_dir, 'H3K9ac_H3K27me3_average_kde.png'), dpi=300)
plt.close()

sns.kdeplot(data=func_df_wide, x='H3K9ac_H3K27me3_ratio', y='H3K27me3', fill=True, clip=None)
plt.savefig(os.path.join(plot_dir, 'H3K9ac_H3K27me3_ratio_average_H3K27me3_kde.png'), dpi=300)
plt.close()

sns.kdeplot(data=func_df_wide, x='H3K9ac_H3K27me3_ratio', y='H3K9ac', fill=True, clip=None)
plt.savefig(os.path.join(plot_dir, 'H3K9ac_H3K27me3_ratio_average_H3K9ac_kde.png'), dpi=300)
plt.close()

# look at what is most correlated with H3K9ac_H3K27me3_ratio
timepoint_wide = timepoint_features.pivot(index='Tissue_ID', columns='feature_name_unique', values='normalized_mean')
corr_df = timepoint_wide.corr(method='spearman')
corr_df_subset = corr_df[['H3K9ac_H3K27me3_ratio+__Cancer', 'H3K9ac_H3K27me3_ratio+__all']]

corr_df_subset = corr_df_subset.sort_values(by='H3K9ac_H3K27me3_ratio+__all', ascending=False)

# look at histone ratio split by site
timepoint_features_subset = timepoint_features.loc[timepoint_features.feature_name_unique == 'H3K9ac_H3K27me3_ratio+__Stroma', :]
timepoint_features_subset = timepoint_features_subset.loc[timepoint_features_subset.Timepoint.isin(['primary_untreated', 'baseline', 'local_recurrence'])]
sns.catplot(data=timepoint_features_subset, x='Timepoint', y='raw_mean', kind='strip')
plt.savefig(os.path.join(plot_dir, 'histone_ratio_Stroma_timepoint.png'), dpi=300)
plt.close()


timepoint_features_subset = timepoint_features_subset.merge(harmonized_metadata[['Tissue_ID', 'Localization']], on='Tissue_ID', how='left')
sns.catplot(data=timepoint_features_subset, x='Localization', y='raw_mean', kind='strip')
plt.savefig(os.path.join(plot_dir, 'histone_ratio_Stroma_localization.png'), dpi=300)
plt.close()

# break out lymphnodes
for cell_type in ['Stroma', 'Cancer', 'CD4T', 'B', 'Fibroblast', 'Monocyte', 'Treg', 'all']:
    timepoint_features_subset = timepoint_features.loc[timepoint_features.feature_name_unique == 'H3K9ac_H3K27me3_ratio+__' + cell_type, :]
    timepoint_features_subset = timepoint_features_subset.loc[timepoint_features_subset.Timepoint.isin(['primary_untreated', 'baseline', 'lymphnode_pos', 'lymphnode_neg'])]
    timepoint_features_subset = timepoint_features_subset.merge(harmonized_metadata[['Tissue_ID', 'Localization']], on='Tissue_ID', how='left')
    timepoint_features_subset = timepoint_features_subset.loc[timepoint_features_subset.Localization.isin(['Lymphnode', 'Breast'])]

    sns.catplot(data=timepoint_features_subset, x='Timepoint', y='raw_mean', kind='strip', order=['primary_untreated', 'lymphnode_pos', 'lymphnode_neg', 'baseline'])
    # put x labels on an angle
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'histone_ratio_' + cell_type + '_ln_timepoint.png'), dpi=300)
    plt.close()



