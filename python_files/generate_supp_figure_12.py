import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from matplotlib_venn import venn3
from python_files.supplementary_plot_helpers import random_feature_generation


BASE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/"
SUPPLEMENTARY_FIG_DIR = os.path.join(BASE_DIR, "supplementary_figs")


## Analyse the significance scores of top features after randomization compared to the TONIC data.
fp_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_12a_files')
if not os.path.exists(fp_dir):
    os.makedirs(fp_dir)

# compute random feature sets
combined_df = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/timepoint_combined_features.csv'))
feature_df = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/feature_ranking.csv'))
feature_metadata = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/feature_metadata.csv'))

repeated_features, repeated_features_num, scores = [], [], []
overlapping_features, random_top_features = [], []

sample_num = 100
np.random.seed(13)

for i, seed in enumerate(random.sample(range(1, 2000), sample_num)):
    print(f'{i+1}/100')
    intersection_of_features, jaccard_score, top_random_features = random_feature_generation(combined_df, seed, feature_df[:100], feature_metadata)

    shared_df = pd.DataFrame({
        'random_seed': [seed] * len(intersection_of_features),
        'repeated_features' : list(intersection_of_features),
        'jaccard_score': [jaccard_score] * len(intersection_of_features)
    })
    overlapping_features.append(shared_df)

    top_random_features['seed'] = seed
    random_top_features.append(top_random_features)

results = pd.concat(overlapping_features)
top_features = pd.concat(random_top_features)
# add TONIC features to data with seed 0
top_features = pd.concat([top_features, feature_df[:100]])
top_features['seed'] = top_features['seed'].fillna(0)

results.to_csv(os.path.join(fp_dir, 'overlapping_features.csv'), index=False)
top_features.to_csv(os.path.join(fp_dir, 'top_features.csv'), index=False)
#'''

top_features = pd.read_csv(os.path.join(fp_dir, 'top_features.csv'))
results = pd.read_csv(os.path.join(fp_dir, 'overlapping_features.csv'))

avg_scores = top_features[['seed', 'pval', 'log_pval', 'fdr_pval', 'med_diff']].groupby(by='seed').mean()
avg_scores['abs_med_diff'] = abs(avg_scores['med_diff'])
top_features['abs_med_diff'] = abs(top_features['med_diff'])

# log p-value & effect size plots
for name, metric in zip(['supp_figure_12a', 'supp_figure_12c'], ['log_pval', 'abs_med_diff']):
    # plot metric dist in top features for TONIC data and one random set
    TONIC = top_features[top_features.seed == 0]
    random = top_features[top_features.seed == 8]
    g = sns.distplot(TONIC[metric], kde=True, color='#1f77b4')
    g = sns.distplot(random[metric], kde=True, color='#ff7f0e')
    g.set(xlim=(0, None))
    plt.xlabel(metric)
    plt.title(f"{metric} Distribution in TONIC vs a Random")
    g.legend(labels=["TONIC", "Randomized"])
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.9, 1))
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(fp_dir, f"{name}.pdf"), dpi=300)
    plt.show()

for name, metric in zip(['supp_figure_12b', 'supp_figure_12d'], ['log_pval', 'abs_med_diff']):
    # plot average metric across top features for each set
    g = sns.distplot(avg_scores[metric][1:], kde=True,  color='#ff7f0e')
    g.axvline(x=avg_scores[metric][0], color='#1f77b4')
    g.set(xlim=(0, avg_scores[metric][0]*1.2))
    plt.xlabel(f'Average {metric} of Top 100 Features')
    plt.title(f"Average {metric} in TONIC vs Random Sets")
    g.legend(labels=["Randomized", "TONIC"])
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.9, 1))
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(fp_dir, f"{name}.pdf"), dpi=300)
    plt.show()

# # general feature overlap plots
# high_features = results.groupby(by='repeated_features').count().sort_values(by='random_seed', ascending=False).reset_index()
# high_features = high_features[high_features.random_seed>3].sort_values(by='random_seed')
# plt.barh(high_features.repeated_features, high_features.random_seed)
# plt.xlabel('How Many Random Sets Contain the Feature')
# plt.title('Repeated Top Features')
# sns.despine()
# plt.savefig(os.path.join(fp_dir, "Repeated_Top_Features.pdf"), dpi=300, bbox_inches='tight')
# plt.show()

repeated_features_num = results.groupby(by='random_seed').count().sort_values(by='repeated_features', ascending=False)
plt.hist(repeated_features_num.repeated_features)
plt.xlabel('Number of TONIC Top Features in each Random Set')
plt.title('Histogram of Overlapping Features')
sns.despine()
plt.savefig(os.path.join(fp_dir, f"supp_figure_12e.pdf"), dpi=300)
plt.show()

# plt.hist(results.jaccard_score, bins=10)
# plt.xlim((0, 0.10))
# plt.title('Histogram of Jaccard Scores')
# sns.despine()
# plt.xlabel('Jaccard Score')
# plt.savefig(os.path.join(fp_dir, "Histogram_of_Jaccard_Scores.pdf"), dpi=300)
# plt.show()

# # compute the correlation between response-associated features
# timepoint_features = pd.read_csv(os.path.join(ANALYSIS_DIR, 'timepoint_combined_features.csv'))
# feature_ranking_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_ranking.csv'))
# feature_ranking_df = feature_ranking_df[np.isin(feature_ranking_df['comparison'], ['primary', 'baseline', 'pre_nivo' , 'on_nivo'])]
# feature_ranking_df = feature_ranking_df.sort_values(by = 'feature_rank_global', ascending=True)
#
# top_features = feature_ranking_df.iloc[:100, :].loc[:, ['feature_name_unique', 'comparison', 'feature_type']]
# top_features.columns = ['feature_name_unique', 'Timepoint', 'feature_type']
#
# remaining_features = feature_ranking_df.iloc[100:, :].loc[:, ['feature_name_unique', 'comparison', 'feature_type']]
# remaining_features.columns = ['feature_name_unique', 'Timepoint', 'feature_type']
#
# def calculate_feature_corr(timepoint_features,
#                             top_features,
#                             remaining_features,
#                             top: bool = True,
#                             n_iterations: int = 1000):
#     """Compares the correlation between
#             1. response-associated features to response-associated features
#             2. response-associated features to remaining features
#         by randomly sampling features with replacement.
#
#     Parameters
#     timepoint_features: pd.DataFrame
#         dataframe containing the feature values for every patient (feature_name_unique, normalized_mean, Patient_ID, Timepoint)
#     top_features: pd.DataFrame
#         dataframe containing the top response-associated features (feature_name_unique, Timepoint)
#     remaining features: pd.DataFrame
#         dataframe containing non response-associated features (feature_name_unique, Timepoint)
#     top: bool (default = True)
#         boolean indicating if the comparison 1. (True) or 2. (False)
#     n_iterations: int (default = 1000)
#         number of features randomly selected for comparison
#     ----------
#     Returns
#     corr_arr: np.array
#         array containing the feature correlation values
#     ----------
#     """
#     corr_arr = []
#     for _ in range(n_iterations):
#         #select feature 1 as a random feature from the top response-associated feature list
#         rand_sample1 = top_features.sample(n = 1)
#         f1 = timepoint_features.iloc[np.where((timepoint_features['feature_name_unique'] == rand_sample1['feature_name_unique'].values[0]) & (timepoint_features['Timepoint'] == rand_sample1['Timepoint'].values[0]))[0], :]
#         if top == True:
#             #select feature 2 as a random feature from the top response-associated list, ensuring f1 != f2
#             rand_sample2 = rand_sample1
#             while (rand_sample2.values == rand_sample1.values).all():
#                 rand_sample2 = top_features.sample(n = 1)
#         else:
#             #select feature 2 as a random feature from the remaining feature list
#             rand_sample2 = remaining_features.sample(n = 1)
#
#         f2 = timepoint_features.iloc[np.where((timepoint_features['feature_name_unique'] == rand_sample2['feature_name_unique'].values[0]) & (timepoint_features['Timepoint'] == rand_sample2['Timepoint'].values[0]))[0], :]
#         merged_features = pd.merge(f1, f2, on = 'Patient_ID') #finds Patient IDs that are shared across timepoints to compute correlation
#         corrval = np.abs(merged_features['normalized_mean_x'].corr(merged_features['normalized_mean_y'])) #regardless of direction
#         corr_arr.append(corrval)
#
#     return np.array(corr_arr)
#
# #C(100, 2) = 100! / [(100-2)! * 2!] = 4950 unique pairwise combinations top 100 features and each other
# corr_within = calculate_feature_corr(timepoint_features, top_features, remaining_features, top=True)
# corr_across = calculate_feature_corr(timepoint_features, top_features, remaining_features, top=False)
#
# corr_across = corr_across[~np.isnan(corr_across)]
# corr_within = corr_within[~np.isnan(corr_within)]
#
# _, axes = plt.subplots(1, 1, figsize = (5, 4), gridspec_kw={'hspace': 0.45, 'wspace': 0.4, 'bottom':0.15})
# g = sns.histplot(corr_within, color='#2089D5', ax = axes, kde = True, bins = 50, label = 'top-top', alpha = 0.5)
# g = sns.histplot(corr_across, color='lightgrey', ax = axes, kde = True, bins = 50, label = 'top-remaining', alpha = 0.5)
# g.tick_params(labelsize=12)
# g.set_ylabel('number of comparisons', fontsize = 12)
# g.set_xlabel('abs(correlation)', fontsize = 12)
# plt.legend(prop={'size':9})
# g.set_xlim(0, 1)
# plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'correlation_response_features', 'correlation_response_associated_features.pdf'), bbox_inches = 'tight', dpi = 300)
# plt.close()