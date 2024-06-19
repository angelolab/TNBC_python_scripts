# File with code for generating supplementary plots
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from ark.utils.plot_utils import cohort_cluster_plot
# from toffy import qc_comp, qc_metrics_plots
from alpineer import io_utils

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests


import python_files.supplementary_plot_helpers as supplementary_plot_helpers

BASE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/"
ANALYSIS_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files"
CHANNEL_DIR = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/'
INTERMEDIATE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files"
OUTPUT_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/output_files"
METADATA_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files/metadata"
SEG_DIR = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output'
SUPPLEMENTARY_FIG_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/supplementary_figs"
SEQUENCE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/sequencing_data"







# False positive analysis
## Analyse the significance scores of top features after randomization compared to the TONIC data.
fp_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, 'false_positive_analysis')
if not os.path.exists(fp_dir):
    os.makedirs(fp_dir)

# compute random feature sets
'''
combined_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'timepoint_combined_features.csv'))
feature_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_ranking.csv'))
feature_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_metadata.csv'))

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
'''

top_features = pd.read_csv(os.path.join(fp_dir, 'top_features.csv'))
results = pd.read_csv(os.path.join(fp_dir, 'overlapping_features.csv'))

avg_scores = top_features[['seed', 'pval', 'log_pval', 'fdr_pval', 'med_diff']].groupby(by='seed').mean()
avg_scores['abs_med_diff'] = abs(avg_scores['med_diff'])
top_features['abs_med_diff'] = abs(top_features['med_diff'])

# log p-value & effect size plots
for name, metric in zip(['Log p-value', 'Effect Size'], ['log_pval', 'abs_med_diff']):
    # plot metric dist in top features for TONIC data and one random set
    TONIC = top_features[top_features.seed == 0]
    random = top_features[top_features.seed == 8]
    g = sns.distplot(TONIC[metric], kde=True, color='#1f77b4')
    g = sns.distplot(random[metric], kde=True, color='#ff7f0e')
    g.set(xlim=(0, None))
    plt.xlabel(name)
    plt.title(f"{name} Distribution in TONIC vs a Random")
    g.legend(labels=["TONIC", "Randomized"])
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.9, 1))
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(fp_dir, f"{metric}_dists.pdf"), dpi=300)
    plt.show()

    # plot average metric across top features for each set
    g = sns.distplot(avg_scores[metric][1:], kde=True,  color='#ff7f0e')
    g.axvline(x=avg_scores[metric][0], color='#1f77b4')
    g.set(xlim=(0, avg_scores[metric][0]*1.2))
    plt.xlabel(f'Average {name} of Top 100 Features')
    plt.title(f"Average {name} in TONIC vs Random Sets")
    g.legend(labels=["Randomized", "TONIC"])
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.9, 1))
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(fp_dir, f"{metric}_avg_per_set.pdf"), dpi=300)
    plt.show()

# general feature overlap plots
high_features = results.groupby(by='repeated_features').count().sort_values(by='random_seed', ascending=False).reset_index()
high_features = high_features[high_features.random_seed>3].sort_values(by='random_seed')
plt.barh(high_features.repeated_features, high_features.random_seed)
plt.xlabel('How Many Random Sets Contain the Feature')
plt.title('Repeated Top Features')
sns.despine()
plt.savefig(os.path.join(fp_dir, "Repeated_Top_Features.pdf"), dpi=300, bbox_inches='tight')
plt.show()

repeated_features_num = results.groupby(by='random_seed').count().sort_values(by='repeated_features', ascending=False)
plt.hist(repeated_features_num.repeated_features)
plt.xlabel('Number of TONIC Top Features in each Random Set')
plt.title('Histogram of Overlapping Features')
sns.despine()
plt.savefig(os.path.join(fp_dir, f"Histogram_of_Overlapping_Features.pdf"), dpi=300)
plt.show()

plt.hist(results.jaccard_score, bins=10)
plt.xlim((0, 0.10))
plt.title('Histogram of Jaccard Scores')
sns.despine()
plt.xlabel('Jaccard Score')
plt.savefig(os.path.join(fp_dir, "Histogram_of_Jaccard_Scores.pdf"), dpi=300)
plt.show()





# volcano plot for RNA features
ranked_features_df = pd.read_csv(os.path.join(SEQUENCE_DIR, 'genomics_outcome_ranking.csv'))
ranked_features_df = ranked_features_df.loc[ranked_features_df.data_type == 'RNA', :]
ranked_features_df = ranked_features_df.sort_values(by='combined_rank', ascending=True)

ranked_features_df[['feature_name_unique']].to_csv(os.path.join(save_dir, 'RNA_features.csv'), index=False)

# plot  volcano
fig, ax = plt.subplots(figsize=(3,3))
sns.scatterplot(data=ranked_features_df, x='med_diff', y='log_pval', alpha=1, hue='importance_score', palette=sns.color_palette("icefire", as_cmap=True),
                s=2.5, edgecolor='none', ax=ax)
ax.set_xlim(-3, 3)
sns.despine()

# add gradient legend
norm = plt.Normalize(ranked_features_df.importance_score.min(), ranked_features_df.importance_score.max())
sm = plt.cm.ScalarMappable(cmap="icefire", norm=norm)
ax.get_legend().remove()
ax.figure.colorbar(sm, ax=ax)
plt.tight_layout()

plt.savefig(os.path.join(save_dir, 'RNA_volcano.pdf'))
plt.close()

# Breakdown of features by timepoint
top_features = ranked_features_df.iloc[:100, :]
#top_features = ranked_features_df.loc[ranked_features_df.fdr_pval < 0.1, :]

# by comparison
top_features_by_comparison = top_features[['data_type', 'comparison']].groupby(['comparison']).size().reset_index()
top_features_by_comparison.columns = ['comparison', 'num_features']
top_features_by_comparison = top_features_by_comparison.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_comparison, x='comparison', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(save_dir, 'Num_features_per_comparison_rna.pdf'))
plt.close()

# by data type
ranked_features_df = pd.read_csv(os.path.join(SEQUENCE_DIR, 'genomics_outcome_ranking.csv'))
top_features = ranked_features_df.iloc[:100, :]

top_features_by_data_type = top_features[['data_type', 'comparison']].groupby(['data_type']).size().reset_index()
top_features_by_data_type.columns = ['data_type', 'num_features']
top_features_by_data_type = top_features_by_data_type.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_data_type, x='data_type', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(save_dir, 'Num_features_per_data_type_genomics.pdf'))
plt.close()

# diagnostic plots for multivariate modeling
save_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, 'multivariate_modeling')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

all_model_rankings = pd.read_csv(os.path.join(BASE_DIR, 'multivariate_lasso/intermediate_results', 'all_model_rankings.csv'))

# plot top features
sns.stripplot(data=all_model_rankings.loc[all_model_rankings.top_ranked, :], x='timepoint', y='importance_score', hue='modality')
plt.title('Top ranked features')
plt.ylim([0, 1.05])
plt.savefig(os.path.join(save_dir, 'top_ranked_features_by_modality_and_timepoint.pdf'))
plt.close()

# plot number of times features are selected
sns.histplot(data=all_model_rankings.loc[all_model_rankings.top_ranked, :], x='count', color='grey', multiple='stack',
             binrange=(1, 10), discrete=True)
plt.title('Number of times features are selected')
plt.savefig(os.path.join(save_dir, 'feature_counts_top_ranked.pdf'))
plt.close()

sns.histplot(data=all_model_rankings, x='count', color='grey', multiple='stack',
             binrange=(1, 10), discrete=True)
plt.title('Number of times features are selected')
plt.savefig(os.path.join(save_dir, 'feature_counts_all.pdf'))
plt.close()

# plot venn diagram
from matplotlib_venn import venn3

rna_rankings_top = all_model_rankings.loc[np.logical_and(all_model_rankings.modality == 'RNA', all_model_rankings.top_ranked), :]
rna_baseline = rna_rankings_top.loc[rna_rankings_top.timepoint == 'baseline', 'feature_name_unique'].values
rna_nivo = rna_rankings_top.loc[rna_rankings_top.timepoint == 'on_nivo', 'feature_name_unique'].values
rna_induction = rna_rankings_top.loc[rna_rankings_top.timepoint == 'post_induction', 'feature_name_unique'].values

venn3([set(rna_baseline), set(rna_nivo), set(rna_induction)], ('Baseline', 'Nivo', 'Induction'))
plt.title('RNA top ranked features')
plt.savefig(os.path.join(save_dir, 'Figure6_RNA_top_ranked_venn.pdf'))
plt.close()

mibi_rankings_top = all_model_rankings.loc[np.logical_and(all_model_rankings.modality == 'MIBI', all_model_rankings.top_ranked), :]
mibi_baseline = mibi_rankings_top.loc[mibi_rankings_top.timepoint == 'baseline', 'feature_name_unique'].values
mibi_nivo = mibi_rankings_top.loc[mibi_rankings_top.timepoint == 'on_nivo', 'feature_name_unique'].values
mibi_induction = mibi_rankings_top.loc[mibi_rankings_top.timepoint == 'post_induction', 'feature_name_unique'].values

venn3([set(mibi_baseline), set(mibi_nivo), set(mibi_induction)], ('Baseline', 'Nivo', 'Induction'))
plt.title('MIBI top ranked features')
plt.savefig(os.path.join(save_dir, 'Figure6_MIBI_top_ranked_venn.pdf'))
plt.close()

# compare correlations between top ranked features
ranked_features_univariate = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_ranking.csv'))

nivo_features_model = all_model_rankings.loc[np.logical_and(all_model_rankings.timepoint == 'on_nivo', all_model_rankings.top_ranked), :]
nivo_features_model = nivo_features_model.loc[nivo_features_model.modality == 'MIBI', 'feature_name_unique'].values

nivo_features_univariate = ranked_features_univariate.loc[np.logical_and(ranked_features_univariate.comparison == 'on_nivo',
                                                                         ranked_features_univariate.feature_rank_global <= 100), :]

timepoint_features = pd.read_csv(os.path.join(ANALYSIS_DIR, 'timepoint_combined_features.csv'))
timepoint_features = timepoint_features.loc[timepoint_features.Timepoint == 'on_nivo', :]

timepoint_features_model = timepoint_features.loc[timepoint_features.feature_name_unique.isin(nivo_features_model), :]
timepoint_features_model = timepoint_features_model[['feature_name_unique', 'normalized_mean', 'Patient_ID']]
timepoint_features_model = timepoint_features_model.pivot(index='Patient_ID', columns='feature_name_unique', values='normalized_mean')

# get values
model_corr = timepoint_features_model.corr()
model_corr = model_corr.where(np.triu(np.ones(model_corr.shape), k=1).astype(np.bool)).values.flatten()
model_corr = model_corr[~np.isnan(model_corr)]

# get univariate features
timepoint_features_univariate = timepoint_features.loc[timepoint_features.feature_name_unique.isin(nivo_features_univariate.feature_name_unique), :]
timepoint_features_univariate = timepoint_features_univariate[['feature_name_unique', 'normalized_mean', 'Patient_ID']]
timepoint_features_univariate = timepoint_features_univariate.pivot(index='Patient_ID', columns='feature_name_unique', values='normalized_mean')

# get values
univariate_corr = timepoint_features_univariate.corr()
univariate_corr = univariate_corr.where(np.triu(np.ones(univariate_corr.shape), k=1).astype(np.bool)).values.flatten()
univariate_corr = univariate_corr[~np.isnan(univariate_corr)]

corr_values = pd.DataFrame({'correlation': np.concatenate([model_corr, univariate_corr]),
                            'model': ['model'] * len(model_corr) + ['univariate'] * len(univariate_corr)})

# plot correlations by model
fig, ax = plt.subplots(1, 1, figsize=(3, 4))
# sns.stripplot(data=corr_values, x='model', y='correlation',
#               color='black', ax=ax)
sns.boxplot(data=corr_values, x='model', y='correlation',
            color='grey', ax=ax, showfliers=False)

ax.set_title('Feature correlation')
# ax.set_ylim([-1, 1])
#ax.set_xticklabels(['Top ranked', 'Other'])

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Figure6_feature_correlation_by_model.pdf'))
plt.close()


# supplementary tables
save_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, 'supplementary_tables')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# sample summary
harmonized_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'harmonized_metadata.csv'))
wes_metadata = pd.read_csv(os.path.join(SEQUENCE_DIR, 'preprocessing/TONIC_WES_meta_table.tsv'), sep='\t')
rna_metadata = pd.read_csv(os.path.join(SEQUENCE_DIR, 'preprocessing/TONIC_tissue_rna_id.tsv'), sep='\t')
rna_metadata = rna_metadata.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint']].drop_duplicates(), on='Tissue_ID', how='left')

harmonized_metadata = harmonized_metadata.loc[harmonized_metadata.MIBI_data_generated, :]

modality = ['MIBI'] * 4 + ['RNA'] * 3 + ['DNA'] * 1
timepoint = ['primary_untreated', 'baseline', 'post_induction', 'on_nivo'] + ['baseline', 'post_induction', 'on_nivo'] + ['baseline']

sample_summary_df = pd.DataFrame({'modality': modality, 'timepoint': timepoint, 'sample_num': [0] * 8, 'patient_num': [0] * 8})

# populate dataframe
for idx, row in sample_summary_df.iterrows():
    if row.modality == 'MIBI':
        sample_summary_df.loc[idx, 'sample_num'] = len(harmonized_metadata.loc[harmonized_metadata.Timepoint == row.timepoint, :])
        sample_summary_df.loc[idx, 'patient_num'] = len(harmonized_metadata.loc[harmonized_metadata.Timepoint == row.timepoint, 'Patient_ID'].unique())
    elif row.modality == 'RNA':
        sample_summary_df.loc[idx, 'sample_num'] = len(rna_metadata.loc[rna_metadata.Timepoint == row.timepoint, :])
        sample_summary_df.loc[idx, 'patient_num'] = len(rna_metadata.loc[rna_metadata.Timepoint == row.timepoint, 'Patient_ID'].unique())
    elif row.modality == 'DNA':
        sample_summary_df.loc[idx, 'sample_num'] = len(wes_metadata.loc[wes_metadata.timepoint == row.timepoint, :])
        sample_summary_df.loc[idx, 'patient_num'] = len(wes_metadata.loc[wes_metadata.timepoint == row.timepoint, 'Individual.ID'].unique())

sample_summary_df.to_csv(os.path.join(save_dir, 'Supplementary_Table_3.csv'), index=False)

# feature metadata
feature_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_metadata.csv'))

feature_metadata.columns = ['Feature name', 'Feature name including compartment', 'Compartment the feature is calculated in',
                            'Cell types used to calculate feature', 'Level of clustering granularity for cell types',
                            'Type of feature', 'Additional information about the feature', 'Additional information about the feature']

feature_metadata.to_csv(os.path.join(save_dir, 'Supplementary_Table_4.csv'), index=False)

# nivo outcomes supplementary plots
# summarize overlap of top features
top_features_by_feature = top_features[['feature_name_unique', 'comparison']].groupby('feature_name_unique').count().reset_index()
feature_counts = top_features_by_feature.groupby('comparison').count().reset_index()
feature_counts.columns = ['num_comparisons', 'num_features']

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=feature_counts, x='num_comparisons', y='num_features', color='grey', ax=ax)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Figure5_num_comparisons_per_feature.pdf'))
plt.close()



# get overlap between static and evolution top features
overlap_type_dict = {'global': [['primary_untreated', 'baseline', 'post_induction', 'on_nivo'],
                                ['primary__baseline', 'baseline__post_induction', 'baseline__on_nivo', 'post_induction__on_nivo']],
                     'primary': [['primary_untreated'], ['primary__baseline']],
                     'baseline': [['baseline'], ['primary__baseline', 'baseline__post_induction', 'baseline__on_nivo']],
                     'post_induction': [['post_induction'], ['baseline__post_induction', 'post_induction__on_nivo']],
                     'on_nivo': [['on_nivo'], ['baseline__on_nivo', 'post_induction__on_nivo']]}

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
for overlap_type, results in overlap_results.items():
    static_ids = results['static_ids']
    evolution_ids = results['evolution_ids']
    venn2([set(static_ids), set(evolution_ids)], set_labels=('Static', 'Evolution'))
    plt.title(overlap_type)
    plt.savefig(os.path.join(plot_dir, 'Figure6_top_features_overlap_{}.pdf'.format(overlap_type)))
    plt.close()

# compute the correlation between response-associated features
timepoint_features = pd.read_csv(os.path.join(ANALYSIS_DIR, 'timepoint_combined_features.csv'))
feature_ranking_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_ranking.csv'))
feature_ranking_df = feature_ranking_df[np.isin(feature_ranking_df['comparison'], ['primary', 'baseline', 'pre_nivo' , 'on_nivo'])]
feature_ranking_df = feature_ranking_df.sort_values(by = 'feature_rank_global', ascending=True)

top_features = feature_ranking_df.iloc[:100, :].loc[:, ['feature_name_unique', 'comparison', 'feature_type']]
top_features.columns = ['feature_name_unique', 'Timepoint', 'feature_type']

remaining_features = feature_ranking_df.iloc[100:, :].loc[:, ['feature_name_unique', 'comparison', 'feature_type']]
remaining_features.columns = ['feature_name_unique', 'Timepoint', 'feature_type']

def calculate_feature_corr(timepoint_features,
                            top_features,
                            remaining_features,
                            top: bool = True,
                            n_iterations: int = 1000):
    """Compares the correlation between 
            1. response-associated features to response-associated features
            2. response-associated features to remaining features
        by randomly sampling features with replacement. 

    Parameters
    timepoint_features: pd.DataFrame
        dataframe containing the feature values for every patient (feature_name_unique, normalized_mean, Patient_ID, Timepoint)
    top_features: pd.DataFrame
        dataframe containing the top response-associated features (feature_name_unique, Timepoint)
    remaining features: pd.DataFrame
        dataframe containing non response-associated features (feature_name_unique, Timepoint)
    top: bool (default = True)
        boolean indicating if the comparison 1. (True) or 2. (False)
    n_iterations: int (default = 1000)
        number of features randomly selected for comparison
    ----------
    Returns
    corr_arr: np.array 
        array containing the feature correlation values
    ----------
    """
    corr_arr = []
    for _ in range(n_iterations):
        #select feature 1 as a random feature from the top response-associated feature list
        rand_sample1 = top_features.sample(n = 1)
        f1 = timepoint_features.iloc[np.where((timepoint_features['feature_name_unique'] == rand_sample1['feature_name_unique'].values[0]) & (timepoint_features['Timepoint'] == rand_sample1['Timepoint'].values[0]))[0], :]
        if top == True:
            #select feature 2 as a random feature from the top response-associated list, ensuring f1 != f2
            rand_sample2 = rand_sample1
            while (rand_sample2.values == rand_sample1.values).all():
                rand_sample2 = top_features.sample(n = 1)
        else:
            #select feature 2 as a random feature from the remaining feature list
            rand_sample2 = remaining_features.sample(n = 1)

        f2 = timepoint_features.iloc[np.where((timepoint_features['feature_name_unique'] == rand_sample2['feature_name_unique'].values[0]) & (timepoint_features['Timepoint'] == rand_sample2['Timepoint'].values[0]))[0], :]
        merged_features = pd.merge(f1, f2, on = 'Patient_ID') #finds Patient IDs that are shared across timepoints to compute correlation
        corrval = np.abs(merged_features['normalized_mean_x'].corr(merged_features['normalized_mean_y'])) #regardless of direction
        corr_arr.append(corrval)

    return np.array(corr_arr)

#C(100, 2) = 100! / [(100-2)! * 2!] = 4950 unique pairwise combinations top 100 features and each other
corr_within = calculate_feature_corr(timepoint_features, top_features, remaining_features, top=True)
corr_across = calculate_feature_corr(timepoint_features, top_features, remaining_features, top=False)

corr_across = corr_across[~np.isnan(corr_across)]
corr_within = corr_within[~np.isnan(corr_within)]

_, axes = plt.subplots(1, 1, figsize = (5, 4), gridspec_kw={'hspace': 0.45, 'wspace': 0.4, 'bottom':0.15})
g = sns.histplot(corr_within, color='#2089D5', ax = axes, kde = True, bins = 50, label = 'top-top', alpha = 0.5)
g = sns.histplot(corr_across, color='lightgrey', ax = axes, kde = True, bins = 50, label = 'top-remaining', alpha = 0.5)
g.tick_params(labelsize=12)
g.set_ylabel('number of comparisons', fontsize = 12)
g.set_xlabel('abs(correlation)', fontsize = 12)
plt.legend(prop={'size':9})
g.set_xlim(0, 1)
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'correlation_response_features', 'correlation_response_associated_features.pdf'), bbox_inches = 'tight', dpi = 300)
plt.close()

# feature differences by timepoint
timepoints = ['primary', 'baseline', 'pre_nivo' , 'on_nivo']

timepoint_features = pd.read_csv(os.path.join(ANALYSIS_DIR, 'timepoint_combined_features.csv'))
feature_ranking_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_ranking.csv'))
feature_ranking_df = feature_ranking_df[np.isin(feature_ranking_df['comparison'], timepoints)]
feature_ranking_df = feature_ranking_df.sort_values(by = 'feature_rank_global', ascending=True)

#access the top response-associated features (unique because a feature could be in the top in multiple timepoints) 
top_features = np.unique(feature_ranking_df.loc[:, 'feature_name_unique'][:100])

#compute the 90th percentile of importance scores and plot the distribution
perc = np.percentile(feature_ranking_df.importance_score, 90)
_, axes = plt.subplots(1, 1, figsize = (4.5, 3.5), gridspec_kw={'hspace': 0.45, 'wspace': 0.4, 'bottom':0.15})
g = sns.histplot(np.abs(feature_ranking_df.importance_score), ax = axes, color = '#1885F2')
g.tick_params(labelsize=12)
g.set_xlabel('importance score', fontsize = 12)
g.set_ylabel('count', fontsize = 12)
axes.axvline(perc, color = 'k', ls = '--', lw = 1, label = '90th percentile')
g.legend(bbox_to_anchor=(0.98, 0.95), loc='upper right', borderaxespad=0, prop={'size':10})
plt.show()

#subset data based on the 90th percentile
feature_ranking_df = feature_ranking_df[feature_ranking_df['importance_score'] > perc]

#min max scale the importance scores (scales features from 0 to 1)
from sklearn.preprocessing import MinMaxScaler
scaled_perc_scores = MinMaxScaler().fit_transform(feature_ranking_df['importance_score'].values.reshape(-1,1))
feature_ranking_df.loc[:, 'scaled_percentile_importance'] = scaled_perc_scores

#pivot the dataframe for plotting (feature x timepoint)
pivot_df = feature_ranking_df.loc[:, ['scaled_percentile_importance', 'feature_name_unique', 'comparison']].pivot(index = 'feature_name_unique', columns = 'comparison')
pivot_df.columns = pivot_df.columns.droplevel(0)
pivot_df = pivot_df.loc[:, timepoints] #reorder
pivot_df.fillna(0, inplace = True) #set features with nan importance scores (i.e. not in the top 90th percentile) to 0

#subset according to top response-associated features
pivot_df = pivot_df.loc[top_features, :]

#plot clustermap
cmap = ['#D8C198', '#D88484', '#5AA571', '#4F8CBE']
sns.set_style('ticks')
g = sns.clustermap(data = pivot_df, yticklabels=True, cmap = 'Blues', vmin = 0, vmax = 1, row_cluster = True,
                   col_cluster = False, figsize = (7, 18), cbar_pos=(1, .03, .02, .1), dendrogram_ratio=0.1, colors_ratio=0.01,
                   col_colors=cmap)
g.tick_params(labelsize=12)

ax = g.ax_heatmap
ax.set_ylabel('Response-associated Features', fontsize = 12)
ax.set_xlabel('Timepoint', fontsize = 12)

ax.axvline(x=0, color='k',linewidth=2.5)
ax.axvline(x=1, color='k',linewidth=1.5)
ax.axvline(x=2, color='k',linewidth=1.5)
ax.axvline(x=3, color='k',linewidth=1.5)
ax.axvline(x=4, color='k',linewidth=2.5)

ax.axhline(y=0, color='k',linewidth=2.5)
ax.axhline(y=len(pivot_df), color='k',linewidth=2.5)

x0, _y0, _w, _h = g.cbar_pos
for spine in g.ax_cbar.spines:
    g.ax_cbar.spines[spine].set_color('k')
    g.ax_cbar.spines[spine].set_linewidth(1)
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'differences_significant_features_timepoint', 'top_features_time_clustermap.pdf'), bbox_inches = 'tight', dpi =300)

#outlier patient analysis
timepoints = ['primary', 'baseline', 'pre_nivo' , 'on_nivo']

timepoint_features = pd.read_csv(os.path.join(ANALYSIS_DIR, 'timepoint_combined_features.csv'))
feature_ranking_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_ranking.csv'))
feature_ranking_df = feature_ranking_df[np.isin(feature_ranking_df['comparison'], timepoints)]
feature_ranking_df = feature_ranking_df.sort_values(by = 'feature_rank_global', ascending=True)

#subset by the top response-associated features
feature_ranking_df = feature_ranking_df.iloc[:100, :]

#merge dataframes for patient-level analysis (feature, raw_mean, Patient_ID, Timepoint, Clinical_benefit) 
feature_ranking_df.rename(columns = {'comparison':'Timepoint'}, inplace = True)
merged_df = pd.merge(timepoint_features, feature_ranking_df, on = ['feature_name_unique', 'Timepoint'])

#create a dictionary mapping patients to their clinical benefit status
status_df = merged_df.loc[:, ['Patient_ID', 'Clinical_benefit']].drop_duplicates()
status_dict = dict(zip(status_df['Patient_ID'], status_df['Clinical_benefit']))

#for each feature, identify outlier patients 
outliers = dict()
feat_list = list(zip(feature_ranking_df['feature_name_unique'], feature_ranking_df['Timepoint']))
for i in range(0, len(merged_df['Clinical_benefit'].unique())): 
    for feat in feat_list:
        try:
            if i == 0:
                df_subset = merged_df.iloc[np.where((merged_df['Clinical_benefit'] == 'Yes') & (merged_df['feature_name_unique'] == feat[0]) & (merged_df['Timepoint'] == feat[1]))[0]].copy()
                df_subset_op = merged_df.iloc[np.where((merged_df['Clinical_benefit'] == 'No') & (merged_df['feature_name_unique'] == feat[0]) & (merged_df['Timepoint'] == feat[1]))[0]].copy()
            else:
                df_subset = merged_df.iloc[np.where((merged_df['Clinical_benefit'] == 'No') & (merged_df['feature_name_unique'] == feat[0]) & (merged_df['Timepoint'] == feat[1]))[0]].copy()
                df_subset_op = merged_df.iloc[np.where((merged_df['Clinical_benefit'] == 'Yes') & (merged_df['feature_name_unique'] == feat[0]) & (merged_df['Timepoint'] == feat[1]))[0]].copy()

            two_std = df_subset['raw_mean'].std() * 2

            #patient considered to be an outlier for this feature if 2 std from the mean in the direction of the opposite clinical benefit group
            outliers_indices = df_subset['raw_mean'] > df_subset['raw_mean'].mean() + two_std if df_subset_op['raw_mean'].mean() > df_subset['raw_mean'].mean() else df_subset['raw_mean'] < df_subset['raw_mean'].mean() - two_std
            outlier_patients = list(df_subset[outliers_indices]['Patient_ID'].values)

            for patient in outlier_patients:
                if patient not in outliers:
                    outliers[patient] = [(feat[0], feat[1])]
                else:
                    outliers[patient].append((feat[0], feat[1]))
        except:
            continue

#count the number of times a patient is an outlier for the top response-associated features
outlier_counts = pd.DataFrame()
counts = []
patients = []
for patient, features in outliers.items():
    counts.append(len(outliers[patient]))
    patients.append(patient.astype('int64'))

outlier_counts = pd.DataFrame(counts, index = patients, columns = ['outlier_counts'])
outlier_counts['Clinical_benefit'] = pd.Series(outlier_counts.index).map(status_dict).values

#plot the distribution indicating the number of times a patient has discordant feature values for the top response-associated features
sns.set_style('ticks')
_, axes = plt.subplots(1, 1, figsize = (5, 4), gridspec_kw={'hspace': 0.45, 'wspace': 0.4, 'bottom':0.15})
g = sns.histplot(data = outlier_counts, legend=False, ax = axes, bins = 30, color = '#AFA0BA', alpha=1)
g.tick_params(labelsize=10)
g.set_xlabel('number of outliers', fontsize = 12)
g.set_ylabel('number of patients', fontsize = 12)
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'outlier_analysis', 'number_outliers_per_patient.pdf'), bbox_inches = 'tight', dpi =300)

#convert dictionary into a dataframe consisting of (Patient_ID, feature_name_unique, Timepoint, feature_type)
reshaped_data = []
for patient, records in outliers.items():
    for feature, timepoint in records:
        reshaped_data.append((patient, feature, timepoint))

outlier_df = pd.DataFrame(reshaped_data, columns = ['Patient_ID', 'feature_name_unique', 'Timepoint'])
outlier_df = pd.merge(outlier_df, feature_ranking_df.loc[:, ['feature_name_unique', 'feature_type', 'Timepoint']], on = ['feature_name_unique', 'Timepoint'])
outlier_df.to_csv(os.path.join(SUPPLEMENTARY_FIG_DIR, 'outlier_analysis', 'TONIC_outlier_counts.csv'), index=False)

#subset dataframe by patients that have discordant feature values in more than 4 response-associated features 
sig_outlier_patients = outlier_df.groupby('Patient_ID').size()[outlier_df.groupby('Patient_ID').size() > 4].index
sig_outlier_df = outlier_df[np.isin(outlier_df['Patient_ID'], sig_outlier_patients)].copy()

#plot the distribution of the feature classes for patients that have discordant feature values in more than 4 response-associated features 
df_pivot = sig_outlier_df.groupby(['Patient_ID', 'feature_type']).size().unstack().reset_index().melt(id_vars = 'Patient_ID').pivot(index='Patient_ID', columns='feature_type', values='value')

df_pivot.plot(kind='bar', stacked=True, figsize=(6,6))
plt.ylabel('Count', fontsize = 12)
plt.xlabel('Patient ID', fontsize = 12)
g.tick_params(labelsize=10)
plt.legend(bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'outlier_analysis', 'feature_classes_outlier_patients.pdf'), bbox_inches = 'tight', dpi =300)

#are any features are consistently discordant for patients considered to be outliers (i.e. have discordant feature values in more than 4 response-associated features)?
cross_feat_counts = pd.DataFrame(sig_outlier_df.groupby('feature_name_unique').size().sort_values(ascending = False), columns = ['count'])
_, axes = plt.subplots(1, 1, figsize = (6, 16), gridspec_kw={'hspace': 0.45, 'wspace': 0.4, 'bottom':0.15})
g = sns.barplot(x = 'count', y =cross_feat_counts.index, data = cross_feat_counts, ax = axes, color = 'lightgrey')
g.tick_params(labelsize=10)
g.set_ylabel('Feature', fontsize = 12)
g.set_xlabel('Number of outlier patients', fontsize = 12)
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'outlier_analysis', 'features_outlier_patients.pdf'), bbox_inches = 'tight', dpi =300)
