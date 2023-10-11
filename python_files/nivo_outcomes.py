import os

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr, ttest_ind, ttest_rel

from python_files.utils import find_conserved_features, compare_timepoints, compare_populations
from python_files.utils import summarize_population_enrichment, summarize_timepoint_enrichment, compute_feature_enrichment

from statsmodels.stats.multitest import multipletests


plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
patient_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/TONIC_data_per_patient.csv'))
feature_metadata = pd.read_csv(os.path.join(data_dir, 'feature_metadata.csv'))
timepoint_features = pd.read_csv(os.path.join(data_dir, 'timepoint_features_filtered.csv'))

#patient_metadata.loc[patient_metadata.Patient_ID.isin([33, 40, 75, 85, 100, 105, 109]), 'iRECIST_response'] = 'noinduction_responders'
# func_df_timepoint = pd.read_csv(os.path.join(data_dir, 'functional_df_per_timepoint_filtered_deduped.csv'))
# func_df_timepoint = func_df_timepoint.loc[(func_df_timepoint.cell_type == 'Mono_Mac') & (func_df_timepoint.subset == 'all') &
#                                           (func_df_timepoint.functional_marker == 'PDL1') & (func_df_timepoint.metric == 'cluster_broad_freq') &
#                                           (func_df_timepoint.MIBI_data_generated), :]
# # add total mono_mac PDL1 expression to df
# func_df_timepoint['feature_name'] = 'Mono_Mac__PDL1+'
# func_df_timepoint['feature_name_unique'] = 'Mono_Mac__PDL1+'
# func_df_timepoint['compartment'] = 'all'
# func_df_timepoint['cell_pop_level'] = 'broad'
# func_df_timepoint['feature_type'] = 'functional_marker'
# func_df_timepoint = func_df_timepoint.rename(columns={'mean': 'raw_mean', 'std': 'raw_std', 'cell_type': 'cell_pop'})
# timepoint_features = timepoint_features.append(func_df_timepoint[['Tissue_ID', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop_level', 'feature_type', 'raw_mean', 'raw_std']])


# # create combined df
# timepoint_features = timepoint_features.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint', 'primary__baseline',
#                                                                    'baseline__on_nivo', 'baseline__post_induction', 'post_induction__on_nivo']].drop_duplicates(), on='Tissue_ID')
# timepoint_features = timepoint_features.merge(patient_metadata[['Patient_ID', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']].drop_duplicates(), on='Patient_ID', how='left')
#
# # Hacky, remove once metadata is updated
# timepoint_features = timepoint_features.loc[timepoint_features.Clinical_benefit.isin(['Yes', 'No']), :]
# timepoint_features = timepoint_features.loc[timepoint_features.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
# timepoint_features = timepoint_features[['Tissue_ID', 'feature_name', 'feature_name_unique', 'raw_mean', 'raw_std', 'normalized_mean', 'normalized_std', 'Patient_ID', 'Timepoint', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]
#
# # # copy lymphnode data and make a combined version
# # lymphnode_df = timepoint_features.loc[timepoint_features.Timepoint.isin(['lymphnode_pos', 'lymphnode_neg']), :].copy()
# # lymphnode_df['Timepoint'] = 'lymphnode'
# #
# # timepoint_features = pd.concat([timepoint_features, lymphnode_df])
#
# # # rename induction timepoint based on treatment
# # timepoint_features.loc[timepoint_features.Timepoint == 'post_induction', 'Timepoint'] = timepoint_features.loc[timepoint_features.Timepoint == 'post_induction', 'Induction_treatment'] + '__post_induction'
# #
# # # rename induction treatment based on no-induction or induction
# # timepoint_features.loc[(timepoint_features.Induction_treatment == 'No induction') & (timepoint_features.Timepoint == 'post_induction'), 'Timepoint'] = 'induction_control'
#
# # look at evolution
# evolution_df = pd.read_csv(os.path.join(data_dir, 'nivo_outcomes/evolution_df.csv'))
# evolution_df = evolution_df.merge(patient_metadata[['Patient_ID', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']].drop_duplicates(), on='Patient_ID', how='left')
# evolution_df = evolution_df.rename(columns={'raw_value': 'raw_mean', 'normalized_value': 'normalized_mean', 'comparison': 'Timepoint'})
# evolution_df = evolution_df[['feature_name_unique', 'raw_mean', 'normalized_mean', 'Patient_ID', 'Timepoint', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]
#
# # combine together into single df
# combined_df = timepoint_features.copy()
# combined_df = combined_df[['feature_name_unique', 'raw_mean', 'normalized_mean', 'Patient_ID', 'Timepoint', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]
# combined_df = pd.concat([combined_df, evolution_df[['feature_name_unique', 'raw_mean', 'normalized_mean',
#                                                     'Patient_ID', 'Timepoint', 'Induction_treatment', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]])
# combined_df['combined_name'] = combined_df.feature_name_unique + '__' + combined_df.Timepoint
#
# combined_df.to_csv(os.path.join(data_dir, 'nivo_outcomes/combined_df.csv'), index=False)

# load previously computed results
combined_df = pd.read_csv(os.path.join(data_dir, 'nivo_outcomes/combined_df.csv'))


# # look at change due to nivo
# for comparison in ['baseline__on_nivo', 'baseline__post_induction', 'post_induction__on_nivo']:
#
#     # compare pre and post therapy
#     pop_1, pop_2 = comparison.split('__')
#     compare_df = compare_timepoints(feature_df=timepoint_features, timepoint_1_name=pop_1, timepoint_1_list=[pop_1],
#                                     timepoint_2_name=pop_2, timepoint_2_list=[pop_2], paired=comparison)
#     # plot results
#     output_dir = plot_dir + '/evolution_{}'.format(comparison)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     summarize_timepoint_enrichment(input_df=compare_df, feature_df=timepoint_features, timepoints=[pop_1, pop_2],
#                                  pval_thresh=2, diff_thresh=0.3, output_dir=output_dir)

# generate a single set of top hits across all comparisons

# settings for generating hits
plot_hits = False
method = 'ttest'

# placeholder for all values
total_dfs = []

for comparison in combined_df.Timepoint.unique():
    population_df = compare_populations(feature_df=combined_df, pop_col='Clinical_benefit',
                                        timepoints=[comparison], pop_1='No', pop_2='Yes', method=method)

    if plot_hits:
        current_plot_dir = os.path.join(plot_dir, 'responders_nonresponders_{}'.format(comparison))
        if not os.path.exists(current_plot_dir):
            os.makedirs(current_plot_dir)
        summarize_population_enrichment(input_df=population_df, feature_df=combined_df, timepoints=[comparison],
                                        pop_col='Clinical_benefit', output_dir=current_plot_dir, sort_by='med_diff')

    if np.sum(~population_df.log_pval.isna()) == 0:
        continue
    long_df = population_df[['feature_name_unique', 'log_pval', 'mean_diff', 'med_diff']]
    long_df['comparison'] = comparison
    long_df = long_df.dropna()
    long_df['pval'] = 10 ** (-long_df.log_pval)
    long_df['fdr_pval'] = multipletests(long_df.pval, method='fdr_bh')[1]
    total_dfs.append(long_df)


# total_dfs_continuous = []
# for comparison in ['baseline', 'post_induction', 'on_nivo', 'baseline__post_induction', 'baseline__on_nivo', 'post_induction__on_nivo']:
#     input_df = combined_df[combined_df.Timepoint == comparison]
#     continuous_df = compare_continuous(feature_df=input_df, variable_col='Time_to_progression_weeks_RECIST1.1')
#
#     if plot_hits:
#         current_plot_dir = os.path.join(plot_dir, 'responders_nonresponders_continuous_{}'.format(comparison))
#         if not os.path.exists(current_plot_dir):
#             os.makedirs(current_plot_dir)
#         summarize_continuous_enrichment(input_df=continuous_df, feature_df=combined_df, timepoint=comparison,
#                                         variable_col='Time_to_progression_weeks_RECIST1.1', output_dir=current_plot_dir, min_score=0.95)

# summarize hits from all comparisons
total_dfs = pd.concat(total_dfs)
total_dfs['log10_qval'] = -np.log10(total_dfs.fdr_pval)

# create importance score
# get ranking of each row by log_pval
total_dfs['pval_rank'] = total_dfs.log_pval.rank(ascending=False)
total_dfs['cor_rank'] = total_dfs.med_diff.abs().rank(ascending=False)
total_dfs['combined_rank'] = (total_dfs.pval_rank.values + total_dfs.cor_rank.values) / 2

# plot top X features per comparison
num_features = 20

for comparison in total_dfs.comparison.unique():
    current_plot_dir = os.path.join(plot_dir, 'top_features_{}'.format(comparison))
    if not os.path.exists(current_plot_dir):
        os.makedirs(current_plot_dir)

    current_df = total_dfs.loc[total_dfs.comparison == comparison, :]
    current_df = current_df.sort_values('combined_rank', ascending=True)
    current_df = current_df.iloc[:num_features, :]

    # plot results
    for feature_name, rank in zip(current_df.feature_name_unique.values, current_df.combined_rank.values):
        plot_df = combined_df.loc[(combined_df.feature_name_unique == feature_name) &
                                  (combined_df.Timepoint == comparison), :]

        g = sns.catplot(data=plot_df, x='Clinical_benefit', y='raw_mean', kind='strip')
        g.fig.suptitle(feature_name)
        g.savefig(os.path.join(current_plot_dir, 'rank_{}_feature_{}.png'.format(rank, feature_name)))
        plt.close()

# generate importance score
max_rank = len(~total_dfs.med_diff.isna())
normalized_rank = total_dfs.combined_rank / max_rank
total_dfs['importance_score'] = 1 - normalized_rank

total_dfs = total_dfs.sort_values('importance_score', ascending=False)
# total_dfs = total_dfs.sort_values('fdr_pval', ascending=True)

# generate signed version of score
total_dfs['signed_importance_score'] = total_dfs.importance_score * np.sign(total_dfs.med_diff)

# add feature type
total_dfs = total_dfs.merge(feature_metadata, on='feature_name_unique', how='left')

feature_type_dict = {'functional_marker': 'phenotype', 'linear_distance': 'interactions',
                     'density': 'density', 'cell_diversity': 'diversity', 'density_ratio': 'density',
                     'mixing_score': 'interactions', 'region_diversity': 'diversity',
                     'compartment_area_ratio': 'compartment', 'density_proportion': 'density',
                      'morphology': 'phenotype', 'pixie_ecm': 'ecm', 'fiber': 'ecm', 'ecm_cluster': 'ecm',
                        'compartment_area': 'compartment', 'ecm_fraction': 'ecm'}
total_dfs['feature_type_broad'] = total_dfs.feature_type.map(feature_type_dict)

# identify top features
total_dfs['top_feature'] = False
total_dfs.iloc[:100, -1] = True

# saved formatted df
total_dfs.to_csv(os.path.join(data_dir, 'nivo_outcomes/outcomes_df.csv'), index=False)


# compare subsets of features to see effect of dox only

# # create different input dfs
# default_df = timepoint_features.copy()
# control_df = timepoint_features.loc[timepoint_features.Induction_treatment != 'No induction', :]
# dox_df = timepoint_features.loc[timepoint_features.Induction_treatment == 'Doxorubicin', :]
#
# len(dox_df.loc[(dox_df.Timepoint == 'post_induction'), 'Patient_ID'].unique())
# induction_pats = default_df.loc[(default_df.Timepoint == 'post_induction'), 'Patient_ID'].unique()
# len(induction_pats)
#
# # pick random subset from induction patients
# np.random.seed(42)
# induction_pat_subsets = [np.random.choice(induction_pats, size=29, replace=False) for _ in range(10)]
#
# dfs = [[default_df, 'default'], [control_df, 'control'], [dox_df, 'dox_only']]
# random_dfs = [[default_df.loc[default_df.Patient_ID.isin(vals), :], 'subset_{}'.format(idx)] for idx, vals in enumerate(induction_pat_subsets)]
#
# dfs = dfs + random_dfs
#
# # generate top hits for each dataset
# total_dfs = []
# for input_df, name in dfs:
#     population_df = compare_populations(feature_df=input_df, pop_col='Clinical_benefit',
#                                             timepoints=['post_induction'], pop_1='No', pop_2='Yes', method='ttest')
#
#
#     long_df = population_df[['feature_name_unique', 'log_pval', 'mean_diff', 'med_diff']]
#     long_df['comparison'] = name
#     long_df = long_df.dropna()
#
#     long_df['pval_rank'] = long_df.log_pval.rank(ascending=False)
#     long_df['cor_rank'] = long_df.med_diff.abs().rank(ascending=False)
#     long_df['combined_rank'] = (long_df.pval_rank.values + long_df.cor_rank.values) / 2
#
#     # add to overall df
#     total_dfs.append(long_df)
#
# total_dfs = pd.concat(total_dfs)
#
# wide_df = total_dfs.pivot(index='feature_name_unique', columns='comparison', values='combined_rank')
# wide_df = wide_df.reset_index()
# wide_df = wide_df.loc[wide_df.default < 100]
#
# # plot results
# plot_df = wide_df.loc[:, ['default', 'subset_5']]
# plot_df.dropna(axis=0, inplace=True)
# sns.scatterplot(data=plot_df, x='default', y='subset_5')
# plt.title('All samples vs. subset 5')
#
# corr, _ = spearmanr(plot_df.default, plot_df.subset_5)
# plt.text(0.5, 0.5, 'Spearman R = {:.2f}'.format(corr), transform=plt.gca().transAxes)
# plt.savefig(os.path.join(plot_dir, 'Feature_Correlation_subset_5.pdf'))
# plt.close()
#
# corrs = []
# for i in range(10):
#     col = 'subset_{}'.format(i)
#     plot_df = wide_df.loc[:, ['default', col]]
#     plot_df.dropna(axis=0, inplace=True)
#     corr, _ = spearmanr(plot_df.default, plot_df[col])
#     corrs.append(corr)
#
# cor_df = pd.DataFrame(corrs, columns=['correlation'])
#
# plot_df = wide_df.loc[:, ['default', 'dox_only']]
# plot_df.dropna(axis=0, inplace=True)
# dox_cor, _ = spearmanr(plot_df.default, plot_df.dox_only)
#
# cor_df = pd.concat([cor_df, (pd.DataFrame([dox_cor], columns=['correlation']))])
# cor_df['type'] = 'randomized'
# cor_df.iloc[-1, -1] = 'dox_only'
#
# sns.stripplot(data=cor_df, x='type', y='correlation')
# plt.savefig(os.path.join(plot_dir, 'Feature_Correlation_summary.pdf'))
# plt.close()