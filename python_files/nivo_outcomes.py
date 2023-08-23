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


local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
patient_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/TONIC_data_per_patient.csv'))
feature_metadata = pd.read_csv(os.path.join(data_dir, 'feature_metadata.csv'))
timepoint_features = pd.read_csv(os.path.join(data_dir, 'timepoint_features_filtered.csv'))
patient_metadata['iRECIST_response'] = 'non-responders'
patient_metadata.loc[(patient_metadata.BOR_iRECIST.isin(['iCR', 'iPR', 'iSD'])), 'iRECIST_response'] = 'responders'

# create mask where boolean arrays are not equal
patient_metadata['survival_diff'] = np.equal(patient_metadata.iRECIST_response.values == 'responders',  patient_metadata.Clinical_benefit.values == 'Yes')


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
# timepoint_features = timepoint_features.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint',
#                                                                    'baseline__on_nivo', 'baseline__post_induction', 'post_induction__on_nivo']].drop_duplicates(), on='Tissue_ID')
# timepoint_features = timepoint_features.merge(patient_metadata[['Patient_ID', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']].drop_duplicates(), on='Patient_ID', how='left')
#
# # Hacky, remove once metadata is updated
# timepoint_features = timepoint_features.loc[timepoint_features.Clinical_benefit.isin(['Yes', 'No']), :]
# timepoint_features = timepoint_features.loc[timepoint_features.Timepoint.isin(['baseline', 'post_induction', 'on_nivo']), :]
# timepoint_features = timepoint_features[['Tissue_ID', 'feature_name', 'feature_name_unique', 'raw_mean', 'raw_std', 'normalized_mean', 'normalized_std', 'Patient_ID', 'Timepoint', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]
#
#
# # look at evolution
# evolution_df = pd.read_csv(os.path.join(data_dir, 'evolution/evolution_df.csv'))
# evolution_df = evolution_df.merge(patient_metadata[['Patient_ID', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']].drop_duplicates(), on='Patient_ID', how='left')
# evolution_df = evolution_df.rename(columns={'raw_value': 'raw_mean', 'normalized_value': 'normalized_mean', 'comparison': 'Timepoint'})
# evolution_df = evolution_df[['feature_name_unique', 'raw_mean', 'normalized_mean', 'Patient_ID', 'Timepoint', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]
#
# # combine together into single df
# combined_df = timepoint_features.copy()
# combined_df = combined_df[['feature_name_unique', 'raw_mean', 'normalized_mean', 'Patient_ID', 'Timepoint', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']]
# combined_df = combined_df.append(evolution_df[['feature_name_unique', 'raw_mean', 'normalized_mean', 'Patient_ID', 'Timepoint', 'Time_to_progression_weeks_RECIST1.1', 'Censoring_PFS_RECIST1.1', 'Clinical_benefit']])
# combined_df['combined_name'] = combined_df.feature_name_unique + '__' + combined_df.Timepoint
#
# combined_df.to_csv(os.path.join(data_dir, 'nivo_outcomes/combined_df_metacluster.csv'), index=False)

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
plot_hits = True
method = 'ttest'

# placeholder for all values
total_dfs = []

for comparison in ['baseline', 'post_induction', 'on_nivo', 'baseline__post_induction', 'baseline__on_nivo', 'post_induction__on_nivo']:
    population_df = compare_populations(feature_df=combined_df, pop_col='Clinical_benefit',
                                        timepoints=[comparison], pop_1='No', pop_2='Yes', method=method)

    if plot_hits:
        current_plot_dir = os.path.join(plot_dir, 'responders_nonresponders_{}'.format(comparison))
        if not os.path.exists(current_plot_dir):
            os.makedirs(current_plot_dir)
        summarize_population_enrichment(input_df=population_df, feature_df=combined_df, timepoints=[comparison],
                                        pop_col='Clinical_benefit', output_dir=current_plot_dir, sort_by='med_diff')

    long_df = population_df[['feature_name_unique', 'log_pval', 'mean_diff', 'med_diff']]
    long_df['comparison'] = comparison
    long_df = long_df.dropna()
    long_df['pval'] = 10 ** (-long_df.log_pval)
    long_df['fdr_pval'] = multipletests(long_df.pval, method='fdr_bh')[1]
    total_dfs.append(long_df)


total_dfs_continuous = []
for comparison in ['baseline', 'post_induction', 'on_nivo', 'baseline__post_induction', 'baseline__on_nivo', 'post_induction__on_nivo']:
    input_df = combined_df[combined_df.Timepoint == comparison]
    continuous_df = compare_continuous(feature_df=input_df, variable_col='Time_to_progression_weeks_RECIST1.1')

    if plot_hits:
        current_plot_dir = os.path.join(plot_dir, 'responders_nonresponders_continuous_{}'.format(comparison))
        if not os.path.exists(current_plot_dir):
            os.makedirs(current_plot_dir)
        summarize_continuous_enrichment(input_df=continuous_df, feature_df=combined_df, timepoint=comparison,
                                        variable_col='Time_to_progression_weeks_RECIST1.1', output_dir=current_plot_dir, min_score=0.95)

# summarize hits from all comparisons
total_dfs = pd.concat(total_dfs)
total_dfs['log10_qval'] = -np.log10(total_dfs.fdr_pval)

# create importance score
# get ranking of each row by log_pval
total_dfs['pval_rank'] = total_dfs.log_pval.rank(ascending=False)
total_dfs['cor_rank'] = total_dfs.med_diff.abs().rank(ascending=False)
total_dfs['combined_rank'] = (total_dfs.pval_rank.values + total_dfs.cor_rank.values) / 2

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
total_dfs.to_csv(os.path.join(data_dir, 'nivo_outcomes/outcomes_df_metacluster.csv'), index=False)





# look at top features across all patients
top_features = total_dfs.loc[total_dfs.top_feature, :]
top_features['combined_name'] = top_features.feature_name_unique + '__' + top_features.comparison

top_feature_df = combined_df.loc[combined_df.combined_name.isin(top_features.combined_name.values), :]

patient_feature_df = top_feature_df.pivot(index=['Patient_ID', 'iRECIST_response'], columns='combined_name', values='normalized_mean')
patient_feature_df = patient_feature_df.reset_index()

patient_feature_df.fillna(0, inplace=True)
patient_feature_df['response_status'] = (patient_feature_df.iRECIST_response == 'responders').astype(int)

plot_df = patient_feature_df.drop(['Patient_ID', 'iRECIST_response'], axis=1)
#plot_df = plot_df.loc[plot_df.response_status == 1, :]
sns.clustermap(plot_df, figsize=(20, 20),
                cmap='RdBu_r', vmin=-5, vmax=5, center=0)
plt.savefig(os.path.join(plot_dir, 'top_features_clustermap.pdf'))
plt.close()

from scipy.stats import fisher_exact

# create fake data
fake_data = pd.DataFrame(np.array([[0, 10], [200, 3000]]), columns=['top_features', 'all_features'], index=['feature_category', 'other_categories'])
fisher_exact(fake_data)

