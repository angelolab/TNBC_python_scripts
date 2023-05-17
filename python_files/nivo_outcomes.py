import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr, ttest_ind, ttest_rel

from python_files.utils import find_conserved_features, compare_timepoints, compare_populations
from python_files.utils import summarize_population_enrichment, summarize_timepoint_enrichment


local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
patient_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/TONIC_data_per_patient.csv'))
patient_metadata = patient_metadata.loc[~patient_metadata.MIBI_evolution_set.isna(), :]
patient_metadata['iRECIST_response'] = 'non-responders'
patient_metadata.loc[(patient_metadata.BOR_iRECIST.isin(['iCR', 'iPR', 'iSD'])), 'iRECIST_response'] = 'responders'

timepoint_features = pd.read_csv(os.path.join(data_dir, 'timepoint_features_no_compartment.csv'))
timepoint_features = timepoint_features.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint', 'primary__baseline']].drop_duplicates(), on='Tissue_ID')


# look at nivo response
means_df = compute_timepoint_enrichment(feature_df=timepoint_features, timepoint_1_name='baseline',
                                            timepoint_1_list=['baseline'], timepoint_2_name='nivo',
                                            timepoint_2_list=['on_nivo'])

summarize_timepoint_enrichment(input_df=means_df, feature_df=timepoint_features, timepoints=['baseline', 'on_nivo'],
                                 pval_thresh=2, diff_thresh=0.3, output_dir=plot_dir + '/baseline_nivo_evolution')



responder_change_df = compute_timepoint_enrichment(feature_df=timepoint_features.loc[timepoint_features.Patient_ID.isin(responders), :], timepoint_1_name='baseline',
                                            timepoint_1_list=['baseline'], timepoint_2_name='nivo',
                                            timepoint_2_list=['on_nivo'])

summarize_timepoint_enrichment(input_df=responder_change_df, feature_df=timepoint_features.loc[timepoint_features.Patient_ID.isin(responders), :], timepoints=['baseline', 'on_nivo'],
                                    pval_thresh=2, diff_thresh=0.3, output_dir=plot_dir + '/baseline_nivo_evolution_responders')

# loop over different populations
pop_df_means = pd.DataFrame({'feature_name': timepoint_features.feature_name.unique()})
keep_rows = []
for population in ['primary_untreated', 'baseline', 'post_induction', 'on_nivo']:
    population_df = compare_populations(feature_df=timepoint_features, pop_col='iRECIST_response', timepoints=[population],
                                        pop_1='responders', pop_2='non-responders')
    pval_thresh = 2
    diff_thresh = 0.3
    population_df_filtered = population_df.loc[(population_df.log_pval > pval_thresh) & (np.abs(population_df.mean_diff) > diff_thresh), :]
    keep_rows.extend(population_df_filtered.feature_name.tolist())

    current_plot_dir = os.path.join(plot_dir, 'new_{}_responders_nonresponders'.format(population))
    if not os.path.exists(current_plot_dir):
        os.makedirs(current_plot_dir)
    summarize_population_enrichment(input_df=population_df, feature_df=timepoint_features, timepoints=[population],
                                    pop_col='iRECIST_response', output_dir=current_plot_dir)

    population_df = population_df.rename(columns={'mean_diff': (population)})
    pop_df_means = pop_df_means.merge(population_df.loc[:, ['feature_name', population]], on='feature_name', how='left')

pop_df_means = pop_df_means.loc[pop_df_means.feature_name.isin(keep_rows), :]
pop_df_means = pop_df_means.set_index('feature_name')
pop_df_means = pop_df_means.fillna(0)

# make clustermap 20 x 10
g = sns.clustermap(pop_df_means, cmap='RdBu_r', vmin=-2, vmax=2)
plt.savefig(os.path.join(plot_dir, 'clustermap_responders_nonresponders.png'))
plt.close()


# look at evolution

evolution_df = pd.read_csv(os.path.join(data_dir, 'evolution/evolution_df.csv'))
evolution_df = evolution_df.merge(patient_metadata[['Patient_ID', 'iRECIST_response']].drop_duplicates(), on='Patient_ID', how='left')

change_df_means = pd.DataFrame({'feature_name': evolution_df.feature_name.unique()})
keep_rows = []
for comparison in ['primary__baseline', 'baseline__post_induction',
       'baseline__on_nivo', 'post_induction__on_nivo']:
    pop_1, pop_2 = comparison.split('__')
    if pop_1 == 'primary':
        pop_1 = 'primary_untreated'

    # subset to the comparison
    input_df = evolution_df.loc[evolution_df.comparison == comparison, :]
    input_df['Timepoint'] = pop_1

    population_df = compare_populations(feature_df=input_df, pop_col='iRECIST_response', timepoints=[pop_1, pop_2],
                                            pop_1='responders', pop_2='non-responders', feature_suff='value')

    pval_thresh = 2
    diff_thresh = 0.3
    population_df_filtered = population_df.loc[(population_df.log_pval > pval_thresh) & (np.abs(population_df.mean_diff) > diff_thresh), :]
    keep_rows.extend(population_df_filtered.feature_name.tolist())

    current_plot_dir = os.path.join(plot_dir, 'evolution_{}_responders_nonresponders'.format(comparison))
    if not os.path.exists(current_plot_dir):
        os.makedirs(current_plot_dir)
    summarize_population_enrichment(input_df=population_df, feature_df=input_df, timepoints=[pop_1, pop_2],
                                    pop_col='iRECIST_response', output_dir=current_plot_dir)

    population_df = population_df.rename(columns={'mean_diff': (comparison)})
    change_df_means = change_df_means.merge(population_df.loc[:, ['feature_name', comparison]], on='feature_name', how='left')


change_df_means = change_df_means.loc[change_df_means.feature_name.isin(keep_rows), :]
change_df_means = change_df_means.set_index('feature_name')
change_df_means = change_df_means.fillna(0)

# make clustermap 20 x 10
g = sns.clustermap(change_df_means, cmap='RdBu_r', vmin=-2, vmax=2)


# create connected dotplot between timepoints by patient
feature_name = 'Ki67+__CD4T'
timepoint_1 = 'baseline'
timepoint_2 = 'on_nivo'
timepoint_3 = 'on_nivo'

pats = harmonized_metadata.loc[harmonized_metadata.baseline__on_nivo, 'Patient_ID'].unique().tolist()
#pats2 = harmonized_metadata.loc[harmonized_metadata.baseline__on_nivo, 'Patient_ID'].unique().tolist()
#pats = list(set(pats) & set(pats2))

plot_df = timepoint_features.loc[(timepoint_features.feature_name == feature_name) &
                                    (timepoint_features.Timepoint.isin([timepoint_1, timepoint_2, timepoint_3]) &
                                     timepoint_features.Patient_ID.isin(pats)), :]
plot_df_1 = plot_df.loc[plot_df.iRECIST_response != 'responders', :]
plot_df_2 = plot_df.loc[plot_df.iRECIST_response == 'responders', :]
fig, ax = plt.subplots(1, 3, figsize=(15, 10))
sns.lineplot(data=plot_df_1, x='Timepoint', y='raw_mean', units='Patient_ID', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[0])
sns.lineplot(data=plot_df_2, x='Timepoint', y='raw_mean', units='Patient_ID', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[1])
sns.lineplot(data=plot_df, x='Timepoint', y='raw_mean', units='Patient_ID',  hue='iRECIST_response', estimator=None, alpha=0.5, marker='o', ax=ax[2])

# add responder and non-responder titles
ax[0].set_title('non-responders')
ax[1].set_title('responders')
ax[2].set_title('combined')



test = evolution_df.loc[(evolution_df.feature_name == feature_name) &
                        (evolution_df.comparison == timepoint_1 + '__' + timepoint_2), :]

test_2 = test.loc[test.iRECIST_response == 'responders', :]

plot_df_missing = timepoint_features.loc[(timepoint_features.feature_name == feature_name) &
                                         (timepoint_features.Timepoint.isin([timepoint_1, timepoint_2, timepoint_3])) &
                                         (timepoint_features.Patient_ID == 75), :]