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
patient_metadata = patient_metadata.loc[~patient_metadata.MIBI_evolution_set.isna(), :]
patient_metadata['iRECIST_response'] = 'non-responders'
patient_metadata.loc[(patient_metadata.BOR_iRECIST.isin(['iCR', 'iPR', 'iSD'])), 'iRECIST_response'] = 'responders'
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

timepoint_features = timepoint_features.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID', 'Timepoint',
                                                                   'baseline__on_nivo', 'baseline__post_induction', 'post_induction__on_nivo']].drop_duplicates(), on='Tissue_ID')
timepoint_features = timepoint_features.merge(patient_metadata[['Patient_ID', 'iRECIST_response']].drop_duplicates(), on='Patient_ID', how='left')

# Hacky, remove once metadata is updated
timepoint_features = timepoint_features.loc[~timepoint_features.iRECIST_response.isna(), :]
timepoint_features = timepoint_features.loc[timepoint_features.Timepoint.isin(['baseline', 'post_induction', 'on_nivo']), :]
timepoint_features = timepoint_features[['Tissue_ID', 'feature_name', 'feature_name_unique', 'raw_mean', 'raw_std', 'normalized_mean', 'normalized_std', 'Patient_ID', 'Timepoint', 'iRECIST_response']]

daisy_dir = '/Users/noahgreenwald/Downloads/daisy_data'
timepoint_features.to_csv(os.path.join(daisy_dir, 'timepoint_features.csv'), index=False)

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

# loop over different populations
pop_df_means = pd.DataFrame({'feature_name_unique': timepoint_features.feature_name_unique.unique()})
keep_rows = []
total_dfs = []

for population in ['baseline', 'post_induction', 'on_nivo']:
    population_df = compare_populations(feature_df=timepoint_features, pop_col='iRECIST_response', timepoints=[population],
                                        pop_1='non-responders', pop_2='responders')
    pval_thresh = 2
    diff_thresh = 0.3
    population_df_filtered = population_df.loc[(population_df.log_pval > pval_thresh) & (np.abs(population_df.mean_diff) > diff_thresh), :]
    keep_rows.extend(population_df_filtered.feature_name_unique.tolist())

    # current_plot_dir = os.path.join(plot_dir, 'responders_nonresponders_timepoint_{}'.format(population))
    # if not os.path.exists(current_plot_dir):
    #     os.makedirs(current_plot_dir)
    # summarize_population_enrichment(input_df=population_df, feature_df=timepoint_features, timepoints=[population],
    #                                 pop_col='iRECIST_response', output_dir=current_plot_dir, sort_by='med_diff')

    long_df = population_df[['feature_name_unique', 'log_pval', 'mean_diff', 'med_diff']]
    long_df['comparison'] = population
    long_df = long_df.dropna()
    long_df['pval'] = 10 ** (-long_df.log_pval)
    long_df['fdr_pval'] = multipletests(long_df.pval, method='fdr_bh')[1]

    total_dfs.append(long_df)


    population_df = population_df.rename(columns={'mean_diff': (population)})
    pop_df_means = pop_df_means.merge(population_df.loc[:, ['feature_name_unique', population]], on=['feature_name_unique'], how='left')


pop_df_means = pop_df_means.loc[pop_df_means.feature_name_unique.isin(keep_rows), :]
pop_df_means = pop_df_means.set_index('feature_name_unique')
pop_df_means = pop_df_means.fillna(0)

# make clustermap 20 x 10
g = sns.clustermap(pop_df_means, cmap='RdBu_r', vmin=-2, vmax=2)
plt.savefig(os.path.join(plot_dir, 'responders_nonresponders_timepoint_clustermap.png'))
plt.close()


# look at evolution

evolution_df = pd.read_csv(os.path.join(data_dir, 'evolution/evolution_df.csv'))
evolution_df = evolution_df.merge(patient_metadata[['Patient_ID', 'iRECIST_response']].drop_duplicates(), on='Patient_ID', how='left')
evolution_df = evolution_df.rename(columns={'raw_value': 'raw_mean', 'normalized_value': 'normalized_mean'})

evolution_df.to_csv(os.path.join(daisy_dir, 'evolution_df.csv'), index=False)

change_df_means = pd.DataFrame({'feature_name_unique': evolution_df.feature_name_unique.unique()})
keep_rows = []
for comparison in [ 'baseline__post_induction', 'baseline__on_nivo', 'post_induction__on_nivo']:
    pop_1, pop_2 = comparison.split('__')
    if pop_1 == 'primary':
        pop_1 = 'primary_untreated'

    # subset to the comparison
    input_df = evolution_df.loc[evolution_df.comparison == comparison, :]
    input_df['Timepoint'] = pop_1

    population_df = compare_populations(feature_df=input_df, pop_col='iRECIST_response', timepoints=[pop_1, pop_2],
                                            pop_1='non-responders', pop_2='responders', feature_suff='mean')

    pval_thresh = 2
    diff_thresh = 0.3
    population_df_filtered = population_df.loc[(population_df.log_pval > pval_thresh) & (np.abs(population_df.mean_diff) > diff_thresh), :]
    keep_rows.extend(population_df_filtered.feature_name_unique.tolist())

    # current_plot_dir = os.path.join(plot_dir, 'response_evolution_{}'.format(comparison))
    # if not os.path.exists(current_plot_dir):
    #     os.makedirs(current_plot_dir)
    # summarize_population_enrichment(input_df=population_df, feature_df=input_df, timepoints=[pop_1, pop_2],
    #                                 pop_col='iRECIST_response', output_dir=current_plot_dir, sort_by='med_diff')

    long_df = population_df[['feature_name_unique', 'log_pval', 'mean_diff', 'med_diff']]
    long_df['comparison'] = comparison
    long_df = long_df.dropna()
    long_df['pval'] = 10 ** (-long_df.log_pval)
    long_df['fdr_pval'] = multipletests(long_df.pval, method='fdr_bh')[1]
    total_dfs.append(long_df)


    population_df = population_df.rename(columns={'mean_diff': (comparison)})
    change_df_means = change_df_means.merge(population_df.loc[:, ['feature_name_unique', comparison]], on='feature_name_unique', how='left')


change_df_means = change_df_means.loc[change_df_means.feature_name_unique.isin(keep_rows), :]
change_df_means = change_df_means.set_index('feature_name_unique')
change_df_means = change_df_means.fillna(0)

# make clustermap 20 x 10
g = sns.clustermap(change_df_means, cmap='RdBu_r', vmin=-2, vmax=2)
plt.savefig(os.path.join(plot_dir, 'response_evolution_clustermap.png'))
plt.close()

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

# generate signed version of score
total_dfs['signed_importance_score'] = total_dfs.importance_score * np.sign(total_dfs.med_diff)

# add feature type
feature_type_dict = {'functional_marker': 'phenotype', 'linear_distance': 'interactions',
                     'density': 'density', 'cell_diversity': 'diversity', 'density_ratio': 'density',
                     'mixing_score': 'interactions', 'region_diversity': 'diversity',
                     'compartment_area_ratio': 'compartment', 'density_proportion': 'density',
                      'morphology': 'phenotype', 'pixie_ecm': 'ecm', 'fiber': 'ecm', 'ecm_cluster': 'ecm',
                        'compartment_area': 'compartment', 'ecm_fraction': 'ecm'}
total_dfs['feature_type_broad'] = total_dfs.feature_type.map(feature_type_dict)

# identify top features
total_dfs['top_feature'] = total_dfs.importance_score > 0.975

# saved formatted df
total_dfs.to_csv(os.path.join(data_dir, 'evolution/evolution_df_outcomes.csv'), index=False)

total_dfs = pd.read_csv(os.path.join(data_dir, 'evolution/evolution_df_outcomes.csv'))


# plot total volcano
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(data=total_dfs, x='med_diff', y='log_pval', alpha=0.5, color='grey')
ax.set_xlim(-3, 3)
sns.despine()
plt.savefig(os.path.join(plot_dir, 'outcomes_volcano.pdf'))
plt.close()


# plot diversity volcano
total_dfs = total_dfs.merge(feature_metadata, on='feature_name_unique', how='left')
total_dfs['diversity'] = total_dfs.feature_type.isin(['mixing_score', 'region_diversity', 'cell_diversity'])

fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(data=total_dfs, x='med_diff', y='log_pval', hue='diversity', alpha=0.5, palette=['lightgrey', 'black'])
ax.set_xlim(-3, 3)
sns.despine()
plt.savefig(os.path.join(plot_dir, 'outcomes_volcano_diversity.pdf'))
plt.close()

# plot phenotype volcano
total_dfs['phenotype'] = total_dfs.feature_type_broad == 'phenotype'

fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(data=total_dfs, x='med_diff', y='log_pval', hue='phenotype', alpha=0.5, palette=['lightgrey', 'black'])
ax.set_xlim(-3, 3)
sns.despine()
plt.savefig(os.path.join(plot_dir, 'outcomes_volcano_phenotype.pdf'))
plt.close()

# look at enriched features
enriched_features = compute_feature_enrichment(feature_df=total_dfs, inclusion_col='top_feature', analysis_col='feature_type_broad')


# plot as a barplot
enriched_features = enriched_features.iloc[:-2, :]
fig, ax = plt.subplots(figsize=(10,8))
sns.barplot(data=enriched_features, x='log2_ratio', y='feature_type_broad', color='grey', ax=ax)
plt.xlabel('Log2 ratio of proportion of top features')
ax.set_xlim(-1.5, 1.5)
sns.despine()
plt.savefig(os.path.join(plot_dir, 'top_feature_enrichment.pdf'))
plt.close()


# look at enriched cell types
enriched_features = compute_feature_enrichment(feature_df=total_dfs, inclusion_col='top_feature', analysis_col='cell_pop')

# plot as a barplot
fig, ax = plt.subplots(figsize=(10,8))
sns.barplot(data=enriched_features, x='log2_ratio', y='cell_pop', color='grey')
plt.xlabel('Log2 ratio of proportion of top features')
ax.set_xlim(-1.5, 1.5)
sns.despine()
plt.savefig(os.path.join(plot_dir, 'top_feature_celltype_enrichment.pdf'))
plt.close()

# plot top features
#top_features = total_dfs.loc[total_dfs.top_feature, :]
top_features = total_dfs.iloc[:100, :]
top_features = top_features.sort_values('importance_score', ascending=False)

for idx, (feature_name, comparison) in enumerate(zip(top_features.feature_name_unique, top_features.comparison)):
    if '__' in comparison:
        source_df = evolution_df
        source_df = source_df.loc[source_df.comparison == comparison, :]

    else:
        source_df = timepoint_features
        source_df = source_df.loc[source_df.Timepoint == comparison, :]

    plot_df = source_df.loc[source_df.feature_name_unique == feature_name, :]

    # plot
    sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'noinduction_responders', 'non-responders'],
                color='grey')
    plt.title(feature_name + ' in ' + comparison)
    plt.savefig(os.path.join(plot_dir, 'top_features_noinduction', f'{idx}_{feature_name}.png'))
    plt.close()


# summarize distribution of top features
top_features_by_comparison = top_features[['feature_name_unique', 'comparison']].groupby('comparison').count().reset_index()
top_features_by_comparison.columns = ['comparison', 'num_features']
top_features_by_comparison = top_features_by_comparison.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_comparison, x='comparison', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'top_features_by_comparison.pdf'))
plt.close()


# summarize overlap of top features
top_features_by_feature = top_features[['feature_name_unique', 'comparison']].groupby('feature_name_unique').count().reset_index()
feature_counts = top_features_by_feature.groupby('comparison').count().reset_index()
feature_counts.columns = ['num_comparisons', 'num_features']

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=feature_counts, x='num_comparisons', y='num_features', color='grey', ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'top_features_by_feature.pdf'))
plt.close()


# plot top featurse across all comparisons
all_top_features = total_dfs.loc[total_dfs.feature_name_unique.isin(top_features.feature_name_unique), :]
all_top_features = all_top_features.loc[~all_top_features.comparison.isin(['primary', 'primary__baseline'])]
all_top_features = all_top_features.pivot(index='feature_name_unique', columns='comparison', values='signed_importance_score')
all_top_features = all_top_features.fillna(0)

sns.clustermap(data=all_top_features, cmap='RdBu_r', vmin=-1, vmax=1, figsize=(10, 10))
plt.savefig(os.path.join(plot_dir, 'top_features_clustermap_all.pdf'))
plt.close()




# create connected dotplot between timepoints by patient
feature_name = 'CD69+__CD4T'
feature_name = 'PDL1+__APC'
feature_name = 'PDL1+__M2_Mac'
feature_name = 'Mono_Mac__PDL1+'
timepoint_1 = 'baseline'
timepoint_2 = 'post_induction'
timepoint_3 = 'on_nivo'

pats = harmonized_metadata.loc[harmonized_metadata.baseline__on_nivo, 'Patient_ID'].unique().tolist()
pats2 = harmonized_metadata.loc[harmonized_metadata.post_induction__on_nivo, 'Patient_ID'].unique().tolist()
pats = set(pats).intersection(set(pats2))

plot_df = timepoint_features.loc[(timepoint_features.feature_name == feature_name) &
                                    (timepoint_features.Timepoint.isin([timepoint_1, timepoint_2, timepoint_3]) &
                                     timepoint_features.Patient_ID.isin(pats)), :]

plot_df_wide = plot_df.pivot(index=['Patient_ID', 'iRECIST_response'], columns='Timepoint', values='raw_mean')
plot_df_wide.dropna(inplace=True)
# divide each row by the baseline value
#plot_df_wide = plot_df_wide.divide(plot_df_wide.loc[:, 'baseline'], axis=0)
#plot_df_wide = plot_df_wide.subtract(plot_df_wide.loc[:, 'baseline'], axis=0)
plot_df_wide = plot_df_wide.reset_index()

plot_df_norm = pd.melt(plot_df_wide, id_vars=['Patient_ID', 'iRECIST_response'], value_vars=['baseline', 'post_induction', 'on_nivo'])

plot_df_1 = plot_df_norm.loc[plot_df_norm.iRECIST_response != 'responders', :]
plot_df_2 = plot_df_norm.loc[plot_df_norm.iRECIST_response == 'responders', :]
fig, ax = plt.subplots(1, 3, figsize=(15, 10))
sns.lineplot(data=plot_df_1, x='Timepoint', y='value', units='Patient_ID', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[0])
sns.lineplot(data=plot_df_2.loc[plot_df_2.Patient_ID == 33], x='Timepoint', y='value', units='Patient_ID', estimator=None, color='grey', alpha=0.5, marker='o', ax=ax[1])
sns.lineplot(data=plot_df_norm, x='Timepoint', y='value', units='Patient_ID',  hue='iRECIST_response', estimator=None, alpha=0.5, marker='o', ax=ax[2])

# set ylimits
ax[0].set_ylim([-0.6, 0.6])
ax[1].set_ylim([-0.6, 0.6])
ax[2].set_ylim([-0.6, 0.6])

# add responder and non-responder titles
ax[0].set_title('non-responders')
ax[1].set_title('responders')
ax[2].set_title('combined')
plt.savefig(os.path.join(plot_dir, 'longitudinal_response_raw_{}.png'.format(feature_name)))
plt.close()

# pat 19, 93 have peak in inudction
# pat 71,108 have high on niov
test = evolution_df.loc[(evolution_df.feature_name == feature_name) &
                        (evolution_df.comparison == timepoint_1 + '__' + timepoint_2), :]

test_2 = test.loc[test.iRECIST_response == 'responders', :]

plot_df_missing = timepoint_features.loc[(timepoint_features.feature_name == feature_name) &
                                         (timepoint_features.Timepoint.isin([timepoint_1, timepoint_2, timepoint_3])) &
                                         (timepoint_features.Patient_ID == 75), :]

# investigate iron in CD8s and Bs
patients = [37, 56]
timepoints = ['baseline', 'post_induction']

select_metadata = harmonized_metadata.loc[harmonized_metadata.Patient_ID.isin(patients) & harmonized_metadata.Timepoint.isin(timepoints), :]
select_metadata = select_metadata.loc[select_metadata.MIBI_data_generated, :]

# copy relevant images
output_dir = os.path.join(plot_dir, 'iron_changes')
channel_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples'
mask_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay'

for i in range(len(select_metadata)):
    pat_id, timepoint, fov = select_metadata.iloc[i, :][['Patient_ID', 'Timepoint', 'fov']]
    output_string = '{}_{}_{}'.format(pat_id, timepoint, fov)
    shutil.copy(os.path.join(channel_dir, fov, 'Fe.tiff'), os.path.join(output_dir, output_string + '_Fe.tiff'))
    shutil.copy(os.path.join(mask_dir, fov + '.png'), os.path.join(output_dir, output_string + '_mask.png'))


# plot specific top features

# PDL1+__APC in induction
feature_name = 'PDL1+__APC'
timepoint = 'post_induction'

plot_df = timepoint_features.loc[(timepoint_features.feature_name_unique == feature_name) &
                                    (timepoint_features.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='grey', ax=ax)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([0, 1])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'response_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()

# change in CD8T density in cancer border
feature_name = 'CD8T__cluster_density__cancer_border'
comparison = 'post_induction__on_nivo'

plot_df = evolution_df.loc[(evolution_df.feature_name_unique == feature_name) &
                            (evolution_df.comparison == comparison), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='grey', ax=ax)
ax.set_title(feature_name + ' ' + comparison)
ax.set_ylim([-.1, .2])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'response_{}_{}.pdf'.format(feature_name, comparison)))
plt.close()

# diversity of stroma in on nivo
feature_name = 'diversity_cell_cluster__Stroma'
timepoint = 'on_nivo'

plot_df = timepoint_features.loc[(timepoint_features.feature_name_unique == feature_name) &
                                    (timepoint_features.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='grey', ax=ax)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([0, 2.5])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'response_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()


# ratio of stroma to t cells
feature_name = 'Stroma__T__ratio__cancer_core'
timepoint = 'on_nivo'

plot_df = timepoint_features.loc[(timepoint_features.feature_name_unique == feature_name) &
                                    (timepoint_features.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(2, 4))
sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='grey', ax=ax)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([-10, 10])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'response_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()


# ratio of cancer to t cells
feature_name = 'Cancer__T__ratio'
timepoint = 'on_nivo'

plot_df = timepoint_features.loc[(timepoint_features.feature_name_unique == feature_name) &
                                    (timepoint_features.Timepoint == timepoint), :]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
sns.stripplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='black', ax=ax)
sns.boxplot(data=plot_df, x='iRECIST_response', y='raw_mean', order=['responders', 'non-responders'],
                color='grey', ax=ax)
ax.set_title(feature_name + ' ' + timepoint)
ax.set_ylim([-10, 15])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'response_{}_{}.pdf'.format(feature_name, timepoint)))
plt.close()


