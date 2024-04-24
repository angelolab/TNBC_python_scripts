import os

import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

ranked_features_univariate = pd.read_csv(os.path.join(base_dir, 'analysis_files/feature_ranking.csv'))
ranked_features_univariate_genomic = pd.read_csv(os.path.join(base_dir, 'sequencing_data/genomics_outcome_ranking.csv'))


# organize scores from cross validation
cv_scores = pd.read_csv(os.path.join(base_dir, 'multivariate_lasso', 'results_protein_rna_0205.csv'))
cv_scores['fold'] = len(cv_scores)

cv_scores_long = pd.melt(cv_scores, id_vars=['fold'], value_vars=cv_scores.columns)
cv_scores_long['assay'] = cv_scores_long['variable'].apply(lambda x: 'rna' if 'rna' in x else 'mibi')
cv_scores_long['variable'] = cv_scores_long['variable'].apply(lambda x: x.replace('_rna', '').replace('_mibi', ''))

# generate boxplots with MIBI scores
fig, ax = plt.subplots(1, 1, figsize=(3, 4))
order = ['primary', 'post_induction', 'baseline', 'on_nivo']
sns.stripplot(data=cv_scores_long.loc[cv_scores_long.assay == 'mibi', :], x='variable', y='value',
              order=order, color='black', ax=ax)
sns.boxplot(data=cv_scores_long.loc[cv_scores_long.assay == 'mibi', :], x='variable', y='value',
            order=order, color='grey', ax=ax, showfliers=False)

ax.set_title('AUC')
ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure6_AUC_MIBI.pdf'))
plt.close()

# generate boxplots with RNA scores
fig, ax = plt.subplots(1, 1, figsize=(3, 4))
order = ['post_induction', 'baseline', 'on_nivo']
sns.stripplot(data=cv_scores_long.loc[cv_scores_long.assay == 'rna', :], x='variable', y='value',
              order=order, color='black', ax=ax)
sns.boxplot(data=cv_scores_long.loc[cv_scores_long.assay == 'rna', :], x='variable', y='value',
            order=order, color='grey', ax=ax, showfliers=False)

ax.set_title('AUC')
ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure6_AUC_RNA.pdf'))
plt.close()


# generate boxplots with both RNA and MIBI scores

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
order = ['primary', 'baseline', 'post_induction','on_nivo']
sns.stripplot(data=cv_scores_long, x='variable', y='value', hue='assay',
              order=order, ax=ax, dodge=True)
sns.boxplot(data=cv_scores_long, x='variable', y='value', hue='assay',
            order=order, ax=ax, showfliers=False)

ax.set_title('AUC')
ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure6_AUC_combined.pdf'))
plt.close()


# look at top features
for timepoint in ['baseline', 'on_nivo', 'post_induction']:
    for modality in ['MIBI', 'RNA']:
        # read in top features
        top_features = pd.read_csv(os.path.join(base_dir, 'multivariate_lasso', 'top_features_results_{}_{}.csv'.format(timepoint, modality)))

        # take columns and append into single df
        top_features_long = pd.DataFrame()

        for i in range(10):
            current_fold = top_features.iloc[:, (i * 2):((i * 2) + 2)].reset_index(drop=True)
            current_fold.columns = ['feature', 'coef']
            current_fold = current_fold.loc[~current_fold.feature.isna(), :]
            current_fold['fold'] = i
            current_fold['rank'] = current_fold.index + 1
            current_fold['rank_norm'] = current_fold['rank'] / current_fold.shape[0]
            current_fold['coef_norm'] = current_fold['coef'].abs() / current_fold['coef'].abs().max()
            top_features_long = pd.concat([top_features_long, current_fold])

        # change remove periods from R script modifications
        top_features_long['feature'] = top_features_long['feature'].apply(lambda x: x.replace('.', '+'))

        # # look at top 10 features
        # top_10_features = top_features_long.loc[top_features_long['rank'] <= 10, :]
        # top_10_grouped = top_10_features.groupby('feature').size().sort_values(ascending=False).head(20)
        #
        # # plot barplot of top 10 features grouped
        # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # top_10_grouped.plot(kind='bar', ax=ax)
        # ax.set_title('Top 10 features')
        #
        # plt.tight_layout()
        # plt.savefig(os.path.join(plot_dir, 'Figure6_top_10_features_{}_{}.pdf'.format(timepoint, modality)))
        # plt.close()

        # plot number of occurrences of each feature
        feature_counts = top_features_long.groupby('feature').size().sort_values(ascending=False)
        feature_counts = feature_counts.head(20)
        feature_counts = feature_counts.reset_index()
        feature_counts.columns = ['feature', 'count']

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        feature_counts.plot(kind='bar', x='feature', y='count', ax=ax)
        ax.set_title('Feature occurrences')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Figure6_feature_occurrences_{}_{}.pdf'.format(timepoint, modality)))
        plt.close()

        # look at average rank
        ranked_features = top_features_long.groupby('feature').agg({'rank_norm': 'mean', 'coef_norm': 'mean', 'rank': 'mean'}).sort_values('rank_norm', ascending=True)
        ranked_features = ranked_features.reset_index()

        ranked_features = ranked_features.merge(feature_counts, on='feature')
        ranked_features.rename(columns={'feature': 'feature_name_unique'}, inplace=True)

        # compare with univariate
        if modality == 'MIBI':
            ranked_features_univ_sub = ranked_features_univariate.loc[ranked_features_univariate.feature_name_unique.isin(ranked_features.feature_name_unique), :]
            ranked_features_univ_sub = ranked_features_univ_sub.loc[ranked_features_univ_sub.comparison == timepoint, :]

            # merge
            combined_rankings = ranked_features_univ_sub[['feature_name_unique', 'feature_rank_global', 'importance_score']].merge(ranked_features, on='feature_name_unique')
            combined_rankings = combined_rankings.sort_values('rank_norm')

        else:
            ranked_features_univ_sub_genomic = ranked_features_univariate_genomic.loc[ranked_features_univariate_genomic.feature_name_unique.isin(ranked_features.feature_name_unique), :]
            ranked_features_univ_sub_genomic = ranked_features_univ_sub_genomic.loc[ranked_features_univ_sub_genomic.comparison == timepoint, :]

            # merge
            combined_rankings = ranked_features_univ_sub_genomic[['feature_name_unique', 'feature_rank_global', 'importance_score']].merge(ranked_features, on='feature_name_unique')
            combined_rankings = combined_rankings.sort_values('rank_norm')

        # save csv
        combined_rankings['top_ranked'] = np.logical_and(combined_rankings.coef_norm.values >= 0.3,
                                                         combined_rankings['count'] >= 3)
        combined_rankings.to_csv(os.path.join(plot_dir, 'combined_feature_rankings_{}_{}.csv'.format(timepoint, modality)), index=False)

        # plot
        sns.scatterplot(data=combined_rankings, x='feature_rank_global', y='rank_norm')
        plt.title('Univariate vs multivariate ranking for all nonzero model features')

        plt.savefig(os.path.join(plot_dir, 'Figure6_univariate_vs_multivariate_ranking_{}_{}.pdf'.format(timepoint, modality)))
        plt.close()

        # plot feature score broken down by ranking


        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        sns.stripplot(data=combined_rankings, x='top_ranked', y='importance_score',
                      color='black', ax=ax)
        sns.boxplot(data=combined_rankings, x='top_ranked', y='importance_score',
                    color='grey', ax=ax, showfliers=False)

        ax.set_title('Feature importance score')
        # ax.set_ylim([0, 1.1])
        #ax.set_xticklabels(['Top ranked', 'Other'])

        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Figure6_feature_importance_by_model_{}_{}.pdf'.format(timepoint, modality)))
        plt.close()

# aggregate combined rankings from each timepoint and modality
all_model_rankings = pd.DataFrame()


for timepoint in ['baseline', 'on_nivo', 'post_induction']:
    for modality in ['MIBI', 'RNA']:
        combined_rankings = pd.read_csv(os.path.join(plot_dir, 'combined_feature_rankings_{}_{}.csv'.format(timepoint, modality)))
        combined_rankings['timepoint'] = timepoint
        combined_rankings['modality'] = modality
        combined_rankings = combined_rankings[['timepoint', 'modality', 'feature_name_unique', 'feature_rank_global', 'rank', 'rank_norm', 'importance_score', 'count', 'coef_norm', 'top_ranked']]
        all_model_rankings = pd.concat([all_model_rankings, combined_rankings])

# plot top features
sns.stripplot(data=all_model_rankings.loc[all_model_rankings.top_ranked, :], x='timepoint', y='importance_score', hue='modality')
plt.title('Top ranked features')
plt.ylim([0, 1.05])
plt.savefig(os.path.join(plot_dir, 'Figure6_top_ranked_features.pdf'))
plt.close()

# plot number of times features are selected
sns.histplot(data=all_model_rankings.loc[all_model_rankings.top_ranked, :], x='count', color='grey', multiple='stack',
             binrange=(1, 10), discrete=True)
plt.title('Number of times features are selected')
plt.savefig(os.path.join(plot_dir, 'Figure6_top_ranked_feature_counts.pdf'))
plt.close()

sns.histplot(data=all_model_rankings, x='count', color='grey', multiple='stack',
             binrange=(1, 10), discrete=True)
plt.title('Number of times features are selected')
plt.savefig(os.path.join(plot_dir, 'Figure6_all_feature_counts.pdf'))
plt.close()

# plot venn diagram
from matplotlib_venn import venn3

rna_rankings_top = all_model_rankings.loc[np.logical_and(all_model_rankings.modality == 'RNA', all_model_rankings.top_ranked), :]
rna_baseline = rna_rankings_top.loc[rna_rankings_top.timepoint == 'baseline', 'feature_name_unique'].values
rna_nivo = rna_rankings_top.loc[rna_rankings_top.timepoint == 'on_nivo', 'feature_name_unique'].values
rna_induction = rna_rankings_top.loc[rna_rankings_top.timepoint == 'post_induction', 'feature_name_unique'].values

venn3([set(rna_baseline), set(rna_nivo), set(rna_induction)], ('Baseline', 'Nivo', 'Induction'))
plt.title('RNA top ranked features')
plt.savefig(os.path.join(plot_dir, 'Figure6_RNA_top_ranked_venn.pdf'))
plt.close()

mibi_rankings_top = all_model_rankings.loc[np.logical_and(all_model_rankings.modality == 'MIBI', all_model_rankings.top_ranked), :]
mibi_baseline = mibi_rankings_top.loc[mibi_rankings_top.timepoint == 'baseline', 'feature_name_unique'].values
mibi_nivo = mibi_rankings_top.loc[mibi_rankings_top.timepoint == 'on_nivo', 'feature_name_unique'].values
mibi_induction = mibi_rankings_top.loc[mibi_rankings_top.timepoint == 'post_induction', 'feature_name_unique'].values

venn3([set(mibi_baseline), set(mibi_nivo), set(mibi_induction)], ('Baseline', 'Nivo', 'Induction'))
plt.title('MIBI top ranked features')
plt.savefig(os.path.join(plot_dir, 'Figure6_MIBI_top_ranked_venn.pdf'))
plt.close()

# compare correlations between top ranked features
nivo_features_model = all_model_rankings.loc[np.logical_and(all_model_rankings.timepoint == 'on_nivo', all_model_rankings.top_ranked), :]
nivo_features_model = nivo_features_model.loc[nivo_features_model.modality == 'MIBI', 'feature_name_unique'].values

nivo_features_univariate = ranked_features_univariate.loc[np.logical_and(ranked_features_univariate.comparison == 'on_nivo',
                                                                         ranked_features_univariate.feature_rank_global <= 100), :]

timepoint_features = pd.read_csv(os.path.join(base_dir, 'analysis_files/timepoint_combined_features.csv'))
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
plt.savefig(os.path.join(plot_dir, 'Figure6_feature_correlation_by_model.pdf'))
plt.close()