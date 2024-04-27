import os

import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns


base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/'

ranked_features_univariate = pd.read_csv(os.path.join(base_dir, 'analysis_files/feature_ranking.csv'))


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
top_features = pd.read_csv(os.path.join(base_dir, 'multivariate_lasso', 'top_features_results_on_nivo_MIBI.csv'))

# take columns and append into single df
top_features_long = pd.DataFrame()

for i in range(10):
    current_fold = top_features.iloc[:, (i * 2):((i * 2) + 2)].reset_index(drop=True)
    current_fold.columns = ['feature', 'coef']
    current_fold['fold'] = i
    current_fold['rank'] = current_fold.index + 1
    top_features_long = pd.concat([top_features_long, current_fold])

# remove nans
top_features_long = top_features_long.loc[~top_features_long.feature.isna(), :]
top_features_long['feature'] = top_features_long['feature'].apply(lambda x: x.replace('.', '+'))

# look at top 10 features
top_10_features = top_features_long.loc[top_features_long['rank'] <= 10, :]
top_10_features.groupby('feature').size().sort_values(ascending=False).head(20)

# look at average rank
ranked_features = top_features_long.groupby('feature').agg({'rank': 'mean'}).sort_values('rank', ascending=True)

# compare with univariate
ranked_features_univ_sub = ranked_features_univariate.loc[ranked_features_univariate.feature_name_unique.isin(ranked_features.index), :]
ranked_features_univ_sub = ranked_features_univ_sub.loc[ranked_features_univ_sub.comparison == 'on_nivo', :]
