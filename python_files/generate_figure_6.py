import os

import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns


base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/'

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

# look at top features
