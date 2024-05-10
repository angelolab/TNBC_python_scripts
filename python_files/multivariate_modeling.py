import os

import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind


base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
multivariate_dir = os.path.join(base_dir, 'multivariate_lasso')

ranked_features_univariate = pd.read_csv(os.path.join(base_dir, 'analysis_files/feature_ranking.csv'))
ranked_features_univariate_genomic = pd.read_csv(os.path.join(base_dir, 'sequencing_data/genomics_outcome_ranking.csv'))


# organize scores from cross validation
cv_scores = pd.read_csv(os.path.join(multivariate_dir, 'results_protein_rna_0205.csv'))
cv_scores['fold'] = len(cv_scores)

cv_scores_long = pd.melt(cv_scores, id_vars=['fold'], value_vars=cv_scores.columns)
cv_scores_long['assay'] = cv_scores_long['variable'].apply(lambda x: 'rna' if 'rna' in x else 'mibi')
cv_scores_long['variable'] = cv_scores_long['variable'].apply(lambda x: x.replace('_rna', '').replace('_mibi', ''))

cv_scores_long.to_csv(os.path.join(base_dir, 'multivariate_lasso', 'formatted_cv_scores.csv'), index=False)
