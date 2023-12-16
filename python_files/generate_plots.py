import os
import shutil

import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn2
from ark.utils.plot_utils import cohort_cluster_plot, color_segmentation_by_stat
import ark.settings as settings
import skimage.io as io


base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
metadata_dir = os.path.join(base_dir, 'intermediate_files/metadata')
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/'
harmonized_metadata = pd.read_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'))
seg_dir = os.path.join(base_dir, 'segmentation_data/deepcell_output')
image_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/'

study_fovs = harmonized_metadata.loc[harmonized_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), 'fov'].values






# Figure 7
cv_scores = pd.read_csv(os.path.join(base_dir, 'multivariate_lasso', 'results_1112_cv.csv'))
cv_scores['fold'] = len(cv_scores)

cv_scores_long = pd.melt(cv_scores, id_vars=['fold'], value_vars=cv_scores.columns)


fig, ax = plt.subplots(1, 1, figsize=(3, 4))
order = ['primary', 'post_induction', 'baseline', 'on_nivo']
         #'primary_AND_baseline', 'primary_AND_post_induction', 'baseline_AND_post_induction']
sns.stripplot(data=cv_scores_long, x='variable', y='value', order=order,
                color='black', ax=ax)
sns.boxplot(data=cv_scores_long, x='variable', y='value', order=order,
                color='grey', ax=ax, showfliers=False)

ax.set_title('AUC')
ax.set_ylim([0, 1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure6_AUC.pdf'))
plt.close()



# testing args
# width = 0.8, linewidth=None, gap=0,
fig, ax = plt.subplots(1, 1, figsize=(3, 4))
order = ['primary', 'post_induction', 'baseline', 'on_nivo']
sns.boxplot(data=cv_scores_long, x='variable', y='value', order=order,
                color='grey', ax=ax, showfliers=False, width=0.3)
