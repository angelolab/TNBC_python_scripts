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


## COMPARTMENT COMPARISONS
# all feature plot
timepoint_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'timepoint_features.csv'))
timepoint_df['long_name'] = timepoint_df['Tissue_ID'] + '//' + timepoint_df['feature_name']

t = timepoint_df.pivot(index='long_name', columns='compartment')['raw_mean']
t = t[t.isnull().sum(axis=1) < 4]
t = t[~t['all'].isna()]

# 2^x for previous log2 scores
t[np.logical_or(t.index.str.contains('__ratio'), t.index.str.contains('H3K9ac_H3K27me3_ratio+'),
                t.index.str.contains('CD45RO_CD45RB_ratio+'))] =\
    2 ** t[np.logical_or(t.index.str.contains('__ratio'), t.index.str.contains('H3K9ac_H3K27me3_ratio+'),
                         t.index.str.contains('CD45RO_CD45RB_ratio+'))]

comp_t = t.divide(t['all'], axis=0)
comp_t.index = [idx.split('//')[1] for idx in comp_t.index]
comp_t['feature_name'] = comp_t.index

df = comp_t.groupby(by=['feature_name']).mean()
df = np.log2(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()
df = df[['all', 'cancer_core', 'cancer_border', 'stroma_border', 'stroma_core']]

sns.set(font_scale=1)
plt.figure(figsize=(8, 30))
heatmap = sns.clustermap(
    df, cmap="vlag", vmin=-2, vmax=2, col_cluster=False, cbar_pos=(1.03, 0.07, 0.015, 0.2),
)
heatmap.tick_params(labelsize=8)
plt.setp(heatmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
x0, _y0, _w, _h = heatmap.cbar_pos
for spine in heatmap.ax_cbar.spines:
    heatmap.ax_cbar.spines[spine].set_color('k')
    heatmap.ax_cbar.spines[spine].set_linewidth(1)

ax = heatmap.ax_heatmap
ax.axvline(x=0, color='k', linewidth=0.8)
ax.axvline(x=1, color='k', linewidth=0.8)
ax.axvline(x=2, color='k', linewidth=0.8)
ax.axvline(x=3, color='k', linewidth=0.8)
ax.axvline(x=4, color='k', linewidth=0.8)
ax.axvline(x=5, color='k', linewidth=0.8)
ax.axhline(y=0, color='k', linewidth=1)
ax.axhline(y=len(df), color='k', linewidth=1.5)
ax.set_ylabel("Feature")
ax.set_xlabel("Compartment")

features_of_interest = [361, 107, 92, 110, 90, 258, 373, 311, 236, 266, 385, 83, 327, 61, 132, 150]
feature_names = [df.index[i] for i in features_of_interest]
reorder = heatmap.dendrogram_row.reordered_ind
new_positions = [reorder.index(i) for i in features_of_interest]
plt.setp(heatmap.ax_heatmap.yaxis.set_ticks(new_positions))
plt.setp(heatmap.ax_heatmap.yaxis.set_ticklabels(feature_names))
plt.tight_layout()

plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'compartment_comparisons/compartment_comparisons.pdf'), dpi=300, bbox_inches="tight")

# high/low standard deviation feature plot
df_copy = df.copy()
df_copy['row_std'] = df_copy.std(axis=1)
df_copy = df_copy.sort_values(by='row_std')

low_std = df_copy[:90]
high_std = df_copy[-90:]
all_std_data = pd.concat([high_std, low_std]).sort_values(by='row_std', ascending=False)
all_std_data = all_std_data[df.columns]

sns.set(font_scale=1)
plt.figure(figsize=(4, 17))
heatmap = sns.heatmap(
    all_std_data, cmap="vlag", vmin=-2, vmax=2, yticklabels=True, cbar_kws={'shrink': 0.1}
)
heatmap.tick_params(labelsize=6)
heatmap.hlines([len(all_std_data)/2], *ax.get_xlim(), ls='--', color='black', linewidth=0.5,)
ax.set_ylabel("Feature")
ax.set_xlabel("Compartment")
plt.tight_layout()

plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'compartment_comparisons/compartments-high_low_std.pdf'), dpi=300, bbox_inches="tight")

# histogram of standard deviations
plt.style.use("default")
g = sns.histplot(df_copy.row_std)
g.set(xlabel='Standard Deviation', ylabel='Feature Counts')
sns.despine()
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'compartment_comparisons/compartments_standard_deviation_hist.pdf'), dpi=300, bbox_inches="tight")
