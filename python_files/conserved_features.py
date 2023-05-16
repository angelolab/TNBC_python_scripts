import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import skimage.io as io

from python_files import utils

local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'
channel_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/'


#
# Preprocess the data to be in paired format
#

feature_df = pd.read_csv(os.path.join(data_dir, 'fov_features_filtered.csv'))
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))

# fov_metadata = harmonized_metadata[harmonized_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'on_nivo', 'post_induction'])]
# fov_metadata = fov_metadata[fov_metadata.fov.isin(feature_df.fov.unique())]
# keep_fov_1 = []
# keep_fov_2 = []
# for name, items in fov_metadata[['fov', 'Tissue_ID']].groupby('Tissue_ID'):
#     if len(items) > 1:
#         keep_fov_1.append(items.iloc[0].fov)
#         keep_fov_2.append(items.iloc[1].fov)
#
# # save fovs
# fov_pairs = pd.DataFrame({'fov_1': keep_fov_1, 'fov_2': keep_fov_2})
# fov_pairs.to_csv(os.path.join(data_dir, 'fov_pairs.csv'), index=False)

# load the fov pairs
fov_pairs = pd.read_csv(os.path.join(data_dir, 'conserved_features/fov_pairs.csv'))
keep_fov_1 = fov_pairs.fov_1
keep_fov_2 = fov_pairs.fov_2

# annotate the feature df
keep_fovs = pd.concat([fov_pairs.fov_1, fov_pairs.fov_2], axis=0)
paired_df = feature_df[feature_df.fov.isin(keep_fovs)].copy()
paired_df['placeholder'] = paired_df.fov.isin(keep_fov_1).astype(int)
paired_df['placeholder'] = paired_df['placeholder'].replace({0: 'fov1', 1: 'fov2'})

# pivot the df to wide format with matching fovs
index_cols = paired_df.columns
index_cols = index_cols.drop(['fov', 'raw_value', 'normalized_value', 'placeholder'])
paired_df = paired_df.pivot(index=index_cols, columns='placeholder', values=['raw_value', 'normalized_value'])
paired_df = paired_df.reset_index()

# clean up columns
paired_df.columns = ['_'.join(col).strip() for col in paired_df.columns.values]
paired_df.columns = [x[:-1] if x.endswith('_') else x for x in paired_df.columns]

paired_df.to_csv(os.path.join(data_dir, 'conserved_features/paired_df_filtered.csv'), index=False)

#
# Using preprocessed features, find conserved features
#

paired_df = pd.read_csv(os.path.join(data_dir, 'conserved_features/paired_df_filtered.csv'))

ranked_features = utils.find_conserved_features(paired_df=paired_df, sample_name_1='raw_value_fov1',
                                                sample_name_2='raw_value_fov2', min_samples=10)
ranked_features = ranked_features.merge(paired_df[['feature_name', 'feature_name_unique',
                                                   'compartment', 'cell_pop', 'cell_pop_level',
                                                   'feature_type']].drop_duplicates(), on='feature_name_unique', how='left')

# generate plot for best ranked features
min_score = 0.85
plot_features = ranked_features.loc[ranked_features.consistency_score >= min_score, :]

# sort by combined rank
plot_features.sort_values(by='consistency_score', inplace=True, ascending=False)

for i in range(len(plot_features))[:20]:
    feature_name = plot_features.iloc[i].feature_name_unique
    feature_score = plot_features.iloc[i].consistency_score
    values = paired_df[(paired_df.feature_name_unique == feature_name)].copy()
    values.dropna(inplace=True)

    # add leading 0 for plot
    str_pos = str(i)
    if i < 10:
        str_pos = '0' + str_pos

    # plot
    sns.scatterplot(data=values, x='normalized_value_fov1', y='normalized_value_fov2')
    correlation, p_val = spearmanr(values.normalized_value_fov1, values.normalized_value_fov2)
    plt.title(f'{feature_name} (score={feature_score:.2f}, r={correlation:.2f}, p={p_val:.2f})')
    plt.savefig(os.path.join(plot_dir, f'conserved_features/{str_pos}_{feature_name}.png'))
    plt.close()

    # compare raw and log-normalized values
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #
    # feature_name = plot_features.iloc[i].feature_name
    # values = paired_df[(paired_df.feature_name == feature_name)].copy()
    # values.dropna(inplace=True)
    #
    # sns.scatterplot(data=values, x='normalized_value_fov1', y='normalized_value_fov2', ax=ax[0])
    # #sns.scatterplot(data=values, x='raw_value_fov1', y='raw_value_fov2', ax=ax[0])
    # #sns.scatterplot(data=values, x='value_fov1', y='value_fov2', ax=ax[0])
    # correlation, p_val = spearmanr(values.normalized_value_fov1, values.normalized_value_fov2)
    # ax[0].set_xlabel('untransformed')
    #
    # logged_values = values.copy()
    # min_val = min(values.normalized_value_fov1.min(), values.normalized_value_fov2.min())
    # if min_val > 0:
    #     increment = 0
    # elif min_val == 0:
    #     increment = 0.01
    # else:
    #     increment = min_val * -1.01
    # logged_values.normalized_value_fov1 = np.log10(values.normalized_value_fov1.values + increment)
    # logged_values.normalized_value_fov2 = np.log10(values.normalized_value_fov2.values + increment)
    # sns.scatterplot(data=logged_values, x='normalized_value_fov1', y='normalized_value_fov2', ax=ax[1])
    # ax[1].set_xlabel('log10 transformed')
    #
    # # set title for whole figure
    # fig.suptitle(feature_name + ' correlation: {:.2f} rank: {}'.format(correlation, plot_features.iloc[i].combined_rank))
    #
    # # convert int rank to string with two leading zeros
    # int_rank = int(plot_features.iloc[i].combined_rank)
    # str_rank = str(int_rank)
    # if int_rank < 10:
    #     str_rank = '0' + str_rank
    #
    # plt.savefig(os.path.join(plot_dir, 'Correlation_{}_{}.png'.format(str_rank, feature_name)))
    # plt.close()


# plot barplot of consistency scores
fig, ax = plt.subplots(figsize=(10, 5))
g = sns.barplot(data=plot_features.iloc[:20, :], x='consistency_score', y='feature_name_unique', ax=ax,
            color='lightblue')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'conserved_features/consistency_scores.png'))
plt.close()

# plot a specified row
row = 25
row = np.where(ranked_features.feature_name == 'TCF1+__Cancer__all')[0][0]
name = ranked_features.loc[row, 'feature_name']
correlation = ranked_features.loc[row, 'cor']
p_val = ranked_features.loc[row, 'p_val']
values = paired_df[(paired_df.feature_name == name)]
values.dropna(inplace=True)
fig, ax = plt.subplots()
sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax)
ax.text(0.05, 0.95, f'cor: {correlation:.2f}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top')
ax.text(0.65, 0.95, f'p: {p_val:.2e}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top')
plt.title(f'{name}')
plt.savefig(os.path.join(plot_dir, 'conserved_features_{}.png'.format(name)))
plt.close()


# determine which features to summarize by
summary_features = ['feature_type', 'cell_pop_level', 'cell_pop']

for feature in summary_features:
    # # get counts of each feature type for conserved vs non-conserved features
    # grouped = ranked_features[[feature, 'conserved']].groupby(feature).mean().reset_index()
    #
    # # plot as a barplot
    # g = sns.barplot(data=grouped, x=feature, y='conserved', color='grey')
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    # g.set_xlabel(feature)
    # g.set_ylabel('Proportion conserved')
    # g.set_title('Proportion conserved by {}'.format(feature))
    # plt.tight_layout()
    # plt.savefig(os.path.join(plot_dir, 'proportion_conserved_by_{}.png'.format(feature)))
    # plt.close()

    # get rank value for each feature type
    grouped = ranked_features[[feature, 'consistency_score']].groupby(feature).mean().reset_index()
    grouped.sort_values(by='consistency_score', inplace=True, ascending=False)

    # plot as a barplot
    g = sns.barplot(data=grouped, x=feature, y='consistency_score', color='grey')
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_xlabel(feature)
    g.set_ylabel('Average consistency score')
    g.set_title('Average consistency by {}'.format(feature))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'conserved_features_average_score_by_{}.png'.format(feature)))
    plt.close()


# compare functional markers
func_features = ranked_features[ranked_features.feature_type == 'functional_marker']
func_features['func_name'] = func_features.feature_name_unique.apply(lambda x: x.split('+')[0])

# remove features with __
func_features = func_features.loc[~func_features.func_name.str.contains('__'), :]

grouped = func_features[['func_name', 'consistency_score']].groupby('func_name').mean().reset_index()
grouped.sort_values(by='consistency_score', inplace=True, ascending=False)

# plot as a barplot
g = sns.barplot(data=grouped, x='func_name', y='consistency_score', color='grey')
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Functional marker')
g.set_ylabel('Average consistency score')
g.set_title('Average consistency by functional marker')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'conserved_features_average_score_by_functional_marker.png'))
plt.close()

# create overlays of specific features
feature_name = 'H3K9ac_H3K27me3_ratio+__all__all'
paired_df.loc[(paired_df.feature_name_unique == feature_name) & (paired_df.normalized_value_fov1 < -2) & (paired_df.normalized_value_fov2 < -2), ]
id = 'T16-00774'

fov_1, fov_2 = harmonized_metadata.loc[(harmonized_metadata.Tissue_ID == id) & harmonized_metadata.fov.isin(keep_fovs), 'fov'].values
fov_1, fov_2 = 'TONIC_TMA3_R4C1', 'TONIC_TMA3_R3C6'

# load images
fov_1_H3K9ac = io.imread(os.path.join(channel_dir, fov_1, 'H3K9ac.tiff'))
fov_1_H3K9ac /= 0.25
fov_1_H3K27me3 = io.imread(os.path.join(channel_dir, fov_1, 'H3K27me3.tiff'))
fov_1_H3K27me3 /= 0.4

fov_2_H3K9ac = io.imread(os.path.join(channel_dir, fov_2, 'H3K9ac.tiff'))
fov_2_H3K9ac /= 0.25
fov_2_H3K27me3 = io.imread(os.path.join(channel_dir, fov_2, 'H3K27me3.tiff'))
fov_2_H3K27me3 /= 0.4

# save images
io.imsave(os.path.join(plot_dir, 'feature_overlays/histone_low', 'H3K9ac_{}.tiff'.format(fov_1)), fov_1_H3K9ac)
io.imsave(os.path.join(plot_dir, 'feature_overlays/histone_low', 'H3K27me3_{}.tiff'.format(fov_1)), fov_1_H3K27me3)
io.imsave(os.path.join(plot_dir, 'feature_overlays/histone_low','H3K9ac_{}.tiff'.format(fov_2)), fov_2_H3K9ac)
io.imsave(os.path.join(plot_dir, 'feature_overlays/histone_low', 'H3K27me3_{}.tiff'.format(fov_2)), fov_2_H3K27me3)


paired_df.loc[(paired_df.feature_name_unique == feature_name) & (paired_df.normalized_value_fov1 > 2) & (paired_df.normalized_value_fov2 > 2), ]
id = 'T18-60540'

fov_1, fov_2 = harmonized_metadata.loc[(harmonized_metadata.Tissue_ID == id) & harmonized_metadata.fov.isin(keep_fovs), 'fov'].values
fov_1, fov_2 = 'TONIC_TMA23_R7C1', 'TONIC_TMA23_R7C2'

fov_1_H3K9ac = io.imread(os.path.join(channel_dir, fov_1, 'H3K9ac.tiff'))
fov_1_H3K9ac /= 0.25
fov_1_H3K27me3 = io.imread(os.path.join(channel_dir, fov_1, 'H3K27me3.tiff'))
fov_1_H3K27me3 /= 0.4

fov_2_H3K9ac = io.imread(os.path.join(channel_dir, fov_2, 'H3K9ac.tiff'))
fov_2_H3K9ac /= 0.25
fov_2_H3K27me3 = io.imread(os.path.join(channel_dir, fov_2, 'H3K27me3.tiff'))
fov_2_H3K27me3 /= 0.4

# save images
io.imsave(os.path.join(plot_dir, 'feature_overlays/histone_high', 'H3K9ac_{}.tiff'.format(fov_1)), fov_1_H3K9ac)
io.imsave(os.path.join(plot_dir, 'feature_overlays/histone_high', 'H3K27me3_{}.tiff'.format(fov_1)), fov_1_H3K27me3)
io.imsave(os.path.join(plot_dir, 'feature_overlays/histone_high','H3K9ac_{}.tiff'.format(fov_2)), fov_2_H3K9ac)
io.imsave(os.path.join(plot_dir, 'feature_overlays/histone_high', 'H3K27me3_{}.tiff'.format(fov_2)), fov_2_H3K27me3)

# ratio overlays
feature_name = 'Mono_Mac__Stroma__ratio__all'
paired_df.loc[(paired_df.feature_name_unique == feature_name) & (paired_df.normalized_value_fov1 < -2) & (paired_df.normalized_value_fov2 < -2), ]
id = 'T16-02009'

fov_1, fov_2 = harmonized_metadata.loc[(harmonized_metadata.Tissue_ID == id) & harmonized_metadata.fov.isin(keep_fovs), 'fov'].values
fov_1, fov_2 = 'TONIC_TMA3_R3C4', 'TONIC_TMA3_R3C5'

image_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay'

for fov in [fov_1, fov_2]:
    shutil.copy(os.path.join(image_dir, '{}.png'.format(fov)),
                os.path.join(plot_dir, 'feature_overlays/mac_low', '{}.png'.format(fov)))


paired_df.loc[(paired_df.feature_name_unique == feature_name) & (paired_df.normalized_value_fov1 > 2) & (paired_df.normalized_value_fov2 > 2), ]
id = 'T17-05618'

fov_1, fov_2 = harmonized_metadata.loc[(harmonized_metadata.Tissue_ID == id) & harmonized_metadata.fov.isin(keep_fovs), 'fov'].values
fov_1, fov_2 = 'TONIC_TMA13_R3C1', 'TONIC_TMA13_R3C2'


for fov in [fov_1, fov_2]:
    shutil.copy(os.path.join(image_dir, '{}.png'.format(fov)),
                os.path.join(plot_dir, 'feature_overlays/mac_high', '{}.png'.format(fov)))


# compare ranking based on subsets of the data
timepoints = [{'timepoints': ['primary_untreated', 'primary', 'biopsy'],
               'timepoint_name': 'primary',
               'tissue': ['Other']},
              {'timepoints': ['baseline', 'post_induction', 'on_nivo', 'metastasis'],
               'timepoint_name': 'met_ln',
               'tissue': ['Lymphnode']},
              {'timepoints': ['baseline', 'post_induction', 'on_nivo', 'metastasis'],
               'timepoint_name': 'met_other',
                'tissue': ['Other', 'Liver', 'Unknown', 'Skin', 'Bone', 'Gut', 'Bladder', 'Oesafageal']},
              {'timepoints': ['all'],
               'timepoint_name': 'all'}]
all_ranked_features = []
for timepoint_df in timepoints:
    current_timepoints = timepoint_df['timepoints']
    if current_timepoints == ['all']:
        valid_paired_df = paired_df.copy()
    else:
        valid_tissue_ids = harmonized_metadata[harmonized_metadata.Timepoint.isin(current_timepoints)].Tissue_ID.unique()
        valid_tissue_ids2 = harmonized_metadata[harmonized_metadata.Localization.isin(timepoint_df['tissue'])].Tissue_ID.unique()
        valid_tissue_ids = np.intersect1d(valid_tissue_ids, valid_tissue_ids2)
        valid_paired_df = paired_df[paired_df.Tissue_ID.isin(valid_tissue_ids)].copy()
    valid_ranked_features = utils.find_conserved_features(paired_df=valid_paired_df, sample_name_1='raw_value_fov1',
                                              sample_name_2='raw_value_fov2', min_samples=10)

    valid_ranked_features['timepoint'] = timepoint_df['timepoint_name']
    all_ranked_features.append(valid_ranked_features)

ranked_features = pd.concat(all_ranked_features, axis=0)

# combine with feature metadata
ranked_features = ranked_features.merge(paired_df[['feature_name', 'feature_name_unique',
                                                   'compartment', 'cell_pop', 'cell_pop_level',
                                                   'feature_type']].drop_duplicates(), on='feature_name', how='left')

# compare ranking among timepoints
wide_ranked_features = ranked_features.pivot(index='feature_name', columns='timepoint', values='combined_rank')
sns.heatmap(wide_ranked_features)
plt.savefig(os.path.join(plot_dir, 'conserved_features_heatmap.png'))
plt.close()

sns.scatterplot(data=wide_ranked_features, x='primary', y='all')
plt.savefig(os.path.join(plot_dir, 'conserved_features_primary_vs_all.png'))
plt.close()
sns.scatterplot(data=wide_ranked_features, x='met_ln', y='all')
plt.savefig(os.path.join(plot_dir, 'conserved_features_met_ln_vs_all.png'))
plt.close()
sns.scatterplot(data=wide_ranked_features, x='met_other', y='all')
plt.savefig(os.path.join(plot_dir, 'conserved_features_met_other_vs_all.png'))
plt.close()




ranked_features['conserved'] = ranked_features.combined_rank <= 100
ranked_features.to_csv(os.path.join(data_dir, 'conserved_features/ranked_features_no_compartment.csv'), index=False)

ranked_features = pd.read_csv(os.path.join(data_dir, 'conserved_features/ranked_features_no_compartment.csv'))

sns.scatterplot(data=ranked_features, x='cor', y='log_pval', hue='combined_rank', palette='viridis')
plt.savefig(os.path.join(plot_dir, 'conserved_features_volcano.png'))
plt.close()