import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr

plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
sequence_dir = os.path.join(base_dir, 'sequencing_data')


# load datasets
genomics_features = pd.read_csv(os.path.join(data_dir, 'genomics/2022-10-04_molecular_summary_table.txt'), sep='\t')
genomics_features = genomics_features.rename(columns={'Individual.ID': 'Patient_ID', 'timepoint': 'Timepoint'})
image_feature_df = pd.read_csv(os.path.join(base_dir, 'analysis_files/timepoint_combined_features.csv'))
#image_features = pd.read_csv(os.path.join(data_dir, 'pca_data_df_grouped.csv'))

harmonized_metadata = pd.read_csv(os.path.join(base_dir, 'analysis_files/harmonized_metadata.csv'))
#image_features = image_features.merge(harmonized_metadata[['Patient_ID', 'Timepoint', 'Tissue_ID']].drop_duplicates(), on=['Tissue_ID'], how='left')

drop_idx = genomics_features['Experiment.System.ID'].isna()
genomics_features = genomics_features[~drop_idx]
genomics_features['subclonal_clonal_ratio'] = genomics_features['SUBCLONAL'] / genomics_features['CLONAL']

binarize_muts = ['PIK3CA_SNV', 'TP53_SNV', 'PTEN_SNV', 'RB1_SNV']

for mut in binarize_muts:
    genomics_features[mut + '_bin'] = (~genomics_features[mut].isna()).astype(int)

# simplify categorical features
#np.unique(genomics_features[cat_cols[4]].values, return_counts=True)
keep_dict = {'IC10_rna': [4, 10], 'IC10_rna_cna': [4., 10.], 'PAM50': ['Basal', 'Her2'], 'IntClust': ['ic10', 'ic4']}

for col in keep_dict.keys():
    genomics_features[col + '_simplified'] = genomics_features[col].apply(lambda x: x if x in keep_dict[col] else 'Other')

# convert categorical to categorical type
cat_cols = ['wgd', 'IC10_rna_simplified', 'IC10_rna_cna_simplified', 'PAM50_simplified', 'IntClust_simplified']

for col in cat_cols:
    genomics_features[col] = genomics_features[col].astype('category')

# identify features to use for correlations
genomics_include = ['purity', 'ploidy', 'fga', 'wgd', 'frac_loh', 'missense_count', 'IC10_rna_simplified', 'IC10_rna_cna_simplified', 'PAM50_simplified', 'IntClust_simplified', 'HBMR_norm', 'subclonal_clonal_ratio'] + [mut + '_bin' for mut in binarize_muts]

genomics_subset = genomics_features[['Patient_ID', 'Timepoint'] + genomics_include]

# preprocess data for primary timepoint
image_features_primary = image_features[image_features.Timepoint == 'baseline']

shared_patients = np.intersect1d(genomics_subset.Patient_ID, image_features_primary.Patient_ID)

genomics_feature_primary = genomics_subset[genomics_subset.Patient_ID.isin(shared_patients)]
image_features_primary = image_features_primary[image_features_primary.Patient_ID.isin(shared_patients)]

genomics_feature_primary = pd.get_dummies(genomics_feature_primary, columns=cat_cols, drop_first=True)

# # check for columns with only a single value
# single_value_cols = []
# for col in genomics_feature_primary.columns:
#     if len(genomics_feature_primary[col].unique()) == 1:
#         single_value_cols.append(col)
#
# genomics_feature_primary = genomics_feature_primary.drop(columns=single_value_cols)

image_features_wide = image_features_primary.pivot(index='Patient_ID', columns='feature_name', values='mean')
image_features_wide = image_features_wide.reset_index()
np.all(image_features_wide['Patient_ID'].values == genomics_feature_primary['Patient_ID'].values)

# calculate all pairwise correlations
image_feature_list, genomic_feature_list, corr_list, pval_list = [], [], [], []
min_samples = 10
for image_col in image_features_wide.columns[1:]:
    for genomic_col in genomics_feature_primary.columns[2:]:
        # drop NaNs
        combined_df = pd.DataFrame({image_col: image_features_wide[image_col].values, genomic_col: genomics_feature_primary[genomic_col].values})
        combined_df = combined_df.dropna()

        # append to lists
        image_feature_list.append(image_col)
        genomic_feature_list.append(genomic_col)
        if len(combined_df) > min_samples:
            corr, pval = spearmanr(combined_df[image_col].values, combined_df[genomic_col].values)
            corr_list.append(corr)
            pval_list.append(pval)
        else:
            corr_list.append(np.nan)
            pval_list.append(np.nan)

corr_df = pd.DataFrame({'image_feature': image_feature_list, 'genomic_feature': genomic_feature_list, 'cor': corr_list, 'pval': pval_list})
corr_df['log_pval'] = -np.log10(corr_df.pval)

# combined pval and correlation rank
corr_df['pval_rank'] = corr_df.log_pval.rank(ascending=False)
corr_df['cor_rank'] = corr_df.cor.abs().rank(ascending=False)
corr_df['combined_rank'] = (corr_df.pval_rank.values + corr_df.cor_rank.values) / 2

# fdr correction
from statsmodels.stats.multitest import multipletests
corr_df = corr_df.dropna(axis=0)
corr_df['fdr'] = multipletests(corr_df.pval.values, method='fdr_bh')[1]

sns.scatterplot(data=corr_df, x='cor', y='log_pval', hue='combined_rank', palette='viridis')
plt.xlabel('Spearman Correlation')
plt.ylabel('-log10(p-value)')
plt.title('Correlation between Genomic and Image Features')
plt.savefig(os.path.join(plot_dir, 'genomics_correlation_scatter.png'), dpi=300)
plt.close()

corr_df = corr_df.sort_values(by='combined_rank', ascending=True)

# plot top 10 correlations
top_corr = corr_df.head(20)

for i in range(top_corr.shape[0]):
    image_col = top_corr.iloc[i].image_feature
    genomic_col = top_corr.iloc[i].genomic_feature
    sns.scatterplot(x=image_features_wide[image_col].values, y=genomics_feature_primary[genomic_col].values)
    plt.xlabel(image_col)
    plt.ylabel(genomic_col)
    plt.title('Correlation: ' + str(round(top_corr.iloc[i].cor, 3)) + ', p-value: ' + str(round(top_corr.iloc[i].pval, 3)))
    plt.savefig(os.path.join(plot_dir, 'genomics_correlation_scatter_met' + str(i) + '.png'), dpi=300)
    plt.close()


# look at RNA correlations with imaging data
RNA_feature_df = pd.read_csv(os.path.join(sequence_dir, 'TONIC_immune_sig_score_table.tsv'), sep='\t')
RNA_feature_df = RNA_feature_df.rename(columns={'sample_identifier': 'rna_seq_sample_id'})
RNA_feature_df = RNA_feature_df.merge(harmonized_metadata[['rna_seq_sample_id', 'Clinical_benefit', 'Timepoint', 'Patient_ID']].drop_duplicates(),
                                    on='rna_seq_sample_id', how='left')

# check which columns are associated with response
plot_cols = [col for col in RNA_feature_df.columns if col not in ['rna_seq_sample_id', 'Clinical_benefit', 'Timepoint', 'TME_subtype']]

for col in plot_cols:
    fig, ax = plt.subplots(1, 3)
    for idx, timepoint in enumerate(['baseline', 'post_induction', 'on_nivo']):
        sns.stripplot(data=immune_scores.loc[immune_scores.Timepoint == timepoint, :],
                      x='Clinical_benefit', y=col, ax=ax[idx], color='black')
        sns.boxplot(data=immune_scores.loc[immune_scores.Timepoint == timepoint, :],
                    x='Clinical_benefit', y=col, ax=ax[idx], color='grey', showfliers=False)
        ax[idx].set_title(timepoint)
        ax[idx].set_xticklabels(ax[idx].get_xticklabels(), rotation=90)
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(plot_dir, 'RNA_features_{}.png'.format(col)))
    plt.close()

# look at correlatoins with image data
image_feature_df_wide = image_feature_df.pivot(index=['Patient_ID', 'Timepoint'], columns='feature_name_unique', values='raw_mean')
image_feature_df_wide = image_feature_df_wide.reset_index()
image_feature_df_wide.sort_values(by='Patient_ID', inplace=True)

RNA_feature_df.sort_values(by='Patient_ID', inplace=True)
RNA_feature_list = [col for col in RNA_feature_df.columns if col not in ['rna_seq_sample_id', 'Clinical_benefit', 'Timepoint', 'TME_subtype', 'Patient_ID', 'Tissue_ID']]
image_feature_list = [col for col in image_feature_df_wide.columns if col not in ['Patient_ID', 'Timepoint']]

# calculate all pairwise correlations
image_features, RNA_features, corr_list, pval_list = [], [], [], []
min_samples = 10

timepoint = 'baseline'
shared_patients = np.intersect1d(image_feature_df_wide.loc[image_feature_df_wide.Timepoint == timepoint, 'Patient_ID'].values,
                                 RNA_feature_df.loc[RNA_feature_df.Timepoint == timepoint, 'Patient_ID'].values)
image_features_shared = image_feature_df_wide.loc[np.logical_and(image_feature_df_wide.Patient_ID.isin(shared_patients),
                                                                 image_feature_df_wide.Timepoint == timepoint), :]
RNA_features_shared = RNA_feature_df.loc[np.logical_and(RNA_feature_df.Patient_ID.isin(shared_patients),
                                                        RNA_feature_df.Timepoint == timepoint), :]

for image_col in image_feature_list:
    for RNA_col in RNA_feature_list:
        # drop NaNs
        combined_df = pd.DataFrame({image_col: image_features_shared[image_col].values, RNA_col: RNA_features_shared[RNA_col].values})
        combined_df = combined_df.dropna()

        # append to lists
        image_features.append(image_col)
        RNA_features.append(RNA_col)
        if len(combined_df) > min_samples:
            corr, pval = spearmanr(combined_df[image_col].values, combined_df[RNA_col].values)
            corr_list.append(corr)
            pval_list.append(pval)
        else:
            corr_list.append(np.nan)
            pval_list.append(np.nan)

corr_df = pd.DataFrame({'image_feature': image_features, 'RNA_feature': RNA_features, 'cor': corr_list, 'pval': pval_list})
corr_df['log_pval'] = -np.log10(corr_df.pval)

# combined pval and correlation rank
corr_df['pval_rank'] = corr_df.log_pval.rank(ascending=False)
corr_df['cor_rank'] = corr_df.cor.abs().rank(ascending=False)
corr_df['combined_rank'] = (corr_df.pval_rank.values + corr_df.cor_rank.values) / 2

# fdr correction
from statsmodels.stats.multitest import multipletests
corr_df = corr_df.dropna(axis=0)
corr_df['fdr'] = multipletests(corr_df.pval.values, method='fdr_bh')[1]

sns.scatterplot(data=corr_df, x='cor', y='log_pval', hue='combined_rank', palette='viridis')
plt.xlabel('Spearman Correlation')
plt.ylabel('-log10(p-value)')
plt.title('Correlation between RNA and Image Features')
plt.savefig(os.path.join(plot_dir, 'RNA_correlation_scatter.png'), dpi=300)
plt.close()

corr_df = corr_df.sort_values(by='combined_rank', ascending=True)

# plot top 10 correlations
top_corr = corr_df.head(100)

for i in range(top_corr.shape[0]):
    image_col = top_corr.iloc[i].image_feature
    RNA_col = top_corr.iloc[i].RNA_feature
    sns.scatterplot(x=image_features_shared[image_col].values, y=RNA_features_shared[RNA_col].values)
    plt.xlabel(image_col)
    plt.ylabel(RNA_col)
    plt.title('Correlation: ' + str(round(top_corr.iloc[i].cor, 3)) + ', p-value: ' + str(round(top_corr.iloc[i].pval, 3)))
    plt.savefig(os.path.join(plot_dir, 'RNA_Image_correlation_' + str(i) + '.png'), dpi=300)
    plt.close()

