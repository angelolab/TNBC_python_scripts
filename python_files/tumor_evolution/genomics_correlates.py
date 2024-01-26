import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr

plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
sequence_dir = os.path.join(base_dir, 'sequencing_data')

harmonized_metadata = pd.read_csv(os.path.join(base_dir, 'analysis_files/harmonized_metadata.csv'))

#
# clean up genomics features
#

# load data
genomics_feature_df = pd.read_csv(os.path.join(sequence_dir, 'TONIC_WES_meta_table.tsv'), sep='\t')
genomics_feature_df = genomics_feature_df.rename(columns={'Individual.ID': 'Patient_ID', 'timepoint': 'Timepoint'})

# drop columns which aren't needed for analysis
drop_cols = ['Localization', 'induction_arm', 'Experiment.System.ID', 'B2']
genomics_feature_df = genomics_feature_df.drop(columns=drop_cols)

# check which columns have non-numeric values
non_numeric_cols = []
for col in genomics_feature_df.columns:
    if genomics_feature_df[col].dtype == 'object':
        non_numeric_cols.append(col)

# recode mutation features to yes/no
binarize_muts = [col for col in genomics_feature_df.columns if '_SNV' in col]
for mut in binarize_muts:
    genomics_feature_df[mut] = (~genomics_feature_df[mut].isna()).astype(int)

# change binary features to 0/1
genomics_feature_df['tail'] = (genomics_feature_df['tail'] == True).astype(int)
genomics_feature_df['wgd'] = (genomics_feature_df['wgd'] == True).astype(int)
genomics_feature_df['hla_loh'] = (genomics_feature_df['hla_loh'] == 'yes').astype(int)

# recode copy number features
amp_cols = [col for col in genomics_feature_df.columns if '_AMP' in col]
del_cols = [col for col in genomics_feature_df.columns if '_DEL' in col]
loh_cols = [col for col in genomics_feature_df.columns if '_LOH' in col]
cn_cols = amp_cols + del_cols + loh_cols

for col in cn_cols:
    genomics_feature_df[col] = (genomics_feature_df[col].values == 'yes').astype(int)

# simplify categorical features
np.unique(genomics_feature_df['IC10_rna'].values, return_counts=True)
keep_dict = {'IC10_rna': [4, 9, 10], 'IC10_rna_cna': [4., 9., 10.], 'PAM50': ['Basal', 'Her2'],
             'IntClust': ['ic10', 'ic4', 'ic9'], 'A1': ['A*01:01', 'A*02:01', 'A*03:01'],
             'A2': ['A*02:01', 'A*03:01', 'A*24:02'], 'B1': ['B*07:02', 'B*08:01', 'B*40:01'],
             'C1': ['C*03:04', 'C*04:01', 'C*07:01', 'C*07:02'], 'C2': ['C*07:01', 'C*07:02']}

for col in keep_dict.keys():
    genomics_feature_df[col] = genomics_feature_df[col].apply(lambda x: x if x in keep_dict[col] else 'Other')
    genomics_feature_df[col] = genomics_feature_df[col].astype('category')

# add computed features
genomics_feature_df['subclonal_clonal_ratio'] = genomics_feature_df['SUBCLONAL'] / genomics_feature_df['CLONAL']

# change categorical features to dummy variables
genomics_feature_df = pd.get_dummies(genomics_feature_df, columns=keep_dict.keys(), drop_first=True)

# save processed genomics features
genomics_feature_df.to_csv(os.path.join(sequence_dir, 'TONIC_WES_meta_table_processed.csv'), index=False)

# transform to long format

genomics_feature_df = pd.read_csv(os.path.join(sequence_dir, 'TONIC_WES_meta_table_processed.csv'))

genomics_feature_df = pd.merge(genomics_feature_df, harmonized_metadata[['Patient_ID', 'Timepoint', 'Tissue_ID']].drop_duplicates(),
                               on=['Patient_ID', 'Timepoint'], how='left')

genomics_feature_df_long = pd.melt(genomics_feature_df, id_vars=['Patient_ID', 'Timepoint', 'Tissue_ID', 'Clinical_benefit'],
                                   var_name='feature_name', value_name='feature_value')


# look at correlatoins with image data
image_feature_df_wide = image_feature_df.pivot(index=['Patient_ID', 'Timepoint'], columns='feature_name_unique', values='raw_mean')
image_feature_df_wide = image_feature_df_wide.reset_index()
image_feature_df_wide.sort_values(by='Patient_ID', inplace=True)

genomics_feature_df.sort_values(by='Patient_ID', inplace=True)
DNA_feature_list = [col for col in genomics_feature_df.columns if col not in ['Clinical_benefit', 'Timepoint', 'Patient_ID']]
image_feature_list = [col for col in image_feature_df_wide.columns if col not in ['Patient_ID', 'Timepoint']]

# calculate all pairwise correlations


timepoint = 'baseline'
shared_patients = np.intersect1d(image_feature_df_wide.loc[image_feature_df_wide.Timepoint == timepoint, 'Patient_ID'].values,
                                 genomics_feature_df.loc[genomics_feature_df.Timepoint == timepoint, 'Patient_ID'].values)
image_features_shared = image_feature_df_wide.loc[np.logical_and(image_feature_df_wide.Patient_ID.isin(shared_patients),
                                                                 image_feature_df_wide.Timepoint == timepoint), :]
DNA_features_shared = genomics_feature_df.loc[np.logical_and(genomics_feature_df.Patient_ID.isin(shared_patients),
                                                        genomics_feature_df.Timepoint == timepoint), :]

# check for columns with only a single value
single_value_cols = []
for col in DNA_feature_list:
    if len(DNA_features_shared[col].unique()) == 1:
        single_value_cols.append(col)

DNA_features_shared = DNA_features_shared.drop(columns=single_value_cols)
DNA_feature_list = [col for col in DNA_feature_list if col not in single_value_cols]
# calculate all pairwise correlations
image_features, DNA_features, corr_list, pval_list = [], [], [], []
min_samples = 10

for image_col in image_feature_list:
    for DNA_col in DNA_feature_list:
        # drop NaNs
        combined_df = pd.DataFrame({image_col: image_features_shared[image_col].values, DNA_col: DNA_features_shared[DNA_col].values})
        combined_df = combined_df.dropna()

        # append to lists
        image_features.append(image_col)
        DNA_features.append(DNA_col)
        if len(combined_df) > min_samples:
            corr, pval = spearmanr(combined_df[image_col].values, combined_df[DNA_col].values)
            corr_list.append(corr)
            pval_list.append(pval)
        else:
            corr_list.append(np.nan)
            pval_list.append(np.nan)

corr_df = pd.DataFrame({'image_feature': image_features, 'DNA_feature': DNA_features, 'cor': corr_list, 'pval': pval_list})
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
plt.title('Correlation between DNA and Image Features')
plt.savefig(os.path.join(plot_dir, 'DNA_correlation_scatter.png'), dpi=300)
plt.close()

corr_df = corr_df.sort_values(by='combined_rank', ascending=True)

# plot top 100 correlations
top_corr = corr_df.head(100)

for i in range(top_corr.shape[0]):
    image_col = top_corr.iloc[i].image_feature
    DNA_col = top_corr.iloc[i].DNA_feature
    sns.scatterplot(x=image_features_shared[image_col].values, y=DNA_features_shared[DNA_col].values)
    plt.xlabel(image_col)
    plt.ylabel(DNA_col)
    plt.title('Correlation: ' + str(round(top_corr.iloc[i].cor, 3)) + ', p-value: ' + str(round(top_corr.iloc[i].pval, 3)))
    plt.savefig(os.path.join(plot_dir, 'DNA_Image_correlation_' + str(i) + '.png'), dpi=300)
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

