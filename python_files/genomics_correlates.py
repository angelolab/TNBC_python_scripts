import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr

local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

# load datasets
genomics_features = pd.read_csv(os.path.join(data_dir, 'genomics/2022-10-04_molecular_summary_table.txt'), sep='\t')
genomics_features = genomics_features.rename(columns={'Individual.ID': 'Patient_ID', 'timepoint': 'Timepoint'})
image_features = pd.read_csv(os.path.join(data_dir, 'timepoint_features_no_compartment.csv'))
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
image_features = image_features.merge(harmonized_metadata[['Patient_ID', 'Timepoint', 'Tissue_ID']].drop_duplicates(), on=['Tissue_ID'], how='left')

genomics_features['subclonal_clonal_ratio'] = genomics_features['SUBCLONAL'] / genomics_features['CLONAL']

binarize_muts = ['PIK3CA_SNV', 'TP53_SNV', 'PTEN_SNV', 'RB1_SNV']

for mut in binarize_muts:
    genomics_features[mut + '_bin'] = (~genomics_features[mut].isna()).astype(int)

# identify features to use for correlations
genomics_include = ['purity', 'ploidy', 'fga', 'wgd', 'frac_loh', 'missense_count', 'IC10_rna', 'IC10_rna_cna', 'PAM50', 'IntClust', 'HBMR_norm', 'subclonal_clonal_ratio'] + [mut + '_bin' for mut in binarize_muts]

# convert categorical to categorical type
cat_cols = ['wgd', 'IC10_rna', 'IC10_rna_cna', 'PAM50', 'IntClust']

for col in cat_cols:
    genomics_features[col] = genomics_features[col].astype('category')

genomics_subset = genomics_features[['Patient_ID', 'Timepoint'] + genomics_include]

drop_idx = genomics_subset.purity.isna()
genomics_subset = genomics_subset[~drop_idx]

# preprocess data for primary timepoint
image_features_primary = image_features[image_features.Timepoint == 'baseline']

shared_patients = np.intersect1d(genomics_subset.Patient_ID, image_features_primary.Patient_ID)

genomics_feature_primary = genomics_subset[genomics_subset.Patient_ID.isin(shared_patients)]
image_features_primary = image_features_primary[image_features_primary.Patient_ID.isin(shared_patients)]

genomics_feature_primary = pd.get_dummies(genomics_feature_primary, columns=cat_cols, drop_first=True)

# check for columns with only a single value
single_value_cols = []
for col in genomics_feature_primary.columns:
    if len(genomics_feature_primary[col].unique()) == 1:
        single_value_cols.append(col)

genomics_feature_primary = genomics_feature_primary.drop(columns=single_value_cols)

image_features_wide = image_features_primary.pivot(index='Patient_ID', columns='feature_name_unique', values='raw_mean')
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

