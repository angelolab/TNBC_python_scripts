import os
import re

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


# process RNA-seq data
RNA_feature_df = pd.read_csv(os.path.join(sequence_dir, 'TONIC_immune_sig_score_table.tsv'), sep='\t')
RNA_feature_df_genes = pd.read_csv(os.path.join(sequence_dir, 'TONIC_immune_sig_gene_TPM_table.tsv'), sep='\t')
RNA_feature_df = pd.merge(RNA_feature_df, RNA_feature_df_genes, on='sample_identifier', how='left')
RNA_feature_df = RNA_feature_df.rename(columns={'sample_identifier': 'rna_seq_sample_id'})
RNA_feature_df = RNA_feature_df.merge(harmonized_metadata[['rna_seq_sample_id', 'Clinical_benefit', 'Timepoint', 'Patient_ID', 'Tissue_ID']].drop_duplicates(),
                                    on='rna_seq_sample_id', how='left')
RNA_feature_df = RNA_feature_df.drop(columns=['rna_seq_sample_id'])
RNA_feature_df['TME_subtype'] = RNA_feature_df['TME_subtype'].astype('category')
RNA_feature_df = pd.get_dummies(RNA_feature_df, columns=['TME_subtype'], drop_first=True)
RNA_feature_df.to_csv(os.path.join(sequence_dir, 'TONIC_immune_sig_score_and_genes_processed.csv'), index=False)

# transform to long format

genomics_feature_df = pd.read_csv(os.path.join(sequence_dir, 'TONIC_WES_meta_table_processed.csv'))

genomics_feature_df = pd.merge(genomics_feature_df, harmonized_metadata[['Patient_ID', 'Timepoint', 'Tissue_ID']].drop_duplicates(),
                               on=['Patient_ID', 'Timepoint'], how='left')

genomics_feature_df_long = pd.melt(genomics_feature_df, id_vars=['Patient_ID', 'Timepoint', 'Tissue_ID', 'Clinical_benefit'],
                                   var_name='feature_name', value_name='feature_value')

# label summary metrics
summary_metrics = ['purity', 'ploidy', 'fga', 'wgd', 'frac_loh']
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(summary_metrics), 'data_type'] = 'DNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(summary_metrics), 'feature_type'] = 'cn_summary'

# label gene cns
gene_cns = [col for col in genomics_feature_df.columns if col.endswith('_cn')]
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(gene_cns), 'data_type'] = 'DNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(gene_cns), 'feature_type'] = 'gene_cn'

# label gene rna
gene_rna = [col for col in genomics_feature_df.columns if col.endswith('_rna')]
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(gene_rna), 'data_type'] = 'RNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(gene_rna), 'feature_type'] = 'gene_rna'

# label mutation summary
mut_summary = ['snv_count', 'missense_count', 'synonymous_count', 'frameshift_count']
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(mut_summary), 'data_type'] = 'DNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(mut_summary), 'feature_type'] = 'mut_summary'

# label mutations
mutations = [col for col in genomics_feature_df.columns if col.endswith('_SNV')]
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(mutations), 'data_type'] = 'DNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(mutations), 'feature_type'] = 'mutation'

# label clonality
clones = ['CLONAL', 'SUBCLONAL', 'INDETERMINATE', 'clones', 'tail', 'subclonal_clonal_ratio']
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(clones), 'data_type'] = 'DNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(clones), 'feature_type'] = 'clonality'

# label antigen information
antigens = ['antigens', 'antigens_clonal', 'antigens_subclonal', 'antigens_indeterminate',
             'antigens_variants', 'HBMR_norm', 'HBMR_norm_p', 'HBMR_norm_cilo', 'HBMR_norm_ciup',
             'hla_tcn', 'hla_lcn', 'hla_loh']
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(antigens), 'data_type'] = 'DNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(antigens), 'feature_type'] = 'antigen'

# label hlas
hlas = [col for col in genomics_feature_df.columns if re.match('[ABC][12]_', col) is not None]
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(hlas), 'data_type'] = 'DNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(hlas), 'feature_type'] = 'hla'

# label mutation signatures
mut_signatures = [col for col in genomics_feature_df.columns if col.startswith('Signature')]
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(mut_signatures), 'data_type'] = 'DNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(mut_signatures), 'feature_type'] = 'mut_signature'

# label chromosome changes
chrom_cn = [col for col in genomics_feature_df.columns if col.endswith('AMP') or col.endswith('DEL') or col.endswith('LOH')]
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(chrom_cn), 'data_type'] = 'DNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(chrom_cn), 'feature_type'] = 'chrom_cn'

# label IC subtypes
ic_subtypes = [col for col in genomics_feature_df.columns if col.startswith('IC') or col.startswith('IntClust')]
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(ic_subtypes), 'data_type'] = 'RNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(ic_subtypes), 'feature_type'] = 'ic_subtype'

# label PAM50 subtypes
pam = ['PAM50_Her2', 'PAM50_Other']
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(pam), 'data_type'] = 'RNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(pam), 'feature_type'] = 'pam_subtype'


# transform RNA features to long format
RNA_feature_df = pd.read_csv(os.path.join(sequence_dir, 'TONIC_immune_sig_score_and_genes_processed.csv'))
RNA_feature_df_long = pd.melt(RNA_feature_df, id_vars=['Patient_ID', 'Timepoint', 'Tissue_ID', 'Clinical_benefit'],
                                   var_name='feature_name', value_name='feature_value')

# label cell type signatures
cell_signatures = ['NK_cells', 'T_cells', 'B_cells', 'Treg', 'Neutrophil_signature', 'MDSC', 'Macrophages',
                   'CAF', 'Matrix', 'Endothelium', 'TME_subtype_F', 'TME_subtype_IE', 'TME_subtype_IE/F']
RNA_feature_df_long.loc[RNA_feature_df_long.feature_name.isin(cell_signatures), 'data_type'] = 'RNA'
RNA_feature_df_long.loc[RNA_feature_df_long.feature_name.isin(cell_signatures), 'feature_type'] = 'cell_signature'

# label functional signatures
functional_signatures = ['T_cell_traffic', 'MHCI', 'MHCII', 'Coactivation_molecules', 'Effector_cells',
                         'T_cell_traffic', 'M1_signatures', 'Th1_signature', 'Antitumor_cytokines',
                         'Checkpoint_inhibition', 'T_reg_traffic', 'Granulocyte_traffic', 'MDSC_traffic',
                         'Macrophage_DC_traffic', 'Th2_signature', 'Protumor_cytokines', 'Matrix_remodeling',
                         'Angiogenesis', 'Proliferation_rate', 'EMT_signature', 'CAscore', 'GEP_mean']
RNA_feature_df_long.loc[RNA_feature_df_long.feature_name.isin(functional_signatures), 'data_type'] = 'RNA'
RNA_feature_df_long.loc[RNA_feature_df_long.feature_name.isin(functional_signatures), 'feature_type'] = 'functional_signature'

# label individual genes
unlabeled = RNA_feature_df_long.loc[RNA_feature_df_long.data_type.isna(), 'feature_name'].unique()
RNA_feature_df_long.loc[RNA_feature_df_long.feature_name.isin(unlabeled), 'data_type'] = 'RNA'
RNA_feature_df_long.loc[RNA_feature_df_long.feature_name.isin(unlabeled), 'feature_type'] = 'gene_rna'

# combine together
genomics_feature_df_long = pd.concat([genomics_feature_df_long, RNA_feature_df_long])
genomics_feature_df_long.to_csv(os.path.join(sequence_dir, 'processed_genomics_features.csv'), index=False)

#
# look at correlations with imaging data
#

# # check which columns are associated with response
# plot_cols = [col for col in RNA_feature_df.columns if col not in ['rna_seq_sample_id', 'Clinical_benefit', 'Timepoint', 'TME_subtype']]
#
# for col in plot_cols:
#     fig, ax = plt.subplots(1, 3)
#     for idx, timepoint in enumerate(['baseline', 'post_induction', 'on_nivo']):
#         sns.stripplot(data=immune_scores.loc[immune_scores.Timepoint == timepoint, :],
#                       x='Clinical_benefit', y=col, ax=ax[idx], color='black')
#         sns.boxplot(data=immune_scores.loc[immune_scores.Timepoint == timepoint, :],
#                     x='Clinical_benefit', y=col, ax=ax[idx], color='grey', showfliers=False)
#         ax[idx].set_title(timepoint)
#         ax[idx].set_xticklabels(ax[idx].get_xticklabels(), rotation=90)
#     plt.tight_layout()
#     sns.despine()
#     plt.savefig(os.path.join(plot_dir, 'RNA_features_{}.png'.format(col)))
#     plt.close()

# look at correlatoins with image data
image_feature_df = pd.read_csv(os.path.join(base_dir, 'analysis_files/timepoint_combined_features.csv'))
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

