import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from python_files.utils import compare_populations
from statsmodels.stats.multitest import multipletests

from scipy.stats import spearmanr

plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
sequence_dir = os.path.join(base_dir, 'sequencing_data')

harmonized_metadata = pd.read_csv(os.path.join(base_dir, 'analysis_files/harmonized_metadata.csv'))

#
# clean up genomics features
#

# load data
genomics_feature_df = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_WES_meta_table.tsv'), sep='\t')
genomics_feature_df = genomics_feature_df.rename(columns={'Individual.ID': 'Patient_ID', 'timepoint': 'Timepoint'})

# add in new IC10 predictions
new_ic10 = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/tonic_predictions.txt'), sep='\t')
new_ic10['Experiment.System.ID'] = new_ic10['Sample'].apply(lambda x: x.split('_vs')[0])
new_ic10 = new_ic10.rename(columns={'voting': 'Eniclust_2'})
genomics_feature_df = pd.merge(genomics_feature_df, new_ic10[['Experiment.System.ID', 'Eniclust_2']], on='Experiment.System.ID', how='left')

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
np.unique(genomics_feature_df['Eniclust_2'].values, return_counts=True)
keep_dict = {'IC10_rna': [4, 9, 10], 'IC10_rna_cna': [4., 9., 10.], 'PAM50': ['Basal', 'Her2'],
             'IntClust': ['ic10', 'ic4', 'ic9'], 'A1': ['A*01:01', 'A*02:01', 'A*03:01'],
             'A2': ['A*02:01', 'A*03:01', 'A*24:02'], 'B1': ['B*07:02', 'B*08:01', 'B*40:01'],
             'C1': ['C*03:04', 'C*04:01', 'C*07:01', 'C*07:02'], 'C2': ['C*07:01', 'C*07:02'],
             'Eniclust_2': ['ic4', 'ic9', 'ic10']}

for col in keep_dict.keys():
    genomics_feature_df[col] = genomics_feature_df[col].apply(lambda x: x if x in keep_dict[col] else 'Other')
    genomics_feature_df[col] = genomics_feature_df[col].astype('category')

# add computed features
genomics_feature_df['subclonal_clonal_ratio'] = genomics_feature_df['SUBCLONAL'] / genomics_feature_df['CLONAL']

# change categorical features to dummy variables
genomics_feature_df = pd.get_dummies(genomics_feature_df, columns=keep_dict.keys(), drop_first=True)

# save processed genomics features
genomics_feature_df.to_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_WES_meta_table_processed.csv'), index=False)


# process RNA-seq data
RNA_feature_df = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_immune_sig_score_table.tsv'), sep='\t')
RNA_feature_df_genes = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_immune_sig_gene_TPM_table.tsv'), sep='\t')
RNA_feature_df = pd.merge(RNA_feature_df, RNA_feature_df_genes, on='sample_identifier', how='left')
RNA_feature_df = RNA_feature_df.rename(columns={'sample_identifier': 'rna_seq_sample_id'})
RNA_feature_df = RNA_feature_df.merge(harmonized_metadata[['rna_seq_sample_id', 'Clinical_benefit', 'Timepoint', 'Patient_ID', 'Tissue_ID']].drop_duplicates(),
                                    on='rna_seq_sample_id', how='left')
RNA_feature_df = RNA_feature_df.drop(columns=['rna_seq_sample_id'])
RNA_feature_df['TME_subtype_immune'] = RNA_feature_df['TME_subtype'].apply(lambda x: 0 if x == 'F' or x == 'D' else 1)
RNA_feature_df['TME_subtype_fibrotic'] = RNA_feature_df['TME_subtype'].apply(lambda x: 1 if x == 'F' or x == 'IE/F' else 0)
RNA_feature_df = RNA_feature_df.drop(columns=['TME_subtype'])

RNA_feature_df.to_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_immune_sig_score_and_genes_processed.csv'), index=False)

# transform to long format

genomics_feature_df = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_WES_meta_table_processed.csv'))

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
ic_subtypes = [col for col in genomics_feature_df.columns if col.startswith('IC') or col.startswith('IntClust') or col.startswith('Eniclust')]
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(ic_subtypes), 'data_type'] = 'RNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(ic_subtypes), 'feature_type'] = 'ic_subtype'

# label PAM50 subtypes
pam = ['PAM50_Her2', 'PAM50_Other']
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(pam), 'data_type'] = 'RNA'
genomics_feature_df_long.loc[genomics_feature_df_long.feature_name.isin(pam), 'feature_type'] = 'pam_subtype'


# transform RNA features to long format
RNA_feature_df = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_immune_sig_score_and_genes_processed.csv'))
RNA_feature_df_long = pd.melt(RNA_feature_df, id_vars=['Patient_ID', 'Timepoint', 'Tissue_ID', 'Clinical_benefit'],
                                   var_name='feature_name', value_name='feature_value')

# label cell type signatures
cell_signatures = ['NK_cells', 'T_cells', 'B_cells', 'Treg', 'Neutrophil_signature', 'MDSC', 'Macrophages',
                   'CAF', 'Matrix', 'Endothelium', 'TME_subtype_immune', 'TME_subtype_fibrotic']
# recode fibrotic to immune yes/no, fibrotic yes/no
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
RNA_feature_df_long.loc[RNA_feature_df_long.feature_type == 'gene_rna', 'feature_name'] = RNA_feature_df_long.loc[RNA_feature_df_long.feature_type == 'gene_rna', 'feature_name'].apply(lambda x: x + '_rna')

# find RNAs already present in genomics features
RNA_names = genomics_feature_df_long.loc[genomics_feature_df_long.feature_type == 'gene_rna', 'feature_name'].unique()
RNA_feature_df_long = RNA_feature_df_long.loc[~RNA_feature_df_long.feature_name.isin(RNA_names), :]

# combine together
genomics_feature_df_long = pd.concat([genomics_feature_df_long, RNA_feature_df_long])
genomics_feature_df_long.to_csv(os.path.join(sequence_dir, 'processed_genomics_features.csv'), index=False)

#
# look at correlations with imaging data
#

# read in  data
genomics_df = pd.read_csv(os.path.join(sequence_dir, 'processed_genomics_features.csv'))
image_feature_df = pd.read_csv(os.path.join(base_dir, 'analysis_files/timepoint_combined_features.csv'))

# start with just baseline samples
genomics_df = genomics_df.loc[genomics_df.Timepoint == 'baseline', :]
image_feature_df = image_feature_df.loc[image_feature_df.Timepoint == 'baseline', :]

# subset to only include shared samples
shared_patients = np.intersect1d(image_feature_df.Patient_ID.unique(), genomics_df.Patient_ID.unique())
image_feature_df = image_feature_df.loc[image_feature_df.Patient_ID.isin(shared_patients), :]
genomics_df = genomics_df.loc[genomics_df.Patient_ID.isin(shared_patients), :]

genomics_df = genomics_df.loc[genomics_df.data_type == 'RNA', :]


# calculate all pairwise correlations
image_features, genomics_features, corr_list, pval_list = [], [], [], []

for image_col in image_feature_df.feature_name_unique.unique():
    for genom_col in genomics_df.feature_name.unique():

        # subset to specific feature
        image_features_shared = image_feature_df.loc[image_feature_df.feature_name_unique == image_col, :]
        genomics_features_shared = genomics_df.loc[genomics_df.feature_name == genom_col, :]

        # merge together
        combined_df = pd.merge(image_features_shared[['Patient_ID', 'raw_mean']],
                               genomics_features_shared[['Patient_ID', 'feature_value']],
                               on='Patient_ID', how='inner')

        # checks:  number of unique values per condition, number of samples per condition
        # append to lists
        image_features.append(image_col)
        genomics_features.append(genom_col)

        # check that all conditions are met for computing correlation

        # total number of samples included
        min_total_samples = len(combined_df) > 10

        # number of unique values for imaging and genomics features
        min_unique_vals = len(combined_df.raw_mean.unique()) > 5 and len(combined_df.feature_value.unique()) > 1

        if min_total_samples and min_unique_vals:
            corr, pval = spearmanr(combined_df.raw_mean.values, combined_df.feature_value.values)
            corr_list.append(corr)
            pval_list.append(pval)
        else:
            corr_list.append(np.nan)
            pval_list.append(np.nan)

corr_df = pd.DataFrame({'image_feature': image_features, 'genomic_features': genomics_features,
                        'cor': corr_list, 'pval': pval_list})
corr_df['log_pval'] = -np.log10(corr_df.pval)

# combined pval and correlation rank
corr_df['pval_rank'] = corr_df.log_pval.rank(ascending=False)
corr_df['cor_rank'] = corr_df.cor.abs().rank(ascending=False)
corr_df['combined_rank'] = (corr_df.pval_rank.values + corr_df.cor_rank.values) / 2

# fdr correction
from statsmodels.stats.multitest import multipletests
corr_df = corr_df.dropna(axis=0)
corr_df['fdr'] = multipletests(corr_df.pval.values, method='fdr_bh')[1]

corr_df = corr_df.sort_values(by='combined_rank', ascending=True)
corr_df.to_csv(os.path.join(sequence_dir, 'genomics_image_correlation_RNA.csv'), index=False)

corr_df_sig = corr_df.loc[~(corr_df.genomic_features.apply(lambda x: 'rna' in x)), :]
sns.scatterplot(data=corr_df, x='cor', y='log_pval', hue='combined_rank', palette='viridis')
plt.xlabel('Spearman Correlation')
plt.ylabel('-log10(p-value)')
plt.title('Correlation between RNA and Image Features')
plt.savefig(os.path.join(plot_dir, 'RNA_correlation_scatter.png'), dpi=300)
plt.close()



# plot top 10 correlations
top_corr = corr_df_sig.head(100)

for i in range(top_corr.shape[0]):
    image_col = top_corr.iloc[i].image_feature
    RNA_col = top_corr.iloc[i].genomic_features

    image_features_shared = image_feature_df.loc[image_feature_df.feature_name_unique == image_col, :]
    genomics_features_shared = genomics_df.loc[genomics_df.feature_name == RNA_col, :]

    # merge together
    combined_df = pd.merge(image_features_shared[['Patient_ID', 'raw_mean']],
                           genomics_features_shared[['Patient_ID', 'feature_value']],
                           on='Patient_ID', how='inner')

    sns.scatterplot(data=combined_df, x='raw_mean', y='feature_value')


    plt.xlabel(image_col)
    plt.ylabel(RNA_col)
    plt.title('Correlation: ' + str(round(top_corr.iloc[i].cor, 3)) + ', p-value: ' + str(round(top_corr.iloc[i].pval, 3)))
    plt.savefig(os.path.join(plot_dir, 'RNA_Image_correlation_' + str(i) + '.png'), dpi=300)
    plt.close()

# look for association with outcome
genomics_df = pd.read_csv(os.path.join(sequence_dir, 'processed_genomics_features.csv'))

plot_hits = False
method = 'ttest'

genomics_df = genomics_df.loc[genomics_df.Timepoint != 'on_nivo_1_cycle', :]
genomics_df = genomics_df.rename(columns={'feature_name': 'feature_name_unique', 'feature_value': 'raw_mean'})
genomics_df['normalized_mean'] = genomics_df['raw_mean']
genomics_df = genomics_df.loc[genomics_df.feature_type != 'gene_rna', :]

# placeholder for all values
total_dfs = []

for comparison in genomics_df.Timepoint.unique():
    population_df = compare_populations(feature_df=genomics_df, pop_col='Clinical_benefit',
                                        timepoints=[comparison], pop_1='No', pop_2='Yes', method=method)

    if plot_hits:
        current_plot_dir = os.path.join(plot_dir, 'responders_nonresponders_{}'.format(comparison))
        if not os.path.exists(current_plot_dir):
            os.makedirs(current_plot_dir)
        summarize_population_enrichment(input_df=population_df, feature_df=combined_df, timepoints=[comparison],
                                        pop_col='Clinical_benefit', output_dir=current_plot_dir, sort_by='med_diff')

    if np.sum(~population_df.log_pval.isna()) == 0:
        continue
    long_df = population_df[['feature_name_unique', 'log_pval', 'mean_diff', 'med_diff']]
    long_df['comparison'] = comparison
    long_df = long_df.dropna()
    long_df['pval'] = 10 ** (-long_df.log_pval)
    long_df['fdr_pval'] = multipletests(long_df.pval, method='fdr_bh')[1]
    total_dfs.append(long_df)

# summarize hits from all comparisons
ranked_features_df = pd.concat(total_dfs)
ranked_features_df['log10_qval'] = -np.log10(ranked_features_df.fdr_pval)

# create importance score
# get ranking of each row by log_pval
ranked_features_df['pval_rank'] = ranked_features_df.log_pval.rank(ascending=False)
ranked_features_df['cor_rank'] = ranked_features_df.med_diff.abs().rank(ascending=False)
ranked_features_df['combined_rank'] = (ranked_features_df.pval_rank.values + ranked_features_df.cor_rank.values) / 2

ranked_features_df = ranked_features_df.sort_values(by='combined_rank', ascending=True)
ranked_features_df.to_csv(os.path.join(sequence_dir, 'genomics_outcome_ranking.csv'), index=False)

# show breakdown of top features
ranked_features_df = pd.read_csv(os.path.join(sequence_dir, 'genomics_outcome_ranking.csv'))
ranked_features_df = pd.merge(genomics_df[['feature_name_unique', 'feature_type', 'data_type']].drop_duplicates(), ranked_features_df, on='feature_name_unique', how='left')
top_features = ranked_features_df.iloc[:25, :]
top_features = ranked_features_df.loc[ranked_features_df.fdr_pval < 0.1, :]


# by comparison
top_features_by_comparison = top_features[['data_type', 'comparison']].groupby(['comparison']).size().reset_index()
top_features_by_comparison.columns = ['comparison', 'num_features']
top_features_by_comparison = top_features_by_comparison.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_comparison, x='comparison', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Num_features_per_comparison_genomics.pdf'))
plt.close()

# by data type
top_features_by_data_type = top_features[['data_type', 'comparison']].groupby(['data_type']).size().reset_index()
top_features_by_data_type.columns = ['data_type', 'num_features']
top_features_by_data_type = top_features_by_data_type.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_data_type, x='data_type', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(plot_dir, 'Num_features_per_data_type_genomics.pdf'))
plt.close()




# plot top X features per comparison
num_features = 50

for comparison in ranked_features_df.comparison.unique():
    current_plot_dir = os.path.join(plot_dir, 'top_features_{}'.format(comparison))
    if not os.path.exists(current_plot_dir):
        os.makedirs(current_plot_dir)

    current_df = ranked_features_df.loc[ranked_features_df.comparison == comparison, :]
    current_df = current_df.sort_values('combined_rank', ascending=True)
    current_df = current_df.iloc[:num_features, :]

    # plot results
    for feature_name, rank in zip(current_df.feature_name_unique.values, current_df.combined_rank.values):
        plot_df = genomics_df.loc[(genomics_df.feature_name_unique == feature_name) &
                                  (genomics_df.Timepoint == comparison), :]

        g = sns.catplot(data=plot_df, x='Clinical_benefit', y='raw_mean', kind='strip')
        g.fig.suptitle(feature_name)
        g.savefig(os.path.join(current_plot_dir, 'rank_{}_feature_{}.png'.format(rank, feature_name)))
        plt.close()


# plot lineage agreement from RNA data
rna_correlations = pd.read_csv(os.path.join(sequence_dir, 'genomics_image_correlation_RNA.csv'))
rna_correlations = rna_correlations.loc[~(rna_correlations.genomic_features.apply(lambda x: 'rna' in x)), :]


populations, rna_feature, image_feature, values = [], [], [], []

pairings = {'T cells': [['T_cells', 'T_cell_traffic'], ['T__cluster_broad_density']],
            'CD8T cells': [['Effector_cells'], ['CD8T__cluster_density']],
            'APC': [['MHCII', 'Macrophage_DC_traffic'], ['APC__cluster_density']],
            'B cells': [['B_cells'], ['B__cluster_broad_density']],
            'NK cells': [['NK_cells'], ['NK__cluster_broad_density']],
            'T regs': [['T_reg_traffic', 'Treg'], ['Treg__cluster_density']],
            'Endothelium': [['Endothelium'], ['Endothelium__cluster_density']],
            'Macrophages': [['Macrophages', 'Macrophage_DC_traffic', 'M1_signatures'],
                            ['M1_Mac__cluster_density', 'M2_Mac__cluster_density', 'Monocyte__cluster_density', 'Mac_Other__cluster_density']],
            'Granulocytes': [['Granulocyte_traffic', 'Neutrophil_signature'], ['Neutrophil__cluster_density', 'Mast__cluster_density']],
            'ECM': [['CAF', 'Matrix_remodeling', 'Matrix'], ['Fibroblast__cluster_density', 'Stroma__cluster_density']]}

# look through  pairings and pull out correlations
for pop, pairs in pairings.items():
    for pair_1 in pairs[0]:
        for pair_2 in pairs[1]:
            pop_df = rna_correlations.loc[(rna_correlations.image_feature == pair_2) &
                                          (rna_correlations.genomic_features == pair_1), :]
            if len(pop_df) == 0:
                print('No data for: ', pop, pair_1, pair_2)
                continue
            populations.append(pop)
            rna_feature.append(pair_1)
            image_feature.append(pair_2)
            values.append(pop_df.cor.values[0])

pop_correlations = pd.DataFrame({'populations': populations, 'correlation': values, 'rna_feature': rna_feature,
                                 'image_feature': image_feature})

# plot results
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.stripplot(data=pop_correlations, x='populations', y='correlation', dodge=True, ax=ax)
ax.set_ylabel('RNA vs image lineage correlation')
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'lineage_correlation_stripplot.png'), dpi=300)
plt.close()


# Same thing for DNA data
dna_correlations = pd.read_csv(os.path.join(sequence_dir, 'genomics_image_correlation_DNA.csv'))
dna_correlations['log_qval'] = -np.log10(dna_correlations.fdr)

lineage_features = [pairings[x][1] for x in pairings.keys()]
lineage_features = [item for sublist in lineage_features for item in sublist]

SNVs = [x for x in dna_correlations.genomic_features.unique() if 'SNV' in x]

dna_correlations = dna_correlations.loc[dna_correlations.image_feature.isin(lineage_features), :]
dna_correlations = dna_correlations.loc[dna_correlations.genomic_features.isin(SNVs), :]

sns.stripplot(data=dna_correlations, y='cor', dodge=True)
plt.ylabel('DNA vs image lineage correlation')
plt.savefig(os.path.join(plot_dir, 'DNA_lineage_correlation_stripplot.png'), dpi=300)
plt.close()


sns.scatterplot(data=dna_correlations, x='cor', y='log_pval')
plt.xlabel('Spearman Correlation')
plt.ylabel('-log10(p-value)')
plt.title('Correlation between DNA and Image Features')
plt.ylim(0, 10)
plt.savefig(os.path.join(plot_dir, 'DNA_correlation_volcano.png'), dpi=300)
plt.close()