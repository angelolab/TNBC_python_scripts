import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from python_files.utils import compare_populations


from scipy.stats import spearmanr

plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
sequence_dir = os.path.join(base_dir, 'sequencing_data')

harmonized_metadata = pd.read_csv(os.path.join(base_dir, 'analysis_files/harmonized_metadata.csv'))

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

# show breakdown of top features
ranked_features_df = pd.read_csv(os.path.join(sequence_dir, 'genomics_outcome_ranking.csv'))
ranked_features_df = pd.merge(genomics_df[['feature_name_unique', 'feature_type', 'data_type']].drop_duplicates(), ranked_features_df, on='feature_name_unique', how='left')
ranked_features_df = ranked_features_df.sort_values(by='combined_rank', ascending=True)
top_features = ranked_features_df.iloc[:50, :]
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

rna_correlations_random_subset = rna_correlations.sample(10000)
sns.scatterplot(data=rna_correlations_random_subset, x='cor', y='log_pval')
plt.xlabel('Spearman Correlation')
plt.ylabel('-log10(p-value)')
plt.title('Correlation between RNA and Image Features')
plt.ylim(0, 10)
plt.savefig(os.path.join(plot_dir, 'RNA_correlation_volcano.pdf'), dpi=300)
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

dna_correlations_random_subset = dna_correlations.sample(10000)
sns.scatterplot(data=dna_correlations_random_subset, x='cor', y='log_pval')
plt.xlabel('Spearman Correlation')
plt.ylabel('-log10(p-value)')
plt.title('Correlation between DNA and Image Features')
plt.ylim(0, 10)
plt.savefig(os.path.join(plot_dir, 'DNA_correlation_volcano.pdf'), dpi=300)
plt.close()


## protein vs rna plots
# calculate total (cell) signal in each image & normalize by cell area
'''
img_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples'
seg_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output'

fovs = io_utils.list_folders(img_dir, substrs='TONIC')
chans = io_utils.list_files(os.path.join(img_dir, fovs[0]), substrs='.tiff')
chans = io_utils.remove_file_extensions(chans)
for chan in ['FOXP3_nuc_include', 'ECAD_smoothed', 'chan_141', 'CD11c_nuc_exclude',
             'CK17_smoothed', 'chan_48', 'chan_45', 'Noodle', 'chan_115', 'chan_39']:
    chans.remove(chan)

# calculate total cell signal in images
total_df = []
with tqdm(total=len(fovs), desc="Signal Calculation", unit="FOVs") as progress:
    for fov in fovs:
        # read in channel img and cell mask
        seg_path = os.path.join(seg_dir, fov + '_whole_cell.tiff')
        img_data = load_utils.load_imgs_from_tree(data_dir=img_dir, fovs=[fov], channels=chans)
        seg_mask = io.imread(seg_path)

        # binarize cell mask and calculate area
        seg_mask[seg_mask > 0] = 1
        fov_area = np.sum(seg_mask)
        chan_signal = []

        for chan in chans:
            img = np.array(img_data.loc[fov, :, :, chan])

            # remove signal outside of cells and get total intensity
            cell_img = img * seg_mask
            cell_signal = np.sum(cell_img)

            if chan == chans[0]:
                chan_df = pd.DataFrame({'fov': [fov],
                                        'cell_areas': [fov_area],
                                        f'{chan}_cell_total_intensity': [cell_signal]})
                fov_df = chan_df
            else:
                fov_df[f'{chan}_cell_total_intensity'] = [cell_signal]

        total_df.append(fov_df)
        progress.update(1)

total_df = pd.concat(total_df)
for chan in chans:
    total_df[f'{chan}_normalized'] = total_df[f'{chan}_cell_total_intensity'] / total_df['cell_areas']
total_df.to_csv(os.path.join(sequence_dir, 'analysis/MIBI_cell_signal_stats_all.csv'), index=False)
'''

# map of mibi channel name to gene name
marker_to_rna = {'CD14': 'CD14', 'CD38': 'CD38', 'HLA1': 'HLA-A', 'PDL1': 'CD274',
                 'HLADR': 'HLA-DRA', 'Ki67': 'MKI67', 'FAP': 'FAP',
                 'Collagen1': 'COL1A1' , 'CD45': 'PTPRC', 'GLUT1': 'SLC2A1',
                 'CD69': 'CD69', 'CK17': 'KRT17', 'CD68': 'CD68',
                 'TBET': 'TBX21', 'CD163': 'CD163', 'FOXP3': 'FOXP3',
                 'Fibronectin': 'FN1', 'CD11c': 'ITGAX', 'Vim': 'VIM', 'CD8': 'CD8A',
                 'CD4': 'CD4', 'H3K9ac': 'KAT2A', 'ECAD': 'CDH1', 'Calprotectin': 'S100A8',
                 'LAG3': 'LAG3', 'SMA': 'SMN1', 'CD31': 'PECAM1', 'IDO': 'IDO1',
                 'TCF1': 'TCF7', 'CD57': 'B3GAT1', 'CD20': 'MS4A1', 'TIM3': 'HAVCR2',
                 'CD56': 'NCAM1', 'PD1': 'PDCD1', 'CD3': 'CD3D', 'ChyTr': 'CMA1'}


meta_data = pd.read_csv(os.path.join(base_dir, 'analysis_files/harmonized_metadata.csv'))
meta_data_baseline = meta_data[meta_data.Timepoint == 'baseline'].dropna(subset=['rna_seq_sample_id'])
meta_data_baseline = meta_data_baseline[['fov', 'rna_seq_sample_id', 'Timepoint']]

rna_data = pd.read_csv(os.path.join(sequence_dir, 'analysis/TONIC_gene_expression_sub.csv'))
rna_baseline = rna_data.merge(meta_data_baseline[['rna_seq_sample_id', 'Timepoint']], on=['rna_seq_sample_id']).drop_duplicates()

mibi_data = pd.read_csv(os.path.join(sequence_dir, 'analysis/MIBI_cell_signal_stats_all.csv'))
norm_cols = [col for col in mibi_data.columns if '_normalized' in col]
mibi_baseline = mibi_data[['fov'] + norm_cols].merge(meta_data_baseline, on=['fov'])

# create correlation plots
markers = list(marker_to_rna.keys())
for chan in markers:
    rna = marker_to_rna[chan]

    # average fov values across patients
    mibi = mibi_baseline[[f'{chan}_normalized', 'rna_seq_sample_id']]
    mibi = mibi.groupby('rna_seq_sample_id').mean()

    rna_data = rna_baseline[['rna_seq_sample_id', rna]]
    combined_df = mibi.merge(rna_data, on=['rna_seq_sample_id'])

    x = combined_df[f'{chan}_normalized']
    y = combined_df[rna]
    r, p = spearmanr(x, y)

    plt.figure()
    plt.scatter(x=x, y=y, s=15)
    plt.title(f'Correlation: {round(r, 3)}')
    plt.xlabel(f'{chan} normalized total intensity')
    plt.ylabel(f'{rna} expression')
    plt.savefig(os.path.join(sequence_dir, f'analysis/plots/{chan}_{rna}_comparison.png'))
