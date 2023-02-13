# script to generate summary stats for each fov
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# load datasets
cluster_df_core = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
metadata_df_core = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
functional_df_core = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core_filtered.csv'))
harmonized_metadata_df = pd.read_csv(os.path.join(data_dir, 'harmonized_metadata.csv'))

#
# The commented out section is from an initial version which used selected cell population
# frequencies for plotting, picking only a subset of the relevant relationships
#


def compute_celltype_ratio(input_data, celltype_1, celltype_2):
    wide_df = pd.pivot(input_data, index='fov', columns=['cell_type'], values='value')
    wide_df.reset_index(inplace=True)

    # if celltypes are lists, create columns which are a sum of individual elements
    if isinstance(celltype_1, list):
        wide_df['celltype_1'] = wide_df[celltype_1].sum(axis=1)
        celltype_1 = 'celltype_1'

    if isinstance(celltype_2, list):
        wide_df['celltype_2'] = wide_df[celltype_2].sum(axis=1)
        celltype_2 = 'celltype_2'

    # replace zeros with minimum non-vero value
    celltype_1_min = np.min(wide_df[celltype_1].array[wide_df[celltype_1] > 0])
    celltype_2_min = np.min(wide_df[celltype_2].array[wide_df[celltype_2] > 0])
    celltype_1_threshold = np.where(wide_df[celltype_1] > 0, wide_df[celltype_1], celltype_1_min)
    celltype_2_threshold = np.where(wide_df[celltype_2] > 0, wide_df[celltype_2], celltype_2_min)

    wide_df['value'] = np.log2(celltype_1_threshold / celltype_2_threshold)
    wide_df = wide_df[['fov', 'value']]

    return wide_df


# compute shannon diversity from list of proportions
def shannon_diversity(proportions):
    proportions = [prop for prop in proportions if prop > 0]
    return -np.sum(proportions * np.log2(proportions))


# # list to hold each fov, metric, value dataframe
# fov_data = []
#
# #
# # Immune related features
# #
#
# # CD4/CD8 ratio
# input_df = cluster_df_core[cluster_df_core['metric'].isin(['cluster_freq'])]
# CD4_CD8_ratio = compute_celltype_ratio(input_data=input_df, celltype_1='CD4T', celltype_2='CD8T')
# CD4_CD8_ratio['metric'] = 'CD4_CD8_ratio'
# CD4_CD8_ratio['category'] = 'immune'
# fov_data.append(CD4_CD8_ratio)
#
# # M1/M2 ratio
# input_df = cluster_df_core[cluster_df_core['metric'].isin(['cluster_freq'])]
# M1_M2_ratio = compute_celltype_ratio(input_data=input_df, celltype_1='M1_Mac', celltype_2='M2_Mac')
# M1_M2_ratio['metric'] = 'M1_M2_ratio'
# M1_M2_ratio['category'] = 'immune'
# fov_data.append(M1_M2_ratio)
#
# # Lymphoid/Myeloid ratio
# input_df = cluster_df_core[cluster_df_core['metric'].isin(['cluster_broad_freq'])]
# Lymphoid_Myeloid_ratio = compute_celltype_ratio(input_data=input_df, celltype_1=['B', 'T'],
#                                                 celltype_2=['Mono_Mac', 'Granulocyte'])
# Lymphoid_Myeloid_ratio['metric'] = 'Myeloid_Lymphoid_ratio'
# Lymphoid_Myeloid_ratio['category'] = 'immune'
# fov_data.append(Lymphoid_Myeloid_ratio)
#
# # Treg proportion T cells
# input_df = cluster_df_core[cluster_df_core['metric'].isin(['tcell_freq'])]
# input_df = input_df[input_df['cell_type'].isin(['Treg'])]
# input_df['metric'] = 'Treg_Tcell_prop'
# input_df['category'] = 'immune'
# input_df = input_df[['fov', 'value', 'metric', 'category']]
# fov_data.append(input_df)
#
# # Treg proportion immune cells
# input_df = cluster_df_core[cluster_df_core['metric'].isin(['immune_freq'])]
# input_df = input_df[input_df['cell_type'].isin(['Treg'])]
# input_df['metric'] = 'Treg_immune_prop'
# input_df['category'] = 'immune'
# input_df = input_df[['fov', 'value', 'metric', 'category']]
# fov_data.append(input_df)
#
# # Tcell proportion immune cells
# input_df = cluster_df_core[cluster_df_core['metric'].isin(['immune_freq'])]
# input_df = input_df[input_df['cell_type'].isin(['CD4T', 'CD8T', 'Treg', 'T_Other'])]
# input_df['metric'] = 'Tcell_immune_prop'
# input_df['category'] = 'immune'
# input_df = input_df[['fov', 'value', 'metric', 'category']]
# fov_data.append(input_df)
#
# # Diversity of immune cell types
# input_df = cluster_df_core[cluster_df_core['metric'].isin(['immune_freq'])]
# wide_df = pd.pivot(input_df, index='fov', columns=['cell_type'], values='value')
# wide_df['value'] = wide_df.apply(shannon_diversity, axis=1)
# wide_df.reset_index(inplace=True)
# wide_df['metric'] = 'immune_diversity'
# wide_df['category'] = 'immune'
# wide_df = wide_df[['fov', 'value', 'metric', 'category']]
# fov_data.append(wide_df)
#
# # functional markers in Tregs
# markers = ['Ki67', 'PD1']
# for marker in markers:
#     input_df = functional_df_core[functional_df_core['metric'].isin(['avg_per_cluster'])]
#     input_df = input_df[input_df['cell_type'].isin(['Treg'])]
#     input_df = input_df[input_df['functional_marker'].isin([marker])]
#     input_df['metric'] = f'{marker}_Treg'
#     input_df['category'] = 'immune'
#     input_df = input_df[['fov', 'value', 'metric', 'category']]
#     fov_data.append(input_df)
#
# # functional markers in CD8s
# markers = ['Ki67', 'PD1',  'TBET', 'TCF1', 'CD69', 'TIM3']
# for marker in markers:
#     input_df = functional_df_core[functional_df_core['metric'].isin(['avg_per_cluster'])]
#     input_df = input_df[input_df['cell_type'].isin(['CD8T'])]
#     input_df = input_df[input_df['functional_marker'].isin([marker])]
#     input_df['metric'] = f'{marker}_CD8T'
#     input_df['category'] = 'immune'
#     input_df = input_df[['fov', 'value', 'metric', 'category']]
#     fov_data.append(input_df)
#
# # functional markers in macrophages
# markers = ['IDO', 'TIM3', 'PDL1']
# for marker in markers:
#     input_df = functional_df_core[functional_df_core['metric'].isin(['avg_per_cluster_broad'])]
#     input_df = input_df[input_df['cell_type'].isin(['Mono_Mac'])]
#     input_df = input_df[input_df['functional_marker'].isin([marker])]
#     input_df['metric'] = f'{marker}_Mono_Mac'
#     input_df['category'] = 'immune'
#     input_df = input_df[['fov', 'value', 'metric', 'category']]
#     fov_data.append(input_df)
#
# #
# # stromal features
# #
#
# # functional markers in fibroblasts
# markers = ['HLADR', 'IDO', 'PDL1', 'Ki67', 'GLUT1']
# for marker in markers:
#     input_df = functional_df_core[functional_df_core['metric'].isin(['avg_per_cluster_broad'])]
#     input_df = input_df[input_df['cell_type'].isin(['Stroma'])]
#     input_df = input_df[input_df['functional_marker'].isin([marker])]
#     input_df['metric'] = f'{marker}_Stroma'
#     input_df['category'] = 'stromal'
#     input_df = input_df[['fov', 'value', 'metric', 'category']]
#     fov_data.append(input_df)
#
#
# #
# # cancer features
# #
#
# # cancer cell proportions
# cancer_populations = ['Cancer_CD56', 'Cancer_CK17', 'Cancer_Ecad', 'Cancer_SMA', 'Cancer_Vim',
#                       'Cancer_Other', 'Cancer_Mono']
#
# for cancer_population in cancer_populations:
#     input_df = cluster_df_core[cluster_df_core['metric'].isin(['cancer_freq'])]
#     input_df = input_df[input_df['cell_type'].isin([cancer_population])]
#     input_df['metric'] = f'{cancer_population}_cancer_prop'
#     input_df['category'] = 'cancer'
#     input_df = input_df[['fov', 'value', 'metric', 'category']]
#     fov_data.append(input_df)
#
# # cancer diversity
# input_df = cluster_df_core[cluster_df_core['metric'].isin(['cancer_freq'])]
# wide_df = pd.pivot(input_df, index='fov', columns=['cell_type'], values='value')
# wide_df['value'] = wide_df.apply(shannon_diversity, axis=1)
# wide_df.reset_index(inplace=True)
# wide_df['metric'] = 'cancer_diversity'
# wide_df['category'] = 'cancer'
# wide_df = wide_df[['fov', 'value', 'metric', 'category']]
# fov_data.append(wide_df)
#
#
# # functional markers in cancer cells
# markers = ['PDL1', 'PDL1_cancer_dim', 'GLUT1', 'Ki67', 'HLA1', 'HLADR']
# for marker in markers:
#     input_df = functional_df_core[functional_df_core['metric'].isin(['avg_per_cluster_broad'])]
#     input_df = input_df[input_df['cell_type'].isin(['Cancer'])]
#     input_df = input_df[input_df['functional_marker'].isin([marker])]
#     input_df['metric'] = f'{marker}_Cancer'
#     input_df['category'] = 'cancer'
#     input_df = input_df[['fov', 'value', 'metric', 'category']]
#     fov_data.append(input_df)
#
# #
# # global features
# #
#
# # immune infiltration
# immune_df = cluster_df_core.loc[(cluster_df_core.metric == 'cluster_broad_freq') &
#                                 (cluster_df_core.cell_type.isin(
#                                     ['Mono_Mac', 'B', 'T', 'Granulocyte', 'NK'])), :]
# immune_df = immune_df.loc[:, ['fov', 'value']]
# immune_grouped = immune_df.groupby('fov').agg(np.sum)
# immune_grouped.reset_index(inplace=True)
# immune_grouped['metric'] = 'immune_infiltration'
# immune_grouped['category'] = 'global'
# fov_data.append(immune_grouped)
#
#
#
# # combine all dfs together, add Tissue_ID metadata
# fov_data_df = pd.concat(fov_data)
# temp_metadata = cluster_df_core[cluster_df_core.metric == 'cluster_freq'][['fov', 'Tissue_ID', 'Timepoint']]
# temp_metadata = temp_metadata.drop_duplicates()
#
# fov_data_df = fov_data_df.merge(temp_metadata, on='fov', how='left')
# fov_data_df = fov_data_df[fov_data_df.Timepoint.isin(['primary_untreated'])]
#
# # convert to wide format for plotting in seaborn clustermap
# wide_df = fov_data_df.pivot(index='fov', columns='metric', values='value')
#
# # replace Nan with 0
# wide_df = wide_df.fillna(0)
#
# sns.clustermap(wide_df, z_score=1)
#
# # same thing for timepoint aggregation
# timepoint_data_df = fov_data_df.groupby(['Tissue_ID', 'metric']).agg(np.mean)
# timepoint_data_df.reset_index(inplace=True)
# timepoint_data_df = timepoint_data_df.pivot(index='Tissue_ID', columns='metric', values='value')
#
# # replace Nan with 0
# timepoint_data_df = timepoint_data_df.fillna(0)
#
# sns.clustermap(timepoint_data_df, z_score=1, cmap='vlag', vmin=-3, vmax=3)
#

#
# This version of the code is for extracting features across all cell x functional marker combos
#

# create lookup table for mapping individual cell types to broader categories
broad_to_narrow = {'Cancer': ['Cancer_CD56', 'Cancer_CK17', 'Cancer_Ecad', 'Cancer_SMA',
                              'Cancer_Vim', 'Cancer_Other', 'Cancer_Mono', 'Cancer', 'Cancer_EMT'],
                   'Immune': ['CD68', 'CD163', 'CD68_CD163_DP', 'CD4_Mono', 'CD14', 'CD11c_HLADR',
                              'CD20', 'CD31', 'CD31_VIM', 'FAP', 'FAP_SMA', 'SMA', 'CD56',
                              'Neutrophil', 'Mast', 'CD4T', 'CD4T_HLADR', 'CD8T', 'Treg', 'CD3_DN',
                              'CD4T_CD8T_DP', 'Immune_Other', 'M1_Mac', 'M2_Mac', 'Mac_Other',
                              'Monocyte', 'APC', 'B', 'CD4T', 'CD8T', 'Treg', 'T_Other',
                              'Neutrophil', 'Mast', 'NK', 'Mono_Mac', 'Granulocyte', 'T'],
                   'Stroma': ['Stroma_Collagen', 'Stroma_Fibronectin', 'VIM', 'Other',
                              'Endothelium', 'Fibroblast', 'Stroma'],
                   }
# reverse lookup table
narrow_to_broad = {}
for broad, narrow in broad_to_narrow.items():
    for n in narrow:
        narrow_to_broad[n] = broad

fov_data = []

# compute diversity of different levels of granularity
diversity_features = [['cluster_broad_freq', 'cluster_broad_diversity'],
                      ['immune_freq', 'immune_diversity'],
                      ['cancer_freq', 'cancer_diversity'],
                      ['stroma_freq', 'stroma_diversity']]

for cluster_name, feature_name in diversity_features:
    input_df = cluster_df_core[cluster_df_core['metric'].isin([cluster_name])]
    for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
        compartment_df = input_df[input_df.subset == compartment]
        wide_df = pd.pivot(compartment_df, index='fov', columns=['cell_type'], values='value')
        wide_df['value'] = wide_df.apply(shannon_diversity, axis=1)
        wide_df.reset_index(inplace=True)
        wide_df['feature_name'] = feature_name + '_' + compartment
        wide_df['compartment'] = compartment

        if cluster_name == 'cluster_broad_freq':
            cell_pop = 'all'
        else:
            cell_pop = cluster_name.split('_')[0]

        wide_df['cell_pop'] = cell_pop
        wide_df['feature_type'] = 'diversity'
        wide_df = wide_df[['fov', 'value', 'feature_name', 'compartment', 'cell_pop', 'feature_type']]
        fov_data.append(wide_df)


# compute abundance of cell types
abundance_features = [['cluster_density', 'cluster_density']]
for cluster_name, feature_name in abundance_features:
    input_df = cluster_df_core[cluster_df_core['metric'].isin([cluster_name])]
    for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
        compartment_df = input_df[input_df.subset == compartment]
        compartment_df['feature_name'] = compartment_df.cell_type + '_' + feature_name + '_' + compartment
        compartment_df = compartment_df.rename(columns={'subset': 'compartment'})
        compartment_df['cell_pop'] = compartment_df.cell_type.apply(lambda x: narrow_to_broad[x])
        compartment_df['feature_type'] = cluster_name.split('_')[-1]
        compartment_df = compartment_df[['fov', 'value', 'feature_name', 'compartment', 'cell_pop', 'feature_type']]
        fov_data.append(compartment_df)


# compute functional marker positivity for different levels of granularity
functional_features = ['cluster_freq']
for functional_name in functional_features:
    input_df = functional_df_core[functional_df_core['metric'].isin([functional_name])]
    for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
        compartment_df = input_df[input_df.subset == compartment]
        compartment_df['feature_name'] = compartment_df.functional_marker + '+_' + compartment_df.cell_type + '_' + compartment
        compartment_df = compartment_df.rename(columns={'subset': 'compartment'})
        compartment_df['cell_pop'] = compartment_df.cell_type.apply(lambda x: narrow_to_broad[x])
        compartment_df['feature_type'] = 'functional_marker'
        compartment_df = compartment_df[['fov', 'value', 'feature_name', 'compartment', 'cell_pop', 'feature_type']]
        fov_data.append(compartment_df)


fov_data_df = pd.concat(fov_data)
fov_data_df = pd.merge(fov_data_df, harmonized_metadata_df[['Tissue_ID', 'fov']], on='fov', how='left')
fov_data_df.to_csv(os.path.join(data_dir, 'fov_features.csv'), index=False)


# create fov_data that contains ratio of features across timepoints
timepoint_data_df = fov_data_df.groupby(['Tissue_ID', 'metric']).agg(np.mean)
timepoint_data_df.reset_index(inplace=True)

harmonized_metadata_df = harmonized_metadata_df.drop('fov', axis=1).drop_duplicates()
timepoint_data_df = pd.merge(timepoint_data_df, harmonized_metadata_df, on='Tissue_ID', how='left')
timepoint_data_df = timepoint_data_df.loc[timepoint_data_df.primary_baseline == True]

evolution_dfs = []
for patient in timepoint_data_df.TONIC_ID.unique():
    temp_df = timepoint_data_df[timepoint_data_df.TONIC_ID == patient]
    temp_df = temp_df[temp_df.Timepoint.isin(['primary_untreated', 'baseline'])]
    wide_df = temp_df.pivot(index='metric', columns='Timepoint', values='value')


# create dictionary of functional markers to keep for each cell type
lymphocyte = ['B', 'CD4T', 'CD8T', 'Immune_Other', 'NK', 'T_Other', 'Treg']
cancer = ['Cancer', 'Cancer_EMT', 'Cancer_Other']
monocyte = ['APC', 'M1_Mac', 'M2_Mac', 'Mono_Mac', 'Monocyte', 'Mac_Other']
stroma = ['Fibroblast', 'Stroma', 'Endothelium']
granulocyte = ['Mast', 'Neutrophil']

keep_dict = {'CD38': ['B', 'Immune_other', 'NK', 'Endothelium'], 'CD45RB': lymphocyte, 'CD45RO': lymphocyte,
             'CD57': lymphocyte + cancer, 'CD69': lymphocyte,
             'GLUT1': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'HLA1': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'HLADR': lymphocyte + monocyte, 'IDO': ['APC', 'B'], 'Ki67': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'LAG3': ['B'], 'PD1': lymphocyte, 'PDL1_combined': lymphocyte + monocyte + granulocyte + cancer,
             'TBET': lymphocyte, 'TCF1': lymphocyte, 'TIM3': lymphocyte + monocyte + granulocyte}


