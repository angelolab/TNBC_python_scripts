# script to generate summary stats for each fov
import os

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# load datasets
cluster_df_core = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
metadata_df_core = pd.read_csv(os.path.join(data_dir, 'metadata/TONIC_data_per_core.csv'))
functional_df_core = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core_filtered.csv'))
harmonized_metadata_df = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
compartment_area = pd.read_csv(os.path.join(data_dir, 'post_processing/fov_annotation_mask_area.csv'))

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
    #for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
    for compartment in ['all']:
        compartment_df = input_df[input_df.subset == compartment].copy()
        wide_df = pd.pivot(compartment_df, index='fov', columns=['cell_type'], values='value')
        wide_df['value'] = wide_df.apply(shannon_diversity, axis=1)
        wide_df.reset_index(inplace=True)
        wide_df['feature_name'] = feature_name
        wide_df['feature_name_unique'] = feature_name + '__' + compartment
        wide_df['compartment'] = compartment

        if cluster_name == 'cluster_broad_freq':
            cell_pop = 'all'
            cell_pop_level = 'broad'
        else:
            cell_pop = cluster_name.split('_')[0]
            cell_pop = cell_pop[0].upper() + cell_pop[1:]
            cell_pop_level = 'broad'

        wide_df['cell_pop'] = cell_pop
        wide_df['cell_pop_level'] = cell_pop_level
        wide_df['feature_type'] = 'diversity'
        wide_df = wide_df[['fov', 'value', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop', 'cell_pop_level',
                           'feature_type']]
        fov_data.append(wide_df)


# compute abundance of cell types
abundance_features = [['cluster_density', 'cluster_density', 'med']]
for cluster_name, feature_name, cell_pop_level in abundance_features:
    input_df = cluster_df_core[cluster_df_core['metric'].isin([cluster_name])]
    #for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
    for compartment in ['all']:
        compartment_df = input_df[input_df.subset == compartment].copy()
        compartment_df['feature_name'] = compartment_df.cell_type + '__' + feature_name
        compartment_df['feature_name_unique'] = compartment_df.cell_type + '__' + feature_name + '__' + compartment
        compartment_df = compartment_df.rename(columns={'subset': 'compartment'})
        compartment_df['cell_pop'] = compartment_df.cell_type.apply(lambda x: narrow_to_broad[x])
        compartment_df['cell_pop_level'] = cell_pop_level
        compartment_df['feature_type'] = cluster_name.split('_')[-1]
        compartment_df = compartment_df[['fov', 'value', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                                         'cell_pop_level', 'feature_type']]
        fov_data.append(compartment_df)

# compute ratio of broad cell type abundances
input_df = cluster_df_core[cluster_df_core['metric'].isin(['cluster_broad_density'])]
# for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
for compartment in ['all']:
    compartment_df = input_df[input_df.subset == compartment].copy()
    cell_types = compartment_df.cell_type.unique()
    for cell_type1, cell_type2 in itertools.combinations(cell_types, 2):
        cell_type1_df = compartment_df[compartment_df.cell_type == cell_type1].copy()
        cell_type2_df = compartment_df[compartment_df.cell_type == cell_type2].copy()

        # only keep FOVS with at least one cell type over the minimum density
        cell_type1_mask = cell_type1_df.value > 0.02
        cell_type2_mask = cell_type2_df.value > 0.02
        cell_mask = cell_type1_mask.values | cell_type2_mask.values

        cell_type1_df = cell_type1_df[cell_mask]
        cell_type2_df = cell_type2_df[cell_mask]

        cell_type1_df['value'] = np.log2((cell_type1_df.value.values + 0.005) /
                                         (cell_type2_df.value.values + 0.005))
        cell_type1_df['feature_name'] = cell_type1 + '__' + cell_type2 + '__ratio__'
        cell_type1_df['feature_name_unique'] = cell_type1 + '__' + cell_type2 + '__ratio__' + compartment
        cell_type1_df['compartment'] = compartment
        cell_type1_df['cell_pop'] = 'multiple'
        cell_type1_df['cell_pop_level'] = 'broad'
        cell_type1_df['feature_type'] = 'density_ratio'
        cell_type1_df = cell_type1_df[['fov', 'value', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                                       'cell_pop_level', 'feature_type']]
        fov_data.append(cell_type1_df)

# compute ratio of specific cell type abundances
cell_ratios = [('CD4T', 'CD8T'), ('CD4T', 'Treg'), ('CD8T', 'Treg'), ('M1_Mac', 'M2_Mac')]
input_df = cluster_df_core[cluster_df_core.metric == 'cluster_density'].copy()
# for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
for compartment in ['all']:
    compartment_df = input_df[input_df.subset == compartment].copy()
    for cell_type1, cell_type2 in cell_ratios:
        cell_type1_df = compartment_df[compartment_df.cell_type == cell_type1].copy()
        cell_type2_df = compartment_df[compartment_df.cell_type == cell_type2].copy()

        # only keep FOVS with at least one cell type over the minimum density
        cell_type1_mask = cell_type1_df.value > 0.02
        cell_type2_mask = cell_type2_df.value > 0.02
        cell_mask = cell_type1_mask.values | cell_type2_mask.values

        cell_type1_df = cell_type1_df[cell_mask]
        cell_type2_df = cell_type2_df[cell_mask]

        cell_type1_df['value'] = np.log2((cell_type1_df.value.values + 0.005) /
                                         (cell_type2_df.value.values + 0.005))
        cell_type1_df['feature_name'] = cell_type1 + '__' + cell_type2 + '__ratio__'
        cell_type1_df['feature_name_unique'] = cell_type1 + '__' + cell_type2 + '__ratio__' + compartment
        cell_type1_df['compartment'] = compartment
        cell_type1_df['cell_pop'] = 'Immune'
        cell_type1_df['cell_pop_level'] = 'med'
        cell_type1_df['feature_type'] = 'density_ratio'
        cell_type1_df = cell_type1_df[['fov', 'value', 'feature_name', 'feature_name_unique','compartment', 'cell_pop',
                                       'cell_pop_level',  'feature_type']]
        fov_data.append(cell_type1_df)

# compute functional marker positivity for different levels of granularity
functional_features = [['cluster_freq', 'med']]
for functional_name, cell_pop_level in functional_features:
    input_df = functional_df_core[functional_df_core['metric'].isin([functional_name])]
    #for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
    for compartment in ['all']:
        compartment_df = input_df[input_df.subset == compartment].copy()
        compartment_df['feature_name'] = compartment_df.functional_marker + '+__' + compartment_df.cell_type
        compartment_df['feature_name_unique'] = compartment_df.functional_marker + '+__' + compartment_df.cell_type + '__' + compartment
        compartment_df = compartment_df.rename(columns={'subset': 'compartment'})
        compartment_df['cell_pop'] = compartment_df.cell_type.apply(lambda x: narrow_to_broad[x])
        compartment_df['cell_pop_level'] = cell_pop_level
        compartment_df['feature_type'] = 'functional_marker'
        compartment_df = compartment_df[['fov', 'value', 'feature_name','feature_name_unique','compartment', 'cell_pop',
                                         'cell_pop_level', 'feature_type']]
        fov_data.append(compartment_df)

# compute compartment abundance and ratios
compartments = ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border']
for idx, compartment in enumerate(compartments):
    compartment_df = compartment_area[compartment_area.compartment == compartment].copy()
    total_area = compartment_area[compartment_area.compartment == 'all']
    compartment_df['value'] = compartment_df.area.values / total_area.area.values
    compartment_df['feature_name'] = compartment + '__proportion'
    compartment_df['feature_name_unique'] = compartment + '__proportion'
    compartment_df['compartment'] = compartment
    compartment_df['cell_pop'] = 'all'
    compartment_df['cell_pop_level'] = 'broad'
    compartment_df['feature_type'] = 'spatial'
    compartment_df = compartment_df[['fov', 'value', 'feature_name', 'feature_name_unique','compartment', 'cell_pop',
                                     'cell_pop_level', 'feature_type']]
    fov_data.append(compartment_df)

    # now look at combinations of compartments
    if idx == 3:
        continue
    compartment2 = compartments[idx + 1]
    compartment2_df = compartment_area[compartment_area.compartment == compartment2].copy()
    compartment2_df['value'] = compartment2_df.area.values / total_area.area.values
    compartment1_keep_mask = compartment_df.value > 0.05
    compartment2_keep_mask = compartment2_df.value > 0.05
    keep_mask = compartment1_keep_mask.values | compartment2_keep_mask.values
    compartment_df = compartment_df[keep_mask]
    compartment2_df = compartment2_df[keep_mask]
    compartment2_df['value'] = np.log2((compartment_df.value.values + 0.01) / (compartment2_df.value.values + 0.01))

    # add metadata
    compartment2_df['feature_name'] = compartment + '__' + compartment2 + '__log2_ratio'
    compartment2_df['feature_name_unique'] = compartment + '__' + compartment2 + '__log2_ratio'
    compartment2_df['compartment'] = 'all'
    compartment2_df['cell_pop'] = 'all'
    compartment2_df['cell_pop_level'] = 'broad'
    compartment2_df['feature_type'] = 'spatial'
    compartment2_df = compartment2_df[['fov', 'value', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                                       'cell_pop_level', 'feature_type']]
    fov_data.append(compartment2_df)

# combine metrics together
fov_data_df = pd.concat(fov_data)
fov_data_df = pd.merge(fov_data_df, harmonized_metadata_df[['Tissue_ID', 'fov']], on='fov', how='left')
fov_data_df.to_csv(os.path.join(data_dir, 'fov_features_no_compartment.csv'), index=False)

# create timepoint-level stats file
grouped = fov_data_df.groupby(['Tissue_ID', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                               'cell_pop_level', 'feature_type']).agg({'value': ['mean', 'std']})
grouped.columns = ['mean', 'std']
grouped = grouped.reset_index()

grouped.to_csv(os.path.join(data_dir, 'timepoint_features_no_compartment.csv'), index=False)
