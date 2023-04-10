# script to generate summary stats for each fov
import os

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

local_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'

# load datasets
cluster_df_core = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
metadata_df_core = pd.read_csv(os.path.join(data_dir, 'metadata/TONIC_data_per_core.csv'))
functional_df_core = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core_filtered.csv'))
harmonized_metadata_df = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))
compartment_area = pd.read_csv(os.path.join(data_dir, 'post_processing/fov_annotation_mask_area.csv'))
mixing_df = pd.read_csv(os.path.join(data_dir, 'spatial_analysis/mixing_score/mixing_df.csv'))
ecm_df = pd.read_csv(os.path.join(data_dir, 'ecm/fov_cluster_counts.csv'))


# compute shannon diversity from list of proportions
def shannon_diversity(proportions):
    proportions = [prop for prop in proportions if prop > 0]
    return -np.sum(proportions * np.log2(proportions))


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
abundance_features = [['cluster_density', 'cluster_density', 'med'],
                      ['total_cell_density', 'total_density', 'broad']]
for cluster_name, feature_name, cell_pop_level in abundance_features:
    input_df = cluster_df_core[cluster_df_core['metric'].isin([cluster_name])]
    #for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
    for compartment in ['all']:
        compartment_df = input_df[input_df.subset == compartment].copy()
        compartment_df['feature_name'] = compartment_df.cell_type + '__' + feature_name
        compartment_df['feature_name_unique'] = compartment_df.cell_type + '__' + feature_name + '__' + compartment
        compartment_df = compartment_df.rename(columns={'subset': 'compartment'})
        if cluster_name != 'total_cell_density':
            compartment_df['cell_pop'] = compartment_df.cell_type.apply(lambda x: narrow_to_broad[x])
        else:
            compartment_df['cell_pop'] = 'all'

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
        minimum_density = 0.0005
        cell_type1_mask = cell_type1_df.value > minimum_density
        cell_type2_mask = cell_type2_df.value > minimum_density
        cell_mask = cell_type1_mask.values | cell_type2_mask.values

        cell_type1_df = cell_type1_df[cell_mask]
        cell_type2_df = cell_type2_df[cell_mask]

        # add minimum density to avoid log2(0)
        cell_type1_df['ratio'] = np.log2((cell_type1_df.value.values + minimum_density) /
                                         (cell_type2_df.value.values + minimum_density))

        cell_type1_df['value'] = cell_type1_df.ratio.values

        cell_type1_df['feature_name'] = cell_type1 + '__' + cell_type2 + '__ratio'
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
        minimum_density = 0.0005
        cell_type1_mask = cell_type1_df.value > minimum_density
        cell_type2_mask = cell_type2_df.value > minimum_density
        cell_mask = cell_type1_mask.values | cell_type2_mask.values

        cell_type1_df = cell_type1_df[cell_mask]
        cell_type2_df = cell_type2_df[cell_mask]

        cell_type1_df['ratio'] = np.log2((cell_type1_df.value.values + minimum_density) /
                                         (cell_type2_df.value.values + minimum_density))

        cell_type1_df['value'] = cell_type1_df.ratio.values
        cell_type1_df['feature_name'] = cell_type1 + '__' + cell_type2 + '__ratio'
        cell_type1_df['feature_name_unique'] = cell_type1 + '__' + cell_type2 + '__ratio__' + compartment
        cell_type1_df['compartment'] = compartment
        cell_type1_df['cell_pop'] = 'Immune'
        cell_type1_df['cell_pop_level'] = 'med'
        cell_type1_df['feature_type'] = 'density_ratio'
        cell_type1_df = cell_type1_df[['fov', 'value', 'feature_name', 'feature_name_unique','compartment', 'cell_pop',
                                       'cell_pop_level',  'feature_type']]
        fov_data.append(cell_type1_df)


# compute proportion of cells in a given cell type
cell_groups = {'Cancer': ['Cancer', 'Cancer_EMT', 'Cancer_Other'],
               'Mono_Mac': ['M1_Mac', 'M2_Mac', 'Mac_Other', 'Monocyte', 'APC'],
               'T': ['CD4T', 'CD8T', 'Treg', 'T_Other'],
               'Granulocyte': ['Neutrophil', 'Mast'],
               'Stroma': ['Fibroblast', 'Stroma']}

input_df = cluster_df_core[cluster_df_core.metric == 'cluster_density'].copy()
for compartment in ['all']:
    compartment_df = input_df[input_df.subset == compartment].copy()
    for broad_cell_type, cell_types in cell_groups.items():
        # get the total for all cell types
        cell_type_df = compartment_df[compartment_df.cell_type.isin(cell_types)].copy()
        grouped_df = cell_type_df[['fov', 'value']].groupby('fov').sum().reset_index()
        grouped_df.columns = ['fov', 'fov_sum']

        # normalize each cell type by the total
        cell_type_df = cell_type_df.merge(grouped_df, on='fov')
        cell_type_df['value'] = cell_type_df.value / cell_type_df.fov_sum

        cell_type_df['feature_name'] = cell_type_df.cell_type + '__' + broad_cell_type + '__proportion'
        cell_type_df['feature_name_unique'] = cell_type_df.cell_type + '__' + broad_cell_type + '__proportion__' + compartment
        cell_type_df['compartment'] = compartment
        cell_type_df['cell_pop'] = broad_cell_type
        cell_type_df['cell_pop_level'] = 'med'
        cell_type_df['feature_type'] = 'density_proportion'
        cell_type_df = cell_type_df[['fov', 'value', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                                     'cell_pop_level', 'feature_type']]
        fov_data.append(cell_type_df)


# compute functional marker positivity for different levels of granularity
functional_features = [['cluster_freq', 'med'],
                       ['total_freq', 'broad']]
for functional_name, cell_pop_level in functional_features:
    input_df = functional_df_core[functional_df_core['metric'].isin([functional_name])]
    #for compartment in ['cancer_core', 'cancer_border', 'stroma_core', 'stroma_border', 'all']:
    for compartment in ['all']:
        compartment_df = input_df[input_df.subset == compartment].copy()
        compartment_df['feature_name'] = compartment_df.functional_marker + '+__' + compartment_df.cell_type
        compartment_df['feature_name_unique'] = compartment_df.functional_marker + '+__' + compartment_df.cell_type + '__' + compartment
        compartment_df = compartment_df.rename(columns={'subset': 'compartment'})

        if functional_name != 'total_freq':
            compartment_df['cell_pop'] = compartment_df.cell_type.apply(lambda x: narrow_to_broad[x])
        else:
            compartment_df['cell_pop'] = 'all'

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

    minimum_abundance = 0.01
    compartment1_keep_mask = compartment_df.value > minimum_abundance
    compartment2_keep_mask = compartment2_df.value > minimum_abundance
    keep_mask = compartment1_keep_mask.values | compartment2_keep_mask.values
    compartment_df = compartment_df[keep_mask]
    compartment2_df = compartment2_df[keep_mask]
    compartment2_df['value'] = np.log2((compartment_df.value.values + minimum_abundance) /
                                       (compartment2_df.value.values + minimum_abundance))

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


# integrate mixing scores
mixing_df['feature_name'] = mixing_df['mixing_type'] + '_mixing'
mixing_df['feature_name_unique'] = mixing_df['mixing_type'] + '_mixing'
mixing_df['compartment'] = 'all'
mixing_df['cell_pop'] = 'multiple'
mixing_df['cell_pop_level'] = 'broad'
mixing_df['feature_type'] = 'spatial'
mixing_df = mixing_df.rename(columns={'mixing_score': 'value'})
mixing_df = mixing_df[['fov', 'value', 'feature_name', 'feature_name_unique', 'compartment',
                       'cell_pop', 'cell_pop_level', 'feature_type']]
fov_data.append(mixing_df)

# add ecm proportions

# add normalized columns that remove no_ecm contribution
ecm_frac = 1 - ecm_df.No_ECM.values
ecm_df['Cold_Coll_Norm'] = ecm_df.Cold_Coll.values / ecm_frac
ecm_df['Hot_Coll_Norm'] = ecm_df.Hot_Coll.values / ecm_frac
ecm_df.loc[ecm_frac == 0, ['Cold_Coll_Norm', 'Hot_Coll_Norm']] = 0

for col_name in ['Cold_Coll', 'Hot_Coll', 'No_ECM', 'Cold_Coll_Norm', 'Hot_Coll_Norm']:
    ecm_df_subset = ecm_df[['fov', col_name]]
    ecm_df_subset = ecm_df_subset.rename(columns={col_name: 'value'})
    ecm_df_subset['feature_name'] = col_name + '__proportion'
    ecm_df_subset['feature_name_unique'] = col_name + '__proportion'
    ecm_df_subset['compartment'] = 'all'
    ecm_df_subset['cell_pop'] = 'ecm'
    ecm_df_subset['cell_pop_level'] = 'broad'
    ecm_df_subset['feature_type'] = 'ecm'
    ecm_df_subset = ecm_df_subset[['fov', 'value', 'feature_name', 'feature_name_unique',
                                   'compartment', 'cell_pop', 'cell_pop_level', 'feature_type']]
    fov_data.append(ecm_df_subset)


# compute z-scores for each feature
fov_data_df = pd.concat(fov_data)
fov_data_df = fov_data_df.rename(columns={'value': 'raw_value'})
fov_data_wide = fov_data_df.pivot(index='fov', columns='feature_name_unique', values='raw_value')
zscore_df = (fov_data_wide - fov_data_wide.mean()) / fov_data_wide.std()

# add z-scores to fov_data_df
zscore_df = zscore_df.reset_index()
zscore_df_long = pd.melt(zscore_df, id_vars='fov', var_name='feature_name_unique', value_name='normalized_value')
fov_data_df = pd.merge(fov_data_df, zscore_df_long, on=['fov', 'feature_name_unique'], how='left')

# add metadata
fov_data_df = pd.merge(fov_data_df, harmonized_metadata_df[['Tissue_ID', 'fov']], on='fov', how='left')

# rearrange columns
fov_data_df = fov_data_df[['Tissue_ID', 'fov', 'raw_value', 'normalized_value', 'feature_name', 'feature_name_unique',
                            'compartment', 'cell_pop', 'cell_pop_level', 'feature_type']]

fov_data_df.to_csv(os.path.join(data_dir, 'fov_features_no_compartment.csv'), index=False)


# create timepoint-level stats file
grouped = fov_data_df.groupby(['Tissue_ID', 'feature_name', 'feature_name_unique', 'compartment', 'cell_pop',
                               'cell_pop_level', 'feature_type']).agg({'raw_value': ['mean', 'std'],
                                                                       'normalized_value': ['mean', 'std']})
grouped.columns = ['raw_mean', 'raw_std', 'normalized_mean', 'normalized_std']
grouped = grouped.reset_index()

grouped.to_csv(os.path.join(data_dir, 'timepoint_features_no_compartment.csv'), index=False)
