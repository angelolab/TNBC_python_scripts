import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from itertools import combinations

from ark.phenotyping.post_cluster_utils import plot_hist_thresholds, create_mantis_project
from alpineer.io_utils import list_folders


# This file takes the cell table generated by Pixie and performs preprocessing and aggregation to facilitate easier plotting

post_processing_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/intermediate_files/post_processing'
analysis_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/analysis_files'

if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

cell_table_name = 'combined_cell_table_normalized_cell_labels'
cell_table_full = pd.read_csv(os.path.join(post_processing_dir, cell_table_name + '.csv'))

# subset to only include relevant channels for cell assignment to speed up the process
cell_table = cell_table_full[['cell_meta_cluster', 'CD4', 'CD3', 'ChyTr', 'Calprotectin', 'CD56',
                              'CD45', 'CK17', 'CD8', 'CD68', 'ECAD', 'SMA', 'Vim']]

original_cluster_names = ['CD11c_HLADR', 'CD14', 'CD163', 'CD20', 'CD31', 'CD31_VIM', 'CD3_DN', 'CD3_noise_split',
     'CD45', 'CD4T', 'CD4T_CD8T_dp', 'CD4T_HLADR', 'CD4_mono', 'CD56', 'CD68', 'CD68_CD163_DP', 'CD8T',
     'CD8_CD8Tdim', 'ChyTry', 'FAP', 'FAP_SMA', 'FOXP3_CD45_split', 'SMA', 'Treg', 'VIM',
     'calprotectin', 'cd56_dirty', 'ck17_tumor', 'ecad_hladr', 'ecad_vim', 'noise',
     'other_stroma_coll','other_stroma_fibronectin', 'tumor_CD56', 'tumor_ecad', 'tumor_other_mono']

# make sure cluster names are correct
assert set(cell_table['cell_meta_cluster'].unique()) == set(original_cluster_names)

plot_hist_thresholds(cell_table=cell_table, populations=['tumor_ecad', 'tumor_other','VIM'],marker='ECAD', threshold=0.005)

# create CD4+ cells from CD3_noise population
marker = 'CD4'
threshold = 0.001
target_pop = 'CD3_noise_split'
new_pop = 'CD3_noise_CD4s'
selected_idx = cell_table[marker] > threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# create CD3 DN cells from CD3_noise population
marker = 'CD3'
threshold = 0.0005
target_pop = 'CD3_noise_split'
new_pop = 'CD3_noise_CD3DN'
selected_idx = cell_table[marker] > threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# remove negative cells from chytry population
marker = 'ChyTr'
threshold = 0.01
target_pop = 'ChyTry'
new_pop = 'ChyTry_neg'
selected_idx = cell_table[marker] < threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# remove negative cells from calprotectin population
marker = 'Calprotectin'
threshold = 0.001
target_pop = 'calprotectin'
new_pop = 'calprotectin_neg'
selected_idx = cell_table[marker] < threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# remove negative cells from CD56_dirty
marker = 'CD56'
threshold = 0.001
target_pop = 'cd56_dirty'
new_pop = 'cd56_dirty_neg'
selected_idx = cell_table[marker] < threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# get immune cells from noise category
marker = 'CD45'
threshold = 0.001
target_pop = 'noise'
new_pop = 'noise_cd45_pos'
selected_idx = cell_table[marker] > threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# remove negative cells from CK17
marker = 'CK17'
threshold = 0.001
target_pop = 'ck17_tumor'
new_pop = 'ck17_tumor_neg'
selected_idx = cell_table[marker] < threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# create CD4+ cells from CD3_DN population
marker = 'CD4'
threshold = 0.001
target_pop = 'CD3_DN'
new_pop = 'CD3_DN_CD4'
selected_idx = cell_table[marker] > threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# create CD8+ cells from CD3_DN population
marker = 'CD8'
threshold = 0.001
target_pop = 'CD3_DN'
new_pop = 'CD3_DN_CD8'
selected_idx = cell_table[marker] > threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# remove negs cells from CD3_DN population
marker = 'CD3'
threshold = 0.0005
target_pop = 'CD3_DN'
new_pop = 'CD3_DN_noise'
selected_idx = cell_table[marker] < threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# remove other cells from CD8_CD8T_dim population
marker = 'CD8'
threshold = 0.001
target_pop = 'CD8_CD8Tdim'
new_pop = 'CD8_CD8Tdim_other'
selected_idx = cell_table[marker] < threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# remove CD68 neg cells from CD68_CD163_DP population
marker = 'CD68'
threshold = 0.001
target_pop = 'CD68_CD163_DP'
new_pop = 'CD68_CD163_DP_68neg'
selected_idx = cell_table[marker] < threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# remove myoepethelial cells from SMA cluster
marker = 'ECAD'
threshold = 0.001
target_pop = 'SMA'
new_pop = 'SMA_ECAD'
selected_idx = cell_table[marker] > threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop


# remove non-immune cells from CD45_FOXP3 cluster
marker = 'CD45'
threshold = 0.001
target_pop = 'FOXP3_CD45_split'
new_pop = 'FOXP3_CD45_split_pos'
selected_idx = cell_table[marker] > threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop


# find ecad_sma positive cells in ecad
marker = 'SMA'
threshold = 0.01
target_pop = 'tumor_ecad'
new_pop = 'tumor_ecad_sma'
selected_idx = cell_table[marker] > threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop

# find ecad_sma positive cells in ck17
marker = 'SMA'
threshold = 0.01
target_pop = 'ck17_tumor'
new_pop = 'ck17_tumor_sma'
selected_idx = cell_table[marker] > threshold
cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop


# find VIM+ tumor cells
pops = ['tumor_CD56', 'ck17_tumor', 'tumor_ecad', 'tumor_other','tumor_other_mono']
marker = 'Vim'
threshold = 0.005
new_pop = 'tumor_vim'
selected_idx = cell_table[marker] > threshold
cell_table.loc[np.logical_and(np.isin(cell_table['cell_meta_cluster'], pops), selected_idx), 'cell_meta_cluster'] = new_pop

# check to make sure filtering worked
new_pops = ['CD3_noise_CD4s', 'CD3_noise_CD3DN', 'ChyTry_neg', 'calprotectin_neg', 'cd56_dirty_neg',
            'noise_cd45_pos', 'ck17_tumor_neg', 'CD3_DN_CD4', 'CD3_DN_CD8', 'CD3_DN_noise',
            'CD8_CD8Tdim_other', 'CD68_CD163_DP_68neg', 'SMA_ECAD', 'FOXP3_CD45_split_pos',
            'tumor_ecad_sma', 'ck17_tumor_sma', 'tumor_vim']

assert np.all(np.isin(new_pops, cell_table['cell_meta_cluster'].unique()))

assert set(cell_table['cell_meta_cluster'].unique()) == set(original_cluster_names + new_pops)


# update cell table with post-inspection decisions
cell_table.loc[cell_table['cell_meta_cluster'] == 'noise', 'cell_meta_cluster'] = 'tumor_other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD8_CD8Tdim', 'cell_meta_cluster'] = 'CD8T'
cell_table.loc[cell_table['cell_meta_cluster'] == 'noise_cd45_pos', 'cell_meta_cluster'] = 'immune_other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'FOXP3_CD45_split_pos', 'cell_meta_cluster'] = 'CD45'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD45', 'cell_meta_cluster'] = 'immune_other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'ecad_hladr',  'cell_meta_cluster'] = 'tumor_other_mono'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD8_CD8Tdim_other', 'cell_meta_cluster'] = 'tumor_other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD3_noise_CD4s', 'cell_meta_cluster'] = 'CD4T'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD3_noise_CD3DN', 'cell_meta_cluster'] = 'immune_other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD3_noise_split', 'cell_meta_cluster'] = 'other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'ChyTry_neg', 'cell_meta_cluster'] = 'other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD3_DN_CD4', 'cell_meta_cluster'] = 'CD4T'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD3_DN_CD8', 'cell_meta_cluster'] = 'CD8T'
cell_table.loc[cell_table['cell_meta_cluster'] == 'cd56_dirty', 'cell_meta_cluster'] = 'tumor_CD56'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD3_DN_noise', 'cell_meta_cluster'] = 'immune_other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'cd56_dirty_neg', 'cell_meta_cluster'] = 'tumor_other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'ck17_tumor_neg', 'cell_meta_cluster'] = 'tumor_ecad'
cell_table.loc[cell_table['cell_meta_cluster'] == 'calprotectin_neg', 'cell_meta_cluster'] = 'other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD68_CD163_DP_68neg', 'cell_meta_cluster'] = 'CD163'
cell_table.loc[cell_table['cell_meta_cluster'] == 'FOXP3_CD45_split', 'cell_meta_cluster'] = 'other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'tumor_ecad_sma', 'cell_meta_cluster'] = 'tumor_sma'
cell_table.loc[cell_table['cell_meta_cluster'] == 'SMA_ECAD', 'cell_meta_cluster'] = 'tumor_sma'
cell_table.loc[cell_table['cell_meta_cluster'] == 'ck17_tumor_sma', 'cell_meta_cluster'] = 'tumor_sma'

# make sure replacements are all updated
removed_pops = ['noise','CD8_CD8Tdim', 'noise_cd45_pos', 'CD45', 'ecad_hladr', 'CD8_CD8Tdim_other',
                'CD3_noise_CD4s', 'CD3_noise_CD3DN', 'CD3_noise_split', 'ChyTry_neg', 'CD3_DN_CD4',
                'CD3_DN_CD8', 'cd56_dirty', 'CD3_DN_noise', 'cd56_dirty_neg', 'ck17_tumor_neg',
                'calprotectin_neg', 'FOXP3_CD45_split_pos', 'CD68_CD163_DP_68neg',
                'FOXP3_CD45_split', 'tumor_ecad_sma', 'SMA_ECAD', 'ck17_tumor_sma']

assert np.all(np.isin(removed_pops, cell_table['cell_meta_cluster'].unique()) == False)


# generate consistent names for clusters: Tumor -> Cancer, and capitalize all names for plotting
replacements = [('tumor_other', 'Cancer_Other'),
                ('ck17_tumor', 'Cancer_CK17'),
                ('ecad_vim', 'Cancer_Vim'),
                ('tumor_vim', 'Cancer_Vim'),
                ('tumor_ecad', 'Cancer_Ecad'),
                ('tumor_other_mono', 'Cancer_Mono'),
                ('tumor_sma', 'Cancer_SMA'),
                ('tumor_CD56', 'Cancer_CD56'),
                ('CD4_mono', 'CD4_Mono'),
                ('CD4T_CD8T_dp', 'CD4T_CD8T_DP'),
                ('ChyTry', 'Mast'),
                ('calprotectin', 'Neutrophil'),
                ('immune_other', 'Immune_Other'),
                ('other', 'Other'),
                ('other_stroma_coll', 'Stroma_Collagen'),
                ('other_stroma_fibronectin', 'Stroma_Fibronectin')]

for old_name, new_name in replacements:
    cell_table = cell_table.replace({'cell_meta_cluster': old_name},
                                    {'cell_meta_cluster': new_name})

all_clusters = ['CD11c_HLADR', 'CD14', 'CD163', 'CD20', 'CD31', 'CD31_VIM',
                'CD3_DN', 'CD4T', 'CD4T_CD8T_DP', 'CD4T_HLADR', 'CD4_Mono', 'CD56',
                'CD68', 'CD68_CD163_DP', 'CD8T', 'Cancer_CD56', 'Cancer_CK17',
                'Cancer_Ecad', 'Cancer_Mono', 'Cancer_Other', 'Cancer_SMA',
                'Cancer_Vim', 'FAP', 'FAP_SMA', 'Immune_Other', 'Mast',
                'Neutrophil', 'Other', 'SMA', 'Stroma_Collagen',
                'Stroma_Fibronectin', 'Treg', 'VIM']

# check that all names were converted correctly
assert set(all_clusters) == set(cell_table['cell_meta_cluster'].unique())

assignment_dict = {'Cancer': ['Cancer_CD56', 'Cancer_CK17', 'Cancer_Ecad'],
                   'Cancer_EMT': ['Cancer_SMA', 'Cancer_Vim'],
                   'Cancer_Other': ['Cancer_Other', 'Cancer_Mono'],
                   'M1_Mac': ['CD68'],
                   'M2_Mac': ['CD163'],
                   'Mac_Other': ['CD68_CD163_DP'],
                   'Monocyte': ['CD4_Mono', 'CD14'],
                   'APC': ['CD11c_HLADR'],
                   'B':  ['CD20'],
                   'Endothelium': ['CD31', 'CD31_VIM'],
                   'Fibroblast': ['FAP', 'FAP_SMA', 'SMA'],
                   'Stroma': ['Stroma_Collagen', 'Stroma_Fibronectin', 'VIM'],
                   'NK': ['CD56'],
                   'Neutrophil': ['Neutrophil'],
                   'Mast': ['Mast'],
                   'CD4T': ['CD4T','CD4T_HLADR'],
                   'CD8T': ['CD8T'],
                   'Treg': ['Treg'],
                   'T_Other': ['CD3_DN','CD4T_CD8T_DP'],
                   'Immune_Other': ['Immune_Other'],
                   'Other': ['Other']}

for new_name in assignment_dict:
    pops = assignment_dict[new_name]
    idx = np.isin(cell_table['cell_meta_cluster'].values, pops)
    cell_table.loc[idx,  'cell_cluster'] = new_name

assignment_dict_2 = {'Cancer': ['Cancer', 'Cancer_EMT', 'Cancer_Other'],
                     'Mono_Mac': ['M1_Mac', 'M2_Mac', 'Mac_Other', 'Monocyte', 'APC'],
                     'B': ['B'],
                     'T': ['CD4T', 'CD8T', 'Treg', 'T_Other'],
                     'Granulocyte': ['Neutrophil', 'Mast'],
                     'Stroma': ['Endothelium', 'Fibroblast', 'Stroma'],
                     'NK': ['NK'],
                     'Other': ['Immune_Other', 'Other']}

for new_name in assignment_dict_2:
    pops = assignment_dict_2[new_name]
    idx = np.isin(cell_table['cell_cluster'].values, pops)
    cell_table.loc[idx,  'cell_cluster_broad'] = new_name

# update original cell table
cell_table_full['cell_meta_cluster'] = cell_table['cell_meta_cluster']
cell_table_full['cell_cluster'] = cell_table['cell_cluster']
cell_table_full['cell_cluster_broad'] = cell_table['cell_cluster_broad']

# save updated cell table
cell_table_full.to_csv(os.path.join(analysis_dir, cell_table_name + '_updated.csv'), index=False)

#
# functional marker thresholding
#

# create df to hold thresholded values
cell_table_func = cell_table_full[['fov', 'label', 'cell_cluster_broad', 'cell_cluster', 'cell_meta_cluster']].copy()

threshold_list = [['Ki67', 0.002], ['CD38', 0.004], ['CD45RB', 0.001], ['CD45RO', 0.002],
                  ['CD57', 0.002], ['CD69', 0.002], ['GLUT1', 0.002], ['IDO', 0.001],
                  ['LAG3', 0.002], ['PD1', 0.0005], ['PDL1', 0.001],
                  ['HLA1', 0.001], ['HLADR', 0.001], ['TBET', 0.0015], ['TCF1', 0.001],
                  ['TIM3', 0.001], ['Vim', 0.002], ['Fe', 0.1]]

for marker, threshold in threshold_list:
    cell_table_func[marker] = cell_table_full[marker].values >= threshold


# set specific threshold for PDL1+ dim tumor cells
PDL1_mask = np.logical_and(cell_table_full['PDL1'].values >= 0.0005, cell_table_full['PDL1'].values < 0.001)
tumor_mask = cell_table_full['cell_cluster_broad'] == 'Cancer'
PDL1_cancer_dim_threshold = np.logical_and(PDL1_mask, tumor_mask)

# set threshold for all PDL1+ cells
cell_table_func['PDL1'] = np.logical_or(cell_table_func['PDL1'].values, PDL1_cancer_dim_threshold)


# create ratios of relevant markers

# # first define minimum values for each marker
# H3K9ac_min = np.percentile(cell_table_full['H3K9ac'].values[cell_table_full['H3K9ac'].values > 0], 5)
# H3K27me3_min = np.percentile(cell_table_full['H3K27me3'].values[cell_table_full['H3K27me3'].values > 0], 5)
# CD45RO_min = np.percentile(cell_table_full['CD45RO'].values[cell_table_full['CD45RO'].values > 0], 5)
# CD45RB_min = np.percentile(cell_table_full['CD45RB'].values[cell_table_full['CD45RB'].values > 0], 5)
#
# # save parameters
# marker_min_df = pd.DataFrame({'H3K9ac': H3K9ac_min, 'H3K27me3': H3K27me3_min,
#                               'CD45RO': CD45RO_min, 'CD45RB': CD45RB_min}, index=[0])
# marker_min_df.to_csv(os.path.join(post_processing_dir, 'marker_min_df.csv'))

# load parameters
marker_min_df = pd.read_csv(os.path.join(post_processing_dir, 'marker_min_df.csv'), index_col=0)

H3K9ac_min = marker_min_df['H3K9ac'].values[0]
H3K27me3_min = marker_min_df['H3K27me3'].values[0]
CD45RO_min = marker_min_df['CD45RO'].values[0]
CD45RB_min = marker_min_df['CD45RB'].values[0]

# create masks for cells to include
valid_H3K = np.logical_or(cell_table_full['H3K9ac'].values >= H3K9ac_min,
                            cell_table_full['H3K27me3'].values >= H3K27me3_min)

valid_CD45 = np.logical_or(cell_table_full['CD45RO'].values >= CD45RO_min,
                            cell_table_full['CD45RB'].values >= CD45RB_min)

# compute the ratios
cell_table_func['H3K9ac_H3K27me3_ratio'] = np.log2((cell_table_full['H3K9ac'].values + H3K9ac_min) /
                                                   (cell_table_full['H3K27me3'].values + H3K27me3_min))
cell_table_func['CD45RO_CD45RB_ratio'] = np.log2((cell_table_full['CD45RO'].values + CD45RO_min) /
                                                 (cell_table_full['CD45RB'].values + CD45RB_min))

# set cells with insufficient counts to nan
cell_table_func.loc[~valid_H3K, 'H3K9ac_H3K27me3_ratio'] = np.nan
cell_table_func.loc[~valid_CD45, 'CD45RO_CD45RB_ratio'] = np.nan

cell_table_func.to_csv(os.path.join(analysis_dir, 'cell_table_func_single_positive.csv'), index=False)

# pairwise marker thresholding
functional_markers = [x[0] for x in threshold_list]
for marker1, marker2 in combinations(functional_markers, 2):
    cell_table_func[marker1 + '__' + marker2] = np.logical_and(cell_table_func[marker1],
                                                               cell_table_func[marker2])

cell_table_func.to_csv(os.path.join(analysis_dir, 'cell_table_func_all.csv'), index=False)


# create consolidated cell table with only cell populations
cell_table_clusters = cell_table_full.loc[:, ['fov', 'label', 'cell_meta_cluster', 'cell_cluster', 'cell_cluster_broad']]
cell_table_clusters.to_csv(os.path.join(analysis_dir, 'cell_table_clusters.csv'), index=False)

# create consolidated cell table with only morphology information
morph_features = ['area', 'centroid_dif', 'convex_area', 'convex_hull_resid', 'eccentricity',
                  'equivalent_diameter', 'major_axis_equiv_diam_ratio', 'major_axis_length',
                  'minor_axis_length', 'num_concavities', 'perim_square_over_area', 'perimeter']

morph_features_nuc = [x + '_nuclear' for x in morph_features]

morph_features = morph_features + morph_features_nuc + ['nc_ratio']

cell_table_morph = cell_table_full.loc[:, ['fov', 'label', 'cell_meta_cluster', 'cell_cluster', 'cell_cluster_broad'] + morph_features]
cell_table_morph.to_csv(os.path.join(analysis_dir, 'cell_table_morph.csv'), index=False)

# create consolidated cell table with only marker counts
marker_list = cell_table_full.columns[:84]
marker_list = [x for x in marker_list if '_nuclear' not in x]
cell_table_counts = cell_table_full.loc[:, ['fov', 'label', 'cell_meta_cluster', 'cell_cluster', 'cell_cluster_broad'] + marker_list]

cell_table_counts.to_csv(os.path.join(analysis_dir, 'cell_table_counts.csv'), index=False)


# # Update labels for mislabeled FOVs
# original_labels = [
#  'TONIC_TMA1_R1C2',
#  'TONIC_TMA1_R1C3',
#  'TONIC_TMA1_R4C1',
#  'TONIC_TMA1_R4C2',
#  'TONIC_TMA1_R4C3',
#  'TONIC_TMA1_R5C1',
#  'TONIC_TMA1_R5C2',
#  'TONIC_TMA1_R5C3',
#  'TONIC_TMA1_R5C4',
#  'TONIC_TMA1_R5C5',
#  'TONIC_TMA1_R6C1',
#  'TONIC_TMA1_R6C2',
#  'TONIC_TMA1_R6C3',
#  'TONIC_TMA1_R7C2',
#  'TONIC_TMA1_R7C3',
#  'TONIC_TMA1_R7C4',
#  'TONIC_TMA1_R7C5',
#  'TONIC_TMA1_R7C6',
#  'TONIC_TMA1_R8C1',
#  'TONIC_TMA1_R8C3',
#  'TONIC_TMA1_R8C4',
#  'TONIC_TMA1_R8C5',
#  'TONIC_TMA1_R10C2',
#  'TONIC_TMA1_R10C3']
#
# # each row is increased by 3
# new_labels = [
#     'TONIC_TMA1_R4C2',
#     'TONIC_TMA1_R4C3',
#     'TONIC_TMA1_R7C1',
#     'TONIC_TMA1_R7C2',
#     'TONIC_TMA1_R7C3',
#     'TONIC_TMA1_R8C1',
#     'TONIC_TMA1_R8C2',
#     'TONIC_TMA1_R8C3',
#     'TONIC_TMA1_R8C4',
#     'TONIC_TMA1_R8C5',
#     'TONIC_TMA1_R9C1',
#     'TONIC_TMA1_R9C2',
#     'TONIC_TMA1_R9C3',
#     'TONIC_TMA1_R10C2',
#     'TONIC_TMA1_R10C3',
#     'TONIC_TMA1_R10C4',
#     'TONIC_TMA1_R10C5',
#     'TONIC_TMA1_R10C6',
#     'TONIC_TMA1_R11C1',
#     'TONIC_TMA1_R11C3',
#     'TONIC_TMA1_R11C4',
#     'TONIC_TMA1_R11C5',
#     'TONIC_TMA1_R13C2',
#     'TONIC_TMA1_R13C3']
#
# # rename image folders. Go in reverse order to avoid overwriting
# image_dir = os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples')
#
# for original_label, new_label in zip(original_labels[::-1], new_labels[::-1]):
#     os.rename(os.path.join(image_dir, original_label), os.path.join(image_dir, new_label))
#
# # update cell table
# cell_table_full = pd.read_csv(os.path.join(data_dir, cell_table_name + '_updated.csv'))
#
# for original_label, new_label in zip(original_labels[::-1], new_labels[::-1]):
#     cell_table_full.loc[cell_table_full['fov'] == original_label, 'fov'] = new_label
#
# cell_table_full.to_csv(os.path.join(data_dir, cell_table_name + '_updated.csv'), index=False)
#
#
# # update csv files
# csv_files = ['/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/post_processing/cell_annotation_mask.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/post_processing/fov_annotation_mask_area.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/ecm/ecm_fraction_fov.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/fiber_segmentation_processed_data/fiber_object_table.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/fiber_segmentation_processed_data/fiber_stats_table.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/fiber_segmentation_processed_data/tile_stats_512/fiber_stats_table-tile_512.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/neighborhood_mats/neighborhood_counts-cell_cluster_broad_radius50.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/neighborhood_mats/neighborhood_counts-cell_cluster_radius50.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/neighborhood_mats/neighborhood_counts-cell_meta_cluster_radius50.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/neighborhood_mats/neighborhood_freqs-cell_cluster_broad_radius50.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/neighborhood_mats/neighborhood_freqs-cell_cluster_radius50.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/neighborhood_mats/neighborhood_freqs-cell_meta_cluster_radius50.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/cell_neighbor_analysis/cell_cluster_broad_avg_dists-nearest_1.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/cell_neighbor_analysis/cell_cluster_broad_avg_dists-nearest_3.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/cell_neighbor_analysis/cell_cluster_broad_avg_dists-nearest_5.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/cell_neighbor_analysis/neighborhood_diversity_radius50.csv',
#              '/Volumes/Shared/Noah Greenwald/ecm_pixel_clustering/fov_pixel_cluster_counts.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/ecm/fov_cluster_counts.csv',
#              '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/mixing_score/cell_cluster_broad/homogeneous_mixing_scores.csv',]
#
# for file in csv_files:
#     df = pd.read_csv(file)
#     for original_label, new_label in zip(original_labels[::-1], new_labels[::-1]):
#         df.loc[df['fov'] == original_label, 'fov'] = new_label
#     df.to_csv(file, index=False)
#
#
# # update folders
# folders = ['/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/ecm/masks',
#            '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/fiber_segmentation_processed_data/tile_stats_512',
#            '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/mask_dir/individual_masks',
#            '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/mask_dir/intermediate_masks',
#            ]
#
# # update images
# '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/fiber_segmentation_processed_data/tile_tiffs'
#
# '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/spatial_analysis/dist_mats'