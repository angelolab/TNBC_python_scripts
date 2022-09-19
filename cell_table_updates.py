import pandas as pd
import numpy as np

from ark.phenotyping.post_cluster_utils import plot_hist_thresholds, create_mantis_project
from ark.utils.io_utils import list_folders


cell_table = pd.read_csv('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/combined_cell_table_normalized_cell_labels.csv')

x = ['CD11c_HLADR', 'CD14', 'CD163', 'CD20', 'CD31', 'CD31_VIM', 'CD3_DN', 'CD3_noise_split',
     'CD45', 'CD4T', 'CD4T_CD8T_dp', 'CD4T_HLADR', 'CD4_mono', 'CD56', 'CD68', 'CD68_CD163_DP', 'CD8T',
     'CD8_CD8Tdim', 'ChyTry', 'FAP', 'FAP_SMA', 'FOXP3_CD45_split', 'SMA', 'Treg', 'VIM',
     'calprotectin', 'cd56_dirty', 'ck17_tumor', 'ecad_hladr', 'ecad_vim', 'noise',
     'other_stroma_coll','other_stroma_fibronectin', 'tumor_CD56', 'tumor_ecad', 'tumor_other_mono']


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



# update cell table with post-inspection decisions
cell_table.loc[cell_table['cell_meta_cluster'] == 'noise', 'cell_meta_cluster'] = 'tumor_other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD8_CD8Tdim', 'cell_meta_cluster'] = 'CD8T'
cell_table.loc[cell_table['cell_meta_cluster'] == 'noise_cd45_pos', 'cell_meta_cluster'] = 'immune_other'
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
cell_table.loc[cell_table['cell_meta_cluster'] == 'FOXP3_CD45_split_pos', 'cell_meta_cluster'] = 'CD45'
cell_table.loc[cell_table['cell_meta_cluster'] == 'CD68_CD163_DP_68neg', 'cell_meta_cluster'] = 'CD163'
cell_table.loc[cell_table['cell_meta_cluster'] == 'FOXP3_CD45_split', 'cell_meta_cluster'] = 'other'
cell_table.loc[cell_table['cell_meta_cluster'] == 'tumor_ecad_sma', 'cell_meta_cluster'] = 'tumor_sma'
cell_table.loc[cell_table['cell_meta_cluster'] == 'SMA_ECAD', 'cell_meta_cluster'] = 'tumor_sma'
cell_table.loc[cell_table['cell_meta_cluster'] == 'ck17_tumor_sma', 'cell_meta_cluster'] = 'tumor_sma'

fovs = list_folders('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/mantis')


create_mantis_project(cell_table, fovs=fovs, seg_dir='/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/segmentation_masks',
                      pop_col='cell_meta_cluster', mask_dir='/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/masks',
                      image_dir='/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/mantis',
                      mantis_dir='/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/mantis')

# generate consistent names for cell clusters
cell_table.loc[cell_table['cell_meta_cluster'] == 'ck17_tumor', 'cell_meta_cluster'] = 'tumor_ck17'
cell_table.loc[cell_table['cell_meta_cluster'] == 'ecad_vim', 'cell_meta_cluster'] = 'tumor_vim'

all_clusters = ['CD11c_HLADR', 'CD14', 'CD163', 'CD20', 'CD31', 'CD31_VIM',
                'CD3_DN', 'CD4T', 'CD4T_CD8T_dp', 'CD4T_HLADR', 'CD4_mono', 'CD56',
                'CD68', 'CD68_CD163_DP', 'CD8T', 'ChyTry', 'FAP', 'FAP_SMA', 'SMA',
                'Treg', 'VIM', 'calprotectin', 'immune_other', 'other',
                'other_stroma_coll', 'other_stroma_fibronectin', 'tumor_CD56',
                'tumor_ck17', 'tumor_ecad', 'tumor_other', 'tumor_other_mono',
                'tumor_sma', 'tumor_vim']

assignment_dict = {'tumor': ['tumor_CD56', 'tumor_ck17', 'tumor_ecad'],
                   'tumor_emt': ['tumor_sma', 'tumor_vim'],
                   'tumor_other': ['tumor_other', 'tumor_other_mono'],
                   'macs': ['CD68', 'CD68_CD163_DP', 'CD163'],
                   'mono': ['CD4_mono', 'CD14'],
                   'apc': ['CD11c_HLADR'],
                   'bcell':  ['CD20'],
                   'endo': ['CD31', 'CD31_VIM'],
                   'fibro': ['FAP', 'FAP_SMA', 'SMA'],
                   'stroma': ['other_stroma_coll', 'other_stroma_fibronectin', 'VIM'],
                   'nk': ['CD56'],
                   'neut': ['calprotectin'],
                   'mast': ['ChyTry'],
                   'CD4T': ['CD4T','CD4T_HLADR'],
                   'CD8T': ['CD8T'],
                   'treg': ['Treg'],
                   't_other': ['CD3_DN','CD4T_CD8T_dp'],
                   'immune': ['immune_other'],
                   'other': ['other']}

for new_name in assignment_dict:
    pops = assignment_dict[new_name]
    idx = np.isin(cell_table['cell_meta_cluster'].values, pops)
    cell_table.loc[idx,  'cell_cluster'] = new_name

assignment_dict_2 = {'tumor': ['tumor', 'tumor_emt', 'tumor_other'],
                     'mono_macs': ['macs', 'mono', 'apc'],
                     'b_cell': ['b_cell'],
                     't_cell': ['CD4', 'CD8', 'treg', 't_other'],
                     'granulocyte': ['neutrophil', 'mast'],
                     'stroma': ['endo', 'fibro', 'stroma'],
                     'nk': ['nk'],
                     'other': ['immune', 'other']}

for new_name in assignment_dict_2:
    pops = assignment_dict_2[new_name]
    idx = np.isin(cell_table['cell_cluster'].values, pops)
    cell_table.loc[idx,  'cell_cluster_broad'] = new_name

# save updated cell table
cell_table.to_csv('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/combined_cell_table_normalized_cell_labels_updated.csv')
cell_table = pd.read_csv('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/combined_cell_table_normalized_cell_labels_updated.csv')
cell_table_testing = cell_table.loc[cell_table.fov.isin(fovs), :]
cell_table_testing.to_csv('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/combined_cell_table_normalized_cell_labels_updated_testing.csv')


# threshold_list = [['Ki67', 0.002], ['CD38', 0.002], ['CD45RB', 0.001], ['CD45RO', 0.002],
#                   ['CD57', 0.002], ['CD69', 0.002], ['GLUT1', 0.002], ['IDO', 0.001],
#                   ['PD1', 0.0005], ['PDL1', 0.0005, "tumors could use either threshold", 0.001],
#                   ['HLA1', 0.001], ['HLADR', 0.001], ['TBET', 0.0015], ['TCF1', 0.001],
#                   ['TIM3', 0.001]]

# create dataframe with counts of the specified markers
marker_counts_df = cell_table_testing.loc[:, ['fov', 'label'] + ['Ki67', 'CD38', 'CD45RB', 'CD45RO', 'CD57',
                                                                 'CD69', 'GLUT1', 'IDO', 'PD1', 'PDL1', 'HLA1', 'HLADR', 'TBET',
                                                                 'TCF1', 'TIM3']]

# save dataframe
marker_counts_df.to_csv('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/example_output/mantis/marker_counts.csv', index=False)