import os
import pandas as pd

from ark.utils.io_utils import list_folders

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'

# create consolidated cell table with only cell populations
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated.csv'))

cell_table = cell_table.loc[:, ['fov', 'cell_meta_cluster', 'label', 'cell_cluster',
                             'cell_cluster_broad']]
cell_table.to_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_clusters_only.csv'),
                  index=False)

# create consolidated cell table with only functional marker freqs
cell_table = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated.csv'))

func_cols = [col for col in cell_table.columns if '_threshold' in col]
cell_table_func = cell_table.loc[:, ['fov', 'cell_cluster_broad', 'cell_cluster', 'cell_meta_cluster'] + func_cols]
cell_table_func.columns = [col.split('_threshold')[0] for col in cell_table_func.columns]
cell_table_func.to_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'),
                       index=False)


# add column denoting which images were acquired
all_fovs = list_folders('/Volumes/Big_Boy/TONIC_Cohort/image_data/samples')
core_df['MIBI_data'] = core_df['fov'].isin(all_fovs)
core_df.to_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))
