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
functional_df_core = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core.csv'))


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


# list to hold each fov, metric, value dataframe
fov_data = []

#
# Immune related features
#

# CD4/CD8 ratio
input_df = cluster_df_core[cluster_df_core['metric'].isin(['cluster_freq'])]
CD4_CD8_ratio = compute_celltype_ratio(input_data=input_df, celltype_1='CD4T', celltype_2='CD8T')
CD4_CD8_ratio['metric'] = 'CD4_CD8_ratio'
CD4_CD8_ratio['category'] = 'immune'
fov_data.append(CD4_CD8_ratio)

# M1/M2 ratio
input_df = cluster_df_core[cluster_df_core['metric'].isin(['cluster_freq'])]
M1_M2_ratio = compute_celltype_ratio(input_data=input_df, celltype_1='M1_Mac', celltype_2='M2_Mac')
M1_M2_ratio['metric'] = 'M1_M2_ratio'
M1_M2_ratio['category'] = 'immune'
fov_data.append(M1_M2_ratio)

# Lymphoid/Myeloid ratio
input_df = cluster_df_core[cluster_df_core['metric'].isin(['cluster_broad_freq'])]
Lymphoid_Myeloid_ratio = compute_celltype_ratio(input_data=input_df, celltype_1=['B', 'T'],
                                                celltype_2=['Mono_Mac', 'Granulocyte'])
Lymphoid_Myeloid_ratio['metric'] = 'Myeloid_Lymphoid_ratio'
Lymphoid_Myeloid_ratio['category'] = 'immune'
fov_data.append(Lymphoid_Myeloid_ratio)

# Treg proportion T cells
input_df = cluster_df_core[cluster_df_core['metric'].isin(['tcell_freq'])]
input_df = input_df[input_df['cell_type'].isin(['Treg'])]
input_df['metric'] = 'Treg_Tcell_prop'
input_df['category'] = 'immune'
input_df = input_df[['fov', 'value', 'metric', 'category']]
fov_data.append(input_df)

# Treg proportion immune cells
input_df = cluster_df_core[cluster_df_core['metric'].isin(['immune_freq'])]
input_df = input_df[input_df['cell_type'].isin(['Treg'])]
input_df['metric'] = 'Treg_immune_prop'
input_df['category'] = 'immune'
input_df = input_df[['fov', 'value', 'metric', 'category']]
fov_data.append(input_df)

# Tcell proportion immune cells
input_df = cluster_df_core[cluster_df_core['metric'].isin(['immune_freq'])]
input_df = input_df[input_df['cell_type'].isin(['CD4T', 'CD8T', 'Treg', 'T_Other'])]
input_df['metric'] = 'Tcell_immune_prop'
input_df['category'] = 'immune'
input_df = input_df[['fov', 'value', 'metric', 'category']]
fov_data.append(input_df)

# Diversity of immune cell types
input_df = cluster_df_core[cluster_df_core['metric'].isin(['immune_freq'])]
wide_df = pd.pivot(input_df, index='fov', columns=['cell_type'], values='value')
wide_df['value'] = wide_df.apply(shannon_diversity, axis=1)
wide_df.reset_index(inplace=True)
wide_df['metric'] = 'immune_diversity'
wide_df['category'] = 'immune'
wide_df = wide_df[['fov', 'value', 'metric', 'category']]
fov_data.append(wide_df)

# functional markers in Tregs
markers = ['Ki67', 'PD1']
for marker in markers:
    input_df = functional_df_core[functional_df_core['metric'].isin(['avg_per_cluster'])]
    input_df = input_df[input_df['cell_type'].isin(['Treg'])]
    input_df = input_df[input_df['functional_marker'].isin([marker])]
    input_df['metric'] = f'{marker}_Treg'
    input_df['category'] = 'immune'
    input_df = input_df[['fov', 'value', 'metric', 'category']]
    fov_data.append(input_df)

# functional markers in CD8s
markers = ['Ki67', 'PD1',  'TBET', 'TCF1', 'CD69', 'TIM3']
for marker in markers:
    input_df = functional_df_core[functional_df_core['metric'].isin(['avg_per_cluster'])]
    input_df = input_df[input_df['cell_type'].isin(['CD8T'])]
    input_df = input_df[input_df['functional_marker'].isin([marker])]
    input_df['metric'] = f'{marker}_CD8T'
    input_df['category'] = 'immune'
    input_df = input_df[['fov', 'value', 'metric', 'category']]
    fov_data.append(input_df)

# functional markers in macrophages
markers = ['IDO', 'TIM3', 'PDL1']
for marker in markers:
    input_df = functional_df_core[functional_df_core['metric'].isin(['avg_per_cluster_broad'])]
    input_df = input_df[input_df['cell_type'].isin(['Mono_Mac'])]
    input_df = input_df[input_df['functional_marker'].isin([marker])]
    input_df['metric'] = f'{marker}_Mono_Mac'
    input_df['category'] = 'immune'
    input_df = input_df[['fov', 'value', 'metric', 'category']]
    fov_data.append(input_df)

#
# stromal features
#

# functional markers in fibroblasts
markers = ['HLADR', 'IDO', 'PDL1', 'Ki67', 'GLUT1']
for marker in markers:
    input_df = functional_df_core[functional_df_core['metric'].isin(['avg_per_cluster_broad'])]
    input_df = input_df[input_df['cell_type'].isin(['Stroma'])]
    input_df = input_df[input_df['functional_marker'].isin([marker])]
    input_df['metric'] = f'{marker}_Stroma'
    input_df['category'] = 'stromal'
    input_df = input_df[['fov', 'value', 'metric', 'category']]
    fov_data.append(input_df)


#
# cancer features
#

# cancer cell proportions
cancer_populations = ['Cancer_CD56', 'Cancer_CK17', 'Cancer_Ecad', 'Cancer_SMA', 'Cancer_Vim',
                      'Cancer_Other', 'Cancer_Mono']

for cancer_population in cancer_populations:
    input_df = cluster_df_core[cluster_df_core['metric'].isin(['cancer_freq'])]
    input_df = input_df[input_df['cell_type'].isin([cancer_population])]
    input_df['metric'] = f'{cancer_population}_cancer_prop'
    input_df['category'] = 'cancer'
    input_df = input_df[['fov', 'value', 'metric', 'category']]
    fov_data.append(input_df)

# cancer diversity
input_df = cluster_df_core[cluster_df_core['metric'].isin(['cancer_freq'])]
wide_df = pd.pivot(input_df, index='fov', columns=['cell_type'], values='value')
wide_df['value'] = wide_df.apply(shannon_diversity, axis=1)
wide_df.reset_index(inplace=True)
wide_df['metric'] = 'cancer_diversity'
wide_df['category'] = 'cancer'
wide_df = wide_df[['fov', 'value', 'metric', 'category']]
fov_data.append(wide_df)


# functional markers in cancer cells
markers = ['PDL1', 'PDL1_cancer_dim', 'GLUT1', 'Ki67', 'HLA1', 'HLADR']
for marker in markers:
    input_df = functional_df_core[functional_df_core['metric'].isin(['avg_per_cluster_broad'])]
    input_df = input_df[input_df['cell_type'].isin(['Cancer'])]
    input_df = input_df[input_df['functional_marker'].isin([marker])]
    input_df['metric'] = f'{marker}_Cancer'
    input_df['category'] = 'cancer'
    input_df = input_df[['fov', 'value', 'metric', 'category']]
    fov_data.append(input_df)

#
# global features
#

# immune infiltration
immune_df = cluster_df_core.loc[(cluster_df_core.metric == 'cluster_broad_freq') &
                                (cluster_df_core.cell_type.isin(
                                    ['Mono_Mac', 'B', 'T', 'Granulocyte', 'NK'])), :]
immune_df = immune_df.loc[:, ['fov', 'value']]
immune_grouped = immune_df.groupby('fov').agg(np.sum)
immune_grouped.reset_index(inplace=True)
immune_grouped['metric'] = 'immune_infiltration'
immune_grouped['category'] = 'global'
fov_data.append(immune_grouped)



# combine all dfs together, add Tissue_ID metadata
fov_data_df = pd.concat(fov_data)
temp_metadata = cluster_df_core[cluster_df_core.metric == 'cluster_freq'][['fov', 'Tissue_ID', 'Timepoint']]
temp_metadata = temp_metadata.drop_duplicates()

fov_data_df = fov_data_df.merge(temp_metadata, on='fov', how='left')
fov_data_df = fov_data_df[fov_data_df.Timepoint.isin(['primary_untreated'])]

# convert to wide format for plotting in seaborn clustermap
wide_df = fov_data_df.pivot(index='fov', columns='metric', values='value')

# replace Nan with 0
wide_df = wide_df.fillna(0)

sns.clustermap(wide_df, z_score=1)

# same thing for timepoint aggregation
timepoint_data_df = fov_data_df.groupby(['Tissue_ID', 'metric']).agg(np.mean)
timepoint_data_df.reset_index(inplace=True)
timepoint_data_df = timepoint_data_df.pivot(index='Tissue_ID', columns='metric', values='value')

# replace Nan with 0
timepoint_data_df = timepoint_data_df.fillna(0)

sns.clustermap(timepoint_data_df, z_score=1, cmap='vlag', vmin=-3, vmax=3)



# create comprehensive features for all major and minor cell types
fov_data = []

# compute diversity of different levels of granularity
diversity_features = [['cluster_broad_freq', 'cluster_broad_diversity', 'broad'],
                      ['immune_freq', 'immune_diversity', 'immune'],
                      ['cancer_freq', 'cancer_diversity', 'cancer'],
                      ['stromal_freq', 'stromal_diversity', 'stromal']]

for cluster_name, feature_name, feature_category in diversity_features:
    input_df = cluster_df_core[cluster_df_core['metric'].isin([cluster_name])]
    wide_df = pd.pivot(input_df, index='fov', columns=['cell_type'], values='value')
    wide_df['value'] = wide_df.apply(shannon_diversity, axis=1)
    wide_df.reset_index(inplace=True)
    wide_df['metric'] = feature_name
    wide_df['category'] = feature_category
    wide_df = wide_df[['fov', 'value', 'metric', 'category']]
    fov_data.append(wide_df)


# compute proportions of cell types for different levels of granularity
proportion_features = [['cluster_broad_freq', 'cluster_broad_prop', 'broad'],
                       #['cluster_freq', 'cluster_prop', 'broad'],
                       ['meta_cluster_freq', 'meta_cluster_prop', 'broad']]
for cluster_name, feature_name, feature_category in proportion_features[2:]:
    input_df = cluster_df_core[cluster_df_core['metric'].isin([cluster_name])]
    input_df['metric'] = input_df.cell_type + '_' + feature_name
    input_df['category'] = feature_category
    input_df = input_df[['fov', 'value', 'metric', 'category']]
    fov_data.append(input_df)


# compute functional marker positivity for different levels of granularity
functional_features = [['cluster_freq', 'broad']]
# functional_features = [['avg_per_cluster_broad', 'broad'],
#                           ['avg_per_cluster', 'broad']]
for functional_name, feature_category in functional_features:
    input_df = functional_df_core[functional_df_core['metric'].isin([functional_name])]
    input_df['metric'] = input_df.functional_marker + '+_' + input_df.cell_type
    input_df['category'] = feature_category

    # create vector of False values
    keep_vector = np.zeros(input_df.shape[0], dtype=bool)
    for marker in keep_dict:
        marker_keep = (input_df['cell_type'].isin(keep_dict[marker])) & (input_df['functional_marker'] == marker)
        keep_vector = keep_vector | marker_keep

    input_df = input_df[keep_vector]
    input_df = input_df[['fov', 'value', 'metric', 'category']]
    fov_data.append(input_df)

fov_data_df = pd.concat(fov_data)
temp_metadata = cluster_df_core[cluster_df_core.metric == 'cluster_freq'][['fov', 'Tissue_ID', 'Timepoint']]
temp_metadata = temp_metadata.drop_duplicates()
fov_data_df = fov_data_df.merge(temp_metadata, on='fov', how='left')

fov_data_df.to_csv(os.path.join(data_dir, 'fov_features.csv'), index=False)


# create dictionary of functional markers to keep for each cell type
lymphocyte = ['B', 'CD4T', 'CD8T', 'Immune_Other', 'NK', 'T_Other', 'Treg']
cancer = ['Cancer', 'Cancer_EMT', 'Cancer_Other']
monocyte = ['APC', 'M1_Mac', 'M2_Mac', 'Mono_Mac', 'Monocyte', 'Mac_Other']
stroma = ['Fibroblast', 'Stroma', 'Endothelium']
granulocyte = ['Mast', 'Neutrophil']

keep_dict = {'CD38': lymphocyte + monocyte + stroma + granulocyte, 'CD45RB': lymphocyte, 'CD45RO': lymphocyte,
             'CD57': lymphocyte + cancer, 'CD69': lymphocyte,
             'GLUT1': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'HLA1': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'HLADR': lymphocyte + monocyte, 'IDO': ['APC', 'B'], 'Ki67': lymphocyte + monocyte + stroma + granulocyte + cancer,
             'PD1': lymphocyte, 'PDL1': lymphocyte + monocyte + granulocyte + cancer, 'PDL1_tumor_dim': cancer,
             'TBET': lymphocyte, 'TCF1': lymphocyte, 'TIM3': lymphocyte + monocyte + granulocyte}

# create mantis_dir to inspect invididual FOVs
func_df = pd.read_csv(os.path.join(data_dir, 'combined_cell_table_normalized_cell_labels_updated_functional_only.csv'))

data_wide.loc[data_wide['GLUT1+_Cancer'] > 0.25, ['GLUT1+_M2_Mac', 'GLUT1+_Fibroblast', 'GLUT1+_Cancer']]
fovs = data_wide.loc[data_wide['HLA1+_Treg'] > 0.5, :].index.tolist()

# select 5th and 9th fov
fovs = fovs[0:2] + fovs[5:7] + fovs[9:11]
fovs.remove('TONIC_TMA23_R10C1')

fovs = ['TONIC_TMA9_R6C3', 'TONIC_TMA9_R7C3', 'TONIC_TMA10_R11C6', 'TONIC_TMA10_R5C5', 'TONIC_TMA16_R3C4', 'TONIC_TMA17_R12C3', 'TONIC_TMA3_R2C4']
fovs = []

marker = 'GLUT1'
cell_types = ['Cancer', 'Fibroblast', 'Cancer_EMT', 'Cancer_Other', 'M2_Mac']
input_cell_table = func_df.copy()
input_cell_table['cell_cluster_new'] = np.where(func_df.cell_cluster.isin(cell_types), func_df.cell_cluster, 'Other')
input_cell_table['cell_cluster_new'] = input_cell_table['cell_cluster_new'].values + input_cell_table[marker].astype('str')
input_cell_table['cell_cluster_new'] = np.where(input_cell_table['PDL1_tumor_dim'].values, input_cell_table['cell_cluster'] + input_cell_table['PDL1_tumor_dim'].astype('str') + '_dim', input_cell_table['cell_cluster_new'])

keep_channels = ['CD38.tiff']
create_mantis_project(cell_table=input_cell_table, fovs=fovs, seg_dir='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output',
                      pop_col='cell_cluster_new', mask_dir='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/mantis_dir/masks',
                      image_dir='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples', mantis_dir='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/mantis_dir/mantis_folders')
all_chans = io_utils.list_files('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/TONIC_TMA3_R4C2', '.tiff')
# TONIC_TMA2_R8C2
for chan in all_chans:
    if chan not in keep_channels:
        for fov in fovs:
            path = os.path.join('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/mantis_dir/mantis_folders', fov, chan)
            if os.path.exists(path):
                os.remove(path)

from ark.utils import data_utils, plot_utils, load_utils, io_utils
from ark.utils.misc_utils import verify_in_list
from ark import settings

import skimage.io as io


def label_cells_by_cluster(fov, all_data, label_map, fov_col=settings.FOV_ID,
                           cell_label_column=settings.CELL_LABEL,
                           cluster_column=settings.KMEANS_CLUSTER):
    """Translates cell-ID labeled images according to the clustering assignment.
    Takes a single FOV, and relabels the image according to the assignment
    of cell IDs to cluster label.
    Args:
        fov (str):
            The FOV to relabel
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers.
        label_map (xarray.DataArray):
            label map for a single FOV
        fov_col (str):
            column with the fovs names in `all_data`.
        cell_label_column (str):
            column with the cell labels in `all_data`.
        cluster_column (str):
            column with the cluster labels in `all_data`.
    Returns:
        numpy.ndarray:
            The image with new designated label assignments
    """

    # verify that fov found in all_data
    # NOTE: label_map fov validation happens in loading function
    verify_in_list(fov_name=[fov], all_data_fovs=all_data[fov_col].unique())

    # subset all_data on the FOV
    df = all_data[all_data[fov_col] == fov]

    # generate the labels to use
    labels_dict = dict(zip(df[cell_label_column], df[cluster_column]))

    # condense extraneous axes
    labeled_img_array = label_map.squeeze().values

    # relabel the array
    relabeled_img_array = data_utils.relabel_segmentation(labeled_img_array, labels_dict)

    return relabeled_img_array


def save_fov_mask(fov, data_dir, mask_data, sub_dir=None, name_suffix=''):
    """Saves a provided cluster label mask overlay for a FOV.
    Args:
        fov (str):
            The FOV to save
        data_dir (str):
            The directory to save the cluster mask
        mask_data (numpy.ndarray):
            The cluster mask data for the FOV
        sub_dir (Optional[str]):
            The subdirectory to save the masks in. If specified images are saved to
            "data_dir/sub_dir". If `sub_dir = None` the images are saved to `"data_dir"`.
            Defaults to `None`.
        name_suffix (str):
            Specify what to append at the end of every fov.
    """

    # data_dir validation
    io_utils.validate_paths(data_dir)

    # ensure None is handled correctly in file path generation
    if sub_dir is None:
        sub_dir = ''

    save_dir = os.path.join(data_dir, sub_dir)

    # make the save_dir if it doesn't already exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # define the file name as the fov name with the name suffix appended
    fov_file = fov + name_suffix + '.tiff'

    # save the image to data_dir
    io.imsave(os.path.join(save_dir, fov_file), mask_data, check_contrast=False)

def create_mantis_project(cell_table, fovs, seg_dir, pop_col,
                          mask_dir, image_dir, mantis_dir) -> None:
    """Create a complete Mantis project for viewing cell labels
    Args:
        cell_table (pd.DataFrame): dataframe of extracted cell features and subtypes
        fovs (list): list of FOVs to use for creating the project
        seg_dir (path): path to the directory containing the segmentations
        pop_col (str): the column containing the distinct cell populations
        mask_dir (path): path to the directory where the masks will be stored
        image_dir (path): path to the directory containing the raw image data
        mantis_dir (path): path to the directory where the mantis project will be created
        seg_suffix_name (str, optional):
            The suffix of the segmentation file. Defaults to "_whole_cell".
    """

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # create small df compatible with FOV function
    small_table = cell_table.loc[:, [pop_col, 'label', 'fov']]

    # generate unique numeric value for each population
    small_table['pop_vals'] = pd.factorize(small_table[pop_col].tolist())[0] + 1

    # label and save the cell mask for each FOV
    for fov in fovs:
        whole_cell_file = [fov + '_feature_0.tiff' for fov in fovs]
        print(whole_cell_file)
        # load the segmentation labels in for the FOV
        label_map = load_utils.load_imgs_from_dir(
            data_dir=seg_dir, files=whole_cell_file, xr_dim_name='compartments',
            xr_channel_names=['feature_0'], trim_suffix='_feature_0'
        ).loc[fov, ...]

        # use label_cells_by_cluster to create cell masks
        mask_data = label_cells_by_cluster(
            fov, small_table, label_map, fov_col='fov',
            cell_label_column='label', cluster_column='pop_vals'
        )

        # save the cell mask for each FOV
        save_fov_mask(
            fov,
            mask_dir,
            mask_data,
            sub_dir=None,
            name_suffix='_cell_mask'
        )

    # rename the columns of small_table
    mantis_df = small_table.rename({'pop_vals': 'metacluster', pop_col: 'mc_name'}, axis=1)

    # create the mantis project
    plot_utils.create_mantis_dir(fovs=fovs, mantis_project_path=mantis_dir,
                                 img_data_path=image_dir, mask_output_dir=mask_dir,
                                 mask_suffix='_cell_mask', mapping=mantis_df,
                                 seg_dir=seg_dir, img_sub_folder='')

