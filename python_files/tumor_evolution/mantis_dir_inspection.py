from ark.utils import data_utils, plot_utils, load_utils, io_utils
from ark.utils.misc_utils import verify_in_list
from ark import settings

import skimage.io as io
import natsort
import shutil
import os
import numpy as np
import pandas as pd
import pathlib
from typing import List, Union
from operator import contains


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


def create_mantis_dir(fovs, mantis_project_path,img_data_path,mask_output_dir,mapping,seg_dir,
                      keep_channels, mask_suffix="_mask",
                      seg_suffix_name = "_feature_0",
                      img_sub_folder = "",):
    """Creates a mantis project directory so that it can be opened by the mantis viewer.
    Copies fovs, segmentation files, masks, and mapping csv's into a new directory structure.
    Here is how the contents of the mantis project folder will look like.
    ```{code-block} sh
    mantis_project
    ├── fov0
    │   ├── cell_segmentation.tiff
    │   ├── chan0.tiff
    │   ├── chan1.tiff
    │   ├── chan2.tiff
    │   ├── ...
    │   ├── population_mask.csv
    │   └── population_mask.tiff
    └── fov1
    │   ├── cell_segmentation.tiff
    │   ├── chan0.tiff
    │   ├── chan1.tiff
    │   ├── chan2.tiff
    │   ├── ...
    │   ├── population_mask.csv
    │   └── population_mask.tiff
    └── ...
    ```
    Args:
        fovs (List[str]):
            A list of FOVs to create a Mantis Project for.
        mantis_project_path (Union[str, pathlib.Path]):
            The folder where the mantis project will be created.
        img_data_path (Union[str, pathlib.Path]):
            The location of the all the fovs you wish to create a project from.
        mask_output_dir (Union[str, pathlib.Path]):
            The folder containing all the masks of the fovs.
        mapping (Union[str, pathlib.Path, pd.DataFrame]):
            The location of the mapping file, or the mapping Pandas DataFrame itself.
        seg_dir (Union[str, pathlib.Path]):
            The location of the segmentation directory for the fovs.
        mask_suffix (str, optional):
            The suffix used to find the mask tiffs. Defaults to "_mask".
        seg_suffix_name (str, optional):
            The suffix of the segmentation file. Defaults to "_whole_cell".
        img_sub_folder (str, optional):
            The subfolder where the channels exist within the `img_data_path`.
            Defaults to "normalized".
    """

    if not os.path.exists(mantis_project_path):
        os.makedirs(mantis_project_path)

    # create key from cluster number to cluster name
    if type(mapping) in {pathlib.Path, str}:
        map_df = pd.read_csv(mapping)
    elif type(mapping) is pd.DataFrame:
        map_df = mapping
    else:
        ValueError("Mapping must either be a path to an already saved mapping csv, \
                   or a DataFrame that is already loaded in.")

    map_df = map_df.loc[:, ['metacluster', 'mc_name']]
    # remove duplicates from df
    map_df = map_df.drop_duplicates()
    map_df = map_df.sort_values(by=['metacluster'])

    # rename for mantis names
    map_df = map_df.rename({'metacluster': 'region_id', 'mc_name': 'region_name'}, axis=1)

    # get names of fovs with masks
    mask_names_loaded = (io_utils.list_files(mask_output_dir, mask_suffix))
    mask_names_delimited = io_utils.extract_delimited_names(mask_names_loaded,
                                                            delimiter=mask_suffix)
    mask_names_sorted = natsort.natsorted(mask_names_delimited)

    # use `fovs`, a subset of the FOVs in `total_fov_names` which
    # is a list of FOVs in `img_data_path`
    fovs = natsort.natsorted(fovs)
    verify_in_list(fovs=fovs, img_data_fovs=mask_names_delimited)

    # Filter out the masks that do not have an associated FOV.
    mask_names = filter(lambda mn: any(contains(mn, f) for f in fovs), mask_names_sorted)

    # create a folder with image data, pixel masks, and segmentation mask
    for fov, mn in zip(fovs, mask_names):
        # set up paths
        img_source_dir = os.path.join(img_data_path, fov, img_sub_folder)
        output_dir = os.path.join(mantis_project_path, fov)

        # copy image data if not already copied in from previous round of clustering
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

            # copy all channels into new folder
            for chan in keep_channels:
                shutil.copy(os.path.join(img_source_dir, chan), os.path.join(output_dir, chan))

        # copy mask into new folder
        mask_name: str = mn + mask_suffix + '.tiff'
        shutil.copy(os.path.join(mask_output_dir, mask_name),
                    os.path.join(output_dir, 'population{}.tiff'.format(mask_suffix)))

        # copy the segmentation files into the output directory
        seg_name: str = fov + seg_suffix_name + '.tiff'
        shutil.copy(os.path.join(seg_dir, seg_name),
                    os.path.join(output_dir, 'cell_segmentation.tiff'))

        # copy mapping into directory
        map_df.to_csv(os.path.join(output_dir, 'population{}.csv'.format(mask_suffix)),
                      index=False)

def create_mantis_project(cell_table, fovs, seg_dir, pop_col,
                          mask_dir, image_dir, mantis_dir, keep_channels) -> None:
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
    create_mantis_dir(fovs=fovs, mantis_project_path=mantis_dir,
                                 img_data_path=image_dir, mask_output_dir=mask_dir,
                                 mask_suffix='_cell_mask', mapping=mantis_df,
                                 seg_dir=seg_dir, img_sub_folder='', keep_channels=keep_channels)


lag3_counts = control_cell_table[['fov', 'lag3_threshold']].groupby('fov').sum('LAG3_threshold')
# sort the dataframe
lag3_counts = lag3_counts.sort_values(by='lag3_threshold', ascending=False)

test_fovs = lag3_counts.index.tolist()[:20]
# code for setting thresholds for functional markers
# test_fovs = cell_table.fov.unique()
# np.random.shuffle(test_fovs)
# test_fovs = test_fovs[:30]
test_fovs = [fov for fov in test_fovs if 'TMA2_' not in fov]
# test_fovs = [fov for fov in test_fovs if  not in fov]
test_fovs = ['TONIC_TMA10_R10C6', 'TONIC_TMA10_R1C3', 'TONIC_TMA10_R1C4', 'TONIC_TMA10_R1C5', 'TONIC_TMA10_R2C4', 'TONIC_TMA10_R3C1', 'TONIC_TMA10_R7C4']
#
# remove_fovs = ['TONIC_TMA5_R9C4', 'TONIC_TMA10_R10C2', 'TONIC_TMA18_R2C3']
# test_fovs = [fov for fov in test_fovs if fov not in remove_fovs]
#

cell_table_testing = control_cell_table.loc[cell_table['fov'].isin(test_fovs), :]
# create dataframe with counts of the specified markers
marker_counts_df = cell_table_testing.loc[:, ['fov', 'label'] + ['LAG3']]

# save dataframe
marker_counts_df.to_csv('/Volumes/Shared/Noah Greenwald/TONIC_Cohort/mantis_dir/mantis_folders/marker_counts.csv', index=False)




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

keep_channels = ['LAG3.tiff']
create_mantis_project(cell_table=cell_table_testing, fovs=test_fovs, seg_dir='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/segmentation_data/deepcell_output',
                      pop_col='cell_meta_cluster', mask_dir='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/mantis_dir/masks',
                      image_dir='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples', mantis_dir='/Volumes/Shared/Noah Greenwald/TONIC_Cohort/mantis_dir/mantis_folders',
                      keep_channels=keep_channels)


for fov in test_fovs:
    #include_chans = ['LAG3.tiff', 'H3K27me3.tiff', 'CD20.tiff']
    include_chans = ['PD1.tiff', 'CD3.tiff', 'CD45.tiff', 'FOXP3.tiff']
    base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/lag3_overlays'
    image_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/controls'
    output_dir = os.path.join(base_dir, fov)
    #os.makedirs(output_dir, exist_ok=True)
    for chan in include_chans:
        shutil.copy(os.path.join(image_dir, fov, chan), os.path.join(output_dir, chan))
