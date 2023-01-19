import itertools
import numpy as np
import os
import pandas as pd
from skimage import morphology
from scipy.ndimage import gaussian_filter
from skimage.measure import label
import skimage.io as io

from ark.utils.misc_utils import verify_in_list
from ark.utils import io_utils, load_utils
from ark.segmentation import marker_quantification



def cluster_df_helper(cell_table, cluster_col_name, result_name, normalize=False):
    """Helper function which creates a df when no subsetting is required

        Args:
            cell_table: the dataframe containing information on each cell
            cluster_col_name: the column name in cell_table that contains the cluster information
            result_name: the name of this statistic in summarized information df
            normalize: whether to report the total or normalized counts in the result

        Returns:
            pd.DataFrame: long format dataframe containing the summarized data"""

    # group each fov by the supplied cluster column, then count and normalize
    grouped = cell_table.groupby(['fov'])
    counts = grouped[cluster_col_name].value_counts(normalize=normalize)
    counts = counts.unstack(level=cluster_col_name, fill_value=0).stack()

    # standardize the column names
    counts = counts.reset_index()
    counts['metric'] = result_name
    counts = counts.rename(columns={cluster_col_name: 'cell_type', 0: 'value'})

    return counts


def create_long_df_by_cluster(cell_table, cluster_col_name, result_name, subset_col=None,
                              normalize=False):
    """Summarize cell counts by cluster, with the option to subset by an additional feature

    Args:
        cell_table (pd.DataFrame): the dataframe containing information on each cell
        cluster_col_name (str): the column name in cell_table that contains the cluster information
        result_name (str): the name of this statistic in the returned df
        subset_col (str): the column name in cell_table to subset by
        normalize (bool): whether to report the total or normalized counts in the result

    Returns:
        pd.DataFrame: long format dataframe containing the summarized data"""

    # first generate df without subsetting
    long_df_all = cluster_df_helper(cell_table, cluster_col_name, result_name, normalize)
    long_df_all['subset'] = 'all'

    # if a subset column is specified, create df stratified by subset
    if subset_col is not None:
        verify_in_list(subset_col=subset_col, cell_table_columns=cell_table.columns)

        # group each fov by fov and cluster
        grouped = cell_table.groupby(['fov', subset_col])
        counts = grouped[cluster_col_name].value_counts(normalize=normalize)

        # unstack and restack to make sure that missing cell populations are filled with zeros
        counts = counts.unstack(level=cluster_col_name, fill_value=0).stack()

        # standardize the column names
        counts = counts.reset_index()
        counts['metric'] = result_name
        counts = counts.rename(columns={cluster_col_name: 'cell_type', subset_col: 'subset',
                                        0: 'value'})

        # combine the two dataframes
        long_df_all = pd.concat([long_df_all, counts], axis=0, ignore_index=True)

    return long_df_all


def functional_df_helper(func_table, cluster_col_name, drop_cols, result_name, normalize=False):
    """Function to summarize functional marker data by cell type

    Args:
        func_table (pd.DataFrame): cell table containing functional markers
        cluster_col_name (str): name of the column in func_table that contains the cluster information
        drop_cols (list): list of columns to drop from func_table
        result_name (str): name of the statistic in the summarized information df
        normalize (bool): whether to report the total or normalized counts in the result

    Returns:
        pd.DataFrame: long format dataframe containing the summarized data"""

    verify_in_list(cell_type_col=cluster_col_name, cell_table_columns=func_table.columns)
    verify_in_list(drop_cols=drop_cols, cell_table_columns=func_table.columns)

    # drop columns from table
    func_table_small = func_table.loc[:, ~func_table.columns.isin(drop_cols)]

    # group by specified columns
    grouped_table = func_table_small.groupby(['fov', cluster_col_name])
    if normalize:
        transformed = grouped_table.agg(np.mean)
    else:
        transformed = grouped_table.agg(np.sum)
    transformed.reset_index(inplace=True)

    # reshape to long df
    long_df = pd.melt(transformed, id_vars=['fov', cluster_col_name], var_name='functional_marker')
    long_df['metric'] = result_name
    long_df = long_df.rename(columns={cluster_col_name: 'cell_type'})

    return long_df


def create_long_df_by_functional(func_table, cluster_col_name, drop_cols, result_name,
                                        subset_col=None, normalize=False):
    """Summarize functional marker positivity by cell type, with the option to subset by an additional feature

    Args:
        func_table (pd.DataFrame): the dataframe containing information on each cell
        cluster_col_name (str): the column name in cell_table that contains the cluster information
        drop_cols (list): list of columns to drop from cell_table
        result_name (str): the name of this statistic in the returned df
        subset_col (str): the column name in cell_table to subset by
        normalize (bool): whether to report the total or normalized counts in the result

    Returns:
        pd.DataFrame: long format dataframe containing the summarized data"""

    # first generate df without subsetting
    drop_cols_all = drop_cols.copy()
    if subset_col is not None:
        drop_cols_all = drop_cols + [subset_col]

    long_df_all = functional_df_helper(func_table, cluster_col_name, drop_cols_all, result_name,
                                       normalize)
    long_df_all['subset'] = 'all'

    # if a subset column is specified, create df stratified by subset
    if subset_col is not None:
        verify_in_list(subset_col=subset_col, cell_table_columns=func_table.columns)

        # drop columns from table
        func_table_small = func_table.loc[:, ~func_table.columns.isin(drop_cols)]

        # group by specified columns
        grouped_table = func_table_small.groupby(['fov', subset_col, cluster_col_name])
        if normalize:
            transformed = grouped_table.agg(np.mean)
        else:
            transformed = grouped_table.agg(np.sum)
        transformed.reset_index(inplace=True)

        # reshape to long df
        long_df = pd.melt(transformed, id_vars=['fov', subset_col, cluster_col_name],
                            var_name='functional_marker')
        long_df['metric'] = result_name
        long_df = long_df.rename(columns={cluster_col_name: 'cell_type', subset_col: 'subset'})

        # combine the two dataframes
        long_df_all = pd.concat([long_df_all, long_df], axis=0, ignore_index=True)

    return long_df_all


def create_channel_mask(img, intensity_thresh, sigma, min_mask_size=0, max_hole_size=100000):
    """Generates a binary mask from a single channel image

    Args:
        img (np.ndarray): image to be masked
        intensity_thresh (float): threshold for the image intensity to use for masking
        sigma (float): sigma for gaussian blur
        min_mask_size (int): minimum size of masked objects to include
        max_hole_size (int): maximum size of holes to leave in masked objects
        """
    # create a binary mask
    img_smoothed = gaussian_filter(img.astype(float), sigma=sigma)
    img_mask = img_smoothed > intensity_thresh

    # if no post-processing return as is
    if min_mask_size == 0:
        return img_mask

    # otherwise, clean up the mask before returning
    label_mask = label(img_mask)
    label_mask = morphology.remove_small_objects(label_mask, min_size=min_mask_size)
    label_mask = morphology.remove_small_holes(label_mask, area_threshold=max_hole_size)

    return label_mask


def create_cell_mask(seg_mask, cell_table, fov_name, cell_types, sigma=10, smooth_thresh=0.3,
                     min_mask_size=0, max_hole_size=100000):
    """Generates a binary from the cells listed in `cell_types`

    args:
        seg_mask (numpy.ndarray): segmentation mask
        cell_table (pandas.DataFrame): cell table containing segmentation IDs and cell types
        fov_name (str): name of the fov to process
        cell_types (list): list of cell types to include in the mask
        sigma (float): sigma for gaussian smoothing
        smooth_thresh (float): threshold for including a pixel in the smoothed mask
        min_mask_size (int): minimum size of a mask to include
        max_hole_size (int): maximum size of a hole to leave without filling

    returns:
        numpy.ndarray: binary mask
    """
    # get cell labels for fov and cell type
    cell_subset = cell_table[cell_table['fov'] == fov_name]
    cell_subset = cell_subset[cell_subset['cell_cluster_broad'].isin(cell_types)]
    cell_labels = cell_subset['label'].values

    # create mask for cell type
    cell_mask = np.isin(seg_mask, cell_labels)

    # postprocess mask
    cell_mask = create_channel_mask(img=cell_mask, intensity_thresh=smooth_thresh,
                                    sigma=sigma, min_mask_size=min_mask_size,
                                    max_hole_size=max_hole_size)

    return cell_mask


def create_cancer_boundary(img, seg_mask, min_mask_size=3500, max_hole_size=1000,
                           border_size=50, channel_thresh=0.0015):
    """Generate masks representing different tumor regions"""
    img_smoothed = gaussian_filter(img, sigma=10)
    img_mask = img_smoothed > channel_thresh

    # clean up mask prior to analysis
    img_mask = np.logical_or(img_mask, seg_mask)
    label_mask = label(img_mask)
    label_mask = morphology.remove_small_objects(label_mask, min_size=min_mask_size)
    label_mask = morphology.remove_small_holes(label_mask, area_threshold=max_hole_size)

    # define external borders
    external_boundary = morphology.binary_dilation(label_mask)

    for _ in range(border_size):
        external_boundary = morphology.binary_dilation(external_boundary)

    external_boundary = external_boundary.astype(int) - label_mask.astype(int)
    # create interior borders
    interior_boundary = morphology.binary_erosion(label_mask)

    for _ in range(border_size):
        interior_boundary = morphology.binary_erosion(interior_boundary)

    interior_boundary = label_mask.astype(int) - interior_boundary.astype(int)

    combined_mask = np.ones_like(img)
    combined_mask[label_mask] = 4
    combined_mask[external_boundary > 0] = 2
    combined_mask[interior_boundary > 0] = 3

    return combined_mask


def calculate_mask_areas(mask_dir, fovs):
    """Calculate the area of each mask per fov

    Args:
        mask_dir (str): path to directory containing masks for each fov
        fovs (list): list of fovs to calculate mask areas for

    Returns
        pd.DataFrame: dataframe containing the area of each mask per fov
    """
    # get list of masks
    mask_files = io_utils.list_files(os.path.join(mask_dir, fovs[0]))
    mask_names = [os.path.splitext(os.path.basename(x))[0] for x in mask_files]

    # loop through fovs and masks to calculate area
    area_dfs = []
    for fov in fovs:
        mask_areas = []
        for mask_file in mask_files:
            mask = io.imread(os.path.join(mask_dir, fov, mask_file))
            mask_areas.append(np.sum(mask))

        area_df = pd.DataFrame({'compartment': mask_names, 'area': mask_areas,
                                'fov': fov})

        # separately calculate size for non-background compartment
        bg_area = area_df[area_df['compartment'] == 'empty_slide']['area'].values[0]
        foreground_area = mask.shape[0] ** 2 - bg_area
        blank_df = pd.DataFrame({'compartment': ['all'], 'area': [foreground_area], 'fov': [fov]})
        area_df = pd.concat([area_df, blank_df], ignore_index=True)

        area_dfs.append(area_df)

    return pd.concat(area_dfs, axis=0)


def assign_cells_to_mask(seg_dir, mask_dir, fovs):
    """Assign cells an image to the mask they overlap most with

    Args:
        seg_dir (str): path to segmentation directory
        mask_dir (str): path to mask directory, with masks for each FOV in a dedicated folder
        fovs (list): list of fovs to process

    Returns:
        pandas.DataFrame: dataframe with cell assignments to masks
    """

    # extract counts of each mask per cell
    normalized_cell_table, _ = marker_quantification.generate_cell_table(segmentation_dir=seg_dir,
                                                                         tiff_dir=mask_dir,
                                                                         fovs=fovs,
                                                                         img_sub_folder='')
    # drop cell_size column
    normalized_cell_table = normalized_cell_table.drop(columns=['cell_size'])

    # move fov column to front
    fov_col = normalized_cell_table.pop('fov')
    normalized_cell_table.insert(0, 'fov', fov_col)

    # remove all columns after label
    normalized_cell_table = normalized_cell_table.loc[:, :'label']

    # move label column to front
    label_col = normalized_cell_table.pop('label')
    normalized_cell_table.insert(1, 'label', label_col)

    # create new column with name of column max for each row
    normalized_cell_table['mask_name'] = normalized_cell_table.iloc[:, 2:].idxmax(axis=1)

    return normalized_cell_table[['fov', 'label', 'mask_name']]


def identify_cell_bounding_box(row_centroid, col_centroid, crop_size, img_shape):
    """Finds the upper-left hand corner of a bounding box surrounding the cell, corrected for edges.

    Args:
        row_coord (int): row coordinate of the cell centroid
        col_coord (int): column coordinate of the cell centroid
        crop_size (int): size of the bounding box
        img_shape (tuple): shape of the image
    """

    # get the image dimensions
    img_height, img_width = img_shape

    # adjust the centroid to be at least crop_size / 2 away from the bottom right corner
    if row_centroid > img_height - crop_size // 2:
        row_centroid = img_height - crop_size // 2
    if col_centroid > img_width - crop_size // 2:
        col_centroid = img_width - crop_size // 2

    # set new coordinates to be crop_size / 2 up and to the left of the centroid
    col_coord = col_centroid - crop_size // 2
    row_coord = row_centroid - crop_size // 2

    # make sure the coordinates are not negative
    col_coord = max(col_coord, 0)
    row_coord = max(row_coord, 0)

    return int(row_coord), int(col_coord)


def generate_cell_crop_coords(cell_table_fov, crop_size, img_shape):
    """Generates the coordinates for cropping each cell in a fov

    Args:
        cell_table_fov (pd.DataFrame): dataframe containing the location of each cell
        crop_size (int): size of the bounding box
        img_shape (tuple): shape of the image

    Returns:
        pd.DataFrame: dataframe containing the coordinates for cropping each cell
    """
    # get the coordinates for each cell
    cell_coords = cell_table_fov[['centroid-0', 'centroid-1']].values

    # calculate the coordinates for the upper left hand corner of the bounding box
    crop_coords = [identify_cell_bounding_box(row_coord, col_coord, crop_size, img_shape)
                   for row_coord, col_coord in cell_coords]

    # create a dataframe with the coordinates
    crop_coords_df = pd.DataFrame(crop_coords, columns=['row_coord', 'col_coord'])

    # add the fov column
    crop_coords_df['fov'] = cell_table_fov['fov'].values[0]

    # add the label column
    crop_coords_df['id'] = cell_table_fov['label'].values

    return crop_coords_df


def generate_tiled_crop_coords(crop_size, img_shape, fov_name):
    """Generate coordinates for uniformly tiled crops

    Args:
        crop_size (int): size of the bounding box
        img_shape (tuple): shape of the image
        fov_name (str): name of the fov

    Returns:
        pd.DataFrame: dataframe containing the coordinates for cropping each tile
    """

    # compute all combinations of start coordinates
    img_height, img_width = img_shape

    row_coords = np.arange(0, img_height, crop_size)
    col_coords = np.arange(0, img_width, crop_size)
    coords = itertools.product(row_coords, col_coords)

    # create a dataframe with the coordinates
    crop_coords_df = pd.DataFrame(coords, columns=['row_coord', 'col_coord'])

    # add the fov column
    crop_coords_df['fov'] = fov_name

    # add a column for the tile combination
    crop_coords_df['id'] = [f'row_{row}_col_{col}' for row, col in zip(crop_coords_df['row_coord'],
                                                                       crop_coords_df['col_coord'])]

    return crop_coords_df


def extract_crop_sums(img_data, crop_size, crop_coords_df):
    """Extracts and sums crops from an image

    Args:
        img_data (np.ndarray): image data for a single fov
        crop_size (int): size of the bounding box around each cell
        crop_coords_df (pd.DataFrame): dataframe containing the coordinates for cropping each tile

    Returns:
        np.ndarray: array of crop sums
    """
    # list to hold crop sums
    crop_sums = []

    for row_coord, col_coord in zip(crop_coords_df['row_coord'], crop_coords_df['col_coord']):
        # crop based on provided coords
        crop = img_data[row_coord:row_coord + crop_size,
                        col_coord:col_coord + crop_size, :]

        # sum the channels within the crop
        crop_sum = crop.sum(axis=(0, 1))

        # add the crop sum to the list
        crop_sums.append(crop_sum)

    return np.array(crop_sums)


def generate_cell_sum_dfs(cell_table, channel_dir, mask_dir, channels, crop_size):
    """Generates dataframes of summed crops around cells for each fov

    Args:
        cell_table (pd.DataFrame): cell table
        channel_dir (str): path to the directory containing image data
        mask_dir (str): path to the directory containing the ecm masks
        channels (list): list of channels to extract crops from
        crop_size (int): size of the bounding box around each cell

    Returns:
        pd.DataFrame: dataframe of summed crops around cells
    """
    # list to hold dataframes
    cell_sum_dfs = []

    for fov in cell_table.fov.unique():
        # load the image data
        img_data = load_utils.load_imgs_from_tree(channel_dir,
                                                  fovs=[fov],
                                                  img_sub_folder='',
                                                  channels=channels)
        ecm_mask = io.imread(os.path.join(mask_dir, fov, 'total_ecm.tiff'))

        # combine the image data and the ecm mask into numpy array
        img_data = np.concatenate((img_data[0].values, ecm_mask[..., None]), axis=-1)

        # subset the cell table to the current fov
        cell_table_fov = cell_table[cell_table.fov == fov]
        cell_table_fov = cell_table_fov.reset_index(drop=True)

        # extract summed counts around each cell
        crop_sums = extract_cell_crop_sums(cell_table_fov=cell_table_fov,
                                           img_data=img_data,
                                           crop_size=crop_size)

        # create a dataframe of the summed counts
        crop_sums_df = pd.DataFrame(crop_sums,
                                    columns=channels + ['ecm_mask'] + ['label'])
        crop_sums_df['fov'] = fov

        # add the dataframe to the list
        cell_sum_dfs.append(crop_sums_df)

    return pd.concat(cell_sum_dfs, ignore_index=True)
