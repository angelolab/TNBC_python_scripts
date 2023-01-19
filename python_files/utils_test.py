import itertools

import numpy as np
import pandas as pd

from python_files import utils


def test_cluster_df_helper():
    """Test create_long_df_by_cluster"""
    # Create test data
    cell_table = pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2", "fov2", "fov3"],
            "cluster": ["cell_type1", "cell_type2", "cell_type2", "cell_type1",
                        "cell_type2", "cell_type2", "cell_type1"]}
    )

    # Test normalize = False
    result = utils.cluster_df_helper(cell_table=cell_table, cluster_col_name="cluster",
                                             result_name="counts_per_cluster", normalize=False)

    result = result.sort_values(by=["cell_type", "fov"]).reset_index(drop=True)

    # first and second fov have 1 cell_type1 cell, third has 1
    # first and second fov have 2 cell_type2 cells, third has 0

    expected = pd.DataFrame(
        {
            "fov": ["fov1", "fov2", "fov3", "fov1", "fov2", "fov3"],
            "cell_type": ["cell_type1", "cell_type1", "cell_type1", "cell_type2", "cell_type2",
                          "cell_type2"],
            "value": [1, 1, 1, 2, 2, 0],
        }
    )
    expected["metric"] = "counts_per_cluster"

    pd.testing.assert_frame_equal(result, expected)

    # Test normalize = True
    result = utils.cluster_df_helper(cell_table=cell_table, cluster_col_name= "cluster",
                                             result_name="freq_per_cluster", normalize=True)
    result = result.sort_values(by=["cell_type", "fov"]).reset_index(drop=True)

    # first and second fovs have three total cells, third has 1
    expected.value = [1/3, 1/3, 1, 2/3, 2/3, 0]
    expected["metric"] = "freq_per_cluster"

    pd.testing.assert_frame_equal(result, expected)


def test_create_long_df_by_cluster_counts():

    # create identical cell table as above, but with additional mask column that has only one value
    cell_table = pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2", "fov2", "fov3"],
            "cluster": ["cell_type1", "cell_type2", "cell_type2", "cell_type1",
                        "cell_type2", "cell_type2", "cell_type1"],
            "mask": ['region1', 'region1', 'region1', 'region1', 'region1', 'region1', 'region1']}
    )

    # Test without supplying a subset column
    result_no_subset = utils.create_long_df_by_cluster(cell_table=cell_table, cluster_col_name="cluster",
                                              result_name="counts_per_cluster", subset_col=None)
    expected = pd.DataFrame(
        {
            "fov": ["fov1", "fov2", "fov3", "fov1", "fov2", "fov3"],
            "cell_type": ["cell_type1", "cell_type1", "cell_type1", "cell_type2", "cell_type2",
                          "cell_type2"],
            "value": [1, 1, 1, 2, 2, 0],
        }
    )

    result_no_subset = result_no_subset.drop(columns=['subset'])
    result_no_subset = result_no_subset.sort_values(by=["cell_type", "fov"]).reset_index(drop=True)

    expected["metric"] = "counts_per_cluster"
    pd.testing.assert_frame_equal(result_no_subset, expected)

    # Test when supplying a subset column, 'all' is equivalent to previous
    result_subset = utils.create_long_df_by_cluster(cell_table=cell_table, cluster_col_name="cluster",
                                                result_name="counts_per_cluster", subset_col='mask')

    result_subset = result_subset.sort_values(by=["cell_type", "fov"]).reset_index(drop=True)

    # extract just the counts per region, discarding names of the subset and index
    result_subset_all = result_subset[result_subset.subset == 'all'].drop(columns=['subset'])
    result_subset_all.reset_index(drop=True, inplace=True)

    result_subset_region1 = result_subset[result_subset.subset == 'region1'].drop(columns=['subset'])
    result_subset_region1.reset_index(drop=True, inplace=True)

    # check that these are equivalent to each other
    pd.testing.assert_frame_equal(result_subset_all, result_subset_region1)

    # check that these are equivalent to the output when not subsetting
    pd.testing.assert_frame_equal(result_subset_region1, result_no_subset)

    # create a single row with region2
    cell_table = cell_table.append({'fov': 'fov1', 'cluster': 'cell_type1', 'mask': 'region2'},
                                   ignore_index=True)

    # check outputs
    result = utils.create_long_df_by_cluster(cell_table=cell_table, cluster_col_name="cluster",
                                                result_name="counts_per_cluster", subset_col='mask')

    result = result.sort_values(by=["cell_type", "fov"]).reset_index(drop=True)

    result_subset_region1 = result[result.subset == 'region1'].drop(columns=['subset'])
    result_subset_region1.reset_index(drop=True, inplace=True)

    result_subset_region2 = result[result.subset == 'region2'].drop(columns=['subset'])
    result_subset_region2.reset_index(drop=True, inplace=True)

    # region1 should be identical to the output when not subsetting
    pd.testing.assert_frame_equal(result_subset_region1, result_no_subset)

    # region2 should only show up for FOV1, with a count of 1 for cell_type1 and 0 for cell_type2
    expected_region2 = pd.DataFrame(
        {
            "fov": ["fov1", "fov1"],
            "cell_type": ["cell_type1", "cell_type2"],
            "value": [1, 0],
            "metric": ["counts_per_cluster", "counts_per_cluster"]
        }
    )

    pd.testing.assert_frame_equal(result_subset_region2, expected_region2)

    # create that mismatch of fovs, masks, and cluters produces expected result
    cell_table = pd.DataFrame(
        {
            "fov": ["fov1", "fov2", "fov2", "fov3", "fov4", "fov5", "fov5", "fov5"],
            "cluster": ["cell_type1", "cell_type2", "cell_type2", "cell_type1", "cell_type3",
                        "cell_type4", "cell_type2", "cell_type2"],
            "mask": ['region1', 'region2', 'region1', 'region3', 'region2', 'region1', 'region1', 'region3']}
    )

    result = utils.create_long_df_by_cluster(cell_table=cell_table, cluster_col_name="cluster",
                                                result_name="counts_per_cluster", subset_col='mask')
    result = result[result.subset != 'all']

    # each unique pairing of fov and mask should contain all clusters
    unique_pairs = result[['fov', 'subset']].drop_duplicates()

    output_len = len(unique_pairs) * len(cell_table.cluster.unique())
    assert len(result) == output_len


# TODO: This code is largely copied from above
def test_create_long_df_by_cluster_freq():
    # create identical cell table as above, but with additional mask column that has only one value
    cell_table = pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2", "fov2", "fov3"],
            "cluster": ["cell_type1", "cell_type2", "cell_type2", "cell_type1",
                        "cell_type2", "cell_type2", "cell_type1"],
            "mask": ['region1', 'region1', 'region1', 'region1', 'region1', 'region1', 'region1']}
    )

    # Test without supplying a subset column
    result_no_subset = utils.create_long_df_by_cluster(cell_table=cell_table,
                                                       cluster_col_name="cluster",
                                                       result_name="freq_per_cluster",
                                                       subset_col=None, normalize=True)
    expected = pd.DataFrame(
        {
            "fov": ["fov1", "fov2", "fov3", "fov1", "fov2", "fov3"],
            "cell_type": ["cell_type1", "cell_type1", "cell_type1", "cell_type2", "cell_type2",
                          "cell_type2"],
            "value": [1/3, 1/3, 1, 2/3, 2/3, 0],
        }
    )

    result_no_subset = result_no_subset.drop(columns=['subset'])
    result_no_subset = result_no_subset.sort_values(by=["cell_type", "fov"]).reset_index(drop=True)

    expected["metric"] = "freq_per_cluster"
    pd.testing.assert_frame_equal(result_no_subset, expected)

    # Test when supplying a subset column, 'all' is equivalent to previous
    result_subset = utils.create_long_df_by_cluster(cell_table=cell_table,
                                                    cluster_col_name="cluster",
                                                    result_name="freq_per_cluster",
                                                    subset_col='mask',
                                                    normalize=True)

    result_subset = result_subset.sort_values(by=["cell_type", "fov"]).reset_index(drop=True)

    # extract just the freq per region, discarding names of the subset and index
    result_subset_all = result_subset[result_subset.subset == 'all'].drop(columns=['subset'])
    result_subset_all.reset_index(drop=True, inplace=True)

    result_subset_region1 = result_subset[result_subset.subset == 'region1'].drop(
        columns=['subset'])
    result_subset_region1.reset_index(drop=True, inplace=True)

    # check that these are equivalent to each other
    pd.testing.assert_frame_equal(result_subset_all, result_subset_region1)

    # check that these are equivalent to the output when not subsetting
    pd.testing.assert_frame_equal(result_subset_region1, result_no_subset)

    # create a single row with region2
    cell_table = cell_table.append({'fov': 'fov1', 'cluster': 'cell_type1', 'mask': 'region2'},
                                   ignore_index=True)

    # check that the output is the same as before, but with an additional row for region2
    result = utils.create_long_df_by_cluster(cell_table=cell_table, cluster_col_name="cluster",
                                             result_name="freq_per_cluster", subset_col='mask',
                                             normalize=True)

    result = result.sort_values(by=["cell_type", "fov"]).reset_index(drop=True)

    result_subset_region1 = result[result.subset == 'region1'].drop(columns=['subset'])
    result_subset_region1.reset_index(drop=True, inplace=True)

    result_subset_region2 = result[result.subset == 'region2'].drop(columns=['subset'])
    result_subset_region2.reset_index(drop=True, inplace=True)

    # region1 should be identical to the output when not subsetting
    pd.testing.assert_frame_equal(result_subset_region1, result_no_subset)

    # region2 should contain only two rows, 100% cell_type1 and 0% cell_type2 in fov1
    expected_region2 = pd.DataFrame(
        {
            "fov": ["fov1", "fov1"],
            "cell_type": ["cell_type1", "cell_type2"],
            "value": [1.0, 0.0],
            "metric": ["freq_per_cluster", "freq_per_cluster"]
        }
    )

    pd.testing.assert_frame_equal(result_subset_region2, expected_region2)


def test_functional_df_helper():
    """Test create_long_df_by_functional"""
    # Create test data
    func_table = pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2", "fov2"],
            "cluster": ["cell_type1", "cell_type2", "cell_type2", "cell_type1",
                        "cell_type2", "cell_type2"],
            "marker2": [True, True, False, False, True, True],
            "marker3": [1, 2, 3, 4, 5, 6],
            "extra_col": [1, 2, 4, 6, 9, 13],

        }
    )

    # Test with normalization
    result = utils.functional_df_helper(func_table=func_table, cluster_col_name="cluster",
                                        drop_cols=['extra_col'],result_name="func_freq",
                                        normalize=True)

    expected = pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov2", "fov2", "fov1", "fov1","fov2", "fov2"],
            "cell_type": ["cell_type1", "cell_type2", "cell_type1", "cell_type2",
                          "cell_type1", "cell_type2", "cell_type1", "cell_type2"],
            "functional_marker": ["marker2", "marker2", "marker2", "marker2",
                                  "marker3", "marker3", "marker3", "marker3"],
            # boolean markers + numeric markers
            "value": [1, 0.5, 0, 1] + [1, 2.5, 4, 5.5],
        }
    )
    expected["metric"] = "func_freq"
    pd.testing.assert_frame_equal(result, expected)

    # Test with no normalization
    result = utils.functional_df_helper(func_table=func_table, cluster_col_name="cluster",
                                        drop_cols=['extra_col'], result_name="func_sum",
                                        normalize=False)

    expected.value = [1, 1, 0, 2, 1, 5, 4, 11]
    expected["metric"] = "func_sum"

    pd.testing.assert_frame_equal(result, expected)


def test_create_long_df_by_functional_counts():

    # create identical cell table as above, but with additional mask column that has only one value
    func_table = pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2", "fov2"],
            "cluster": ["cell_type1", "cell_type2", "cell_type2", "cell_type1",
                        "cell_type2", "cell_type2"],
            "marker2": [True, True, False, False, True, True],
            "marker3": [1, 2, 3, 4, 5, 6],
            "extra_col": [1, 2, 4, 6, 9, 13],
            "mask": ["region1", "region1", "region1", "region1", "region1", "region1"]

        }
    )

    # Test without supplying a subset column
    result_no_subset = utils.create_long_df_by_functional(func_table=func_table, cluster_col_name="cluster",
                                        drop_cols=['extra_col', 'mask'], result_name="func_count",
                                                subset_col=None, normalize=False)

    expected = pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov2", "fov2", "fov1", "fov1", "fov2", "fov2"],
            "cell_type": ["cell_type1", "cell_type2", "cell_type1", "cell_type2",
                          "cell_type1", "cell_type2", "cell_type1", "cell_type2"],
            "functional_marker": ["marker2", "marker2", "marker2", "marker2",
                                  "marker3", "marker3", "marker3", "marker3"],
            # boolean markers + numeric markers
            "value": [1, 1, 0, 2] + [1, 5, 4, 11]
        }
    )
    expected["metric"] = "func_count"
    expected["subset"] = "all"
    pd.testing.assert_frame_equal(result_no_subset, expected)

    # Test with supplying a subset column
    result_subset = utils.create_long_df_by_functional(func_table=func_table, cluster_col_name="cluster",
                                        drop_cols=['extra_col'], result_name="func_count",
                                                subset_col="mask", normalize=False)

    result_subset_all = result_subset[result_subset.subset == 'all']
    result_subset_all.reset_index(drop=True, inplace=True)

    result_subset_region1 = result_subset[result_subset.subset == 'region1']
    result_subset_region1.reset_index(drop=True, inplace=True)

    # all should be identical to the output when not subsetting
    pd.testing.assert_frame_equal(result_subset_all, result_no_subset)

    # region1 should be identical to the output when not subsetting, except for the subset column
    expected_subset_region1 = expected.copy()
    expected_subset_region1.subset = "region1"

    pd.testing.assert_frame_equal(result_subset_region1, expected_subset_region1)

    # create additional rows with two cells in region2
    func_table_region2 = pd.DataFrame(
        {
            "fov": ["fov1", "fov1"],
            "cluster": ["cell_type1", "cell_type1"],
            "marker2": [True, True],
            "marker3": [1, 2],
            "extra_col": [1, 2],
            "mask": ["region2", "region2"]
        }
    )
    func_table = pd.concat([func_table, func_table_region2], axis=0)

    # Test with supplying a subset column
    result_subset = utils.create_long_df_by_functional(func_table=func_table, cluster_col_name="cluster",
                                        drop_cols=['extra_col'], result_name="func_count",
                                                subset_col="mask", normalize=False)

    result_subset_region1 = result_subset[result_subset.subset == 'region1']
    result_subset_region1.reset_index(drop=True, inplace=True)

    result_subset_region2 = result_subset[result_subset.subset == 'region2']
    result_subset_region2.reset_index(drop=True, inplace=True)

    # region1 should be identical to the output when not subsetting, except for the subset column
    pd.testing.assert_frame_equal(result_subset_region1, expected_subset_region1)

    # both cells are positive for marker2, and the sum of marker intensity is 3
    expected_subset_region2 = pd.DataFrame(
        {
            "fov": ["fov1", "fov1"],
            "cell_type": ["cell_type1", "cell_type1"],
            "functional_marker": ["marker2", "marker3"],
            # boolean markers + numeric markers
            "value": [2, 3]
        }
    )

    expected_subset_region2["metric"] = "func_count"
    expected_subset_region2["subset"] = "region2"

    pd.testing.assert_frame_equal(result_subset_region2, expected_subset_region2)


# TODO: This code is largely copied from above
def test_create_long_df_by_functional_freq():

    # create identical cell table as above, but with additional mask column that has only one value
    func_table = pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2", "fov2"],
            "cluster": ["cell_type1", "cell_type2", "cell_type2", "cell_type1",
                        "cell_type2", "cell_type2"],
            "marker2": [True, True, False, False, True, True],
            "marker3": [1, 2, 3, 4, 5, 6],
            "extra_col": [1, 2, 4, 6, 9, 13],
            "mask": ["region1", "region1", "region1", "region1", "region1", "region1"]

        }
    )

    # Test without supplying a subset column
    result_no_subset = utils.create_long_df_by_functional(func_table=func_table, cluster_col_name="cluster",
                                        drop_cols=['extra_col', 'mask'], result_name="func_freq",
                                                subset_col=None, normalize=True)

    expected = pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov2", "fov2", "fov1", "fov1", "fov2", "fov2"],
            "cell_type": ["cell_type1", "cell_type2", "cell_type1", "cell_type2",
                          "cell_type1", "cell_type2", "cell_type1", "cell_type2"],
            "functional_marker": ["marker2", "marker2", "marker2", "marker2",
                                  "marker3", "marker3", "marker3", "marker3"],
            # boolean markers + numeric markers
            "value": [1, 0.5, 0, 1] + [1, 2.5, 4, 5.5],
        }
    )
    expected["metric"] = "func_freq"
    expected["subset"] = "all"
    pd.testing.assert_frame_equal(result_no_subset, expected)

    # Test with supplying a subset column
    result_subset = utils.create_long_df_by_functional(func_table=func_table, cluster_col_name="cluster",
                                        drop_cols=['extra_col'], result_name="func_freq",
                                                subset_col="mask", normalize=True)

    result_subset_all = result_subset[result_subset.subset == 'all']
    result_subset_all.reset_index(drop=True, inplace=True)

    result_subset_region1 = result_subset[result_subset.subset == 'region1']
    result_subset_region1.reset_index(drop=True, inplace=True)

    # all should be identical to the output when not subsetting
    pd.testing.assert_frame_equal(result_subset_all, result_no_subset)

    # region1 should be identical to the output when not subsetting, except for the subset column
    expected_subset_region1 = expected.copy()
    expected_subset_region1.subset = "region1"

    pd.testing.assert_frame_equal(result_subset_region1, expected_subset_region1)

    # create additional rows with two cells in region2
    func_table_region2 = pd.DataFrame(
        {
            "fov": ["fov1", "fov1"],
            "cluster": ["cell_type1", "cell_type1"],
            "marker2": [True, True],
            "marker3": [1, 2],
            "extra_col": [1, 2],
            "mask": ["region2", "region2"]
        }
    )
    func_table = pd.concat([func_table, func_table_region2], axis=0)

    # Test with supplying a subset column
    result_subset = utils.create_long_df_by_functional(func_table=func_table, cluster_col_name="cluster",
                                        drop_cols=['extra_col'], result_name="func_freq",
                                                subset_col="mask", normalize=True)

    result_subset_region1 = result_subset[result_subset.subset == 'region1']
    result_subset_region1.reset_index(drop=True, inplace=True)

    result_subset_region2 = result_subset[result_subset.subset == 'region2']
    result_subset_region2.reset_index(drop=True, inplace=True)

    # region1 should be identical to the output when not subsetting, except for the subset column
    pd.testing.assert_frame_equal(result_subset_region1, expected_subset_region1)

    # both cells are positive for marker2, and the sum of marker intensity is 3
    expected_subset_region2 = pd.DataFrame(
        {
            "fov": ["fov1", "fov1"],
            "cell_type": ["cell_type1", "cell_type1"],
            "functional_marker": ["marker2", "marker3"],
            # boolean markers + numeric markers
            "value": [1, 1.5]
        }
    )

    expected_subset_region2["metric"] = "func_freq"
    expected_subset_region2["subset"] = "region2"

    pd.testing.assert_frame_equal(result_subset_region2, expected_subset_region2)


def test_identify_cell_bounding_box():
    # when the centroid is at least crop_size/2 away from the edge, the bounding box should not change
    # otherwise, the bounding box should be adjusted to be at crop_size/2 away from the edge

    row_coords = [3, 10, 30, 98]
    col_coords = [30, 4, 99, 90]

    bb_row_coords = [0, 0, 20, 80]
    bb_col_coords = [20, 0, 80, 80]

    crop_size = 20
    img_shape = (100, 100)

    for i in range(len(row_coords)):
        row_coord, col_coord = utils.identify_cell_bounding_box(row_coords[i], col_coords[i],
                                                                crop_size, img_shape)
        assert row_coord == bb_row_coords[i]
        assert col_coord == bb_col_coords[i]


def test_generate_cell_crop_coords():
    row_coords = [3, 10, 30, 98]
    col_coords = [30, 4, 99, 90]

    bb_row_coords = [0, 0, 20, 80]
    bb_col_coords = [20, 0, 80, 80]

    crop_size = 20
    img_shape = (100, 100)

    cell_table = pd.DataFrame({'fov': ['fov1', 'fov1', 'fov1', 'fov1'],
                               'label': [1, 2, 3, 4],
                               'centroid-0': row_coords,
                               'centroid-1': col_coords})

    bb_df = utils.generate_cell_crop_coords(cell_table, crop_size, img_shape)
    assert bb_df.row_coord.tolist() == bb_row_coords
    assert bb_df.col_coord.tolist() == bb_col_coords
    assert bb_df.fov.tolist() == ['fov1', 'fov1', 'fov1', 'fov1']
    assert bb_df.id.tolist() == [1, 2, 3, 4]


def test_generate_tiled_crop_coords():
    img_shape = (100, 100)
    crop_size = 20

    # generate combinations of row and col coords
    coords = itertools.product(range(0, img_shape[0], crop_size),
                                 range(0, img_shape[1], crop_size))
    predicted_df = pd.DataFrame(coords, columns=['row_coord', 'col_coord'])


    coord_df = utils.generate_tiled_crop_coords(crop_size, img_shape, 'fov1')

    pd.testing.assert_frame_equal(coord_df[['row_coord', 'col_coord']],
                                  predicted_df[['row_coord', 'col_coord']])


def test_extract_crop_sums():
    # create test image
    chan0 = np.zeros((100, 100))
    chan0[:10, 50:60] = 2
    chan0[10:20, 20:30] = 1

    chan1 = np.zeros((100, 100))
    chan1[50:60, 40:50] = 3

    img = np.stack([chan0, chan1], axis=-1)

    # create crop coordinates
    row_coords = [0, 10, 50, 0, 80]
    col_coords = [50, 20, 40, 55, 80]

    coords_df = pd.DataFrame({"row_coord": row_coords,
                               "col_coord": col_coords})

    # create expected output
    expected = np.stack([[200, 100, 0, 100, 0],
                         [0, 0, 300, 0, 0]], axis=-1)
    # test
    result = utils.extract_crop_sums( img_data=img, crop_size=10, crop_coords_df=coords_df)

    np.testing.assert_array_equal(result, expected)

# def test_generate_cell_sum_dfs(tmpdir):
#     # create image data for first fov
#     fov1_chan1 = np.zeros((100, 100))
#     fov1_chan1[:10, 50:60] = 1
#
#     fov1_chan2 = np.zeros((100, 100))
#     fov1_chan2[50:60, 40:50] = 1
#
#     fov1_mask = np.zeros((100, 100))
#     fov1_mask[:5, 50:60] = 1
#     fov1_mask = fov1_mask == 1
#
#     # create image data for second fov
#     fov2_chan1 = np.zeros((100, 100))
#     fov2_chan1[90:, 20:30] = 2
#
#     fov2_chan2 = np.zeros((100, 100))
#     fov2_chan2[70:80, 20:30] = 1
#
#     fov2_mask = np.zeros((100, 100))
#     fov2_mask[75:80, 20:30] = 1
#     fov2_mask = fov2_mask == 1
#
#     # create directory structure
#     channel_dir = tmpdir.mkdir("channel_dir")
#     fov1_dir = channel_dir.mkdir("fov1")
#     fov2_dir = channel_dir.mkdir("fov2")
#
#     import skimage.io as io
#     for chan_name, chan_mask in zip(["chan1", "chan2", "total_ecm"],
#                                     [fov1_chan1, fov1_chan2, fov1_mask.astype('uint8')]):
#         io.imsave(fov1_dir.join(f"{chan_name}.tiff").strpath, chan_mask, check_contrast=False)
#
#     for chan_name, chan_mask in zip(["chan1", "chan2", "total_ecm"],
#                                     [fov2_chan1, fov2_chan2, fov2_mask.astype('uint8')]):
#         io.imsave(fov2_dir.join(f"{chan_name}.tiff").strpath, chan_mask, check_contrast=False)
#
#     # create cell table
#     cell_ids = [1, 2, 1, 3]
#     fovs = ["fov1", "fov1", "fov2", "fov2"]
#     row_coords = [0, 55, 95, 75]
#     col_coords = [55, 45, 25, 25]
#
#     cell_table = pd.DataFrame({"label": cell_ids, "fov": fovs, "centroid-0": row_coords,
#                                  "centroid-1": col_coords})
#
#     # create expected output
#     expected_output = pd.DataFrame({"chan1": [100, 0, 200, 0],
#                                   "chan2": [0, 100, 0, 100],
#                                   "ecm_mask": [50, 0, 0, 50],
#                                   'label': [1, 2, 1, 3],
#                                   'fov': ["fov1", "fov1", "fov2", "fov2"]})
#
#
#     # test
#     result = utils.generate_cell_sum_dfs(cell_table=cell_table, channel_dir=channel_dir.strpath,
#                                          mask_dir=channel_dir.strpath, channels=['chan1', 'chan2'],
#                                          crop_size=10)
#
#     # compare results
#     expected_output.iloc[:, 0:4] = expected_output.iloc[:, 0:4].astype(float)
#     pd.testing.assert_frame_equal(result, expected_output)
#
