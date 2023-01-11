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

