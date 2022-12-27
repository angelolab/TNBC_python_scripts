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
    result = utils.cluster_df_new(cell_table=cell_table, cluster_col_name="cluster",
                                             result_name="counts_per_cluster", normalize=False)

    # first and second fov have 1 cell_type1 cell, third has 1
    # first and second fov have 2 cell_type2 cells, third has 0

    expected = pd.DataFrame(
        {
            "cell_type": ["cell_type1", "cell_type1", "cell_type1", "cell_type2", "cell_type2",
                          "cell_type2"],
            "fov": ["fov1", "fov2", "fov3", "fov1", "fov2", "fov3"],
            "value": [1, 1, 1, 2, 2, 0],
        }
    )
    expected["metric"] = "counts_per_cluster"
    pd.testing.assert_frame_equal(result, expected)

    # Test normalize = True
    result = utils.cluster_df_new(cell_table=cell_table, cluster_col_name= "cluster",
                                             result_name="freq_per_cluster", normalize='index')

    # first and second fovs have three total cells, third has 1
    expected.value = [1/3, 1/3, 1, 2/3, 2/3, 0]
    expected["metric"] = "freq_per_cluster"

    pd.testing.assert_frame_equal(result, expected)


def test_create_long_df_by_cluster():

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

    expected["metric"] = "counts_per_cluster"
    pd.testing.assert_frame_equal(result_no_subset, expected)

    # Test when supplying a subset column, 'all' is equivalent to previous
    result_subset = utils.create_long_df_by_cluster(cell_table=cell_table, cluster_col_name="cluster",
                                                result_name="counts_per_cluster", subset_col='mask')

    # extract just the counts per region, discarding names of the subset
    result_subset_all = result_subset[result_subset.subset == 'all'].drop(columns=['subset'])
    result_subset_region1 = result_subset[result_subset.subset == 'region1'].drop(columns=['subset'])

    # check that these are equivalent to each other
    pd.testing.assert_frame_equal(result_subset_all, result_subset_region1)

    # check that these are equivalent to the output when not subsetting
    pd.testing.assert_frame_equal(result_subset_region1, result_no_subset)

    # create a single row with region2
    cell_table = cell_table.append({'fov': 'fov1', 'cluster': 'cell_type1', 'mask': 'region2'},
                                   ignore_index=True)

    # check that the output is the same as before, but with an additional row for region2
    result_subset_region2 = utils.create_long_df_by_cluster(cell_table=cell_table, cluster_col_name="cluster",
                                                result_name="counts_per_cluster", subset_col='mask')
    result_subset_region2 = result_subset_region2[result_subset_region2.subset == 'region2'].drop(columns=['subset'])

    # output should be zero everywhere except for first row
    expected_region2 = expected.copy()
    expected_region2['mask'] = 'region2'
    expected_region2['value'] = 0
    expected_region2.iloc[0, 2] = 1
    expected_region2["metric"] = "counts_per_cluster"

    #pd.testing.assert_frame_equal(result_subset_region2, expected_region2)





def test_create_long_df_by_functional():
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

    # Test with transform_func = np.mean
    result = utils.create_long_df_by_functional(func_table=func_table, cluster_col_name="cluster",
                                                drop_cols=['extra_col'],transform_func=np.mean,
                                                result_name="func_freq")

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

    # Test with transform_func = np.sum
    result = utils.create_long_df_by_functional(func_table=func_table, cluster_col_name="cluster",
                                                drop_cols=['extra_col'], transform_func=np.sum,
                                                result_name="func_sum")

    expected.value = [1, 1, 0, 2, 1, 5, 4, 11]
    expected["metric"] = "func_sum"

    pd.testing.assert_frame_equal(result, expected)

