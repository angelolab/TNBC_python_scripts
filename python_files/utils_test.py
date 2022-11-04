import pandas as pd

from python_files import utils


def test_create_long_df_by_cluster():
    """Test create_long_df_by_cluster"""
    # Create test data
    cell_table = pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2", "fov2", "fov3"],
            "cluster": ["cell_type1", "cell_type2", "cell_type2", "cell_type1",
                        "cell_type2", "cell_type2", "cell_type1"]}
    )

    # Test normalize = False
    result = utils.create_long_df_by_cluster(cell_table, "counts_per_cluster", "cluster", normalize=False)

    # first and second fov have 1 cell_type1 cell, third has 1
    # first and second fov have 2 cell_type2 cells, third has 0

    expected = pd.DataFrame(
        {
            "fov": ["fov1", "fov2", "fov3", "fov1", "fov2", "fov3"],
            "cell_type": ["cell_type1", "cell_type1", "cell_type1", "cell_type2", "cell_type2", "cell_type2"],
            "value": [1, 1, 1, 2, 2, 0],
        }
    )
    expected["metric"] = "counts_per_cluster"
    pd.testing.assert_frame_equal(result, expected)

    # Test normalize = True
    result = utils.create_long_df_by_cluster(cell_table, "counts_per_cluster", "cluster", normalize='index')

    # first and second fovs have three total cells, third has 1
    expected.value = [1/3, 1/3, 1, 2/3, 2/3, 0]

    pd.testing.assert_frame_equal(result, expected)


