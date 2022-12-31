import numpy as np
import pandas as pd

from ark.utils.misc_utils import verify_in_list


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



