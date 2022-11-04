import pandas as pd

from ark.utils.misc_utils import verify_in_list


def create_long_df_by_cluster(cell_table, cluster_col_name, result_name, normalize=False):
    """Creates a dataframe summarizing cell clusters across FOVs in long format

    Args:
        cell_table: the dataframe containing information on each cell
        cluster_col_name: the column name in cell_table that contains the cluster information
        result_name: the name of this statistic in summarized information df
        normalize: whether to report the total or normalized counts in the result

    Returns:
        pd.DataFrame: long format dataframe containing the summarized data"""

    # create 2D summary table
    crosstab = pd.crosstab(index=cell_table['fov'], rownames=['fov'],
                           columns=cell_table[cluster_col_name], normalize=normalize)

    # convert to long format
    crosstab['fov'] = crosstab.index
    long_df = pd.melt(crosstab, id_vars=['fov'], var_name='cell_type')
    long_df['metric'] = result_name

    return long_df


def create_long_df_by_functional(func_table, cluster_col_name, drop_cols, transform_func, result_name):
    """Function to summarize functional marker data by cell type

    Args:
        func_table (pd.DataFrame): cell table containing functional markers
        cluster_col_name (str): name of the column in func_table that contains the cluster information
        drop_cols (list): list of columns to drop from func_table
        transform_func (function): function to apply to each column in func_table
        result_name (str): name of the statistic in the summarized information df

    Returns:
        pd.DataFrame: long format dataframe containing the summarized data"""

    verify_in_list(cell_type_col=cluster_col_name, cell_table_columns=func_table.columns)
    verify_in_list(drop_cols=drop_cols, cell_table_columns=func_table.columns)

    # drop columns from table
    func_table_small = func_table.loc[:, ~func_table.columns.isin(drop_cols)]

    # group by specified columns
    grouped_table = func_table_small.groupby(['fov', cluster_col_name])
    transformed = grouped_table.agg(transform_func)
    transformed.reset_index(inplace=True)

    # reshape to long df
    long_df = pd.melt(transformed, id_vars=['fov', cluster_col_name], var_name='functional_marker')
    long_df['metric'] = result_name
    long_df = long_df.rename(columns={cluster_col_name: 'cell_type'})

    return long_df