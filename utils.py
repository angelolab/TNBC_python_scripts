
def create_long_df_by_cluster(cell_table, result_name, col_name, normalize=False):
    """Creats a dataframe summarizing cell clusters across FOVs in long format

    Args:
        cell_table: the dataframe containing information on each cell
        result_name: the name to give the column which will contain the summarized information
        col_name: the name of the column in the cell_table to summarize cell clusters from
        normalize: whether to report the total or normalized counts in the result

    Returns:
        pd.DataFrame: long format dataframe containing the summarized data"""

    # create 2D summary table
    crosstab = pd.crosstab(index=cell_table['fov'], rownames=['fov'],
                           columns=cell_table[col_name], normalize=normalize)

    # convert to long format
    crosstab['fov'] = crosstab.index
    long_df = pd.melt(crosstab, id_vars=['fov'], var_name='cell_type')
    long_df['metric'] = result_name

    return long_df


def create_long_df_by_functional(func_table, cell_type_col, drop_cols,
                                 transform_func, result_name):
    """Function to summarize functional marker data by cell type"""

    verify_in_list(cell_type_col=cell_type_col, cell_table_columns=func_table.columns)
    verify_in_list(drop_cols=drop_cols, cell_table_columns=func_table.columns)

    # drop columns from table
    func_table_small = func_table.loc[:, ~func_table.columns.isin(drop_cols)]

    # group by specified columns
    grouped_table = func_table_small.groupby(['fov', cell_type_col])
    transformed = grouped_table.agg(transform_func)
    transformed.reset_index(inplace=True)

    # reshape to long df
    long_df = pd.melt(transformed, id_vars=['fov', cell_type_col], var_name='functional_marker')
    long_df['metric'] = result_name
    long_df = long_df.rename(columns={cell_type_col: 'cell_type'})

    return long_df