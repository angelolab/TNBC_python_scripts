# script to generate summary stats for each fov

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# load dataset
core_df = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
core_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_core.csv'))

immune_df = core_df.loc[(core_df.metric == 'cluster_broad_freq') &
                        (core_df.cell_type.isin(['mono_macs', 'b_cell', 't_cell', 'granulocyte', 'nk'])), :]
immune_grouped = immune_df.groupby('fov').agg(np.sum)
immune_grouped['fov'] = immune_grouped.index
immune_grouped['metric'] = 'immune_infiltration'

