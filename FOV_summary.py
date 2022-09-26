import matplotlib.pyplot as plt

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dataset
timepoint_df = pd.read_csv(os.path.join(data_dir, 'summary_df_timepoint.csv'))
timepoint_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint.csv'))
timepoint_metadata = timepoint_metadata.loc[:, ['Tissue_ID', 'TONIC_ID', 'Timepoint']]
timepoint_df = timepoint_df.merge(timepoint_metadata, on='Tissue_ID')

plot_df = timepoint_df.loc[timepoint_df.Timepoint == 'primary', :]

# create stacked barplot
plot_cross = pd.pivot(plot_df.loc[plot_df.metric == 'cluster_broad_freq', :], index='TONIC_ID',
                     columns='cell_type', values='mean')

# order columns by prevalance
means = plot_cross.mean(axis=0).sort_values(ascending=False)
plot_cross['TONIC_ID'] = plot_cross.index
plot_cross.columns = pd.CategoricalIndex(plot_cross.columns.values, ordered=True, categories=means.index.tolist() + ['TONIC_ID'])
plot_cross = plot_cross.sort_index(axis=1)

# order rows by count
tumor_counts = plot_cross[means.index[0]].sort_values(ascending=False)
plot_cross.index = pd.CategoricalIndex(plot_cross.index.values, ordered=True, categories=tumor_counts.index.tolist())
plot_cross = plot_cross.sort_index(axis=0)

plot_cross.plot(x='TONIC_ID', kind='bar', stacked=True, figsize=(12, 5))

# reverse legend ordering
handles, labels = plt.gca().get_legend_handles_labels()
order = list(np.arange(len(plot_cross.columns) - 1))
order.reverse()
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

plt.xlabel('Patient ID', fontsize=15)
plt.ylabel('Proportion of total cells', fontsize=15)
plt.title('Frequency of broad clusters across primary tumors', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Primary_tumor_cluster_freq.png'))
plt.close()


sns.catplot(total_df.loc[total_df.metric == 'tcell_freq'], x='cell_type', y='value', hue='Tissue_ID')
